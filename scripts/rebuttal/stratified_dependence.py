"""Task 6: covariate-varying dependence (synthetic, d_z = 1).

Generates the stratified Gaussian shared-frailty DGP of
``configs/rebuttal/stratified_dependence.yaml``, fits DVFM (the synthetic
study's settings, unchanged) and untuned HACSurv-2D on the same cohort and
split per repeat, and reports:

* DVFM's conditional Kendall's tau from decoder samples at reference points in
  each stratum, against the true tau of that stratum (0 or the target);
* HACSurv's single global tau, which cannot vary with x;
* DVFM latent recovery per stratum (Spearman of the posterior mean with the
  true z, sign aligned on all validation subjects, as in the paper);
* oracle IBS, C-index and MAE per stratum for both models.

DVFM training mirrors ``experiments.synthetic.run_synthetic_pilot``: seeded
loader, best post-warmup validation ELBO checkpoint, no time rescaling.
"""

from __future__ import annotations

import argparse
import time as timer
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from dvfm.model import DVFM
from dvfm.prediction import predict_survival_curves
from dvfm.training import train_dvfm
from experiments.config import expand_seed_streams, load_config
from sota.hacsurv import fit_hacsurv_2d
from utility.data import SurvivalData, SurvivalDataset
from utility.metrics import (
    compute_oracle_metrics, learned_conditional_kendall_tau, pearson_correlation, spearman_correlation,
)
from utility.runtime import resolve_device, seed_everything
from utility.splitting import split_survival_data
from utility.synthetic import generate_stratified_gaussian_shared_frailty

from rebuttal_common import PROJECT_ROOT, environment_record, write_json

STRATA = ("independent", "dependent")


def check_dvfm_matches_synthetic_study(dvfm: dict) -> None:
    reference = load_config(PROJECT_ROOT / "configs" / "synthetic.yaml")["models"]["dvfm"]
    differing = sorted(
        key for key in set(reference) | set(dvfm)
        if key not in {"latent_dims", "latent_dim"} and reference.get(key) != dvfm.get(key)
    )
    if differing:
        raise ValueError(f"DVFM settings differ from configs/synthetic.yaml in {differing}")


def fit_synthetic_dvfm(settings, train, validation, model_seed, device):
    seed_everything(int(model_seed))
    train_loader = DataLoader(
        SurvivalDataset(train.X, train.time, train.event),
        batch_size=int(settings["batch_size"]), shuffle=True,
        generator=torch.Generator().manual_seed(int(model_seed)),
    )
    validation_loader = DataLoader(
        SurvivalDataset(validation.X, validation.time, validation.event),
        batch_size=int(settings["batch_size"]), shuffle=False,
    )
    model = DVFM(
        train.X.shape[1], int(settings["latent_dim"]),
        encoder_hidden=list(settings["encoder_hidden"]), decoder_hidden=list(settings["decoder_hidden"]),
        dropout=float(settings["dropout"]), scale_link=str(settings["scale_link"]),
        latent_loading_l1=float(settings["latent_loading_l1"]),
        logvar_min=float(settings["logvar_min"]), logvar_max=float(settings["logvar_max"]),
    ).to(device)
    artifacts = train_dvfm(
        model, train_loader, validation_loader,
        n_epochs=int(settings["epochs"]), lr=float(settings["learning_rate"]),
        beta_max=float(settings["beta_max"]), warmup_epochs=int(settings["warmup_epochs"]),
        free_bits=float(settings["free_bits"]), device=device,
        checkpoint_min_epoch=int(settings["checkpoint_min_epoch"]),
        numerical_failure_threshold=float(settings["numerical_failure_threshold"]),
        weight_decay=float(settings["weight_decay"]), return_artifacts=True,
    )
    model.load_state_dict(artifacts["best_validation_elbo_state"])
    model.to(device)
    return model, artifacts, train_loader


def encode_mean(model, data, device):
    model.eval()
    with torch.no_grad():
        mu, _ = model.encoder(
            torch.as_tensor(data.X, dtype=torch.float32, device=device),
            torch.as_tensor(data.time, dtype=torch.float32, device=device),
            torch.as_tensor(data.event, dtype=torch.float32, device=device),
        )
    return mu[:, 0].cpu().numpy()


def stratum_masks(X, feature, threshold):
    dependent = np.asarray(X)[:, feature] >= threshold
    return {"independent": ~dependent, "dependent": dependent}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=PROJECT_ROOT / "configs" / "rebuttal" / "stratified_dependence.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--repeats", type=int, nargs="*")
    parser.add_argument("--models", nargs="*", default=["dvfm", "hacsurv_2d"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--hacsurv-device", default=None,
                        help="Device for HACSurv only (default: same as DVFM)")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    check_dvfm_matches_synthetic_study(cfg["models"]["dvfm"])
    data_cfg, evaluation = cfg["data"], cfg["evaluation"]
    output_dir = args.output_dir or PROJECT_ROOT / cfg["study"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device or cfg["compute"]["device"])
    hacsurv_device = resolve_device(args.hacsurv_device) if args.hacsurv_device else device
    torch.set_num_threads(int(cfg["compute"]["torch_num_threads"]))
    write_json(output_dir / "environment.json", {
        "dvfm": environment_record(device), "hacsurv": environment_record(hacsurv_device),
    })
    feature, threshold = int(data_cfg["stratify_feature"]), float(data_cfg["threshold"])
    target_tau = float(data_cfg["kendall_tau"])

    dgp_rows, tau_rows, recovery_rows, prediction_rows, manifest = [], [], [], [], []
    for repeat, seeds in enumerate(expand_seed_streams(cfg["seeds"])):
        if args.repeats and repeat not in args.repeats:
            continue
        generated = generate_stratified_gaussian_shared_frailty(
            n_samples=int(data_cfg["n_samples"]), n_features=int(data_cfg["n_features"]),
            kendall_tau=target_tau, censoring_rate=float(data_cfg["censoring_rate"]),
            dgp_seed=int(seeds["dgp"]), sampling_seed=int(seeds["sampling"]),
            calibration_samples=int(data_cfg["calibration_samples"]),
            stratify_feature=feature, threshold=threshold,
        )
        full = SurvivalData(
            generated.X, generated.observed_time, generated.event,
            [f"X{i}" for i in range(generated.X.shape[1])],
            generated.event_time, generated.censor_time, generated.true_z,
        )
        train, validation, test = split_survival_data(full, cfg["split"], int(seeds["split"]))
        grid = np.linspace(
            0.0, float(np.quantile(train.true_event_time, evaluation["grid_max_quantile"])),
            int(evaluation["n_time_points"]),
        )
        base = {"repeat": repeat, "seed": int(seeds["model"])}
        for name in STRATA:
            dgp_rows.append({
                **base, "stratum": name, "true_conditional_kendall_tau": target_tau if name == "dependent" else 0.0,
                "empirical_conditional_kendall_tau": generated.stratum_conditional_kendall_tau[name],
                "censoring_rate": generated.stratum_censoring_rate[name],
                "frailty_loading": generated.frailty_loading if name == "dependent" else 0.0,
                "overall_censoring_rate": generated.achieved_censoring_rate,
            })
        test_masks = stratum_masks(test.X, feature, threshold)
        train_masks = stratum_masks(train.X, feature, threshold)
        x_mean = np.mean(train.X, axis=0)

        def prediction_metrics(model_name, survival):
            for name, mask in {"all": np.ones(len(test.time), bool), **test_masks}.items():
                if mask.sum() < 2:
                    continue
                metrics = compute_oracle_metrics(survival[mask], grid, test.true_event_time[mask])
                prediction_rows.append({**base, "model": model_name, "stratum": name, "n": int(mask.sum()),
                                        **{key: metrics[key] for key in ("oracle_ibs", "oracle_ci", "oracle_mae")}})

        if "dvfm" in args.models:
            settings = cfg["models"]["dvfm"]
            started = timer.perf_counter()
            model, artifacts, train_loader = fit_synthetic_dvfm(
                settings, train, validation, seeds["model"], device
            )
            fit_seconds = timer.perf_counter() - started
            seed_everything(int(seeds["model"]) + 50_000)
            survival = predict_survival_curves(model, test.X, grid, train_loader, int(settings["mc_samples"]), device)
            prediction_metrics("DVFM", survival)
            for stratum in STRATA:
                stratum_x1 = train.X[train_masks[stratum], feature]
                if not len(stratum_x1):
                    continue
                for index, quantile in enumerate(evaluation["reference_quantiles"]):
                    reference = x_mean.copy()
                    reference[feature] = float(np.quantile(stratum_x1, quantile))
                    tau = learned_conditional_kendall_tau(
                        model, reference, int(evaluation["dependence_samples"]), device,
                        int(seeds["model"]) + 1000 * (1 + STRATA.index(stratum)) + index,
                    )
                    tau_rows.append({
                        **base, "model": "DVFM", "stratum": stratum, "reference_quantile": float(quantile),
                        "reference_x1": reference[feature], "learned_kendall_tau": tau,
                        "true_kendall_tau": target_tau if stratum == "dependent" else 0.0,
                    })
            validation_mu, test_mu = encode_mean(model, validation, device), encode_mean(model, test, device)
            correlation = pearson_correlation(validation_mu, validation.true_z)
            sign = -1.0 if correlation < 0 else 1.0
            for stratum, mask in {"all": np.ones(len(test.time), bool), **test_masks}.items():
                for subgroup, sub in (("All", mask), ("Event observed", mask & (test.event == 1)),
                                      ("Censored", mask & (test.event == 0))):
                    if sub.sum() < 3:
                        continue
                    recovery_rows.append({
                        **base, "model": "DVFM", "stratum": stratum, "subgroup": subgroup, "n": int(sub.sum()),
                        "alignment_sign": sign,
                        "spearman": spearman_correlation(sign * test_mu[sub], test.true_z[sub]),
                    })
            manifest.append({**base, "model": "DVFM", "fit_seconds": fit_seconds,
                             "best_validation_elbo_epoch": artifacts["best_validation_elbo_epoch"],
                             "latent_loading_l1_magnitude": float(model.decoder.latent_loading_l1().detach().cpu())})
            print(f"repeat {repeat} DVFM done in {fit_seconds:.0f}s", flush=True)

        if "hacsurv_2d" in args.models:
            settings = cfg["models"]["hacsurv_2d"]
            seed_everything(int(seeds["model"]))
            started = timer.perf_counter()
            survival, info, _, _ = fit_hacsurv_2d(
                train.X, train.time, train.event, validation.X, validation.time, validation.event,
                test.X, grid, seed=int(seeds["model"]), device=hacsurv_device, **settings,
            )
            fit_seconds = timer.perf_counter() - started
            prediction_metrics("HACSurv (untuned)", survival)
            for stratum in STRATA:
                tau_rows.append({
                    **base, "model": "HACSurv (untuned)", "stratum": stratum, "reference_quantile": np.nan,
                    "reference_x1": np.nan, "learned_kendall_tau": info["learned_conditional_kendall_tau"],
                    "true_kendall_tau": target_tau if stratum == "dependent" else 0.0,
                })
            manifest.append({**base, "model": "HACSurv (untuned)", "fit_seconds": fit_seconds,
                             "checkpoint_epoch": info["checkpoint_epoch"], "epochs_completed": info["epochs_completed"],
                             "device": str(hacsurv_device)})
            print(f"repeat {repeat} HACSurv done in {fit_seconds:.0f}s, tau={info['learned_conditional_kendall_tau']:.3f}",
                  flush=True)

        # Rewritten after every repeat so an interrupted run keeps what finished.
        for filename, rows in (("dgp_rows.csv", dgp_rows), ("tau_rows.csv", tau_rows),
                               ("recovery_rows.csv", recovery_rows), ("prediction_rows.csv", prediction_rows),
                               ("run_manifest.csv", manifest)):
            pd.DataFrame(rows).to_csv(output_dir / filename, index=False)
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
