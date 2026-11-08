"""Task 6: covariate-varying dependence (synthetic, d_z = 1).

Generates a Gaussian shared-frailty DGP whose loading depends on x1, either
as a step (``configs/rebuttal/stratified_dependence.yaml``: 0 below the
threshold, calibrated to the target tau above) or smoothly
(``configs/rebuttal/smooth_dependence.yaml``: lambda_max * sigmoid(slope * x1)).
It fits DVFM (the synthetic study's settings, unchanged) and untuned
HACSurv-2D on the same cohort and split per repeat, and reports:

* DVFM's conditional Kendall's tau from decoder samples at reference points
  (within each stratum, or at fixed x1 values for the smooth profile) against
  the true conditional tau there;
* HACSurv's single global tau, which cannot vary with x;
* DVFM latent recovery per stratum or x1 bin (Spearman of the posterior mean
  with the true z, sign aligned on all validation subjects, as in the paper);
* oracle IBS, C-index and MAE per stratum or bin for both models; DVFM's are
  given with z from the aggregate posterior and from the prior
  (``prediction_mode``).

Because the paper computes tau with z from the N(0, 1) prior, DVFM's tau is
also reported with z from the aggregate training posterior and from the
posterior of training subjects in the reference point's stratum or bin, with
per-group posterior summaries against x1. ``--dvfm-variant`` refits DVFM with
one of the config's ``diagnostic_variants``: diagnostics of why tau is or is
not recovered, never a replacement for the paper's settings. Held-out joint
IWAE and the checkpoint's validation ELBO are recorded so variants with
different tau can be compared on how well they fit the observed data.

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

from dvfm.likelihood import encode_posterior, heldout_log_likelihood
from dvfm.model import DVFM
from dvfm.prediction import predict_survival_curves, predict_survival_from_prior
from dvfm.training import train_dvfm
from experiments.config import expand_seed_streams, load_config
from sota.hacsurv import fit_hacsurv_2d
from utility.data import SurvivalData, SurvivalDataset
from utility.metrics import (
    compute_oracle_metrics, learned_conditional_kendall_tau, pearson_correlation, spearman_correlation,
)
from utility.runtime import resolve_device, seed_everything
from utility.splitting import split_survival_data
from scipy.stats import kendalltau

from utility.synthetic import conditional_kendall_tau_for_loading, generate_stratified_gaussian_shared_frailty

from rebuttal_common import PROJECT_ROOT, environment_record, write_json


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
        decoder_architecture=str(settings.get("decoder_architecture", "single")),
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


def group_masks(X, feature, data_cfg, evaluation):
    """Strata (step profile) or x1 bins centred on the reference points (sigmoid)."""
    x1 = np.asarray(X)[:, feature]
    if data_cfg.get("loading_profile", "step") == "step":
        dependent = x1 >= float(data_cfg["threshold"])
        return {"independent": ~dependent, "dependent": dependent}
    points = sorted(float(value) for value in evaluation["reference_x1"])
    edges = [-np.inf, *((a + b) / 2 for a, b in zip(points, points[1:])), np.inf]
    return {f"x1~{point:g}": (x1 >= low) & (x1 < high) for point, low, high in zip(points, edges, edges[1:])}


def reference_points(train_X, feature, data_cfg, evaluation, loading_max, target_tau):
    """(group, x1, true conditional tau) at which DVFM's tau is evaluated."""
    if data_cfg.get("loading_profile", "step") == "step":
        masks = group_masks(train_X, feature, data_cfg, evaluation)
        return [
            (name, float(np.quantile(train_X[mask, feature], quantile)),
             target_tau if name == "dependent" else 0.0)
            for name, mask in masks.items() if mask.any()
            for quantile in evaluation["reference_quantiles"]
        ]
    slope, threshold = float(data_cfg["sigmoid_slope"]), float(data_cfg["threshold"])
    return [
        (f"x1~{float(x1):g}", float(x1), conditional_kendall_tau_for_loading(
            loading_max / (1.0 + np.exp(-slope * (float(x1) - threshold))), seed=0,
            n_samples=int(evaluation.get("true_tau_samples", 200_000)),
        ))
        for x1 in evaluation["reference_x1"]
    ]


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
    parser.add_argument("--dvfm-variant", default="paper",
                        help="paper, or a name under diagnostic_variants in the config")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    check_dvfm_matches_synthetic_study(cfg["models"]["dvfm"])
    if args.dvfm_variant != "paper":
        cfg["models"]["dvfm"] = {**cfg["models"]["dvfm"], **cfg["diagnostic_variants"][args.dvfm_variant]}
    dvfm_label = "DVFM" if args.dvfm_variant == "paper" else f"DVFM [{args.dvfm_variant}]"
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

    dgp_rows, tau_rows, recovery_rows, prediction_rows, posterior_rows, manifest = [], [], [], [], [], []
    for repeat, seeds in enumerate(expand_seed_streams(cfg["seeds"])):
        if args.repeats and repeat not in args.repeats:
            continue
        generated = generate_stratified_gaussian_shared_frailty(
            n_samples=int(data_cfg["n_samples"]), n_features=int(data_cfg["n_features"]),
            kendall_tau=target_tau, censoring_rate=float(data_cfg["censoring_rate"]),
            dgp_seed=int(seeds["dgp"]), sampling_seed=int(seeds["sampling"]),
            calibration_samples=int(data_cfg["calibration_samples"]),
            stratify_feature=feature, threshold=threshold,
            loading_profile=str(data_cfg.get("loading_profile", "step")),
            sigmoid_slope=float(data_cfg.get("sigmoid_slope", 2.0)),
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
        base = {"repeat": repeat, "seed": int(seeds["model"]), "dvfm_variant": args.dvfm_variant,
                "loading_profile": str(data_cfg.get("loading_profile", "step"))}
        event_residual = np.log(generated.event_time) - generated.X @ generated.beta_event
        censor_residual = (np.log(generated.censor_time) - generated.X @ generated.beta_censor
                           - generated.censor_intercept)
        for name, mask in group_masks(generated.X, feature, data_cfg, evaluation).items():
            dgp_rows.append({
                **base, "stratum": name, "n": int(mask.sum()),
                "mean_frailty_loading": float(generated.subject_loading[mask].mean()),
                "true_conditional_kendall_tau_at_mean_loading": conditional_kendall_tau_for_loading(
                    float(generated.subject_loading[mask].mean()), seed=0),
                "empirical_conditional_kendall_tau": float(
                    kendalltau(event_residual[mask], censor_residual[mask]).statistic),
                "censoring_rate": float(1.0 - generated.event[mask].mean()),
                "frailty_loading_max": generated.frailty_loading,
                "overall_censoring_rate": generated.achieved_censoring_rate,
            })
        test_masks = group_masks(test.X, feature, data_cfg, evaluation)
        train_masks = group_masks(train.X, feature, data_cfg, evaluation)
        points = reference_points(train.X, feature, data_cfg, evaluation, generated.frailty_loading, target_tau)
        x_mean = np.mean(train.X, axis=0)

        def prediction_metrics(model_name, survival, mode=None):
            for name, mask in {"all": np.ones(len(test.time), bool), **test_masks}.items():
                if mask.sum() < 2:
                    continue
                metrics = compute_oracle_metrics(survival[mask], grid, test.true_event_time[mask])
                prediction_rows.append({**base, "model": model_name, "prediction_mode": mode, "stratum": name,
                                        "n": int(mask.sum()),
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
            prediction_metrics(dvfm_label, survival, "aggregate_posterior")
            batch = int(settings["batch_size"])
            train_mu, train_std = encode_posterior(model, train.X, train.time, train.event, batch_size=batch, device=device)
            samples = int(evaluation["dependence_samples"])
            for index, (group, x1, true_tau) in enumerate(points):
                reference = x_mean.copy()
                reference[feature] = x1
                tau_seed = int(seeds["model"]) + 1000 + index
                local = train_masks[group]
                tau_rows.append({
                    **base, "model": dvfm_label, "stratum": group, "reference_x1": x1, "true_kendall_tau": true_tau,
                    # The paper's estimator: z from the N(0, 1) prior.
                    "learned_kendall_tau": learned_conditional_kendall_tau(model, reference, samples, device, tau_seed),
                    "learned_kendall_tau_aggregate_posterior": learned_conditional_kendall_tau(
                        model, reference, samples, device, tau_seed, mixture=(train_mu, train_std)),
                    "learned_kendall_tau_local_posterior": learned_conditional_kendall_tau(
                        model, reference, samples, device, tau_seed, mixture=(train_mu[local], train_std[local])),
                })
            x1_train = train.X[:, feature]
            for group, mask in {"all": np.ones(len(train.time), bool), **train_masks}.items():
                for dim in range(train_mu.shape[1]):
                    mu, std = train_mu[mask, dim], train_std[mask, dim]
                    posterior_rows.append({
                        **base, "model": dvfm_label, "split": "train", "stratum": group, "latent_dimension": dim,
                        "n": int(mask.sum()), "mean_posterior_mean": float(mu.mean()),
                        "sd_posterior_mean": float(mu.std()), "mean_posterior_sd": float(std.mean()),
                        "mean_kl_to_prior": float(np.mean(-0.5 * (1 + 2 * np.log(std) - mu ** 2 - std ** 2))),
                        "spearman_posterior_mean_x1": spearman_correlation(mu, x1_train[mask]),
                        "spearman_posterior_mean_true_z": spearman_correlation(mu, train.true_z[mask]),
                    })
            validation_mu, _ = encode_posterior(model, validation.X, validation.time, validation.event,
                                                batch_size=batch, device=device)
            test_mu, _ = encode_posterior(model, test.X, test.time, test.event, batch_size=batch, device=device)
            # As the paper's diagnostics: the dimension and sign best aligned on validation.
            correlations = np.asarray([pearson_correlation(validation_mu[:, dim], validation.true_z)
                                       for dim in range(validation_mu.shape[1])])
            # A fully collapsed encoder gives only NaN correlations; recovery is then NaN.
            selected = int(np.nanargmax(np.abs(correlations))) if np.isfinite(correlations).any() else 0
            sign = -1.0 if correlations[selected] < 0 else 1.0
            for stratum, mask in {"all": np.ones(len(test.time), bool), **test_masks}.items():
                for subgroup, sub in (("All", mask), ("Event observed", mask & (test.event == 1)),
                                      ("Censored", mask & (test.event == 0))):
                    if sub.sum() < 3:
                        continue
                    recovery_rows.append({
                        **base, "model": dvfm_label, "stratum": stratum, "subgroup": subgroup, "n": int(sub.sum()),
                        "selected_latent_dimension": selected, "alignment_sign": sign,
                        "spearman": spearman_correlation(sign * test_mu[sub, selected], test.true_z[sub]),
                    })
            likelihood = heldout_log_likelihood(
                model, test.X, test.time, test.event, mixing_mu=train_mu, mixing_std=train_std,
                n_samples=int(evaluation.get("likelihood_samples", 200)), seed=int(seeds["model"]), device=device,
            )
            manifest.append({**base, "model": dvfm_label, "fit_seconds": fit_seconds,
                             "latent_dim": int(settings["latent_dim"]),
                             "best_validation_elbo_epoch": artifacts["best_validation_elbo_epoch"],
                             "best_validation_negative_elbo": artifacts["best_validation_elbo"],
                             "test_joint_iwae": float(np.mean(likelihood["joint_iwae"])),
                             "test_joint_elbo": float(np.mean(likelihood["joint_elbo"])),
                             "test_event_margin_loglik": float(np.mean(likelihood["event_margin"])),
                             "latent_loading_l1_magnitude": float(model.decoder.latent_loading_l1().detach().cpu())})
            # Prior-sampled curves last, so no earlier output's random stream moves.
            seed_everything(int(seeds["model"]) + 60_000)
            prediction_metrics(dvfm_label, predict_survival_from_prior(
                model, test.X, grid, int(settings["mc_samples"]), device), "prior")
            print(f"repeat {repeat} {dvfm_label} done in {fit_seconds:.0f}s", flush=True)

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
            # One global generator: the same tau at every reference point.
            for group, x1, true_tau in points:
                tau_rows.append({
                    **base, "model": "HACSurv (untuned)", "stratum": group, "reference_x1": x1,
                    "learned_kendall_tau": info["learned_conditional_kendall_tau"], "true_kendall_tau": true_tau,
                })
            manifest.append({**base, "model": "HACSurv (untuned)", "fit_seconds": fit_seconds,
                             "checkpoint_epoch": info["checkpoint_epoch"], "epochs_completed": info["epochs_completed"],
                             "device": str(hacsurv_device)})
            print(f"repeat {repeat} HACSurv done in {fit_seconds:.0f}s, tau={info['learned_conditional_kendall_tau']:.3f}",
                  flush=True)

        # Rewritten after every repeat so an interrupted run keeps what finished.
        for filename, rows in (("dgp_rows.csv", dgp_rows), ("tau_rows.csv", tau_rows),
                               ("recovery_rows.csv", recovery_rows), ("prediction_rows.csv", prediction_rows),
                               ("posterior_rows.csv", posterior_rows),
                               ("run_manifest.csv", manifest)):
            pd.DataFrame(rows).to_csv(output_dir / filename, index=False)
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
