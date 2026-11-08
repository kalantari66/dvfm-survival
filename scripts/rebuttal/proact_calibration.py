"""Task 4: D- and KM-calibration of the PRO-ACT latent ablation (d_z = 0 and 1).

Refits DVFM on the exact splits, seeds and settings of
``configs/proact_latent_ablation.yaml`` (no setting is changed) and scores the
predictions under two mixing distributions for z: the aggregate posterior
(the paper's predictor) and the N(0, 1) prior. Both metrics assume censoring
independent of death given covariates, which is the assumption the paper
questions, so they are descriptive only.

D-calibration uses each subject's predicted survival at their own observed
time, S(t_i | x_i), computed exactly from the Weibull decoder, not
interpolated from the reported grid (which stops at the 0.9 quantile of
training death times). KM-calibration compares the mean predicted curve with
the test-set Kaplan--Meier curve over the full follow-up. Each row also carries
the C-index and IBS-IPCW of the curves and the paper's held-out margin
likelihoods under that mixing distribution. The aggregate-posterior path runs
first and unchanged, so its rows can be checked against the paper run; the
prior uses the same random streams.
"""

from __future__ import annotations

import argparse
import time as timer
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from SurvivalEVAL.Evaluations.DistributionCalibration import d_calibration, km_calibration

from dvfm.likelihood import encode_posterior, heldout_log_likelihood
from dvfm.prediction import get_median_survival_time, predict_survival_curves, predict_survival_from_mixture
from experiments.config import expand_seed_streams, load_config, validate_config
from experiments.real_latent_ablation import _encode_split, apply_missingness_filter, load_real_cohort
from experiments.runner import (
    _models_for_dataset, evaluation_time_grid, fit_dvfm, scale_observed_time, training_time_scale,
)
from utility.data import SurvivalData
from utility.metrics import collect_metrics
from utility.runtime import resolve_device, seed_everything
from utility.splitting import time_event_stratified_split_indices

from rebuttal_common import PROJECT_ROOT, REBUTTAL_ROOT, environment_record, write_json

N_BINS = 10


def survival_at_observed_times(model, X, time, mixture_mu, mixture_std, n_samples, seed, device):
    """Monte Carlo S(t_i | x_i) with z from the aggregate posterior, per subject."""
    generator = torch.Generator(device=torch.device(device).type).manual_seed(int(seed))
    mu = torch.as_tensor(mixture_mu, dtype=torch.float32, device=device)
    std = torch.as_tensor(mixture_std, dtype=torch.float32, device=device)
    x = torch.as_tensor(X, dtype=torch.float32, device=device)
    t = torch.as_tensor(time, dtype=torch.float32, device=device)
    total = torch.zeros(len(x), device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(int(n_samples)):
            index = torch.randint(0, len(mu), (len(x),), device=device, generator=generator)
            noise = torch.randn((len(x), mu.shape[1]), device=device, generator=generator)
            shape, scale, _, _ = model.decoder(x, mu[index] + std[index] * noise)
            total += torch.exp(-(t / scale).pow(shape))
    return (total / int(n_samples)).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "proact_latent_ablation.yaml")
    parser.add_argument("--paper-results", type=Path,
                        default=PROJECT_ROOT / "results" / "proact-latent-ablation" / "results_raw.csv")
    parser.add_argument("--output-dir", type=Path, default=REBUTTAL_ROOT / "task4_proact_calibration")
    parser.add_argument("--repeats", type=int, nargs="*", help="Subset of repeat indices (default: all)")
    parser.add_argument("--device", default=None, help="Override compute.device")
    args = parser.parse_args()

    cfg = load_config(args.config)
    validate_config(cfg)
    eval_cfg = cfg["evaluation"]
    device = resolve_device(args.device or str(cfg["compute"].get("device", "auto")))
    torch.set_num_threads(max(1, int(cfg["compute"].get("torch_num_threads", 1))))
    calibration_samples = int(eval_cfg["likelihood_samples"])
    horizon = eval_cfg.get("likelihood_horizon")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "environment.json", environment_record(device))

    (spec,) = cfg["data"]["datasets"]
    frame, time, event = load_real_cohort(spec)
    spec, _ = apply_missingness_filter(frame, spec)
    cohort = SurvivalData(np.zeros((len(time), 0)), time, event, [])
    model_cfg = _models_for_dataset(cfg, spec["name"])
    latent_dims = sorted(int(dim) for dim in cfg["models"]["dvfm"]["latent_dims"])
    rows, histograms = [], []
    for repeat, seeds in enumerate(expand_seed_streams(cfg["seeds"])):
        if args.repeats and repeat not in args.repeats:
            continue
        split_idx = time_event_stratified_split_indices(cohort, cfg["split"], int(seeds["split"]))
        (x_train, x_validation, x_test), names = _encode_split(frame, spec, *split_idx)
        train, validation, test = (
            SurvivalData(x, time[idx], event[idx], names)
            for x, idx in zip((x_train, x_validation, x_test), split_idx)
        )
        time_points = evaluation_time_grid(train, eval_cfg)
        time_scale = training_time_scale(train)
        model_train, model_validation, model_test = (
            scale_observed_time(part, time_scale) for part in (train, validation, test)
        )
        full_grid = np.linspace(0.0, float(test.time.max()), 400)
        for latent_dim in latent_dims:
            c = {**model_cfg["dvfm"], "latent_dim": latent_dim}
            # Same seeding point as run_real_latent_ablation: immediately before the fit.
            seed_everything(int(seeds["model"]))
            started = timer.perf_counter()
            model, artifacts, train_loader, checkpoint = fit_dvfm(model_train, model_validation, c, device)
            fit_seconds = timer.perf_counter() - started
            # Reported curves, exactly as the paper run computes them.
            survival = predict_survival_curves(
                model, model_test.X, time_points / time_scale, train_loader,
                n_samples=int(c["mc_samples"]), device=device,
            )
            median = get_median_survival_time(survival, time_points / time_scale) * time_scale
            reported = collect_metrics(
                "DVFM", median, survival, test.time, test.event, None, time_points,
                t_train=train.time, e_train=train.event,
            )
            mixture_mu, mixture_std = encode_posterior(
                model, model_train.X, model_train.time, model_train.event,
                batch_size=int(c["batch_size"]), device=device,
            )
            # A single N(0, 1) component turns every mixture-based estimator into
            # its prior-based counterpart.
            mixtures = {
                "aggregate_posterior": (mixture_mu, mixture_std),
                "prior": (np.zeros((1, latent_dim)), np.ones((1, latent_dim))),
            }
            for mode, (mix_mu, mix_std) in mixtures.items():
                if mode == "prior":
                    torch.manual_seed(int(seeds["model"]) + 60_000)
                    survival = predict_survival_from_mixture(
                        model, model_test.X, time_points / time_scale, mix_mu, mix_std,
                        n_samples=int(c["mc_samples"]), device=device,
                    )
                    median = get_median_survival_time(survival, time_points / time_scale) * time_scale
                    reported = collect_metrics(
                        "DVFM", median, survival, test.time, test.event, None, time_points,
                        t_train=train.time, e_train=train.event,
                    )
                probabilities = survival_at_observed_times(
                    model, model_test.X, model_test.time, mix_mu, mix_std,
                    calibration_samples, int(seeds["model"]) + 1, device,
                )
                statistic, p_value, histogram = d_calibration(probabilities, test.event.astype(int), N_BINS)
                # Every fit reseeds first, so reseeding here cannot leak into later fits.
                torch.manual_seed(int(seeds["model"]) + 2)
                mean_curve = predict_survival_from_mixture(
                    model, model_test.X, full_grid / time_scale, mix_mu, mix_std,
                    n_samples=calibration_samples, device=device,
                ).mean(axis=0)
                km_score = km_calibration(mean_curve, full_grid, test.time, test.event.astype(int))
                # As run_real_latent_ablation reports them (same seed and horizon).
                likelihood = heldout_log_likelihood(
                    model, model_test.X, model_test.time, model_test.event,
                    mixing_mu=mix_mu, mixing_std=mix_std, n_samples=calibration_samples,
                    time_scale=time_scale, horizon=None if horizon is None else float(horizon) / time_scale,
                    seed=int(seeds["model"]), device=device,
                )
                expected = len(test.time) / N_BINS
                spread = (mix_mu ** 2 + mix_std ** 2).mean() - mix_mu.mean() ** 2 if latent_dim else np.nan
                rows.append({
                    "repeat": repeat, "latent_dim": latent_dim, "prediction_mode": mode,
                    "model_seed": int(seeds["model"]), "checkpoint": checkpoint,
                    "best_validation_elbo_epoch": artifacts["best_validation_elbo_epoch"],
                    "n_test": len(test.time), "test_event_rate": float(np.mean(test.event)),
                    "d_cal_statistic": float(statistic), "d_cal_p_value": float(p_value),
                    "d_cal_max_abs_bin_deviation": float(np.max(np.abs(histogram - expected)) / len(test.time)),
                    "km_calibration": float(km_score),
                    "refit_c_index": reported["DVFM C-Idx"], "refit_ibs_ipcw": reported["DVFM IBS IPCW"],
                    "test_joint_iwae": float(np.mean(likelihood["joint_iwae"])),
                    "test_event_margin_loglik": float(np.mean(likelihood["event_margin"])),
                    "test_censor_margin_loglik": float(np.mean(likelihood["censor_margin"])),
                    "mixture_mean": float(mix_mu.mean()) if latent_dim else np.nan,
                    "mixture_sd": float(np.sqrt(spread)) if latent_dim else np.nan,
                    "fit_seconds": fit_seconds,
                })
                histograms.extend({
                    "repeat": repeat, "latent_dim": latent_dim, "prediction_mode": mode, "bin": index,
                    "survival_probability_upper": 1.0 - index / N_BINS,
                    "survival_probability_lower": 1.0 - (index + 1) / N_BINS,
                    "count": float(count), "expected": expected,
                } for index, count in enumerate(histogram))
                print(f"repeat {repeat} d_z={latent_dim} {mode}: D-cal p={p_value:.3g}, KM-cal={km_score:.4f}, "
                      f"C-index={reported['DVFM C-Idx']:.4f} ({fit_seconds:.0f}s)", flush=True)

    results = pd.DataFrame(rows)
    pd.DataFrame(histograms).to_csv(args.output_dir / "d_calibration_histograms.csv", index=False)
    if args.paper_results.exists():
        paper = pd.read_csv(args.paper_results)[[
            "Repeat", "latent_dim", "C-Idx", "IBS IPCW", "best_validation_elbo_epoch",
            "test_joint_iwae", "test_event_margin_loglik", "test_censor_margin_loglik",
        ]]
        paper = paper.rename(columns={
            "Repeat": "repeat", "C-Idx": "paper_c_index", "IBS IPCW": "paper_ibs_ipcw",
            "best_validation_elbo_epoch": "paper_best_validation_elbo_epoch",
            "test_joint_iwae": "paper_test_joint_iwae",
            "test_event_margin_loglik": "paper_test_event_margin_loglik",
            "test_censor_margin_loglik": "paper_test_censor_margin_loglik",
        })
        # The paper ran the aggregate posterior only, so only those rows are checked.
        paper["prediction_mode"] = "aggregate_posterior"
        results = results.merge(paper, on=["repeat", "latent_dim", "prediction_mode"], how="left",
                                validate="one_to_one")
        results["c_index_refit_minus_paper"] = results["refit_c_index"] - results["paper_c_index"]
        results["ibs_ipcw_refit_minus_paper"] = results["refit_ibs_ipcw"] - results["paper_ibs_ipcw"]
        results["event_margin_refit_minus_paper"] = (
            results["test_event_margin_loglik"] - results["paper_test_event_margin_loglik"])
    else:
        print(f"Paper results not found at {args.paper_results}; reproduction check skipped.")
    results.to_csv(args.output_dir / "calibration_rows.csv", index=False)
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
