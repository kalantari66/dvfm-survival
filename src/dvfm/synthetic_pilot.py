"""Controlled shared-Gaussian-frailty pilot with paired model comparisons."""

from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lifelines.utils import concordance_index
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .baselines import (
    _fit_cox_and_predict_survival,
    fit_clayton_weibull_aft,
    train_deepsurv,
    train_mtlr,
)
from .config import expand_scenarios
from .data import SurvivalData
from .metrics import compute_oracle_brier_ibs
from .model import DVFM, SurvivalDataset
from .prediction import (
    get_median_survival_time,
    learned_conditional_kendall_tau,
    predict_survival_curves,
    predict_survival_from_prior,
)
from .synthetic import generate_gaussian_shared_frailty
from .training import train_dvfm


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _subset(data: SurvivalData, index: np.ndarray) -> SurvivalData:
    return SurvivalData(
        X=data.X[index].copy(), time=data.time[index].copy(),
        event=data.event[index].copy(), feature_names=list(data.feature_names),
        true_event_time=data.true_event_time[index].copy(),
        true_censor_time=data.true_censor_time[index].copy(),
        true_z=data.true_z[index].copy(),
    )


def _split(data: SurvivalData, cfg: dict, seed: int):
    indices = np.arange(len(data.time))
    train_validation, test = train_test_split(
        indices, test_size=float(cfg["test_fraction"]), random_state=seed,
        stratify=data.event,
    )
    relative_validation = float(cfg["validation_fraction"]) / (
        1.0 - float(cfg["test_fraction"])
    )
    train, validation = train_test_split(
        train_validation, test_size=relative_validation,
        random_state=seed + 1, stratify=data.event[train_validation],
    )
    return _subset(data, train), _subset(data, validation), _subset(data, test)


def _safe_correlation(function, x, y) -> float:
    x, y = np.asarray(x).reshape(-1), np.asarray(y).reshape(-1)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(function(x, y).statistic)


def _encode(model, data: SurvivalData, batch_size: int, device: torch.device):
    if model.encoder is None:
        return np.empty((len(data.time), 0)), np.empty((len(data.time), 0))
    loader = DataLoader(
        SurvivalDataset(data.X, data.time, data.event),
        batch_size=batch_size, shuffle=False,
    )
    mus, logvars = [], []
    model.eval()
    with torch.no_grad():
        for x, observed_time, event in loader:
            mu, logvar = model.encoder(
                x.to(device), observed_time.to(device), event.to(device)
            )
            mus.append(mu.cpu().numpy())
            logvars.append(logvar.cpu().numpy())
    return np.concatenate(mus), np.concatenate(logvars)


def _frailty_diagnostics(
    model, validation, test, batch_size, active_threshold, target_tau, device
):
    validation_mu, validation_logvar = _encode(model, validation, batch_size, device)
    test_mu, _ = _encode(model, test, batch_size, device)
    masks = {
        "all": np.ones(len(test.time), dtype=bool),
        "event_observed": test.event == 1,
        "censored": test.event == 0,
    }
    if validation_mu.shape[1] == 0:
        return [{
            "subgroup": subgroup, "n": int(mask.sum()),
            "frailty_recovery_defined": False,
            "selected_latent_dimension": np.nan,
            "active_latent_dimensions": 0, "mean_kl": 0.0,
            "frailty_pearson": np.nan, "frailty_spearman": np.nan,
            "frailty_r2_calibrated": np.nan, "frailty_rmse_calibrated": np.nan,
        } for subgroup, mask in masks.items()]

    per_dimension_kl = np.mean(
        -0.5 * (1 + validation_logvar - validation_mu ** 2 - np.exp(validation_logvar)),
        axis=0,
    )
    recovery_defined = float(target_tau) > 0.0
    correlations = np.asarray([
        _safe_correlation(pearsonr, validation_mu[:, j], validation.true_z)
        for j in range(validation_mu.shape[1])
    ])
    selected = int(np.nanargmax(np.abs(correlations))) if recovery_defined else 0
    raw_pearson = correlations[selected] if recovery_defined else float("nan")
    sign = 1.0 if not np.isfinite(raw_pearson) or raw_pearson >= 0 else -1.0
    if recovery_defined:
        validation_aligned = sign * validation_mu[:, selected]
        test_aligned = sign * test_mu[:, selected]
        calibrator = LinearRegression().fit(
            validation_aligned.reshape(-1, 1), validation.true_z
        )
        test_calibrated = calibrator.predict(test_aligned.reshape(-1, 1))
    else:
        test_aligned = np.full(len(test.time), np.nan)
        test_calibrated = np.full(len(test.time), np.nan)
        calibrator = None

    rows = []
    for subgroup, mask in masks.items():
        truth, estimate = test.true_z[mask], test_calibrated[mask]
        rows.append({
            "subgroup": subgroup, "n": int(mask.sum()),
            "frailty_recovery_defined": recovery_defined,
            "selected_latent_dimension": selected,
            "active_latent_dimensions": int(np.sum(per_dimension_kl > active_threshold)),
            "mean_kl": float(np.sum(per_dimension_kl)),
            "frailty_pearson": _safe_correlation(pearsonr, test_aligned[mask], truth) if recovery_defined else np.nan,
            "frailty_spearman": _safe_correlation(spearmanr, test_aligned[mask], truth) if recovery_defined else np.nan,
            "frailty_r2_calibrated": float(r2_score(truth, estimate)) if recovery_defined else np.nan,
            "frailty_rmse_calibrated": float(mean_squared_error(truth, estimate) ** 0.5) if recovery_defined else np.nan,
            "validation_alignment_sign": sign if recovery_defined else np.nan,
            "validation_calibration_intercept": float(calibrator.intercept_) if recovery_defined else np.nan,
            "validation_calibration_slope": float(calibrator.coef_[0]) if recovery_defined else np.nan,
        })
    return rows


def _prediction_row(survival, grid, test, context):
    survival = np.asarray(survival, dtype=float)
    if not np.all(np.isfinite(survival)):
        raise FloatingPointError("Survival predictions contain NaN or infinity")
    if np.any(survival < -1e-6) or np.any(survival > 1.0 + 1e-6):
        raise FloatingPointError("Survival predictions fall outside [0, 1]")
    median = get_median_survival_time(survival, grid)
    _, oracle_ibs = compute_oracle_brier_ibs(survival, grid, test.true_event_time)
    censored, observed = test.event == 0, test.event == 1
    return {
        **context, "oracle_ibs": oracle_ibs,
        "oracle_ci": float(concordance_index(test.true_event_time, median)),
        "oracle_mae": float(np.mean(np.abs(test.true_event_time - median))),
        "oracle_mae_censored": float(np.mean(np.abs(test.true_event_time[censored] - median[censored]))),
        "oracle_mae_uncensored": float(np.mean(np.abs(test.true_event_time[observed] - median[observed]))),
    }


def _calibration_rows(survival, grid, truth, context):
    return [{
        **context, "time": float(grid[index]),
        "mean_predicted_survival": float(survival[:, index].mean()),
        "empirical_oracle_survival": float(np.mean(truth > grid[index])),
    } for index in np.linspace(0, len(grid) - 1, 10, dtype=int)]


def _fit_baseline(name, train, test, grid, cfg, device):
    if name == "coxph":
        _, survival = _fit_cox_and_predict_survival(
            train.X, train.time, train.event, test.X, grid
        )
    elif name == "deepsurv":
        settings = cfg["models"]["deepsurv"]
        _, _, survival = train_deepsurv(
            train.X, train.time, train.event, test.X,
            n_epochs=int(settings["epochs"]), batch_size=int(settings["batch_size"]),
            lr=float(settings["learning_rate"]), device=device, eval_time_points=grid,
        )
    elif name == "mtlr":
        settings = cfg["models"]["mtlr"]
        _, _, survival = train_mtlr(
            train.X, train.time, train.event, test.X,
            num_bins=int(settings["bins"]), n_epochs=int(settings["epochs"]),
            lr=float(settings["learning_rate"]), device=device, eval_time_points=grid,
        )
    elif name == "clayton_aft":
        settings = cfg["models"]["clayton_aft"]
        _, survival = fit_clayton_weibull_aft(
            train.X, train.time, train.event, test.X, grid,
            epochs=int(settings["epochs"]), lr=float(settings["learning_rate"]),
            device=device,
        )
    else:
        raise ValueError(f"Unsupported synthetic baseline: {name}")
    return survival


def run_synthetic_pilot(cfg: dict, out_dir: Path, device: torch.device) -> pd.DataFrame:
    """Run the size x tau x censoring x latent-dimension pilot."""
    data_cfg, model_cfg, evaluation = cfg["data"], cfg["models"], cfg["evaluation"]
    seed_cfg = cfg["seeds"]
    rows, histories, diagnostics, calibration, manifest = [], [], [], [], []
    for scenario in expand_scenarios(data_cfg):
        n_samples = int(scenario.get("n_samples", data_cfg.get("n_samples")))
        for repeat, (sampling_seed, split_seed, model_seed) in enumerate(zip(
            seed_cfg["sampling"], seed_cfg["split"], seed_cfg["model"]
        )):
            generated = generate_gaussian_shared_frailty(
                n_samples=n_samples, n_features=int(data_cfg["n_features"]),
                kendall_tau=float(scenario["kendall_tau"]),
                censoring_rate=float(scenario["censoring_rate"]),
                dgp_seed=int(seed_cfg["dgp"]), sampling_seed=int(sampling_seed),
                calibration_samples=int(data_cfg.get("calibration_samples", 50_000)),
            )
            full = SurvivalData(
                generated.X, generated.observed_time, generated.event,
                [f"X{i}" for i in range(generated.X.shape[1])],
                generated.event_time, generated.censor_time, generated.true_z,
            )
            train, validation, test = _split(full, cfg["split"], int(split_seed))
            grid = np.linspace(
                0.0,
                float(np.quantile(train.true_event_time, evaluation["grid_max_quantile"])),
                int(evaluation["n_time_points"]),
            )
            base = {
                "study": cfg["study"]["name"],
                "scenario": f"n-{n_samples}-kendall_tau-{scenario['kendall_tau']}-censoring_rate-{scenario['censoring_rate']}",
                "n_samples": n_samples,
                "target_kendall_tau": float(scenario["kendall_tau"]),
                "empirical_conditional_kendall_tau": generated.empirical_conditional_kendall_tau,
                "empirical_marginal_kendall_tau": generated.empirical_marginal_kendall_tau,
                "target_censoring_rate": float(scenario["censoring_rate"]),
                "achieved_censoring_rate": generated.achieved_censoring_rate,
                "repeat": repeat, "dgp_seed": int(seed_cfg["dgp"]),
                "sampling_seed": int(sampling_seed), "split_seed": int(split_seed),
                "model_seed": int(model_seed),
            }

            for baseline in [str(name).lower() for name in model_cfg["enabled"] if str(name).lower() != "dvfm"]:
                started = time.perf_counter()
                fit_context = {**base, "model": baseline, "latent_dim": np.nan}
                try:
                    _seed(int(model_seed))
                    survival = _fit_baseline(baseline, train, test, grid, cfg, device)
                    context = {
                        **fit_context, "checkpoint": "fixed_epochs",
                        "checkpoint_epoch": model_cfg.get(baseline, {}).get("epochs", np.nan),
                        "is_primary_checkpoint": True, "prediction_mode": "standard",
                    }
                    rows.append(_prediction_row(survival, grid, test, context))
                    calibration.extend(_calibration_rows(
                        survival, grid, test.true_event_time, context
                    ))
                    manifest.append({
                        **fit_context, "status": "success",
                        "runtime_seconds": time.perf_counter() - started, "error": "",
                    })
                except Exception as error:
                    manifest.append({
                        **fit_context, "status": "failed",
                        "runtime_seconds": time.perf_counter() - started,
                        "error": repr(error),
                    })

            settings = model_cfg["dvfm"]
            for latent_dim in settings["latent_dims"]:
                started = time.perf_counter()
                fit_context = {**base, "model": "dvfm", "latent_dim": int(latent_dim)}
                try:
                    _seed(int(model_seed))
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
                        train.X.shape[1], int(latent_dim),
                        encoder_hidden=list(settings.get("encoder_hidden", [64, 32])),
                        decoder_hidden=list(settings.get("decoder_hidden", [32, 64])),
                    ).to(device)
                    artifacts = train_dvfm(
                        model, train_loader, validation_loader,
                        n_epochs=int(settings["epochs"]), lr=float(settings["learning_rate"]),
                        beta_max=float(settings["beta_max"]),
                        warmup_epochs=int(settings["warmup_epochs"]),
                        free_bits=float(settings["free_bits"]), device=device,
                        checkpoint_min_epoch=int(settings["checkpoint_min_epoch"]),
                        numerical_failure_threshold=float(
                            settings["numerical_failure_threshold"]
                        ),
                        return_artifacts=True,
                    )
                    histories.extend([{
                        **fit_context, **item,
                        "train_elbo": item["train_loss"],
                        "validation_elbo": item["validation_loss"],
                        "is_best_validation_elbo_post_warmup": (
                            item["epoch"] == artifacts["best_validation_elbo_epoch"]
                        ),
                    } for item in artifacts["history"]])
                    history_by_epoch = {
                        int(item["epoch"]): item for item in artifacts["history"]
                    }
                    primary_checkpoint = str(settings["primary_checkpoint"])
                    checkpoints = (
                        ("best_validation_elbo_post_warmup", artifacts["best_validation_elbo_state"], artifacts["best_validation_elbo_epoch"]),
                        ("final", artifacts["final_state"], int(settings["epochs"])),
                    )
                    checkpoint_issues = []
                    primary_completed = False
                    for checkpoint, state, checkpoint_epoch in checkpoints:
                        is_primary = checkpoint == primary_checkpoint
                        checkpoint_history = history_by_epoch[int(checkpoint_epoch)]
                        state_valid = all(
                            bool(torch.isfinite(value).all()) for value in state.values()
                        )
                        if not checkpoint_history["numerical_valid"] or not state_valid:
                            message = f"{checkpoint}: invalid objective or state"
                            checkpoint_issues.append(message)
                            if is_primary:
                                raise FloatingPointError(message)
                            continue
                        try:
                            model.load_state_dict(state); model.to(device)
                            seed_offset = 0 if checkpoint == "final" else 50_000
                            learned_tau = learned_conditional_kendall_tau(
                                model, np.mean(train.X, axis=0),
                                int(evaluation["dependence_samples"]), device,
                                int(model_seed) + seed_offset,
                            )
                            if not np.isfinite(learned_tau):
                                raise FloatingPointError("Learned Kendall's tau is non-finite")
                            checkpoint_context = {
                                **fit_context, "checkpoint": checkpoint,
                                "checkpoint_epoch": int(checkpoint_epoch),
                                "is_primary_checkpoint": is_primary,
                                "checkpoint_numerical_valid": True,
                                "checkpoint_validation_elbo": checkpoint_history["validation_loss"],
                                "checkpoint_validation_reconstruction_nll": checkpoint_history["validation_reconstruction"],
                                "checkpoint_validation_kl": checkpoint_history["validation_kl"],
                                "learned_conditional_kendall_tau": learned_tau,
                                "conditional_kendall_tau_error": learned_tau - float(scenario["kendall_tau"]),
                                "absolute_conditional_kendall_tau_error": abs(learned_tau - float(scenario["kendall_tau"])),
                            }
                            diagnostics.extend([{
                                **checkpoint_context, **item,
                            } for item in _frailty_diagnostics(
                                model, validation, test, int(settings["batch_size"]),
                                float(evaluation["active_kl_threshold"]),
                                float(scenario["kendall_tau"]), device,
                            )])
                            _seed(int(model_seed) + seed_offset)
                            predictors = {
                                "prior": lambda: predict_survival_from_prior(
                                    model, test.X, grid, int(settings["mc_samples"]), device
                                ),
                                "aggregate_posterior": lambda: predict_survival_curves(
                                    model, test.X, grid, train_loader,
                                    int(settings["mc_samples"]), device
                                ),
                            }
                            for mode in evaluation["prediction_modes"]:
                                survival = predictors[str(mode)]()
                                context = {**checkpoint_context, "prediction_mode": str(mode)}
                                rows.append(_prediction_row(survival, grid, test, context))
                                calibration.extend(_calibration_rows(
                                    survival, grid, test.true_event_time, context
                                ))
                            primary_completed = primary_completed or is_primary
                        except Exception as checkpoint_error:
                            checkpoint_issues.append(
                                f"{checkpoint}: {checkpoint_error!r}"
                            )
                            if is_primary:
                                raise
                    if not primary_completed:
                        raise RuntimeError("Primary DVFM checkpoint was not evaluated")
                    manifest.append({
                        **fit_context, "status": "success",
                        "primary_checkpoint": primary_checkpoint,
                        "numerically_invalid_epochs": artifacts["numerically_invalid_epochs"],
                        "final_checkpoint_valid": artifacts["final_checkpoint_valid"],
                        "checkpoint_issues": "; ".join(checkpoint_issues),
                        "runtime_seconds": time.perf_counter() - started, "error": "",
                    })
                except Exception as error:
                    manifest.append({
                        **fit_context, "status": "failed",
                        "runtime_seconds": time.perf_counter() - started,
                        "error": repr(error),
                    })

    pd.DataFrame(histories).to_csv(
        out_dir / "training_history.csv.gz", index=False, compression="gzip"
    )
    pd.DataFrame(diagnostics).to_csv(out_dir / "dvfm_diagnostics.csv", index=False)
    pd.DataFrame(calibration).to_csv(
        out_dir / "calibration_curves.csv.gz", index=False, compression="gzip"
    )
    pd.DataFrame(manifest).to_csv(out_dir / "run_manifest.csv", index=False)
    failures = [item for item in manifest if item["status"] == "failed"]
    if failures:
        raise RuntimeError(
            f"{len(failures)} synthetic fits failed; see {out_dir / 'run_manifest.csv'}"
        )
    return pd.DataFrame(rows)


__all__ = ["run_synthetic_pilot"]
