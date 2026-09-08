"""Controlled shared-Gaussian-frailty pilot with paired model comparisons."""

from __future__ import annotations

import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader
from sota.adapters import fit_deephit, fit_sksurv_ensemble, fit_weibull_aft

from sota.baselines import (
    _fit_cox_and_predict_survival,
    fit_clayton_weibull_aft,
    train_deepsurv,
    train_mtlr,
)
from .config import expand_scenarios
from utility.data import SurvivalData
from dvfm.model import DVFM
from utility.data import SurvivalDataset
from sota.hacsurv import fit_hacsurv_2d
from sota.bayesian_cox_gamma_frailty import fit_bayesian_cox_gamma_frailty
from utility.metrics import (
    compute_oracle_metrics,
    frailty_regression_metrics,
    learned_conditional_kendall_tau,
    oracle_calibration_rows,
    oracle_joint_survival_ise,
    predict_dvfm_joint_survival,
    predict_hacsurv_joint_survival,
    prepare_joint_survival_evaluation,
    pearson_correlation,
    spearman_correlation,
)
from dvfm.prediction import (
    predict_survival_curves,
    predict_survival_from_prior,
)
from utility.synthetic import generate_clayton_gamma_frailty, generate_gaussian_shared_frailty
from utility.runtime import seed_everything
from utility.splitting import split_survival_data
from dvfm.training import train_dvfm


def _seed(seed: int) -> None:
    seed_everything(seed)


def _split(data: SurvivalData, cfg: dict, seed: int):
    return split_survival_data(data, cfg, seed)


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
        pearson_correlation(validation_mu[:, j], validation.true_z)
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
        recovery_metrics = (
            frailty_regression_metrics(truth, estimate)
            if recovery_defined else {
                "frailty_r2_calibrated": np.nan,
                "frailty_rmse_calibrated": np.nan,
            }
        )
        rows.append({
            "subgroup": subgroup, "n": int(mask.sum()),
            "frailty_recovery_defined": recovery_defined,
            "selected_latent_dimension": selected,
            "active_latent_dimensions": int(np.sum(per_dimension_kl > active_threshold)),
            "mean_kl": float(np.sum(per_dimension_kl)),
            "frailty_pearson": pearson_correlation(test_aligned[mask], truth) if recovery_defined else np.nan,
            "frailty_spearman": spearman_correlation(test_aligned[mask], truth) if recovery_defined else np.nan,
            **recovery_metrics,
            "validation_alignment_sign": sign if recovery_defined else np.nan,
            "validation_calibration_intercept": float(calibrator.intercept_) if recovery_defined else np.nan,
            "validation_calibration_slope": float(calibrator.coef_[0]) if recovery_defined else np.nan,
        })
    return rows


def _classical_frailty_diagnostics(model, validation, test, target_tau):
    """Evaluate a scalar posterior frailty using the same alignment as DVFM."""
    validation_latent = model.posterior_log_frailty_mean(
        validation.X, validation.time, validation.event
    )
    test_latent = model.posterior_log_frailty_mean(
        test.X, test.time, test.event
    )
    recovery_defined = float(target_tau) > 0.0
    if recovery_defined:
        raw_pearson = pearson_correlation(validation_latent, validation.true_z)
        sign = 1.0 if not np.isfinite(raw_pearson) or raw_pearson >= 0 else -1.0
        validation_aligned = sign * validation_latent
        test_aligned = sign * test_latent
        calibrator = LinearRegression().fit(
            validation_aligned.reshape(-1, 1), validation.true_z
        )
        test_calibrated = calibrator.predict(test_aligned.reshape(-1, 1))
    else:
        sign, calibrator = np.nan, None
        test_aligned = np.full(len(test.time), np.nan)
        test_calibrated = np.full(len(test.time), np.nan)

    posterior_shape, posterior_rate = model.posterior_parameters(
        test.X, test.time, test.event
    )
    posterior_sd = (
        torch.sqrt(posterior_shape) / posterior_rate
    ).detach().cpu().numpy()
    masks = {
        "all": np.ones(len(test.time), dtype=bool),
        "event_observed": test.event == 1,
        "censored": test.event == 0,
    }
    rows = []
    for subgroup, mask in masks.items():
        if recovery_defined:
            recovery = frailty_regression_metrics(
                test.true_z[mask], test_calibrated[mask]
            )
            pearson = pearson_correlation(test_aligned[mask], test.true_z[mask])
            spearman = spearman_correlation(test_aligned[mask], test.true_z[mask])
        else:
            recovery = {
                "frailty_r2_calibrated": np.nan,
                "frailty_rmse_calibrated": np.nan,
            }
            pearson = spearman = np.nan
        rows.append({
            "subgroup": subgroup,
            "n": int(mask.sum()),
            "frailty_recovery_defined": recovery_defined,
            "selected_latent_dimension": 0,
            "active_latent_dimensions": np.nan,
            "mean_kl": np.nan,
            "frailty_pearson": pearson,
            "frailty_spearman": spearman,
            **recovery,
            "validation_alignment_sign": sign,
            "validation_calibration_intercept": (
                float(calibrator.intercept_) if calibrator is not None else np.nan
            ),
            "validation_calibration_slope": (
                float(calibrator.coef_[0]) if calibrator is not None else np.nan
            ),
            "mean_posterior_frailty_sd": float(np.mean(posterior_sd[mask])),
        })
    return rows


def _prediction_row(survival, grid, test, context):
    survival = np.asarray(survival, dtype=float)
    if not np.all(np.isfinite(survival)):
        raise FloatingPointError("Survival predictions contain NaN or infinity")
    if np.any(survival < -1e-6) or np.any(survival > 1.0 + 1e-6):
        raise FloatingPointError("Survival predictions fall outside [0, 1]")
    metrics = compute_oracle_metrics(
        survival, grid, test.true_event_time, test.event
    )
    return {
        **context, **metrics,
    }


def _dvfm_variants(settings: dict) -> list[dict]:
    """Resolve named OFAT variants against the shared DVFM defaults."""
    configured = settings.get("variants")
    if not configured:
        return [{**deepcopy(settings), "name": "default"}]
    shared = {key: deepcopy(value) for key, value in settings.items() if key != "variants"}
    variants = []
    for override in configured:
        resolved = {**deepcopy(shared), **deepcopy(override)}
        variants.append(resolved)
    return variants


def _generate_scenario(data_cfg: dict, scenario: dict, seed_cfg: dict, sampling_seed: int):
    mechanism = str(scenario.get("mechanism", "gaussian_shared_frailty")).lower()
    common = dict(
        n_samples=int(scenario.get("n_samples", data_cfg.get("n_samples"))),
        n_features=int(data_cfg["n_features"]),
        kendall_tau=float(scenario["kendall_tau"]),
        censoring_rate=float(scenario["censoring_rate"]),
        dgp_seed=int(seed_cfg["dgp"]), sampling_seed=int(sampling_seed),
    )
    if mechanism == "gaussian_shared_frailty":
        generated = generate_gaussian_shared_frailty(
            **common,
            calibration_samples=int(data_cfg.get("calibration_samples", 50_000)),
        )
    elif mechanism == "clayton_gamma_frailty":
        generated = generate_clayton_gamma_frailty(**common)
    else:
        raise ValueError(f"Unsupported synthetic mechanism: {mechanism}")
    return mechanism, generated


def _write_hyperparameter_comparison(
    rows: list[dict], diagnostics: list[dict], out_dir: Path, sweep_cfg: dict
) -> None:
    """Write validation-only ranks and paired deltas for a tuning sweep."""
    reference_name = str(sweep_cfg.get("reference_variant", "reference"))
    selection_partition = str(sweep_cfg.get("selection_partition", "validation"))
    selection_mode = str(sweep_cfg.get("selection_prediction_mode", "prior"))
    recovery_tolerance = float(sweep_cfg.get("frailty_spearman_tolerance", 0.03))
    predictions = pd.DataFrame(rows)
    selection = predictions.loc[
        predictions["is_primary_checkpoint"].astype(bool)
        & predictions["partition"].eq(selection_partition)
        & predictions["prediction_mode"].eq(selection_mode)
    ].copy()
    if selection.empty:
        raise RuntimeError("Hyperparameter sweep produced no validation predictions")
    cell = ["scenario", "repeat"]
    selection["validation_ibs_rank"] = selection.groupby(cell)["oracle_ibs"].rank(
        method="average", ascending=True
    )
    selection["validation_ci_rank"] = selection.groupby(cell)["oracle_ci"].rank(
        method="average", ascending=False
    )
    summary = selection.groupby("hyperparameter_variant", as_index=False).agg(
        mean_validation_oracle_ibs=("oracle_ibs", "mean"),
        mean_validation_oracle_ci=("oracle_ci", "mean"),
        mean_oracle_joint_survival_ise=("oracle_joint_survival_ise", "mean"),
        mean_validation_ibs_rank=("validation_ibs_rank", "mean"),
        mean_validation_ci_rank=("validation_ci_rank", "mean"),
        mean_absolute_kendall_tau_error=("absolute_conditional_kendall_tau_error", "mean"),
        numerical_cells=("oracle_ibs", "size"),
    )
    diagnostic_frame = pd.DataFrame(diagnostics)
    scenario_summary = selection.groupby(
        ["mechanism", "scenario", "hyperparameter_variant"], as_index=False
    ).agg(
        mean_validation_oracle_ibs=("oracle_ibs", "mean"),
        std_validation_oracle_ibs=("oracle_ibs", "std"),
        mean_validation_oracle_ci=("oracle_ci", "mean"),
        mean_oracle_joint_survival_ise=("oracle_joint_survival_ise", "mean"),
        mean_absolute_kendall_tau_error=("absolute_conditional_kendall_tau_error", "mean"),
        seeds=("oracle_ibs", "size"),
    )
    if not diagnostic_frame.empty:
        recovery = diagnostic_frame.loc[
            diagnostic_frame["is_primary_checkpoint"].astype(bool)
            & diagnostic_frame["subgroup"].eq("all")
            & diagnostic_frame["frailty_recovery_defined"].astype(bool)
        ].groupby("hyperparameter_variant", as_index=False).agg(
            mean_frailty_spearman=("frailty_spearman", "mean"),
            mean_frailty_r2_calibrated=("frailty_r2_calibrated", "mean"),
            mean_active_latent_dimensions=("active_latent_dimensions", "mean"),
        )
        summary = summary.merge(recovery, on="hyperparameter_variant", how="left")
        scenario_recovery = diagnostic_frame.loc[
            diagnostic_frame["is_primary_checkpoint"].astype(bool)
            & diagnostic_frame["subgroup"].eq("all")
            & diagnostic_frame["frailty_recovery_defined"].astype(bool)
        ].groupby(
            ["mechanism", "scenario", "hyperparameter_variant"], as_index=False
        ).agg(mean_frailty_spearman=("frailty_spearman", "mean"))
        scenario_summary = scenario_summary.merge(
            scenario_recovery,
            on=["mechanism", "scenario", "hyperparameter_variant"], how="left",
        )
    scenario_summary.to_csv(
        out_dir / "hyperparameter_scenario_summary.csv", index=False
    )
    reference_summary = summary.loc[
        summary["hyperparameter_variant"].eq(reference_name)
    ]
    if len(reference_summary) != 1:
        raise RuntimeError("Hyperparameter ranking requires exactly one reference row")
    reference_recovery = float(reference_summary["mean_frailty_spearman"].iloc[0])
    reference_tau_error = float(
        reference_summary["mean_absolute_kendall_tau_error"].iloc[0]
    )
    summary["frailty_recovery_guardrail_pass"] = (
        summary["mean_frailty_spearman"] >= reference_recovery - recovery_tolerance
    )
    summary["dependence_not_worse_than_reference"] = (
        summary["mean_absolute_kendall_tau_error"] <= reference_tau_error
    )
    summary["selection_eligible"] = (
        summary["frailty_recovery_guardrail_pass"]
        & summary["dependence_not_worse_than_reference"]
    )
    summary.sort_values(
        ["selection_eligible", "mean_validation_ibs_rank", "mean_validation_oracle_ibs"],
        ascending=[False, True, True], inplace=True,
    )
    summary.to_csv(out_dir / "hyperparameter_ranking.csv", index=False)

    reference = selection.loc[
        selection["hyperparameter_variant"].eq(reference_name),
        cell + [
            "oracle_ibs", "oracle_ci", "oracle_mae",
            "oracle_joint_survival_ise",
        ],
    ].rename(columns={
        "oracle_ibs": "reference_oracle_ibs",
        "oracle_ci": "reference_oracle_ci",
        "oracle_mae": "reference_oracle_mae",
        "oracle_joint_survival_ise": "reference_oracle_joint_survival_ise",
    })
    if len(reference) != selection[cell].drop_duplicates().shape[0]:
        raise RuntimeError("Reference variant is missing from one or more tuning cells")
    paired = selection.merge(reference, on=cell, how="left", validate="many_to_one")
    paired["delta_oracle_ibs_vs_reference"] = (
        paired["oracle_ibs"] - paired["reference_oracle_ibs"]
    )
    paired["delta_oracle_ci_vs_reference"] = (
        paired["oracle_ci"] - paired["reference_oracle_ci"]
    )
    paired["delta_oracle_mae_vs_reference"] = (
        paired["oracle_mae"] - paired["reference_oracle_mae"]
    )
    paired["delta_oracle_joint_survival_ise_vs_reference"] = (
        paired["oracle_joint_survival_ise"]
        - paired["reference_oracle_joint_survival_ise"]
    )
    if "comparison_parent" in selection:
        parent = selection[
            cell + [
                "hyperparameter_variant", "oracle_ibs", "oracle_ci",
                "oracle_mae", "oracle_joint_survival_ise",
            ]
        ].rename(columns={
            "hyperparameter_variant": "comparison_parent",
            "oracle_ibs": "parent_oracle_ibs",
            "oracle_ci": "parent_oracle_ci",
            "oracle_mae": "parent_oracle_mae",
            "oracle_joint_survival_ise": "parent_oracle_joint_survival_ise",
        })
        paired = paired.merge(
            parent, on=cell + ["comparison_parent"], how="left",
            validate="many_to_one",
        )
        for metric in (
            "oracle_ibs", "oracle_ci", "oracle_mae",
            "oracle_joint_survival_ise",
        ):
            paired[f"delta_{metric}_vs_parent"] = (
                paired[metric] - paired[f"parent_{metric}"]
            )
    paired.to_csv(out_dir / "hyperparameter_paired_deltas.csv", index=False)


def _fit_baseline(
    name, train, validation, test, grid, cfg, device, model_seed,
    joint_evaluation=None,
):
    fit_info, history, fitted_model = {}, [], None
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
    elif name == "deephit":
        settings = cfg["models"]["deephit"]
        _, survival = fit_deephit(
            train.X, train.time, train.event,
            validation.X, validation.time, validation.event,
            test.X, grid, settings, device,
        )
    elif name in {"gbsa", "rsf"}:
        _, survival = fit_sksurv_ensemble(
            name, train.X, train.time, train.event, test.X, grid,
            cfg["models"][name],
        )
    elif name == "weibull_aft":
        _, survival = fit_weibull_aft(
            train.X, train.time, train.event, test.X, grid,
            cfg["models"]["weibull_aft"],
        )
    elif name == "bayesian_cox_gamma_frailty":
        survival, fit_info, history, fitted_model = (
            fit_bayesian_cox_gamma_frailty(
                train.X, train.time, train.event,
                validation.X, validation.time, validation.event,
                test.X, grid,
                cfg["models"]["bayesian_cox_gamma_frailty"], device,
            )
        )
    elif name == "hacsurv_2d":
        settings = cfg["models"]["hacsurv_2d"]
        survival, fit_info, history, fitted_model = fit_hacsurv_2d(
            train.X, train.time, train.event,
            validation.X, validation.time, validation.event,
            test.X, grid,
            epochs=int(settings["epochs"]),
            batch_size=int(settings["batch_size"]),
            learning_rate=float(settings["learning_rate"]),
            copula_learning_rate=float(settings["copula_learning_rate"]),
            copula_start_epoch=int(settings["copula_start_epoch"]),
            early_stopping_patience=int(settings["early_stopping_patience"]),
            minimum_epochs=int(settings.get("minimum_epochs", 0)),
            checkpoint_min_epoch=int(settings.get("checkpoint_min_epoch", 0)),
            generator_samples=int(settings["generator_samples"]),
            validation_generator_samples=int(settings["validation_generator_samples"]),
            hidden_size=int(settings["hidden_size"]),
            hidden_survival=int(settings["hidden_survival"]),
            inverse_iterations=int(settings["inverse_iterations"]),
            inverse_tolerance=float(settings["inverse_tolerance"]),
            scale_regularization=float(settings["scale_regularization"]),
            numerical_failure_threshold=float(settings["numerical_failure_threshold"]),
            dtype=str(settings["dtype"]), seed=int(model_seed), device=device,
        )
        if joint_evaluation is not None:
            joint_prediction = predict_hacsurv_joint_survival(
                fitted_model, joint_evaluation,
                generator_samples=int(cfg["evaluation"]["joint_model_samples"]),
                batch_size=int(cfg["evaluation"]["joint_model_batch_size"]),
                device=device,
            )
            fit_info["oracle_joint_survival_ise"] = oracle_joint_survival_ise(
                joint_prediction, joint_evaluation
            )
    else:
        raise ValueError(f"Unsupported synthetic baseline: {name}")
    return survival, fit_info, history, fitted_model


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
            mechanism, generated = _generate_scenario(
                data_cfg, scenario, seed_cfg, int(sampling_seed)
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
            joint_evaluation = None
            if evaluation.get("compute_oracle_joint_survival_ise", False):
                joint_evaluation = prepare_joint_survival_evaluation(
                    generated, mechanism, test.X,
                    train.true_event_time, train.true_censor_time,
                    n_time_points=int(evaluation["joint_n_time_points"]),
                    max_quantile=float(evaluation["joint_grid_max_quantile"]),
                    n_subjects=int(evaluation["joint_n_subjects"]),
                    dgp_samples=int(evaluation["joint_dgp_samples"]),
                    seed=int(split_seed) + 70_000,
                )
            base = {
                "study": cfg["study"]["name"],
                "scenario": scenario.get(
                    "name",
                    f"{mechanism}-n-{n_samples}-kendall_tau-{scenario['kendall_tau']}-censoring_rate-{scenario['censoring_rate']}",
                ),
                "mechanism": mechanism,
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
                    survival, fit_info, baseline_history, fitted_model = _fit_baseline(
                        baseline, train, validation, test, grid, cfg, device,
                        int(model_seed), joint_evaluation,
                    )
                    histories.extend([
                        {**fit_context, **item} for item in baseline_history
                    ])
                    if "learned_conditional_kendall_tau" in fit_info:
                        fit_info["conditional_kendall_tau_error"] = (
                            float(fit_info["learned_conditional_kendall_tau"])
                            - float(scenario["kendall_tau"])
                        )
                        fit_info["absolute_conditional_kendall_tau_error"] = abs(
                            fit_info["conditional_kendall_tau_error"]
                        )
                    context = {
                        **fit_context,
                        "checkpoint": fit_info.get("checkpoint", "fixed_epochs"),
                        "checkpoint_epoch": fit_info.get(
                            "checkpoint_epoch",
                            model_cfg.get(baseline, {}).get("epochs", np.nan),
                        ),
                        "is_primary_checkpoint": True, "prediction_mode": "standard",
                        "partition": "test",
                        **fit_info,
                    }
                    rows.append(_prediction_row(survival, grid, test, context))
                    if baseline == "bayesian_cox_gamma_frailty":
                        diagnostics.extend([{
                            **context, **item,
                        } for item in _classical_frailty_diagnostics(
                            fitted_model, validation, test,
                            float(scenario["kendall_tau"]),
                        )])
                    calibration.extend(oracle_calibration_rows(
                        survival, grid, test.true_event_time, context
                    ))
                    manifest.append({
                        **fit_context, "status": "success",
                        **fit_info,
                        "runtime_seconds": time.perf_counter() - started, "error": "",
                    })
                except Exception as error:
                    manifest.append({
                        **fit_context, "status": "failed",
                        "runtime_seconds": time.perf_counter() - started,
                        "error": repr(error),
                    })

            shared_settings = model_cfg["dvfm"]
            settings_and_latent_dims = [] if "dvfm" not in {
                str(name).lower() for name in model_cfg["enabled"]
            } else [
                (settings, latent_dim)
                for settings in _dvfm_variants(shared_settings)
                for latent_dim in settings.get(
                    "latent_dims", [settings.get("latent_dim")]
                )
            ]
            for settings, latent_dim in settings_and_latent_dims:
                started = time.perf_counter()
                fit_context = {
                    **base, "model": "dvfm", "latent_dim": int(latent_dim),
                    "hyperparameter_variant": str(settings["name"]),
                    "comparison_parent": str(settings.get("comparison_parent", "reference")),
                    "epochs": int(settings["epochs"]),
                    "batch_size": int(settings["batch_size"]),
                    "dropout": float(settings.get("dropout", 0.0)),
                    "weight_decay": float(settings.get("weight_decay", 0.0)),
                    "encoder_hidden": "-".join(map(str, settings.get("encoder_hidden", [64, 32]))),
                    "decoder_hidden": "-".join(map(str, settings.get("decoder_hidden", [32, 64]))),
                    "scale_link": str(settings.get("scale_link", "softplus")),
                    "latent_path": str(settings.get("latent_path", "nonlinear")),
                    "shape_mode": str(settings.get("shape_mode", "conditional")),
                }
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
                        dropout=float(settings.get("dropout", 0.0)),
                        scale_link=str(settings.get("scale_link", "softplus")),
                        latent_path=str(settings.get("latent_path", "nonlinear")),
                        shape_mode=str(settings.get("shape_mode", "conditional")),
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
                        weight_decay=float(settings.get("weight_decay", 0.0)),
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
                            if is_primary and joint_evaluation is not None:
                                joint_prediction = predict_dvfm_joint_survival(
                                    model, joint_evaluation,
                                    mc_samples=int(evaluation["joint_model_samples"]),
                                    batch_size=int(evaluation["joint_model_batch_size"]),
                                    seed=int(model_seed) + 90_000, device=device,
                                )
                                checkpoint_context["oracle_joint_survival_ise"] = (
                                    oracle_joint_survival_ise(
                                        joint_prediction, joint_evaluation
                                    )
                                )
                            else:
                                checkpoint_context["oracle_joint_survival_ise"] = np.nan
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
                            evaluation_parts = {
                                "test": (test, predictors),
                            }
                            if "validation" in evaluation.get("evaluate_partitions", ["test"]):
                                evaluation_parts["validation"] = (
                                    validation,
                                    {
                                        "prior": lambda: predict_survival_from_prior(
                                            model, validation.X, grid,
                                            int(settings["mc_samples"]), device
                                        ),
                                        "aggregate_posterior": lambda: predict_survival_curves(
                                            model, validation.X, grid, train_loader,
                                            int(settings["mc_samples"]), device
                                        ),
                                    },
                                )
                            for partition, (partition_data, partition_predictors) in evaluation_parts.items():
                                for mode in evaluation["prediction_modes"]:
                                    survival = partition_predictors[str(mode)]()
                                    context = {
                                        **checkpoint_context,
                                        "prediction_mode": str(mode),
                                        "partition": partition,
                                    }
                                    rows.append(_prediction_row(
                                        survival, grid, partition_data, context
                                    ))
                                    calibration.extend(oracle_calibration_rows(
                                        survival, grid, partition_data.true_event_time, context
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
    if model_cfg["dvfm"].get("variants"):
        _write_hyperparameter_comparison(
            rows, diagnostics, out_dir, cfg.get("hyperparameter_sweep", {}),
        )
    return pd.DataFrame(rows)


__all__ = ["run_synthetic_pilot"]
