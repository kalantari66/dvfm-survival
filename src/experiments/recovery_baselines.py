"""Scalar-frailty recovery baselines for the shared-latent comparison.

CoxPH martingale residuals and the Cox--Gamma posterior frailty are scored
against the true latent on the same cohorts, splits and held-out subjects as
DVFM, and exported in the schema of ``experiments.latent_recovery`` so the two
can be compared row for row.

The two baselines are not independent.  Within a censoring stratum the
Cox--Gamma posterior mean ``psi(alpha + delta) - log(alpha + r(x) H0(t))`` and
the martingale residual ``delta - H(t | x)`` are both strictly monotone in the
fitted cumulative event hazard, so they rank subjects identically under a
common hazard fit and differ only through the fit itself.  They are reported
together to show that the gap to DVFM is not an artefact of either baseline's
particular baseline-hazard estimator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from sota.bayesian_cox_gamma_frailty import fit_bayesian_cox_gamma_frailty
from sota.coxph import make_cox_model
from utility.metrics import (
    collect_metrics,
    pearson_correlation,
    spearman_correlation,
)
from utility.runtime import seed_everything
from utility.semisynthetic import fit_semisynthetic_dgp, generate_semisynthetic
from utility.splitting import time_event_stratified_split_indices

from .runner import (
    _models_for_dataset,
    _preprocess_three,
    _subset,
    evaluation_time_grid,
    scale_observed_time,
    training_time_scale,
)

try:
    from sksurv.util import Surv
except ImportError:  # pragma: no cover - mirrors sota.coxph
    Surv = None


def martingale_residuals(model, X, time, event) -> np.ndarray:
    """Return ``delta_i - H(t_i | x_i)`` from a fitted CoxPH model.

    The cumulative hazard is read off each subject's own step function at its
    own observed time, interpolated exactly as survival curves are elsewhere so
    that observed times beyond the training support do not raise.
    """
    functions = model.predict_cumulative_hazard_function(X)
    times = np.asarray(time, dtype=float)
    cumulative = np.asarray([
        np.interp(point, function.x, function.y, left=0.0, right=function.y[-1])
        for function, point in zip(functions, times)
    ], dtype=float)
    return np.asarray(event, dtype=float) - cumulative


def aligned_recovery_rows(
    validation_statistic,
    validation_truth,
    test_statistic,
    test_truth,
    test_event,
    context: dict | None = None,
) -> list[dict]:
    """Score one scalar statistic against the truth, aligned on validation only.

    The sign and the linear calibration are estimated from validation subjects,
    matching ``experiments.synthetic._classical_frailty_diagnostics`` and
    ``experiments.latent_recovery``, so no held-out truth informs the mapping.
    """
    context = dict(context or {})
    validation_statistic = np.asarray(validation_statistic, dtype=float)
    test_statistic = np.asarray(test_statistic, dtype=float)
    validation_truth = np.asarray(validation_truth, dtype=float)
    test_truth = np.asarray(test_truth, dtype=float)
    test_event = np.asarray(test_event)

    raw_pearson = pearson_correlation(validation_statistic, validation_truth)
    sign = 1.0 if not np.isfinite(raw_pearson) or raw_pearson >= 0 else -1.0
    validation_aligned = sign * validation_statistic
    calibrator = LinearRegression().fit(
        validation_aligned.reshape(-1, 1), validation_truth
    )
    test_aligned = sign * test_statistic
    test_calibrated = calibrator.predict(test_aligned.reshape(-1, 1))

    masks = {
        "All": np.ones(len(test_truth), dtype=bool),
        "Event observed": test_event == 1,
        "Censored": test_event == 0,
    }
    representations = (
        ("Aligned, uncalibrated", test_aligned),
        ("Validation-calibrated", test_calibrated),
    )
    rows = []
    for subgroup, mask in masks.items():
        truth = test_truth[mask]
        for representation, estimate in representations:
            prediction = estimate[mask]
            residual = truth - prediction
            rows.append({
                **context,
                "split": "test",
                "subgroup": subgroup,
                "representation": representation,
                "n": int(mask.sum()),
                "pearson": pearson_correlation(prediction, truth),
                "spearman": spearman_correlation(prediction, truth),
                "r2": (
                    float(1.0 - np.sum(residual ** 2) / np.sum((truth - truth.mean()) ** 2))
                    if len(truth) > 1 and np.std(truth) > 0 else np.nan
                ),
                "rmse": float(np.sqrt(np.mean(residual ** 2))) if len(truth) else np.nan,
                "alignment_sign": sign,
                "calibration_intercept": float(calibrator.intercept_),
                "calibration_slope": float(calibrator.coef_[0]),
            })
    return rows


def _fit_coxph(train, config):
    if Surv is None:
        raise ImportError(
            "Recovery baselines require scikit-survival. Install the project "
            "baseline dependencies before fitting CoxPH."
        )
    model = make_cox_model(config)
    model.fit(train.X, Surv.from_arrays(
        event=np.asarray(train.event, dtype=bool),
        time=np.asarray(train.time, dtype=float),
    ))
    return model


def _cox_survival(model, X, time_points) -> np.ndarray:
    grid = np.asarray(time_points, dtype=float)
    return np.vstack([
        np.interp(grid, function.x, function.y, left=1.0, right=function.y[-1])
        for function in model.predict_survival_function(X)
    ])


def recovery_baseline_rows(
    cfg: dict, dataset: dict, copula: str, repeat: int, seeds: dict,
) -> tuple[list[dict], list[dict]]:
    """Refit both baselines on one regenerated cell and score their recovery.

    Returns the recovery rows and the oracle-metric cross-check rows.  The
    cross-check recomputes each baseline's oracle IBS from the refit and is
    compared against the benchmark run: agreement proves the regenerated
    cohort and split are the ones the stored results were produced on.
    """
    target_tau = float(cfg["evaluation"]["recovery_baseline_kendall_tau"])
    if target_tau <= 0.0:
        raise ValueError(
            "Recovery is undefined at tau=0; "
            "evaluation.recovery_baseline_kendall_tau must be positive"
        )
    data_cfg = cfg["data"]
    dgp = fit_semisynthetic_dgp(
        dataset, cox_penalizer=float(data_cfg.get("cox_penalizer", 0.01))
    )
    rate_specification = data_cfg["censoring_rates"]
    target_rates = (
        [1.0 - float(dgp.source_event_rate)]
        if isinstance(rate_specification, str)
        and rate_specification.lower() == "original"
        else rate_specification
    )
    model_cfg = _models_for_dataset(cfg, dataset["name"])

    rows, crosscheck = [], []
    for target_rate in target_rates:
        generated = generate_semisynthetic(
            dgp, kendall_tau=target_tau, censoring_rate=float(target_rate),
            sampling_seed=int(seeds["sampling"]), copula=str(copula),
        )
        if generated.data.true_z is None:
            raise ValueError(
                f"The {copula} copula supplies no recovery target at "
                f"tau={target_tau:g}"
            )
        train_idx, validation_idx, test_idx = time_event_stratified_split_indices(
            generated.data, cfg["split"], int(seeds["split"])
        )
        train, validation, test = _preprocess_three(
            _subset(generated.data, train_idx),
            _subset(generated.data, validation_idx),
            _subset(generated.data, test_idx),
            {**cfg["preprocessing"], "numeric_features": dataset["numeric_features"]},
        )
        time_points = evaluation_time_grid(train, cfg["evaluation"])
        scale = training_time_scale(train)
        model_time_points = time_points / scale
        model_train = scale_observed_time(train, scale)
        model_validation = scale_observed_time(validation, scale)
        model_test = scale_observed_time(test, scale)

        context = {
            "dataset": dataset["name"],
            "copula": str(copula).lower(),
            "target_kendall_tau": target_tau,
            "target_censoring_rate": float(generated.target_censoring_rate),
            "achieved_censoring_rate": float(generated.achieved_censoring_rate),
            "repeat": int(repeat),
            "training_time_scale": float(scale),
        }

        # One model per job in the benchmark, so each fit saw a freshly seeded
        # generator; reproduce that ordering fit by fit.
        seed_everything(int(seeds["model"]))
        cox = _fit_coxph(model_train, model_cfg["coxph"])
        cox_statistic = {
            part_name: martingale_residuals(cox, part.X, part.time, part.event)
            for part_name, part in (
                ("validation", model_validation), ("test", model_test)
            )
        }
        cox_survival = _cox_survival(cox, model_test.X, model_time_points)

        seed_everything(int(seeds["model"]))
        gamma_survival, _, _, gamma = fit_bayesian_cox_gamma_frailty(
            model_train.X, model_train.time, model_train.event,
            model_validation.X, model_validation.time, model_validation.event,
            model_test.X, model_time_points,
            model_cfg["bayesian_cox_gamma_frailty"], device="cpu",
        )
        gamma_statistic = {
            part_name: gamma.posterior_log_frailty_mean(
                part.X, part.time, part.event
            )
            for part_name, part in (
                ("validation", model_validation), ("test", model_test)
            )
        }

        for model_name, statistic in (
            ("CoxPHMartingale", cox_statistic),
            ("BayesianCoxGammaFrailty", gamma_statistic),
        ):
            rows.extend(aligned_recovery_rows(
                statistic["validation"], validation.true_z,
                statistic["test"], test.true_z, test.event,
                {**context, "model": model_name},
            ))

        for model_name, survival in (
            ("CoxPH", cox_survival), ("BayesianCoxGammaFrailty", gamma_survival),
        ):
            # Oracle metrics read medians off the curves themselves, so the
            # median argument is unused on this path.
            metrics = collect_metrics(
                model_name, None, survival, test.time, test.event,
                test.true_event_time, time_points,
                t_train=train.time, e_train=train.event, oracle_only=True,
            )
            crosscheck.append({
                **context, "model": model_name,
                "refit_oracle_ibs": metrics[f"{model_name} IBS Oracle"],
                "refit_oracle_ci": metrics[f"{model_name} CI Oracle"],
                "refit_oracle_mae": metrics[f"{model_name} MAE Oracle"],
            })
    return rows, crosscheck


def run_recovery_baselines(
    cfg: dict, dataset: dict, repeat: int, seeds: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score both baselines on every configured positive copula for one cell."""
    copulas = cfg["data"].get("copulas", [cfg["data"].get("copula", "clayton")])
    rows, crosscheck = [], []
    for copula in copulas:
        cell_rows, cell_crosscheck = recovery_baseline_rows(
            cfg, dataset, str(copula), repeat, seeds
        )
        rows.extend(cell_rows)
        crosscheck.extend(cell_crosscheck)
    return pd.DataFrame(rows), pd.DataFrame(crosscheck)


__all__ = [
    "aligned_recovery_rows",
    "martingale_residuals",
    "recovery_baseline_rows",
    "run_recovery_baselines",
]
