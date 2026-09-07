"""Canonical survival metrics for all DVFM experiments.

SurvivalEVAL supplies survival-curve interpolation and all censoring-aware
metrics. Oracle synthetic metrics use the same evaluator with fully observed
event times. The final section contains the synthetic joint-survival ISE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from SurvivalEVAL import SurvivalEvaluator
from SurvivalEVAL import concordance as survival_eval_concordance


def _as_arrays(survival_curves, time_points, event_times, event_indicators):
    curves = np.asarray(survival_curves, dtype=float)
    grid = np.asarray(time_points, dtype=float)
    times = np.asarray(event_times, dtype=float)
    events = np.asarray(event_indicators, dtype=int)
    if curves.shape != (len(times), len(grid)):
        raise ValueError(
            "survival_curves must have shape (n_subjects, n_time_points)"
        )
    if len(grid) < 2 or np.any(np.diff(grid) <= 0):
        raise ValueError("time_points must be a strictly increasing grid")
    if not np.all(np.isfinite(curves)):
        raise FloatingPointError("survival curves contain non-finite values")
    return np.clip(curves, 0.0, 1.0), grid, times, events


def make_survival_evaluator(
    survival_curves, time_points, event_times, event_indicators,
    train_event_times=None, train_event_indicators=None,
) -> SurvivalEvaluator:
    """Construct the canonical SurvivalEVAL evaluator."""
    curves, grid, times, events = _as_arrays(
        survival_curves, time_points, event_times, event_indicators
    )
    return SurvivalEvaluator(
        curves,
        grid,
        times,
        events,
        None if train_event_times is None else np.asarray(train_event_times, dtype=float),
        None if train_event_indicators is None else np.asarray(train_event_indicators, dtype=int),
        predict_time_method="Median",
        interpolation="Linear",
    )


def compute_oracle_brier_ibs(survival_curves, time_points, true_event_times):
    """Oracle Brier curve and IBS using fully observed synthetic event times."""
    truth = np.asarray(true_event_times, dtype=float)
    evaluator = make_survival_evaluator(
        survival_curves, time_points, truth, np.ones(len(truth), dtype=int)
    )
    grid = np.asarray(time_points, dtype=float)
    brier = np.asarray(
        evaluator.brier_score_multiple_points(grid, IPCW_weighted=False),
        dtype=float,
    )
    ibs = evaluator.integrated_brier_score(
        target_times=grid, IPCW_weighted=False, integration_method="trapz"
    )
    return brier, float(ibs)


def compute_ipcw_brier_ibs(
    survival_curves, time_points, observed_time, event, tau=None,
    train_time=None, train_event=None,
):
    """SurvivalEVAL IPCW Brier curve and IBS on censored outcomes."""
    observed_time = np.asarray(observed_time, dtype=float)
    event = np.asarray(event, dtype=int)
    if train_time is None:
        train_time, train_event = observed_time, event
    evaluator = make_survival_evaluator(
        survival_curves, time_points, observed_time, event,
        train_time, train_event,
    )
    horizon = float(np.quantile(observed_time, 0.8) if tau is None else tau)
    grid = np.asarray(time_points, dtype=float)
    evaluation_grid = grid[grid <= horizon]
    if len(evaluation_grid) < 2:
        evaluation_grid = grid[:2]
        horizon = float(evaluation_grid[-1])
    brier = np.asarray(
        evaluator.brier_score_multiple_points(
            evaluation_grid, IPCW_weighted=True
        ),
        dtype=float,
    )
    ibs = evaluator.integrated_brier_score(
        target_times=evaluation_grid,
        IPCW_weighted=True,
        integration_method="trapz",
    )
    return brier, float(ibs), horizon


def compute_oracle_metrics(
    survival_curves, time_points, true_event_times, event_indicators=None
):
    """Return the oracle synthetic metrics used in result tables."""
    truth = np.asarray(true_event_times, dtype=float)
    evaluator = make_survival_evaluator(
        survival_curves, time_points, truth, np.ones(len(truth), dtype=int)
    )
    predicted = median_survival_time(survival_curves, time_points)
    _, ibs = compute_oracle_brier_ibs(
        survival_curves, time_points, true_event_times
    )
    result = {
        "oracle_ibs": float(ibs),
        "oracle_ci": concordance_index(truth, predicted),
        "oracle_mae": float(np.mean(np.abs(truth - predicted))),
    }
    if event_indicators is not None:
        observed = np.asarray(event_indicators, dtype=int) == 1
        censored = ~observed
        result["oracle_mae_uncensored"] = _masked_mae(truth, predicted, observed)
        result["oracle_mae_censored"] = _masked_mae(truth, predicted, censored)
    return result


def _masked_mae(truth, prediction, mask):
    return (
        float(np.mean(np.abs(truth[mask] - prediction[mask])))
        if np.any(mask) else np.nan
    )


def median_survival_time(survival_curves, time_points):
    """First grid crossing of S(t)<=0.5, with the last grid point as fallback."""
    curves = np.asarray(survival_curves, dtype=float)
    grid = np.asarray(time_points, dtype=float)
    medians = np.full(len(curves), float(grid[-1]), dtype=float)
    for index, curve in enumerate(curves):
        crossing = np.flatnonzero(curve <= 0.5)
        if crossing.size:
            medians[index] = grid[crossing[0]]
    return medians


def censoring_rate(event_indicators):
    events = np.asarray(event_indicators, dtype=float)
    return float(1.0 - np.mean(events)) if events.size else np.nan


def oracle_calibration_rows(survival, time_points, true_event_times, context=None, bins=10):
    """Return population survival calibration points when event times are known."""
    survival = np.asarray(survival, dtype=float)
    time_points = np.asarray(time_points, dtype=float)
    true_event_times = np.asarray(true_event_times, dtype=float)
    context = {} if context is None else context
    return [
        {
            **context,
            "time": float(time_points[index]),
            "mean_predicted_survival": float(survival[:, index].mean()),
            "empirical_oracle_survival": float(
                np.mean(true_event_times > time_points[index])
            ),
        }
        for index in np.linspace(0, len(time_points) - 1, int(bins), dtype=int)
    ]


def concordance_index(event_times, predicted_times, event_indicators=None):
    """SurvivalEVAL Harrell concordance for point predictions."""
    times = np.asarray(event_times, dtype=float)
    events = (
        np.ones(len(times), dtype=int) if event_indicators is None
        else np.asarray(event_indicators, dtype=int)
    )
    return float(survival_eval_concordance(
        np.asarray(predicted_times, dtype=float), times, events,
        method="Harrell", ties="Risk",
    )[0])


def learned_conditional_kendall_tau(
    model, x_reference, n_samples=2000, device="cpu", seed=0
):
    """Estimate event/censoring Kendall's tau from a fitted DVFM decoder."""
    rng = np.random.default_rng(seed)
    x = torch.as_tensor(
        np.repeat(np.asarray(x_reference)[None, :], n_samples, axis=0),
        dtype=torch.float32, device=device,
    )
    generator = torch.Generator(device=torch.device(device).type).manual_seed(seed)
    with torch.no_grad():
        z = torch.randn(
            (n_samples, model.latent_dim), generator=generator, device=device
        )
        shape_e, scale_e, shape_c, scale_c = model.decoder(x, z)
        ue = torch.as_tensor(rng.uniform(size=n_samples), dtype=torch.float32, device=device)
        uc = torch.as_tensor(rng.uniform(size=n_samples), dtype=torch.float32, device=device)
        event_time = scale_e * (-torch.log(ue)) ** (1 / shape_e)
        censor_time = scale_c * (-torch.log(uc)) ** (1 / shape_c)
    return float(kendalltau(
        event_time.cpu().numpy(), censor_time.cpu().numpy()
    ).statistic)


def hacsurv_kendall_tau(model, points: int = 2000) -> float:
    """Numerically evaluate HACSurv tau = 1 - 4∫t[phi'(t)]²dt."""
    model.generator.resample(max(model.generator.samples, 1000))
    rates = model.generator._rates().detach()
    lower_rate = max(float(rates.min().cpu()), 1e-8)
    upper = min(25.0 / lower_rate, 1e6)
    device, dtype = rates.device, rates.dtype
    positive = torch.logspace(
        -7, np.log10(upper), int(points), device=device, dtype=dtype
    )
    grid = torch.cat((torch.zeros(1, device=device, dtype=dtype), positive))
    derivative = model.generator.derivative(grid, order=1)
    integral = torch.trapz(grid * derivative.square(), grid)
    return float(torch.clamp(1.0 - 4.0 * integral, -1.0, 1.0).cpu())


def _safe_correlation(function, x, y) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(function(x, y).statistic)


def pearson_correlation(x, y) -> float:
    """Numerically guarded Pearson correlation."""
    return _safe_correlation(pearsonr, x, y)


def spearman_correlation(x, y) -> float:
    """Numerically guarded Spearman rank correlation."""
    return _safe_correlation(spearmanr, x, y)


def frailty_regression_metrics(truth, estimate) -> dict[str, float]:
    """Calibrated frailty scale-recovery metrics."""
    truth = np.asarray(truth, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    if len(truth) < 2:
        return {"frailty_r2_calibrated": np.nan, "frailty_rmse_calibrated": np.nan}
    return {
        "frailty_r2_calibrated": float(r2_score(truth, estimate)),
        "frailty_rmse_calibrated": float(
            mean_squared_error(truth, estimate) ** 0.5
        ),
    }


def collect_metrics(
    prefix, medians, surv_curves, t_test, e_test, true_t_test, time_points,
    tau=None, t_train=None, e_train=None, oracle_only=False,
):
    """Collect oracle or censoring-aware metrics under stable column names."""
    if true_t_test is not None:
        oracle = compute_oracle_metrics(
            surv_curves, time_points, true_t_test, e_test
        )
    else:
        oracle = {
            "oracle_ibs": np.nan, "oracle_ci": np.nan, "oracle_mae": np.nan,
            "oracle_mae_censored": np.nan, "oracle_mae_uncensored": np.nan,
        }
    oracle_output = {
        f"{prefix} IBS Oracle": oracle["oracle_ibs"],
        f"{prefix} CI Oracle": oracle["oracle_ci"],
        f"{prefix} MAE Oracle": oracle["oracle_mae"],
        f"{prefix} MAE Oracle Censored": oracle["oracle_mae_censored"],
        f"{prefix} MAE Oracle Uncensored": oracle["oracle_mae_uncensored"],
    }
    if oracle_only:
        return oracle_output

    evaluator = make_survival_evaluator(
        surv_curves, time_points, t_test, e_test, t_train, e_train
    )
    _, ibs, horizon = compute_ipcw_brier_ibs(
        surv_curves, time_points, t_test, e_test, tau=tau,
        train_time=t_train, train_event=e_train,
    )
    output = {
        **oracle_output,
        f"{prefix} C-Idx": float(evaluator.concordance(method="Harrell")[0]),
        f"{prefix} CI IPCW": float(
            evaluator.concordance(method="Uno", tau=horizon)[0]
        ),
        f"{prefix} IBS IPCW": float(ibs),
        f"{prefix} MAE Margin": float(evaluator.mae(method="Margin")),
        "evaluation_time_horizon": horizon,
    }
    # Preserve the old generic MAE column while giving it censor-aware meaning.
    output[f"{prefix} MAE"] = output[f"{prefix} MAE Margin"]
    return output


@dataclass
class JointSurvivalEvaluation:
    X: np.ndarray
    event_grid: np.ndarray
    censor_grid: np.ndarray
    truth: np.ndarray


def _joint_from_conditional(event_survival, censor_survival):
    return np.einsum(
        "bme,bmc->bec", event_survival, censor_survival, optimize=True
    ) / event_survival.shape[1]


def prepare_joint_survival_evaluation(
    generated, mechanism: str, test_X: np.ndarray,
    train_event_time: np.ndarray, train_censor_time: np.ndarray,
    *, n_time_points: int, max_quantile: float, n_subjects: int,
    dgp_samples: int, seed: int,
) -> JointSurvivalEvaluation:
    """Evaluate the known DGP joint survival on held-out covariates."""
    rng = np.random.default_rng(seed)
    count = min(int(n_subjects), len(test_X))
    indices = np.sort(rng.choice(len(test_X), size=count, replace=False))
    X = np.asarray(test_X[indices], dtype=np.float64)
    event_grid = np.linspace(
        0.0, float(np.quantile(train_event_time, max_quantile)), int(n_time_points)
    )
    censor_grid = np.linspace(
        0.0, float(np.quantile(train_censor_time, max_quantile)), int(n_time_points)
    )
    batch_size = 32
    surfaces = []
    mechanism = str(mechanism).lower()
    for start in range(0, count, batch_size):
        xb = X[start:start + batch_size]
        if mechanism == "gaussian_shared_frailty":
            latent = rng.normal(size=int(dgp_samples))
            event_scale = np.exp(
                xb @ generated.beta_event[:, None]
                + generated.frailty_loading * latent[None, :]
            )
            censor_scale = np.exp(
                xb @ generated.beta_censor[:, None] + generated.censor_intercept
                + generated.frailty_loading * latent[None, :]
            )
            event_hazard = (
                event_grid[None, None, :] / event_scale[:, :, None]
            ) ** generated.shape_event
            censor_hazard = (
                censor_grid[None, None, :] / censor_scale[:, :, None]
            ) ** generated.shape_censor
            event_survival = np.exp(-event_hazard)
            censor_survival = np.exp(-censor_hazard)
        elif mechanism == "clayton_gamma_frailty":
            frailty = rng.gamma(
                shape=1.0 / generated.clayton_theta,
                scale=1.0, size=int(dgp_samples),
            )
            event_scale = np.exp(xb @ generated.beta_event)[:, None]
            censor_scale = np.exp(
                xb @ generated.beta_censor + generated.censor_intercept
            )[:, None]
            event_hazard = (
                event_grid[None, :] / event_scale
            ) ** generated.shape_event
            censor_hazard = (
                censor_grid[None, :] / censor_scale
            ) ** generated.shape_censor
            event_survival = np.exp(
                -frailty[None, :, None]
                * np.expm1(np.minimum(
                    generated.clayton_theta * event_hazard[:, None, :], 50.0
                ))
            )
            censor_survival = np.exp(
                -frailty[None, :, None]
                * np.expm1(np.minimum(
                    generated.clayton_theta * censor_hazard[:, None, :], 50.0
                ))
            )
        else:
            raise ValueError(f"Unsupported joint-survival DGP: {mechanism}")
        surfaces.append(_joint_from_conditional(event_survival, censor_survival))
    truth = np.concatenate(surfaces, axis=0)
    return JointSurvivalEvaluation(X, event_grid, censor_grid, truth)


def predict_dvfm_joint_survival(
    model, evaluation: JointSurvivalEvaluation, *, mc_samples: int,
    batch_size: int, seed: int, device: torch.device,
) -> np.ndarray:
    """Compute E_z[S_T(t|x,z) S_C(c|x,z)] under the DVFM prior."""
    predictions = []
    dtype = next(model.parameters()).dtype
    generator = torch.Generator(device=device.type).manual_seed(int(seed))
    model.eval()
    with torch.no_grad():
        for start in range(0, len(evaluation.X), int(batch_size)):
            x = torch.as_tensor(
                evaluation.X[start:start + int(batch_size)],
                dtype=dtype, device=device,
            )
            batch = len(x)
            repeated_x = x.repeat_interleave(int(mc_samples), dim=0)
            z = torch.randn(
                (batch * int(mc_samples), model.latent_dim),
                dtype=dtype, device=device, generator=generator,
            )
            shape_t, scale_t, shape_c, scale_c = model.decoder(repeated_x, z)
            event_grid = torch.as_tensor(
                evaluation.event_grid, dtype=dtype, device=device
            )
            censor_grid = torch.as_tensor(
                evaluation.censor_grid, dtype=dtype, device=device
            )
            event_survival = torch.exp(-(
                event_grid[None, :] / scale_t[:, None]
            ).pow(shape_t[:, None])).reshape(batch, int(mc_samples), -1)
            censor_survival = torch.exp(-(
                censor_grid[None, :] / scale_c[:, None]
            ).pow(shape_c[:, None])).reshape(batch, int(mc_samples), -1)
            joint = torch.einsum(
                "bme,bmc->bec", event_survival, censor_survival
            ) / int(mc_samples)
            predictions.append(joint.cpu().numpy())
    return np.concatenate(predictions, axis=0)


def predict_hacsurv_joint_survival(
    model, evaluation: JointSurvivalEvaluation, *, generator_samples: int,
    batch_size: int, device: torch.device,
) -> np.ndarray:
    """Evaluate HACSurv's learned survival copula and neural margins."""
    predictions = []
    dtype = next(model.parameters()).dtype
    event_grid = torch.as_tensor(
        evaluation.event_grid, dtype=dtype, device=device
    )
    censor_grid = torch.as_tensor(
        evaluation.censor_grid, dtype=dtype, device=device
    )
    model.eval()
    with torch.no_grad():
        for start in range(0, len(evaluation.X), int(batch_size)):
            x = torch.as_tensor(
                evaluation.X[start:start + int(batch_size)],
                dtype=dtype, device=device,
            )
            predictions.append(model.joint_survival_grid(
                x, event_grid, censor_grid, int(generator_samples)
            ).cpu().numpy())
    return np.concatenate(predictions, axis=0)


def oracle_joint_survival_ise(
    prediction: np.ndarray, evaluation: JointSurvivalEvaluation
) -> float:
    """Mean subject-specific 2D integrated squared error, normalized by area."""
    squared = np.square(np.asarray(prediction) - evaluation.truth)
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    over_censor = integrate(squared, evaluation.censor_grid, axis=2)
    over_both = integrate(over_censor, evaluation.event_grid, axis=1)
    area = max(
        float(evaluation.event_grid[-1] * evaluation.censor_grid[-1]), 1e-12
    )
    return float(np.mean(over_both / area))


__all__ = [
    "JointSurvivalEvaluation", "censoring_rate", "collect_metrics",
    "compute_ipcw_brier_ibs", "compute_oracle_brier_ibs",
    "compute_oracle_metrics", "concordance_index",
    "frailty_regression_metrics", "hacsurv_kendall_tau",
    "learned_conditional_kendall_tau",
    "make_survival_evaluator", "median_survival_time", "pearson_correlation",
    "spearman_correlation",
    "oracle_joint_survival_ise",
    "predict_dvfm_joint_survival", "predict_hacsurv_joint_survival",
    "prepare_joint_survival_evaluation",
]
