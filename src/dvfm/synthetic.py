"""Controlled synthetic data generators for DVFM experiments."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.stats import kendalltau
from .reference_core import generate_copula_data


@dataclass
class GaussianFrailtySample:
    X: np.ndarray
    observed_time: np.ndarray
    event: np.ndarray
    event_time: np.ndarray
    censor_time: np.ndarray
    true_z: np.ndarray
    frailty_loading: float
    censor_intercept: float
    empirical_conditional_kendall_tau: float
    empirical_marginal_kendall_tau: float
    achieved_censoring_rate: float
    beta_event: np.ndarray
    beta_censor: np.ndarray
    shape_event: float
    shape_censor: float


@dataclass
class ClaytonAFTSample:
    """Sample from the exact parametric family fitted by ClaytonWeibullAFT."""

    X: np.ndarray
    observed_time: np.ndarray
    event: np.ndarray
    event_time: np.ndarray
    censor_time: np.ndarray
    beta_event: np.ndarray
    beta_censor: np.ndarray
    shape_event: float
    shape_censor: float
    clayton_theta: float


@dataclass
class ClaytonGammaFrailtySample:
    """Marshall--Olkin Clayton sample with its generating Gamma frailty."""

    X: np.ndarray
    observed_time: np.ndarray
    event: np.ndarray
    event_time: np.ndarray
    censor_time: np.ndarray
    true_z: np.ndarray
    true_frailty: np.ndarray
    clayton_theta: float
    censor_intercept: float
    empirical_conditional_kendall_tau: float
    empirical_marginal_kendall_tau: float
    achieved_censoring_rate: float
    beta_event: np.ndarray
    beta_censor: np.ndarray
    shape_event: float
    shape_censor: float


def generate_clayton_gamma_frailty(
    *,
    n_samples: int,
    n_features: int,
    kendall_tau: float,
    censoring_rate: float,
    dgp_seed: int,
    sampling_seed: int,
) -> ClaytonGammaFrailtySample:
    """Reproduce the historical Clayton Gamma-frailty construction.

    ``true_z`` is standardized log Gamma frailty, matching the target used in
    the earlier successful subject-level recovery experiment. The marginal
    Weibull regression is synthetic here; the historical SUPPORT experiment
    used fitted Cox margins with the same Marshall--Olkin latent construction.
    """
    tau = float(kendall_tau)
    if not 0.0 < tau < 1.0:
        raise ValueError("Clayton Gamma frailty requires kendall_tau in (0, 1)")
    if not 0.0 < float(censoring_rate) < 1.0:
        raise ValueError("censoring_rate must be between 0 and 1")
    theta = 2.0 * tau / (1.0 - tau)
    parameter_rng = np.random.default_rng(dgp_seed)
    beta_event = parameter_rng.normal(0.0, 0.25, size=n_features)
    beta_censor = parameter_rng.normal(0.0, 0.25, size=n_features)
    rng = np.random.default_rng(sampling_seed)
    X = rng.normal(size=(n_samples, n_features))
    frailty = rng.gamma(shape=1.0 / theta, scale=1.0, size=n_samples)
    event_noise = rng.exponential(size=n_samples)
    censor_noise = rng.exponential(size=n_samples)
    u_event = np.clip((1.0 + event_noise / frailty) ** (-1.0 / theta), 1e-8, 1.0 - 1e-8)
    u_censor = np.clip((1.0 + censor_noise / frailty) ** (-1.0 / theta), 1e-8, 1.0 - 1e-8)
    event_time = np.exp(X @ beta_event) * (-np.log(u_event)) ** (1.0 / 1.5)
    censor_base = np.exp(X @ beta_censor) * (-np.log(u_censor)) ** (1.0 / 1.3)
    censor_intercept = float(np.quantile(
        np.log(event_time) - np.log(censor_base), 1.0 - float(censoring_rate)
    ))
    censor_time = np.exp(censor_intercept) * censor_base
    event = (event_time <= censor_time).astype(int)
    log_frailty = np.log(np.clip(frailty, 1e-12, None))
    true_z = (log_frailty - log_frailty.mean()) / max(log_frailty.std(), 1e-12)
    event_residual = np.log(event_time) - X @ beta_event
    censor_residual = np.log(censor_time) - X @ beta_censor - censor_intercept
    return ClaytonGammaFrailtySample(
        X=X.astype(np.float32), observed_time=np.minimum(event_time, censor_time),
        event=event, event_time=event_time, censor_time=censor_time,
        true_z=true_z, true_frailty=frailty, clayton_theta=theta,
        censor_intercept=censor_intercept,
        empirical_conditional_kendall_tau=float(kendalltau(event_residual, censor_residual).statistic),
        empirical_marginal_kendall_tau=float(kendalltau(event_time, censor_time).statistic),
        achieved_censoring_rate=float(1.0 - event.mean()),
        beta_event=beta_event, beta_censor=beta_censor,
        shape_event=1.5, shape_censor=1.3,
    )


def generate_clayton_aft_data(
    *,
    n_samples: int,
    n_features: int,
    clayton_theta: float,
    seed: int,
    beta_event: np.ndarray | None = None,
    beta_censor: np.ndarray | None = None,
    shape_event: float = 1.5,
    shape_censor: float = 1.3,
) -> ClaytonAFTSample:
    """Generate data exactly matching ClaytonWeibullAFT's likelihood.

    Coefficient vectors include the intercept as their final element, matching
    the baseline implementation's augmented design matrix.
    """
    theta = float(clayton_theta)
    if theta <= 0:
        raise ValueError("clayton_theta must be positive")
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    if beta_event is None:
        beta_event = np.r_[np.linspace(-0.25, 0.25, n_features), 0.2]
    if beta_censor is None:
        beta_censor = np.r_[np.linspace(0.2, -0.2, n_features), 0.0]
    beta_event = np.asarray(beta_event, dtype=float)
    beta_censor = np.asarray(beta_censor, dtype=float)
    if beta_event.shape != (n_features + 1,) or beta_censor.shape != (n_features + 1,):
        raise ValueError("beta_event and beta_censor must each have n_features + 1 entries")

    # Marshall-Olkin sampling gives uniforms with a Clayton copula.
    shared = rng.gamma(shape=1.0 / theta, scale=1.0, size=n_samples)
    u_event = (1.0 + rng.exponential(size=n_samples) / shared) ** (-1.0 / theta)
    u_censor = (1.0 + rng.exponential(size=n_samples) / shared) ** (-1.0 / theta)
    X_aug = np.column_stack([X, np.ones(n_samples)])
    scale_event = np.exp(X_aug @ beta_event)
    scale_censor = np.exp(X_aug @ beta_censor)
    event_time = scale_event * (-np.log1p(-u_event)) ** (1.0 / float(shape_event))
    censor_time = scale_censor * (-np.log1p(-u_censor)) ** (1.0 / float(shape_censor))
    event = (event_time <= censor_time).astype(int)
    return ClaytonAFTSample(
        X=X.astype(np.float32), observed_time=np.minimum(event_time, censor_time), event=event,
        event_time=event_time, censor_time=censor_time, beta_event=beta_event,
        beta_censor=beta_censor, shape_event=float(shape_event), shape_censor=float(shape_censor),
        clayton_theta=theta,
    )


def _conditional_tau(loading: float, seed: int, n: int) -> float:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    e = loading * z + np.log(-np.log(rng.uniform(size=n))) / 1.5
    c = loading * z + np.log(-np.log(rng.uniform(size=n))) / 1.3
    return float(kendalltau(e, c).statistic)


def calibrate_gaussian_frailty_loading(kendall_tau: float, *, seed: int, n_samples: int = 50_000, steps: int = 24) -> float:
    """Calibrate a common log-scale loading to conditional Kendall's tau."""
    target = float(kendall_tau)
    if not 0.0 <= target < 1.0:
        raise ValueError("kendall_tau must be in [0, 1)")
    if target == 0.0:
        return 0.0
    low, high = 0.0, 16.0
    for _ in range(steps):
        mid = (low + high) / 2.0
        if _conditional_tau(mid, seed, n_samples) < target:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def generate_gaussian_shared_frailty(*, n_samples: int, n_features: int, kendall_tau: float, censoring_rate: float, dgp_seed: int, sampling_seed: int, calibration_samples: int = 50_000) -> GaussianFrailtySample:
    """Generate Weibull times with one known Gaussian shared frailty.

    The censor intercept is calibrated once on the generated sample. All model
    variants then receive the identical observed times and censoring indicators.
    """
    if not 0.0 < float(censoring_rate) < 1.0:
        raise ValueError("censoring_rate must be between 0 and 1")
    loading = calibrate_gaussian_frailty_loading(kendall_tau, seed=dgp_seed + 17, n_samples=calibration_samples)
    parameter_rng = np.random.default_rng(dgp_seed)
    beta_event = parameter_rng.normal(0.0, 0.25, size=n_features)
    beta_censor = parameter_rng.normal(0.0, 0.25, size=n_features)
    rng = np.random.default_rng(sampling_seed)
    X = rng.normal(size=(n_samples, n_features))
    z = rng.normal(size=n_samples)
    u_event = np.clip(rng.uniform(size=n_samples), 1e-8, 1.0 - 1e-8)
    u_censor = np.clip(rng.uniform(size=n_samples), 1e-8, 1.0 - 1e-8)
    event_time = np.exp(X @ beta_event + loading * z) * (-np.log(u_event)) ** (1.0 / 1.5)
    censor_base = np.exp(X @ beta_censor + loading * z) * (-np.log(u_censor)) ** (1.0 / 1.3)
    # Censoring occurs when log(T) - log(C_base) exceeds the intercept.
    censor_intercept = float(np.quantile(
        np.log(event_time) - np.log(censor_base), 1.0 - float(censoring_rate)
    ))
    censor_time = np.exp(censor_intercept) * censor_base
    event = (event_time <= censor_time).astype(int)
    event_residual = np.log(event_time) - X @ beta_event
    censor_residual = np.log(censor_time) - X @ beta_censor - censor_intercept
    return GaussianFrailtySample(
        X=X.astype(np.float32), observed_time=np.minimum(event_time, censor_time), event=event,
        event_time=event_time, censor_time=censor_time, true_z=z,
        frailty_loading=loading, censor_intercept=censor_intercept,
        empirical_conditional_kendall_tau=float(kendalltau(event_residual, censor_residual).statistic),
        empirical_marginal_kendall_tau=float(kendalltau(event_time, censor_time).statistic),
        achieved_censoring_rate=float(1.0 - event.mean()),
        beta_event=beta_event, beta_censor=beta_censor,
        shape_event=1.5, shape_censor=1.3,
    )


__all__ = ["ClaytonAFTSample", "ClaytonGammaFrailtySample", "GaussianFrailtySample",
           "generate_clayton_aft_data", "generate_clayton_gamma_frailty",
           "calibrate_gaussian_frailty_loading", "generate_gaussian_shared_frailty",
           "generate_copula_data"]
