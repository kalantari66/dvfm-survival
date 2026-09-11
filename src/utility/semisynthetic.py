"""Cox-margin/copula semi-synthetic survival-data generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.stats import kendalltau
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import SurvivalData


SUPPORT_NUMERIC_FEATURES = ["x0", "x7", "x8", "x9", "x10", "x11", "x12", "x13"]
SUPPORT_CATEGORICAL_FEATURES = ["x1", "x2", "x3", "x4", "x5", "x6"]


@dataclass
class CoxMargin:
    """A fitted Cox margin with an inverse conditional-survival transform."""

    model: CoxPHFitter
    feature_names: list[str]
    baseline_times: np.ndarray
    baseline_cumulative_hazard: np.ndarray

    def relative_risk(self, X: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(np.asarray(X, dtype=float), columns=self.feature_names)
        return self.model.predict_partial_hazard(frame).to_numpy(dtype=float).reshape(-1)

    def inverse_survival(self, uniforms: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Return ``t`` satisfying the fitted Cox survival ``S(t|X)=u``."""
        u = np.clip(np.asarray(uniforms, dtype=float), 1e-12, 1.0 - 1e-12)
        targets = -np.log(u) / np.clip(self.relative_risk(X), 1e-12, None)
        hazard = self.baseline_cumulative_hazard
        times = self.baseline_times
        sampled = np.interp(targets, hazard, times)

        # Breslow's estimate ends at the last observed margin event. Linear
        # cumulative-hazard extrapolation avoids piling extreme draws at that time.
        above = targets > hazard[-1]
        if np.any(above):
            positive = np.flatnonzero(np.diff(hazard) > 0)
            if positive.size:
                start = max(0, int(positive[-1]) - 9)
                delta_h = hazard[-1] - hazard[start]
                delta_t = times[-1] - times[start]
                rate = delta_h / max(delta_t, 1e-12)
            else:
                rate = hazard[-1] / max(times[-1], 1e-12)
            sampled[above] = times[-1] + (targets[above] - hazard[-1]) / max(rate, 1e-12)
        return np.clip(sampled, 1e-8, None)


@dataclass
class SupportSemiSyntheticDGP:
    X: np.ndarray
    feature_names: list[str]
    event_margin: CoxMargin
    censor_margin: CoxMargin
    source_n_samples: int
    source_event_rate: float
    cox_penalizer: float


@dataclass
class SemiSyntheticSample:
    data: SurvivalData
    target_kendall_tau: float
    empirical_copula_kendall_tau: float
    empirical_marginal_kendall_tau: float
    clayton_theta: float
    target_censoring_rate: float
    achieved_censoring_rate: float
    censor_time_scale: float


def _support_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, SUPPORT_NUMERIC_FEATURES),
        ("categorical", categorical, SUPPORT_CATEGORICAL_FEATURES),
    ], verbose_feature_names_out=False)


def _fit_cox_margin(X: np.ndarray, feature_names: list[str], time, event, penalizer: float) -> CoxMargin:
    frame = pd.DataFrame(X, columns=feature_names)
    frame["duration"] = np.asarray(time, dtype=float)
    frame["event"] = np.asarray(event, dtype=int)
    model = CoxPHFitter(penalizer=float(penalizer))
    model.fit(frame, duration_col="duration", event_col="event", show_progress=False)
    baseline = model.baseline_cumulative_hazard_.iloc[:, 0]
    hazard = baseline.to_numpy(dtype=float)
    times = baseline.index.to_numpy(dtype=float)
    keep = np.r_[True, np.diff(hazard) > 0]
    hazard, times = hazard[keep], times[keep]
    if not len(hazard) or hazard[-1] <= 0:
        raise RuntimeError("Cox margin has no positive baseline cumulative hazard")
    return CoxMargin(model, feature_names, times, hazard)


def fit_support_semisynthetic_dgp(path: str | Path, *, cox_penalizer: float = 0.01) -> SupportSemiSyntheticDGP:
    """Load SUPPORT, preprocess its covariates, and fit event/censor Cox margins."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_feather(path)
    required = set(SUPPORT_NUMERIC_FEATURES + SUPPORT_CATEGORICAL_FEATURES + ["duration", "event"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"SUPPORT file is missing columns: {missing}")
    frame = frame.loc[np.isfinite(frame["duration"]) & (frame["duration"] > 0)].reset_index(drop=True)
    event = (pd.to_numeric(frame["event"], errors="coerce").fillna(0) > 0).astype(int).to_numpy()
    preprocessor = _support_preprocessor()
    X = np.asarray(preprocessor.fit_transform(frame), dtype=np.float64)
    feature_names = list(preprocessor.get_feature_names_out())
    duration = frame["duration"].to_numpy(dtype=float)
    event_margin = _fit_cox_margin(X, feature_names, duration, event, cox_penalizer)
    censor_margin = _fit_cox_margin(X, feature_names, duration, 1 - event, cox_penalizer)
    return SupportSemiSyntheticDGP(
        X=X.astype(np.float32), feature_names=feature_names,
        event_margin=event_margin, censor_margin=censor_margin,
        source_n_samples=len(frame), source_event_rate=float(event.mean()),
        cox_penalizer=float(cox_penalizer),
    )


def sample_clayton_uniforms(n_samples: int, kendall_tau: float, seed: int, *, return_frailty=False):
    """Sample bivariate Clayton uniforms using its Gamma frailty representation."""
    tau = float(kendall_tau)
    if not 0.0 <= tau < 1.0:
        raise ValueError("Clayton kendall_tau must be in [0, 1)")
    if tau == 0.0:
        uniforms = np.random.default_rng(seed).uniform(size=(int(n_samples), 2))
        return (uniforms, 0.0, None) if return_frailty else (uniforms, 0.0)
    theta = 2.0 * tau / (1.0 - tau)
    rng = np.random.default_rng(seed)
    frailty = rng.gamma(shape=1.0 / theta, scale=1.0, size=int(n_samples))
    noise = rng.exponential(size=(int(n_samples), 2))
    uniforms = (1.0 + noise / frailty[:, None]) ** (-1.0 / theta)
    result = (np.clip(uniforms, 1e-12, 1.0 - 1e-12), theta)
    return (*result, frailty) if return_frailty else result


def generate_support_semisynthetic(
    dgp: SupportSemiSyntheticDGP,
    *,
    kendall_tau: float,
    censoring_rate: float,
    sampling_seed: int,
) -> SemiSyntheticSample:
    """Resample complete event/censor times while retaining SUPPORT covariates."""
    target_censoring = float(censoring_rate)
    if not 0.0 < target_censoring < 1.0:
        raise ValueError("censoring_rate must be between 0 and 1")
    uniforms, theta, frailty = sample_clayton_uniforms(
        len(dgp.X), kendall_tau, sampling_seed, return_frailty=True
    )
    # Match the diagnostic workflow's cohort-standardized log-frailty target.
    # Independence has no shared random frailty to recover.
    true_z = None
    if frailty is not None:
        log_frailty = np.log(np.clip(frailty, 1e-12, None))
        true_z = (log_frailty - log_frailty.mean()) / max(log_frailty.std(), 1e-12)
    event_time = dgp.event_margin.inverse_survival(uniforms[:, 0], dgp.X)
    censor_base = dgp.censor_margin.inverse_survival(uniforms[:, 1], dgp.X)

    # C=s*C0 is censored exactly when E/C0>s. With continuous draws, this
    # quantile calibration differs from the requested finite-sample rate by at
    # most one subject and is shared by all fitted methods in the scenario.
    ratio = event_time / np.clip(censor_base, 1e-12, None)
    censor_scale = float(np.quantile(ratio, 1.0 - target_censoring))
    censor_time = np.clip(censor_scale * censor_base, 1e-8, None)
    event = (event_time <= censor_time).astype(int)
    observed = np.minimum(event_time, censor_time)
    data = SurvivalData(
        X=dgp.X.copy(), time=observed, event=event,
        feature_names=list(dgp.feature_names),
        true_event_time=event_time, true_censor_time=censor_time,
        true_z=true_z,
    )
    return SemiSyntheticSample(
        data=data, target_kendall_tau=float(kendall_tau),
        empirical_copula_kendall_tau=float(kendalltau(uniforms[:, 0], uniforms[:, 1]).statistic),
        empirical_marginal_kendall_tau=float(kendalltau(event_time, censor_time).statistic),
        clayton_theta=float(theta), target_censoring_rate=target_censoring,
        achieved_censoring_rate=float(1.0 - event.mean()),
        censor_time_scale=censor_scale,
    )


__all__ = [
    "CoxMargin", "SemiSyntheticSample", "SupportSemiSyntheticDGP",
    "fit_support_semisynthetic_dgp", "generate_support_semisynthetic",
    "sample_clayton_uniforms",
]
