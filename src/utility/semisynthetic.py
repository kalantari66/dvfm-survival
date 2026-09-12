"""Cox-margin/copula semi-synthetic survival-data generation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri
from scipy.stats import kendalltau, multivariate_normal
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
    source_time: np.ndarray
    source_event: np.ndarray
    source_n_samples_before_subsampling: int
    subsample_target_size: int | None


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
    copula: str = "clayton"


def _semisynthetic_preprocessor(numeric_features=None, categorical_features=None) -> ColumnTransformer:
    """Impute, z-score numeric covariates, and one-hot encode categoricals.

    The historical name of this helper referred to SUPPORT, but this is the
    common preprocessing contract for every real-cohort semi-synthetic DGP.
    """
    numeric_features = SUPPORT_NUMERIC_FEATURES if numeric_features is None else numeric_features
    categorical_features = SUPPORT_CATEGORICAL_FEATURES if categorical_features is None else categorical_features
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, numeric_features),
        ("categorical", categorical, categorical_features),
    ], verbose_feature_names_out=False)


# Kept private for compatibility with any downstream exploratory notebooks.
_support_preprocessor = _semisynthetic_preprocessor


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
    return fit_semisynthetic_dgp(dict(
        path=path, time_column="duration", event_column="event",
        numeric_features=SUPPORT_NUMERIC_FEATURES,
        categorical_features=SUPPORT_CATEGORICAL_FEATURES,
    ), cox_penalizer=cox_penalizer)


def _load_builtin_semisynthetic_source(loader: str) -> pd.DataFrame:
    """Load cohorts distributed by scikit-survival into the common schema."""
    try:
        from sksurv.datasets import load_flchain, load_whas500
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "Built-in semi-synthetic datasets require scikit-survival. "
            "Install the project environment before running WHAS or FLCHAIN."
        ) from exc
    name = str(loader).lower()
    if name == "whas500":
        covariates, outcome = load_whas500()
        frame = pd.DataFrame(covariates)
        frame["time"] = outcome["lenfol"]
        frame["event"] = outcome["fstat"]
        return frame
    if name == "flchain":
        covariates, outcome = load_flchain()
        frame = pd.DataFrame(covariates)
        frame["time"] = outcome["futime"]
        frame["event"] = outcome["death"]
        return frame
    raise ValueError(f"Unsupported built-in semi-synthetic loader: {loader}")


def _source_label(spec: dict) -> str:
    return str(spec.get("path", f"builtin:{spec.get('loader', 'unknown')}"))


def stratified_time_event_subsample(
    frame: pd.DataFrame,
    *,
    time_column: str,
    event_column: str,
    target_size: int,
    time_bins: int = 10,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Draw an exact sample stratified by event status and quantile time bins.

    This follows the Lillelund et al. semi-synthetic protocol. Exact
    largest-remainder allocation preserves the joint stratum proportions while
    avoiding the size drift introduced by independently rounding each stratum.
    """
    target_size = int(target_size)
    if target_size < 1:
        raise ValueError("subsample.target_size must be positive")
    if target_size >= len(frame):
        return frame.reset_index(drop=True).copy()
    if int(time_bins) < 2:
        raise ValueError("subsample.time_bins must be at least 2")
    event = (pd.to_numeric(frame[event_column], errors="coerce").fillna(0) > 0).astype(int)
    time_bin = pd.qcut(
        frame[time_column], q=min(int(time_bins), len(frame)),
        labels=False, duplicates="drop",
    )
    strata = pd.DataFrame({"event": event, "time_bin": time_bin}, index=frame.index)
    counts = strata.value_counts(sort=False).sort_index()
    expected = counts.to_numpy(dtype=float) * target_size / len(frame)
    allocation = np.floor(expected).astype(int)
    remainder = target_size - int(allocation.sum())
    for position in np.argsort(-(expected - allocation), kind="stable")[:remainder]:
        allocation[position] += 1

    rng = np.random.default_rng(int(random_seed))
    selected: list[np.ndarray] = []
    for ((status, bin_id), _), n_draw in zip(counts.items(), allocation):
        if n_draw == 0:
            continue
        members = strata.index[
            (strata["event"] == status) & (strata["time_bin"] == bin_id)
        ].to_numpy()
        selected.append(rng.choice(members, size=int(n_draw), replace=False))
    indices = np.concatenate(selected)
    return frame.loc[indices[rng.permutation(len(indices))]].reset_index(drop=True).copy()


def load_semisynthetic_source(spec):
    """Read and check a real cohort before fitting semi-synthetic margins."""
    if spec.get("loader"):
        frame = _load_builtin_semisynthetic_source(spec["loader"])
    else:
        path = Path(spec["path"])
        if path.suffix.lower() in {".feather", ".ftr"}:
            frame = pd.read_feather(path)
        elif path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
        elif path.suffix.lower() in {".parquet", ".pq"}:
            frame = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported semi-synthetic input format: {path}")
    rename_columns = spec.get("rename_columns", {})
    if rename_columns:
        frame = frame.rename(columns=rename_columns)
    drop_columns = spec.get("drop_columns", [])
    if drop_columns:
        frame = frame.drop(columns=drop_columns)
    required = set(spec["numeric_features"] + spec["categorical_features"] +
                   [spec["time_column"], spec["event_column"]])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{_source_label(spec)} is missing columns: {missing}")
    return frame


# Compatibility spelling used by the first SUPPORT-only runner.
read_semisynthetic_source = load_semisynthetic_source


def fit_semisynthetic_dgp(spec, *, cox_penalizer=0.01):
    """Fit Cox margins for a dataset with explicitly configured covariates."""
    frame = load_semisynthetic_source(spec)
    time_column, event_column = spec["time_column"], spec["event_column"]
    frame = frame.loc[np.isfinite(frame[time_column]) & (frame[time_column] > 0)].reset_index(drop=True)
    source_n_samples_before_subsampling = len(frame)
    subsample = spec.get("subsample")
    if subsample:
        frame = stratified_time_event_subsample(
            frame, time_column=time_column, event_column=event_column,
            target_size=int(subsample["target_size"]),
            time_bins=int(subsample.get("time_bins", 10)),
            random_seed=int(subsample.get("random_seed", 42)),
        )
    event = (pd.to_numeric(frame[event_column], errors="coerce").fillna(0) > 0).astype(int).to_numpy()
    preprocessor = _semisynthetic_preprocessor(spec["numeric_features"], spec["categorical_features"])
    X = np.asarray(preprocessor.fit_transform(frame), dtype=np.float64)
    feature_names = list(preprocessor.get_feature_names_out())
    duration = frame[time_column].to_numpy(dtype=float)
    event_margin = _fit_cox_margin(X, feature_names, duration, event, cox_penalizer)
    censor_margin = _fit_cox_margin(X, feature_names, duration, 1 - event, cox_penalizer)
    return SupportSemiSyntheticDGP(
        X=X.astype(np.float32), feature_names=feature_names,
        event_margin=event_margin, censor_margin=censor_margin,
        source_n_samples=len(frame), source_event_rate=float(event.mean()),
        cox_penalizer=float(cox_penalizer), source_time=duration,
        source_event=event,
        source_n_samples_before_subsampling=source_n_samples_before_subsampling,
        subsample_target_size=None if not subsample else int(subsample["target_size"]),
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


@lru_cache(maxsize=None)
def frank_theta_from_tau(kendall_tau: float) -> float:
    """Invert Frank's Kendall-tau relationship for a positive dependence target."""
    tau = float(kendall_tau)
    if tau == 0.0:
        return 0.0

    def tau_from_theta(theta: float) -> float:
        # Algebraically x / (exp(x) - 1), evaluated without overflow for large x.
        debye_1 = quad(
            lambda x: 1.0 if x == 0.0 else x * np.exp(-x) / (1.0 - np.exp(-x)),
            0.0, theta, limit=100,
        )[0] / theta
        return 1.0 - 4.0 / theta + 4.0 * debye_1 / theta

    return float(brentq(lambda theta: tau_from_theta(theta) - tau, 1e-8, 1e4))


def sample_semisynthetic_uniforms(n_samples: int, copula: str, kendall_tau: float, seed: int):
    """Sample event/censor survival uniforms for the supported semi-synthetic copulas."""
    name = str(copula).lower()
    tau = float(kendall_tau)
    if not 0.0 <= tau < 1.0:
        raise ValueError("kendall_tau must be in [0, 1)")
    if name == "clayton":
        uniforms, theta, frailty = sample_clayton_uniforms(n_samples, tau, seed, return_frailty=True)
        return uniforms, theta, frailty
    rng = np.random.default_rng(seed)
    if name == "gaussian":
        rho = np.sin(np.pi * tau / 2.0)
        normals = rng.multivariate_normal([0.0, 0.0], [[1.0, rho], [rho, 1.0]], size=int(n_samples))
        return np.clip(ndtr(normals), 1e-12, 1.0 - 1e-12), float(rho), None
    if name == "frank":
        theta = frank_theta_from_tau(tau)
        if theta == 0.0:
            return rng.uniform(size=(int(n_samples), 2)), theta, None
        u, w = rng.uniform(size=(2, int(n_samples)))
        a = np.exp(-theta * u)
        b = np.exp(-theta)
        v = -np.log1p(w * (b - 1.0) / (a - w * (a - 1.0))) / theta
        return np.column_stack([u, np.clip(v, 1e-12, 1.0 - 1e-12)]), theta, None
    if name == "gumbel":
        # Gumbel's Archimedean parameter is theta=1/(1-tau).  Its
        # Marshall--Olkin representation uses a positive alpha-stable shared
        # frailty, alpha=1/theta.  At tau=0 the copula is exactly product.
        if tau == 0.0:
            return rng.uniform(size=(int(n_samples), 2)), 1.0, None
        theta = 1.0 / (1.0 - tau)
        alpha = 1.0 / theta
        angle = rng.uniform(1e-12, np.pi - 1e-12, size=int(n_samples))
        exponential = rng.exponential(size=int(n_samples))
        frailty = (
            np.sin(alpha * angle) / np.power(np.sin(angle), 1.0 / alpha)
            * np.power(np.sin((1.0 - alpha) * angle) / exponential, (1.0 - alpha) / alpha)
        )
        noise = rng.exponential(size=(int(n_samples), 2))
        uniforms = np.exp(-np.power(noise / frailty[:, None], alpha))
        return np.clip(uniforms, 1e-12, 1.0 - 1e-12), float(theta), frailty
    raise ValueError(f"Unsupported semi-synthetic copula: {copula}")


def generate_semisynthetic(
    dgp: SupportSemiSyntheticDGP,
    *,
    kendall_tau: float,
    censoring_rate: float,
    sampling_seed: int,
    copula: str = "clayton",
) -> SemiSyntheticSample:
    """Resample complete event/censor times while retaining source covariates."""
    target_censoring = float(censoring_rate)
    if not 0.0 < target_censoring < 1.0:
        raise ValueError("censoring_rate must be between 0 and 1")
    uniforms, theta, frailty = sample_semisynthetic_uniforms(
        len(dgp.X), copula, kendall_tau, sampling_seed
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
        censor_time_scale=censor_scale, copula=str(copula).lower(),
    )


# Public backwards compatibility for the original SUPPORT-only notebook API.
generate_support_semisynthetic = generate_semisynthetic


def semi_synthetic_joint_survival(dgp, sample, X, event_grid, censor_grid):
    """Exact copula joint survival surface for fitted Cox margins of one DGP."""
    x = np.asarray(X, dtype=float)
    event_hazard = np.interp(event_grid, dgp.event_margin.baseline_times,
                             dgp.event_margin.baseline_cumulative_hazard)
    censor_hazard = np.interp(censor_grid / sample.censor_time_scale, dgp.censor_margin.baseline_times,
                              dgp.censor_margin.baseline_cumulative_hazard)
    u = np.exp(-dgp.event_margin.relative_risk(x)[:, None] * event_hazard[None, :])
    v = np.exp(-dgp.censor_margin.relative_risk(x)[:, None] * censor_hazard[None, :])
    copula = sample.copula
    theta = sample.clayton_theta
    if copula == "clayton":
        return np.maximum(u[:, :, None] ** (-theta) + v[:, None, :] ** (-theta) - 1.0, 1e-12) ** (-1.0 / theta) if theta else u[:, :, None] * v[:, None, :]
    if copula == "frank":
        if theta == 0.0:
            return u[:, :, None] * v[:, None, :]
        numerator = np.expm1(-theta * u[:, :, None]) * np.expm1(-theta * v[:, None, :])
        return -np.log1p(numerator / np.expm1(-theta)) / theta
    if copula == "gaussian":
        rho = theta
        points = np.stack(np.broadcast_arrays(ndtri(u[:, :, None]), ndtri(v[:, None, :])), axis=-1)
        flat = points.reshape(-1, 2)
        return multivariate_normal.cdf(flat, mean=[0.0, 0.0], cov=[[1.0, rho], [rho, 1.0]]).reshape(points.shape[:-1])
    if copula == "gumbel":
        if theta == 1.0:
            return u[:, :, None] * v[:, None, :]
        log_u = -np.log(np.clip(u[:, :, None], 1e-12, 1.0))
        log_v = -np.log(np.clip(v[:, None, :], 1e-12, 1.0))
        return np.exp(-np.power(log_u ** theta + log_v ** theta, 1.0 / theta))
    raise ValueError(f"Unsupported semi-synthetic copula: {copula}")


def plot_event_distribution_comparison(dgp, sample, output_path, *, title: str):
    """Compare source and generated observed-time distributions on shared axes.

    The event and censoring margins are fitted independently from the source
    cohort, so this plot deliberately checks marginals rather than copula
    dependence.  The copula governs their joint association, not either target
    marginal distribution.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from lifelines import KaplanMeierFitter

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_time = np.asarray(dgp.source_time, dtype=float)
    source_event = np.asarray(dgp.source_event, dtype=int)
    generated_time = np.asarray(sample.data.time, dtype=float)
    generated_event = np.asarray(sample.data.event, dtype=int)
    maximum = float(max(source_time.max(), generated_time.max()))
    intervals = math.ceil(math.log2(max(len(source_time), len(generated_time))) + 1)
    bins = np.linspace(0.0, maximum, intervals)
    colors = {"event": "#55A868", "censored": "#4C72B0"}

    fig = Figure(figsize=(12.0, 4.8))
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 2, sharex=True, sharey=True)
    for ax, name, times, events in (
        (axes[0], "Original cohort", source_time, source_event),
        (axes[1], "Semi-synthetic cohort", generated_time, generated_event),
    ):
        km = KaplanMeierFitter().fit(times, event_observed=events)
        ax.step(km.survival_function_.index, km.survival_function_.iloc[:, 0],
                color="#0173B2", linewidth=2.5, where="post", zorder=3)
        ax.set(title=f"{name} (censoring = {100 * (1 - events.mean()):.1f}%)",
               xlabel="Time", ylim=(0, 1.05), xlim=(0, maximum))
        ax.set_ylabel("Survival probability")
        ax.grid(visible=True, which="major", linestyle="--", linewidth=0.5, alpha=0.4)

        counts = ax.twinx()
        counts.hist(
            [times[events == 1], times[events == 0]], bins=bins,
            histtype="barstacked", stacked=True, alpha=0.78,
            color=[colors["event"], colors["censored"]], zorder=2,
            label=["Event", "Censored"],
        )
        counts.set_ylabel("Count")
        counts.yaxis.grid(False)
        counts.legend(loc="upper right", frameon=True)
        ax.set_zorder(counts.get_zorder() + 1)
        ax.patch.set_visible(False)

    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")


__all__ = [
    "CoxMargin", "SemiSyntheticSample", "SupportSemiSyntheticDGP",
    "fit_support_semisynthetic_dgp", "fit_semisynthetic_dgp", "load_semisynthetic_source", "read_semisynthetic_source",
    "stratified_time_event_subsample",
    "generate_semisynthetic", "generate_support_semisynthetic",
    "sample_clayton_uniforms", "sample_semisynthetic_uniforms",
    "semi_synthetic_joint_survival", "plot_event_distribution_comparison",
]
