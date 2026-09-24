"""Go/no-go check for masking a covariate on a real cohort.

Reproduces the paper's cohort for one configured dataset -- the same source
file, time filter, stratified subsample and preprocessing -- then asks whether
the column to be masked is worth masking:

1. how the masked column is distributed, and how censoring varies across it;
2. whether it carries information about the event hazard AND the censoring
   hazard, beyond the covariates that remain;
3. how much of it the remaining covariates already explain, since only the
   unexplained part is recoverable by a latent that is independent of X.

Nothing here fits DVFM or writes an experiment artifact. It reads the existing
configuration without modifying it and prints aggregate statistics only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from lifelines import CoxPHFitter
from scipy.stats import chi2

from utility.semisynthetic import (
    _semisynthetic_preprocessor,
    load_semisynthetic_source,
    stratified_time_event_subsample,
)

FALLBACK_PENALIZER = 0.01


def cohort(spec: dict) -> pd.DataFrame:
    """Return the cohort the paper's pipeline fits, before any masking.

    The ordering matters and matches ``fit_semisynthetic_dgp``: drop
    non-positive durations first, subsample second.
    """
    frame = load_semisynthetic_source(spec)
    time_column = spec["time_column"]
    frame = frame.loc[
        np.isfinite(frame[time_column]) & (frame[time_column] > 0)
    ].reset_index(drop=True)
    subsample = spec.get("subsample")
    if subsample:
        frame = stratified_time_event_subsample(
            frame, time_column=time_column, event_column=spec["event_column"],
            target_size=int(subsample["target_size"]),
            time_bins=int(subsample.get("time_bins", 10)),
            random_seed=int(subsample.get("random_seed", 42)),
        )
    return frame


def describe_levels(frame: pd.DataFrame, spec: dict, column: str) -> pd.DataFrame:
    """Count, censoring rate and median follow-up for each level."""
    event = (
        pd.to_numeric(frame[spec["event_column"]], errors="coerce").fillna(0) > 0
    ).astype(int)
    time = pd.to_numeric(frame[spec["time_column"]], errors="coerce")
    grouped = pd.DataFrame({
        "level": frame[column], "event": event, "time": time,
    }).groupby("level")
    table = grouped.agg(
        n=("event", "size"),
        events=("event", "sum"),
        censoring_rate=("event", lambda values: 1.0 - float(values.mean())),
        median_time=("time", "median"),
        median_time_events=("time", "median"),
    )
    # Median follow-up among the uncensored only, which censoring truncates.
    table["median_time_events"] = grouped.apply(
        lambda part: float(part.loc[part["event"] == 1, "time"].median())
        if (part["event"] == 1).any() else np.nan,
        include_groups=False,
    )
    table["share"] = table["n"] / float(len(frame))
    return table[["n", "share", "events", "censoring_rate", "median_time",
                  "median_time_events"]]


def design(frame: pd.DataFrame, spec: dict, masked: str):
    """Encode the covariates that remain once the masked column is removed."""
    numeric = [name for name in spec["numeric_features"] if name != masked]
    categorical = [name for name in spec["categorical_features"] if name != masked]
    if masked not in spec["numeric_features"] + spec["categorical_features"]:
        raise ValueError(f"{masked!r} is not a configured feature")
    preprocessor = _semisynthetic_preprocessor(
        numeric, categorical,
        numeric_imputation=spec.get("numeric_imputation", "mean"),
    )
    matrix = np.asarray(preprocessor.fit_transform(frame), dtype=float)
    names = list(preprocessor.get_feature_names_out())
    leaked = [name for name in names if masked.lower() in str(name).lower()]
    if leaked:
        raise AssertionError(f"The masked column survived encoding as {leaked}")
    return pd.DataFrame(matrix, columns=names)


def _fit(frame: pd.DataFrame, penalizer: float) -> tuple[CoxPHFitter, float]:
    """Fit a Cox model, relaxing to a small ridge only if the strict fit fails."""
    try:
        model = CoxPHFitter(penalizer=penalizer)
        model.fit(frame, duration_col="duration", event_col="event")
        return model, penalizer
    except Exception as error:
        if penalizer >= FALLBACK_PENALIZER:
            raise
        print(f"    unpenalized fit failed ({type(error).__name__}); "
              f"retrying with penalizer={FALLBACK_PENALIZER}")
        model = CoxPHFitter(penalizer=FALLBACK_PENALIZER)
        model.fit(frame, duration_col="duration", event_col="event")
        return model, FALLBACK_PENALIZER


def hazard_report(
    reduced: pd.DataFrame, block: pd.DataFrame, duration, event, label: str,
    penalizer: float,
) -> dict:
    """Likelihood-ratio test and hazard ratios for one block of terms."""
    base = reduced.copy()
    base["duration"] = np.asarray(duration, dtype=float)
    base["event"] = np.asarray(event, dtype=int)
    full = pd.concat([reduced, block], axis=1)
    full["duration"] = base["duration"]
    full["event"] = base["event"]

    reduced_model, used = _fit(base, penalizer)
    full_model, used_full = _fit(full, penalizer)
    statistic = 2.0 * (full_model.log_likelihood_ - reduced_model.log_likelihood_)
    degrees = block.shape[1]
    p_value = float(chi2.sf(statistic, degrees)) if statistic > 0 else 1.0
    summary = full_model.summary.loc[list(block.columns)]
    print(f"\n  {label}: LR chi2({degrees}) = {statistic:.2f}, p = {p_value:.3g}"
          f"  [penalizer {max(used, used_full):g}]")
    for name, row in summary.iterrows():
        print(f"    {name:<28s} HR = {row['exp(coef)']:.3f} "
              f"[{row['exp(coef) lower 95%']:.3f}, {row['exp(coef) upper 95%']:.3f}]"
              f"  p = {row['p']:.3g}")
    return {
        "label": label, "lr_statistic": float(statistic), "df": int(degrees),
        "p_value": p_value,
        "min_hr": float(summary["exp(coef)"].min()),
        "max_hr": float(summary["exp(coef)"].max()),
    }


def explained_variance(reduced: pd.DataFrame, target: np.ndarray) -> tuple[float, float]:
    """R^2 and adjusted R^2 of the masked column on the covariates that remain."""
    matrix = np.column_stack([np.ones(len(reduced)), reduced.to_numpy(dtype=float)])
    target = np.asarray(target, dtype=float)
    coefficients, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    residual = target - matrix @ coefficients
    total = float(np.sum((target - target.mean()) ** 2))
    r_squared = float(1.0 - np.sum(residual ** 2) / total) if total > 0 else np.nan
    predictors = matrix.shape[1] - 1
    adjusted = 1.0 - (1.0 - r_squared) * (len(target) - 1) / (
        len(target) - predictors - 1
    )
    return r_squared, float(adjusted)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/semi_synthetic.yaml"))
    parser.add_argument("--dataset", default="seer_brain")
    parser.add_argument("--column", default="Grade (thru 2017)",
                        help="The covariate to be masked and recovered.")
    parser.add_argument("--drop-code", type=float, action="append", default=[],
                        help="Exclude rows with this level; repeatable.")
    parser.add_argument("--penalizer", type=float, default=0.0,
                        help="Cox ridge. 0 gives a strict likelihood-ratio test.")
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    specs = [
        item for item in config["data"]["datasets"]
        if str(item["name"]).lower() == args.dataset.lower()
    ]
    if len(specs) != 1:
        raise ValueError(f"Expected one configured dataset named {args.dataset!r}")
    spec = specs[0]

    frame = cohort(spec)
    print(f"Cohort: {args.dataset} | n = {len(frame):,} "
          f"(subsample {spec.get('subsample', {}).get('target_size', 'none')})")

    print(f"\nLevels of {args.column!r} before any exclusion:")
    print(describe_levels(frame, spec, args.column).to_string(
        float_format=lambda value: f"{value:.4g}"
    ))

    if args.drop_code:
        keep = ~frame[args.column].isin(args.drop_code)
        print(f"\nExcluding levels {args.drop_code}: "
              f"{int((~keep).sum()):,} rows removed, {int(keep.sum()):,} remain")
        frame = frame.loc[keep].reset_index(drop=True)
        print(describe_levels(frame, spec, args.column).to_string(
            float_format=lambda value: f"{value:.4g}"
        ))

    event = (
        pd.to_numeric(frame[spec["event_column"]], errors="coerce").fillna(0) > 0
    ).astype(int).to_numpy()
    duration = frame[spec["time_column"]].to_numpy(dtype=float)
    print(f"\nOverall: event rate {event.mean():.4f}, "
          f"censoring rate {1.0 - event.mean():.4f}")

    reduced = design(frame, spec, args.column)
    print(f"Reduced design: {reduced.shape[1]} encoded features, "
          f"{args.column!r} absent (asserted)")

    values = pd.to_numeric(frame[args.column], errors="coerce").to_numpy(dtype=float)
    ordinal = pd.DataFrame({
        "masked_ordinal": (values - values.mean()) / max(values.std(), 1e-12)
    })
    categorical = pd.get_dummies(
        pd.Categorical(frame[args.column]), prefix="masked", drop_first=True,
    ).astype(float).reset_index(drop=True)

    print("\n=== Event hazard ===")
    hazard_report(reduced, ordinal, duration, event,
                  "grade as ordinal (per SD)", args.penalizer)
    hazard_report(reduced, categorical, duration, event,
                  "grade as levels", args.penalizer)

    print("\n=== Censoring hazard ===")
    hazard_report(reduced, ordinal, duration, 1 - event,
                  "grade as ordinal (per SD)", args.penalizer)
    hazard_report(reduced, categorical, duration, 1 - event,
                  "grade as levels", args.penalizer)

    r_squared, adjusted = explained_variance(reduced, values)
    print(f"\n=== Explained by the remaining covariates ===")
    print(f"  R^2 = {r_squared:.4f}   adjusted R^2 = {adjusted:.4f}")
    print(f"  residual SD = {np.sqrt((1 - r_squared)) * values.std():.4f} "
          f"of a total SD of {values.std():.4f}")


if __name__ == "__main__":
    main()
