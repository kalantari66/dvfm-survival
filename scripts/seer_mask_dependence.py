"""Quantify the event/censoring dependence that masking a covariate induces.

Both Cox models are fitted WITH the covariate to be masked, so event and
censoring times are conditionally independent given the full covariate vector.
Marginalising the masked covariate out -- exactly what a model that never sees
it must do -- leaves a dependence between the two times at a fixed reduced-X
profile. Simulating from the fitted Breslow survival functions measures that
dependence as a Kendall's tau, before any model is fitted.

This is the number that says what the masked-covariate experiment can show: a
large tau means the omitted covariate is a genuine shared-frailty mechanism, a
small one means the experiment tests recovery of omitted heterogeneity rather
than of dependence.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from lifelines import CoxPHFitter
from scipy.stats import kendalltau

from seer_mask_check import cohort, design

FALLBACK_PENALIZER = 0.01


def _fit(frame: pd.DataFrame, penalizer: float) -> tuple[CoxPHFitter, float]:
    try:
        model = CoxPHFitter(penalizer=penalizer)
        model.fit(frame, duration_col="duration", event_col="event")
        return model, penalizer
    except Exception as error:
        if penalizer >= FALLBACK_PENALIZER:
            raise
        print(f"  unpenalized fit failed ({type(error).__name__}); "
              f"retrying with penalizer={FALLBACK_PENALIZER}")
        model = CoxPHFitter(penalizer=FALLBACK_PENALIZER)
        model.fit(frame, duration_col="duration", event_col="event")
        return model, FALLBACK_PENALIZER


def baseline_curve(model: CoxPHFitter) -> tuple[np.ndarray, np.ndarray]:
    """Breslow baseline survival as (times, S0), both increasing in index."""
    curve = model.baseline_survival_.iloc[:, 0]
    times = np.asarray(curve.index, dtype=float)
    survival = np.asarray(curve.to_numpy(), dtype=float)
    order = np.argsort(times)
    return times[order], survival[order]


def sample_times(
    times: np.ndarray, survival: np.ndarray, linear_predictor: np.ndarray,
    uniforms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert S(t | x) = S0(t) ** exp(eta) at the given uniform draws.

    Returns the sampled times and a mask marking draws that never reach the
    target survival level inside the fitted support, which are administratively
    truncated at the last observed time rather than extrapolated.
    """
    target = uniforms ** np.exp(-np.asarray(linear_predictor, dtype=float))
    # survival is non-increasing, so negate to search an increasing array.
    index = np.searchsorted(-survival, -target, side="left")
    truncated = index >= len(times)
    index = np.clip(index, 0, len(times) - 1)
    return times[index], truncated


def profiles(reduced: pd.DataFrame, coefficients: pd.Series) -> dict[str, np.ndarray]:
    """Three real subjects: median, low and high risk under the event model."""
    columns = [name for name in reduced.columns if name in coefficients.index]
    score = reduced[columns].to_numpy(dtype=float) @ coefficients[columns].to_numpy()
    order = np.argsort(score)
    picks = {
        "low_risk": order[int(0.10 * (len(order) - 1))],
        "median": order[int(0.50 * (len(order) - 1))],
        "high_risk": order[int(0.90 * (len(order) - 1))],
    }
    return {name: reduced.iloc[index].to_numpy(dtype=float) for name, index in picks.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/semi_synthetic.yaml"))
    parser.add_argument("--dataset", default="seer_brain")
    parser.add_argument("--column", default="Grade (thru 2017)")
    parser.add_argument("--drop-code", type=float, action="append",
                        default=[0.0, 5.0])
    parser.add_argument("--draws", type=int, default=50_000)
    parser.add_argument("--penalizer", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--grade-scale", type=float, default=1.0,
        help="Multiply the fitted grade coefficients. A diagnostic only: it "
             "checks that the simulation detects dependence when it is there.",
    )
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    spec = next(
        item for item in config["data"]["datasets"]
        if str(item["name"]).lower() == args.dataset.lower()
    )
    frame = cohort(spec)
    if args.drop_code:
        frame = frame.loc[
            ~frame[args.column].isin(args.drop_code)
        ].reset_index(drop=True)
    print(f"Cohort: {args.dataset} | n = {len(frame):,} "
          f"| excluded levels {args.drop_code}")

    reduced = design(frame, spec, args.column)
    grade = pd.get_dummies(
        pd.Categorical(frame[args.column]), prefix="grade", drop_first=True,
    ).astype(float).reset_index(drop=True)
    full = pd.concat([reduced, grade], axis=1)
    duration = frame[spec["time_column"]].to_numpy(dtype=float)
    event = (
        pd.to_numeric(frame[spec["event_column"]], errors="coerce").fillna(0) > 0
    ).astype(int).to_numpy()

    fitted = {}
    for name, outcome in (("event", event), ("censoring", 1 - event)):
        design_frame = full.copy()
        design_frame["duration"] = duration
        design_frame["event"] = outcome
        print(f"Fitting the {name} model with {args.column!r} included")
        model, used = _fit(design_frame, args.penalizer)
        fitted[name] = model
        print(f"  penalizer {used:g} | log-likelihood {model.log_likelihood_:.1f}")

    levels = pd.Categorical(frame[args.column]).categories
    weights = (
        frame[args.column].value_counts(normalize=True).reindex(levels).to_numpy()
    )
    print("\nEmpirical grade distribution used for the draws:")
    for level, weight in zip(levels, weights):
        print(f"  level {level}: {weight:.4f}")

    rng = np.random.default_rng(args.seed)
    reference = profiles(reduced, fitted["event"].params_)
    rows = []
    for profile_name, covariates in reference.items():
        drawn = rng.choice(len(levels), size=args.draws, p=weights)
        block = np.zeros((args.draws, grade.shape[1]), dtype=float)
        for position, level_index in enumerate(range(1, len(levels))):
            block[drawn == level_index, position] = 1.0
        sampled = {}
        truncation = {}
        for name in ("event", "censoring"):
            params = fitted[name].params_
            eta = (
                covariates @ params[reduced.columns].to_numpy()
                + args.grade_scale * (block @ params[grade.columns].to_numpy())
            )
            times, survival = baseline_curve(fitted[name])
            uniforms = rng.uniform(size=args.draws)
            sampled[name], truncated = sample_times(times, survival, eta, uniforms)
            truncation[name] = float(truncated.mean())
        statistic = kendalltau(sampled["event"], sampled["censoring"]).statistic
        # The same draw of grade shifts both margins; with grade held fixed the
        # two times are independent by construction, which is the null here.
        control = kendalltau(
            sampled["event"], rng.permutation(sampled["censoring"])
        ).statistic
        rows.append({
            "profile": profile_name,
            "kendall_tau": float(statistic),
            "tau_grade_fixed_control": float(control),
            "median_event_time": float(np.median(sampled["event"])),
            "median_censor_time": float(np.median(sampled["censoring"])),
            "truncated_event": truncation["event"],
            "truncated_censoring": truncation["censoring"],
        })

    table = pd.DataFrame(rows)
    print(f"\nInduced conditional Kendall's tau ({args.draws:,} draws per profile):")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nMean tau across profiles: {table['kendall_tau'].mean():.4f}")


if __name__ == "__main__":
    main()
