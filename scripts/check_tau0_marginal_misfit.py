"""Test whether DVFM's spurious dependence at tau=0 tracks its marginal misfit.

At the independence condition the true Kendall's tau is zero, yet DVFM reports a
nonzero learned conditional tau that grows with cohort size.  One explanation is
that the latent is absorbing marginal misspecification rather than modelling
dependence: the semi-synthetic margins are CoxPH fits with Breslow baselines,
while DVFM decodes a Weibull shape and scale, so a larger cohort supplies more
evidence for a latent-mediated repair of that mismatch.

If that is what is happening, DVFM's learned tau at tau=0 should be larger
exactly where its marginal fit is worse relative to the correctly specified
CoxPH baseline.  This script measures that association.  Oracle IBS is paired
within a seed, since every model sees the identical cohort and split.

A positive rank correlation supports the misspecification-repair explanation; a
null or negative one refutes it, and the inflation needs another account.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "results" / "semi-synthetic"

CELL_KEYS = ["dataset", "copula", "target_kendall_tau", "target_censoring_rate", "repeat"]
MODEL_NAMES = {
    "dvfm": "DVFM", "coxph": "CoxPH", "deepsurv": "DeepSurv", "rsf": "RSF",
    "mtlr": "MTLR", "deephit": "DeepHit", "hacsurv": "HACSurv",
    "hacsurv_2d": "HACSurv", "clayton_aft": "ClaytonAFT", "gbsa": "GBSA",
    "bayesian_cox_gamma_frailty": "BayesianCoxGammaFrailty",
}


def _snake_case(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _canonicalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the same column and model-name normalization as the notebooks."""
    out = frame.rename(columns={column: _snake_case(column) for column in frame.columns}).copy()
    aliases = {
        "dataset_name": "dataset", "target_censoring": "target_censoring_rate",
        "censoring_rate": "target_censoring_rate", "kendall_tau": "target_kendall_tau",
    }
    out = out.rename(columns={key: value for key, value in aliases.items()
                              if key in out and value not in out})
    if "model" in out:
        lowered = out["model"].astype(str).str.lower()
        out["model"] = lowered.map(MODEL_NAMES).fillna(out["model"].astype(str))
    if "copula" in out:
        out["copula"] = out["copula"].astype(str).str.title()
    return out


def load_results(result_root: Path) -> pd.DataFrame:
    """Read the cross-dataset aggregate, or combine per-dataset result files."""
    aggregate = result_root / "results_raw.csv"
    if aggregate.exists():
        return _canonicalize(pd.read_csv(aggregate))
    per_dataset = sorted(result_root.glob("*/results_raw.csv"))
    if not per_dataset:
        raise FileNotFoundError(f"No results_raw.csv under {result_root}")
    frames = (pd.read_csv(path) for path in per_dataset)
    return _canonicalize(pd.concat(frames, ignore_index=True))


def primary_rows(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    """Keep one reported row per run: primary checkpoint, aggregate posterior."""
    out = frame[frame["model"].eq(model)].copy()
    if "is_primary_checkpoint" in out:
        flag = out["is_primary_checkpoint"].astype(str).str.lower()
        out = out[flag.eq("true")]
    if "prediction_mode" in out:
        out = out[out["prediction_mode"].astype(str).str.lower().eq("aggregate_posterior")]
    if "latent_dim" in out and model == "DVFM":
        latent_dim = pd.to_numeric(out["latent_dim"], errors="coerce")
        out = out[latent_dim.eq(1) | latent_dim.isna()]
    duplicated = out.duplicated(CELL_KEYS)
    if duplicated.any():
        raise ValueError(
            f"{model} has {int(duplicated.sum())} duplicate rows per "
            f"{CELL_KEYS}; the checkpoint/mode filter did not reduce to one row per run."
        )
    return out


def build_cells(results: pd.DataFrame, tau: float, baseline: str) -> pd.DataFrame:
    """Return one row per dataset x copula cell, seeds averaged after pairing."""
    selected = results[np.isclose(
        pd.to_numeric(results["target_kendall_tau"], errors="coerce"), tau
    )]
    if selected.empty:
        raise ValueError(f"No rows at target_kendall_tau={tau:g}")

    dvfm = primary_rows(selected, "DVFM")
    reference = primary_rows(selected, baseline)
    if "learned_conditional_kendall_tau" not in dvfm:
        raise ValueError("Results do not contain learned_conditional_kendall_tau")

    paired = dvfm.merge(
        reference[CELL_KEYS + ["oracle_ibs"]],
        on=CELL_KEYS, suffixes=("_dvfm", "_reference"),
    )
    if paired.empty:
        raise ValueError(f"No seeds where both DVFM and {baseline} completed at tau={tau:g}")

    # Positive gap = DVFM predicts the margin worse than the correctly
    # specified baseline, on the identical cohort and split.
    paired["ibs_gap"] = paired["oracle_ibs_dvfm"] - paired["oracle_ibs_reference"]
    paired["learned_tau"] = pd.to_numeric(
        paired["learned_conditional_kendall_tau"], errors="coerce"
    )

    cells = (
        paired.groupby(["dataset", "copula"], as_index=False)
        .agg(learned_tau=("learned_tau", "mean"),
             ibs_gap=("ibs_gap", "mean"),
             seeds=("repeat", "nunique"))
    )
    return cells.dropna(subset=["learned_tau", "ibs_gap"]).sort_values("ibs_gap")


def report(cells: pd.DataFrame, tau: float, baseline: str) -> None:
    print(f"\ntau = {tau:g} | DVFM learned tau vs (DVFM - {baseline}) oracle IBS")
    print(f"{'dataset':<16}{'copula':<14}{'learned tau':>12}{'IBS gap':>12}{'seeds':>7}")
    for row in cells.itertuples():
        print(f"{row.dataset:<16}{row.copula:<14}{row.learned_tau:>12.3f}"
              f"{row.ibs_gap:>12.4f}{row.seeds:>7d}")

    if len(cells) < 3:
        print("\nToo few cells for a correlation.")
        return
    rho, rho_p = spearmanr(cells["ibs_gap"], cells["learned_tau"])
    r, r_p = pearsonr(cells["ibs_gap"], cells["learned_tau"])
    print(f"\nn = {len(cells)} cells")
    print(f"Spearman rho = {rho:+.3f} (p = {rho_p:.3f})")
    print(f"Pearson  r   = {r:+.3f} (p = {r_p:.3f})")
    if rho > 0 and rho_p < 0.05:
        print("Positive and significant: consistent with the latent absorbing marginal misfit.")
    elif rho > 0:
        print("Positive but not significant: suggestive only, do not report as a mechanism.")
    else:
        print("Not positive: the misspecification-repair explanation is not supported.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--tau", type=float, default=0.0,
                        help="Target Kendall's tau to analyse (default: the independence cell)")
    parser.add_argument("--baseline", default="CoxPH",
                        help="Correctly specified marginal reference (default: CoxPH)")
    parser.add_argument("--output", type=Path, help="Optional CSV path for the per-cell table")
    args = parser.parse_args()

    results = load_results(args.result_root)
    cells = build_cells(results, args.tau, args.baseline)
    report(cells, args.tau, args.baseline)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        cells.to_csv(args.output, index=False)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
