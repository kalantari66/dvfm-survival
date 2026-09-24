"""Build the tau=0 vs tau=0.5 model rank-shift table for the paper.

Reuses the ranking protocol of notebooks/semi_synthetic_results.ipynb verbatim:
rank models within each dataset x copula x seed cell, average over seeds within
a copula, average the copulas within a dataset, then take the median over the
12 datasets. Writes paper/tables/semi_synthetic_rank_shift.tex.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

RANK_METRICS = {
    "oracle_ibs": ("Oracle IBS", False),
    "oracle_ci": ("Oracle CI", True),
    "oracle_mae": ("Oracle MAE", False),
}
SEED_KEYS = ["dataset", "copula", "target_kendall_tau", "target_censoring_rate", "repeat"]
# Figure 3 y-axis labels, so the two displays name the models identically.
MODEL_LABELS = {
    "DVFM": "DVFM (ours)",
    "BayesianCoxGammaFrailty": "CG Frailty",
    "ClaytonAFT": "Clayton AFT",
}
# Models that estimate the joint law of (E, C), and so permit dependence
# without requiring it, against those whose likelihood assumes E _||_ C | X.
# This grouping is the table's claim, not a ranking.
ANCHOR_MODEL = "DVFM"  # bolded, and the first row of whichever group it is in
DEPENDENT_MODELS = ["DVFM", "ClaytonAFT", "HACSurv"]
INDEPENDENT_MODELS = ["BayesianCoxGammaFrailty", "CoxPH", "DeepSurv", "RSF", "MTLR"]
# Internal controls that share the benchmark's result files but never rank.
ABLATION_MODELS = ["DVFM-z0"]
# The one cell whose direction runs against its group. The caption names it, so
# it is expected rather than a violation; every other departure still stops the
# table, because the caption would then be asserting something the data denies.
CLAIM_EXCEPTIONS = {("BayesianCoxGammaFrailty", "oracle_ci")}


def snake_case(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def canonicalize(frame):
    """Column and model-name normalization, identical to the notebook's."""
    out = frame.rename(columns={column: snake_case(column) for column in frame.columns}).copy()
    aliases = {
        "dataset_name": "dataset", "target_censoring": "target_censoring_rate",
        "censoring_rate": "target_censoring_rate", "kendall_tau": "target_kendall_tau",
        "oracle_joint_ise": "oracle_joint_survival_ise",
    }
    out = out.rename(columns={k: v for k, v in aliases.items() if k in out and v not in out})
    names = {"dvfm": "DVFM", "coxph": "CoxPH", "deepsurv": "DeepSurv", "rsf": "RSF",
             "gbsa": "GBSA", "mtlr": "MTLR", "hacsurv": "HACSurv",
             "hacsurv_2d": "HACSurv", "clayton_aft": "ClaytonAFT",
             "bayesian_cox_gamma_frailty": "BayesianCoxGammaFrailty",
             "bayesiancoxgammafrailty": "BayesianCoxGammaFrailty"}
    out["model"] = out["model"].astype(str).str.lower().map(names).fillna(out["model"].astype(str))
    # The matched no-frailty ablation is an internal control, not a competing
    # method, so it never takes part in the benchmark ranking.
    out = out[~out["model"].isin(ABLATION_MODELS)].copy()
    out["copula"] = out["copula"].astype(str).str.title()
    return out


def seed_level_metric_ranks(frame, metric, higher_is_better):
    """Rank paired runs, penalizing only the atomic seeds that failed."""
    work = frame[SEED_KEYS + ["model", metric, "numerical_failure"]].copy()
    if work.duplicated(SEED_KEYS + ["model"]).any():
        raise ValueError(f"Duplicate model rows found while ranking {metric}.")
    values = pd.to_numeric(work[metric], errors="coerce")
    failure = work["numerical_failure"].fillna(False)
    if failure.dtype != bool:
        failure = failure.astype(str).str.lower().eq("true")
    valid = np.isfinite(values) & ~failure
    work["_score"] = values.where(valid)
    work["seed_rank"] = work.groupby(SEED_KEYS)["_score"].rank(
        ascending=not higher_is_better, method="min"
    )
    worst_completed = work.groupby(SEED_KEYS)["seed_rank"].transform("max")
    if worst_completed[~valid].isna().any():
        raise ValueError(f"A {metric} seed cell has no successful model to define a failure rank.")
    work.loc[~valid, "seed_rank"] = worst_completed.loc[~valid] + 1
    return work


def dataset_median_ranks(results, tau, expected_repeats):
    """One median-across-datasets rank per model per metric, at one tau."""
    rank_input = results[np.isclose(results["target_kendall_tau"], tau)].copy()
    if rank_input.empty:
        raise ValueError(f"No results for tau={tau:g}.")
    out = {}
    for metric, (_, higher_is_better) in RANK_METRICS.items():
        seed_ranks = seed_level_metric_ranks(rank_input, metric, higher_is_better)
        counts = seed_ranks.groupby(["dataset", "copula", "model"])["repeat"].nunique()
        if not counts.eq(expected_repeats).all():
            raise ValueError(f"Incomplete seed grid at tau={tau:g}, {metric}.")
        scenario = seed_ranks.groupby(["dataset", "copula", "model"], as_index=False)["seed_rank"].mean()
        per_dataset = scenario.groupby(["dataset", "model"])["seed_rank"].mean().unstack("model")
        out[metric] = per_dataset.median(axis=0)
    return pd.DataFrame(out)


def cell(low, high):
    """One 'a -> b' cell, shaded by direction; a falling rank is an improvement."""
    improved = high < low
    shade = "highrank" if improved else "lowrank"
    arrow = "improvGreen" if improved else "improvRed"
    return (f"\\cellcolor{{{shade}}}${low:.2f}\\,"
            f"\\textcolor{{{arrow}}}{{\\to}}\\,{high:.2f}$")


def build_table(low, high):
    lines = [
        "% Requires: booktabs, wrapfig, xcolor, colortbl (for \\cellcolor).",
        "% Colors improvGreen, improvRed, highrank, lowrank are defined in the preamble.",
        "% Generated by scripts/make_rank_shift_table.py -- do not edit by hand.",
        r"\begin{wraptable}{r}{0.47\textwidth}",
        r"\vspace{-\intextsep}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\caption{\textbf{Dependence is what the rank responds to.} Model rank at "
        r"$\tau=0 \to \tau=0.5$, median over the 12 semi-synthetic datasets; rank 1 "
        r"is best, so a fall is an improvement. Every model that relaxes independent "
        r"censoring improves on all three metrics. Every model that assumes it "
        r"degrades on Oracle IBS and Oracle MAE, and all but the Cox--Gamma frailty "
        r"model degrade on Oracle CI as well.}",
        r"\label{tab:semi_synthetic_rank_shift}",
        r"\begin{tabular}{@{}lccc@{}}",
        r"\toprule",
        r"Model & Oracle IBS & Oracle CI & Oracle MAE \\",
        r"\midrule",
        r"\multicolumn{4}{@{}l}{\emph{Relaxes independent censoring}} \\",
    ]
    groups = [
        (DEPENDENT_MODELS, None),
        (INDEPENDENT_MODELS, r"\multicolumn{4}{@{}l}{\emph{Assumes independent censoring}} \\"),
    ]
    for group, header in groups:
        if header is not None:
            lines += [r"\midrule", header]
        # Our model heads its group; the rest follow by IBS improvement, so the
        # ordering below it is still informative.
        ranked = sorted(group, key=lambda m: (
            m != ANCHOR_MODEL, high.loc[m, "oracle_ibs"] - low.loc[m, "oracle_ibs"]
        ))
        for model in ranked:
            label = MODEL_LABELS.get(model, model)
            if model == ANCHOR_MODEL:
                label = r"\textbf{" + label + "}"
            cells = " & ".join(cell(low.loc[model, m], high.loc[model, m]) for m in RANK_METRICS)
            lines.append(f"{label} & {cells} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{wraptable}", ""]
    return "\n".join(lines)


def check_caption_claim(delta):
    """The caption asserts a direction for every cell; verify the data still does.

    The sentence is hard-coded above, so a rerun on new results can quietly
    produce a table whose own caption one of its cells contradicts.  A rank that
    falls is an improvement, so the dependence-aware group must be negative
    throughout and the independence-assuming group positive apart from the cells
    the caption names, which must in turn still be exceptions -- the caption
    calls them out by name, so a cell that rejoins its group falsifies it too.
    """
    claims = (
        (DEPENDENT_MODELS, -1.0, "improve"),
        (INDEPENDENT_MODELS, 1.0, "degrade"),
    )
    violations = []
    for group, expected_sign, claimed in claims:
        for model in group:
            for metric, (label, _) in RANK_METRICS.items():
                if (model, metric) in CLAIM_EXCEPTIONS:
                    continue
                shift = float(delta.loc[model, metric])
                if shift * expected_sign <= 0.0:
                    violations.append(
                        f"  {model} / {label}: rank shift {shift:+.2f}, but the "
                        f"caption claims this group should {claimed}"
                    )
    for model, metric in sorted(CLAIM_EXCEPTIONS):
        expected_sign = -1.0 if model in DEPENDENT_MODELS else 1.0
        shift = float(delta.loc[model, metric])
        if shift * expected_sign > 0.0:
            violations.append(
                f"  {model} / {RANK_METRICS[metric][0]}: rank shift {shift:+.2f} "
                "now matches its group, so the caption should stop naming it as "
                "an exception"
            )
    return violations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    config = yaml.safe_load((args.root / "configs" / "semi_synthetic.yaml").read_text(encoding="utf-8"))
    results = canonicalize(pd.read_csv(args.root / "results" / "semi-synthetic" / "results_raw.csv"))
    expected_repeats = len(config["seeds"])

    low = dataset_median_ranks(results, 0.0, expected_repeats)
    high = dataset_median_ranks(results, 0.5, expected_repeats)
    delta = high - low
    summary = pd.concat({"tau=0": low, "tau=0.5": high, "delta": delta}, axis=1)
    print(summary.round(2).to_string())
    print("\nmean delta over the three metrics:")
    print(delta.mean(axis=1).round(2).sort_values().to_string())

    violations = check_caption_claim(delta)
    if violations:
        raise ValueError(
            "These results do not support the caption's claim, so the table is "
            "not written; reword the claim in build_table() or report the "
            "exceptions:\n" + "\n".join(violations)
        )

    out = args.root / "paper" / "tables" / "semi_synthetic_rank_shift.tex"
    out.write_text(build_table(low, high), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
