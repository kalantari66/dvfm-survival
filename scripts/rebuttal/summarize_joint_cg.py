"""Summarise the joint Cox--Gamma baseline against DVFM and the event-only CG.

Reads completed artifacts only (no fitting) and fails if any is missing:

* rebuttal/joint_cg/fits/<dataset>/seed_<k>/  joint and event-only CG rows;
* rebuttal/task3_recovery_all_datasets/baselines/  event-only CG (must match);
* rebuttal/task1_prior_vs_aggregate/semi_synthetic/  DVFM test posterior means
  (the 600 d_z = 1 refits), signed on validation subjects as Figure 6 does.

Recovery is the Spearman correlation with the true z among test subjects at
tau = 0.5, test split, validation-calibrated (the Figure 6 rows). Per dataset,
the paired difference is averaged over the four copulas within each seed; the
95% interval bootstraps the ten seed means. The across-dataset median gets a
bootstrap interval over datasets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rebuttal_common import DATASET_LABEL, PROJECT_ROOT

KEY = ["dataset", "copula", "repeat", "subgroup"]
SUBGROUPS = ["Censored", "Event observed", "All"]
EXPECTED_ROWS = 12 * 4 * 10 * len(SUBGROUPS)


def baseline_rows(root: Path) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in sorted(root.glob("*/seed_*/recovery_rows.csv"))]
    if not frames:
        raise FileNotFoundError(f"No recovery rows under {root}")
    rows = pd.concat(frames, ignore_index=True)
    return rows[(rows.split == "test") & (rows.representation == "Validation-calibrated")]


def dvfm_rows(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob("*/seed_*/subjects/*.csv.gz")):
        if "independence" in path.name:
            continue
        subjects = pd.read_csv(path)
        validation, test = subjects[subjects.split == "validation"], subjects[subjects.split == "test"]
        sign = -1.0 if np.corrcoef(validation.true_z, validation.mu_0)[0, 1] < 0 else 1.0
        dataset, repeat = path.parts[-4], int(path.parts[-3].removeprefix("seed_"))
        copula = path.name.split("_tau")[0][len(dataset) + 1:]
        for subgroup, mask in (("All", test.event >= 0), ("Event observed", test.event == 1),
                               ("Censored", test.event == 0)):
            rows.append({
                "dataset": dataset, "copula": copula, "repeat": repeat, "subgroup": subgroup, "model": "DVFM",
                "spearman": pd.Series(sign * test.mu_0[mask]).corr(test.true_z[mask], method="spearman"),
            })
    return pd.DataFrame(rows)


def paired(wide: pd.DataFrame, comparison: str, subgroup: str, rng) -> pd.DataFrame:
    cells = wide[wide.subgroup == subgroup].assign(diff=lambda frame: frame.DVFM - frame[comparison])
    out = []
    for dataset, group in cells.groupby("dataset"):
        seed_means = group.groupby("repeat")["diff"].mean().to_numpy()
        boots = rng.choice(seed_means, (10_000, len(seed_means))).mean(axis=1)
        out.append({
            "comparison": comparison, "subgroup": subgroup, "dataset": dataset,
            "dvfm": group.DVFM.mean(), "baseline": group[comparison].mean(),
            "diff": group["diff"].mean(), "lo": np.percentile(boots, 2.5), "hi": np.percentile(boots, 97.5),
            "cells_dvfm_better": int((group["diff"] > 0).sum()), "cells": len(group),
        })
    out = pd.DataFrame(out)
    medians = np.median(rng.choice(out["diff"].to_numpy(), (10_000, len(out))), axis=1)
    return out.assign(median=np.median(out["diff"]), median_lo=np.percentile(medians, 2.5),
                      median_hi=np.percentile(medians, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuttal-dir", type=Path, default=PROJECT_ROOT / "rebuttal")
    args = parser.parse_args()
    out_dir = args.rebuttal_dir / "joint_cg"

    fits = baseline_rows(out_dir / "fits")
    task3 = baseline_rows(args.rebuttal_dir / "task3_recovery_all_datasets" / "baselines")
    dvfm = dvfm_rows(args.rebuttal_dir / "task1_prior_vs_aggregate" / "semi_synthetic")

    # Same cohorts and splits as Task 3: the refitted event-only CG must agree.
    check = fits[fits.model == "BayesianCoxGammaFrailty"].merge(
        task3[task3.model == "BayesianCoxGammaFrailty"], on=KEY + ["representation"], suffixes=("", "_task3"))
    for name, frame in (("joint CG", fits[fits.model == "JointCoxGammaFrailty"]),
                        ("event-only CG check", check), ("DVFM", dvfm)):
        if len(frame) != EXPECTED_ROWS:
            raise RuntimeError(f"{name}: {len(frame)} rows, expected {EXPECTED_ROWS}")
    worst = float((check.spearman - check.spearman_task3).abs().max())
    if not worst <= 1e-9:
        raise RuntimeError(f"Event-only CG differs from Task 3 by {worst:g}: not the same cohorts")

    cells = pd.concat([dvfm, fits[KEY + ["model", "spearman"]]], ignore_index=True)
    wide = cells.pivot_table(index=KEY, columns="model", values="spearman", dropna=False).reset_index()
    if wide[["DVFM", "JointCoxGammaFrailty", "BayesianCoxGammaFrailty"]].isna().any().any():
        raise RuntimeError("Missing or non-finite recovery for some cell")
    rng = np.random.default_rng(20261117)
    result = pd.concat([paired(wide, comparison, subgroup, rng)
                        for comparison in ("JointCoxGammaFrailty", "BayesianCoxGammaFrailty")
                        for subgroup in SUBGROUPS], ignore_index=True)
    result["dataset_label"] = result.dataset.map(DATASET_LABEL)
    wide.to_csv(out_dir / "recovery_cells.csv", index=False)
    result.to_csv(out_dir / "paired_differences.csv", index=False)

    fit_info = pd.concat([pd.read_csv(path)
                          for path in sorted((out_dir / "fits").glob("*/seed_*/ibs_crosscheck.csv"))])
    joint = fit_info[fit_info.model == "JointCoxGammaFrailty"]
    joint.to_csv(out_dir / "joint_fit_info.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"Event-only CG vs Task 3: max |diff| {worst:.2g} over {len(check)} rows")
    for (comparison, subgroup), group in result.groupby(["comparison", "subgroup"], sort=False):
        print(f"\n== DVFM - {comparison}, {subgroup}: median {group['median'].iloc[0]:+.3f} "
              f"[{group.median_lo.iloc[0]:+.3f}, {group.median_hi.iloc[0]:+.3f}]; "
              f"CI > 0 on {(group.lo > 0).sum()}, CI < 0 on {(group.hi < 0).sum()} of {len(group)}")
        print(group[["dataset_label", "dvfm", "baseline", "diff", "lo", "hi", "cells_dvfm_better"]]
              .round(3).to_string(index=False))
    print("\nJoint CG fits: alpha median", round(float(joint.frailty_alpha.median()), 2),
          "| checkpoint at the 500-epoch cap:", int((joint.checkpoint_epoch >= 500).sum()), "of", len(joint))
    print(fit_info.pivot_table(index="dataset", columns="model", values="refit_oracle_ibs").round(4))


if __name__ == "__main__":
    main()
