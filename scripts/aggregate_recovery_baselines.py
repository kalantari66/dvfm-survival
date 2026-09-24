"""Merge recovery-baseline seed jobs and verify they refit the stored cohorts.

The cross-check joins each refit's oracle metrics onto the benchmark row for
the same dataset, copula, censoring rate and repeat.  Both fits are
deterministic, so agreement to floating-point tolerance proves the regenerated
cohort and split are the ones the reported results were produced on; a
mismatch means the recovery rows describe different data and must not be
reported beside DVFM's.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.config import expand_seed_streams, load_config

RELATIVE_TOLERANCE = 1e-6
BENCHMARK_DIRECTORY = {
    "CoxPH": "coxph",
    "BayesianCoxGammaFrailty": "bayesian_cox_gamma_frailty",
}
# The censoring rate is determined by the dataset under
# ``censoring_rates: original`` and is an arbitrary float, so it is verified
# rather than joined on: a float key would have to survive a CSV round trip
# bit for bit on both sides.
JOIN_KEYS = ["dataset", "copula", "target_kendall_tau", "repeat", "model"]
CENSORING_TOLERANCE = 1e-9


def _benchmark_rows(result_root: Path, dataset: str) -> pd.DataFrame:
    frames = []
    for model, directory in BENCHMARK_DIRECTORY.items():
        path = result_root / dataset / directory / "results_raw.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Benchmark results for the cross-check are missing: {path}"
            )
        frame = pd.read_csv(path)
        frame = frame[frame["Model"].eq(model)]
        frames.append(pd.DataFrame({
            "dataset": frame["Dataset"],
            "copula": frame["Copula"].astype(str).str.lower(),
            "target_kendall_tau": pd.to_numeric(frame["Target Kendall Tau"]),
            "repeat": pd.to_numeric(frame["Repeat"]),
            "model": model,
            "benchmark_censoring_rate": pd.to_numeric(frame["Target Censoring Rate"]),
            "benchmark_oracle_ibs": pd.to_numeric(frame["oracle_ibs"]),
        }))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--result-root", required=True, type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    datasets = [
        str(name)
        for name in config["evaluation"]["recovery_baseline_datasets"]
    ]
    seed_count = len(expand_seed_streams(config["seeds"]))
    output_root = args.result_root / "recovery_baselines"
    output_root.mkdir(parents=True, exist_ok=True)

    rows, crosscheck = [], []
    for dataset in datasets:
        for seed_index in range(seed_count):
            seed_root = (
                args.result_root / dataset / "recovery_baselines"
                / f"seed_{seed_index}"
            )
            rows.append(pd.read_csv(seed_root / "recovery_rows.csv"))
            crosscheck.append(pd.read_csv(seed_root / "ibs_crosscheck.csv"))
    recovery = pd.concat(rows, ignore_index=True)
    refits = pd.concat(crosscheck, ignore_index=True)

    benchmark = pd.concat(
        [_benchmark_rows(args.result_root, dataset) for dataset in datasets],
        ignore_index=True,
    )
    merged = refits.merge(benchmark, on=JOIN_KEYS, how="left", validate="one_to_one")
    unmatched = merged["benchmark_oracle_ibs"].isna()
    if unmatched.any():
        raise ValueError(
            "No benchmark row matches these refit cells: "
            f"{merged.loc[unmatched, JOIN_KEYS].to_dict('records')}"
        )
    censoring_gap = np.abs(
        merged["target_censoring_rate"] - merged["benchmark_censoring_rate"]
    )
    if (censoring_gap > CENSORING_TOLERANCE).any():
        raise ValueError(
            "Refit and benchmark cells disagree on the target censoring rate, "
            f"so they are not the same condition (max gap {censoring_gap.max():.3g})"
        )
    merged["relative_deviation"] = np.abs(
        merged["refit_oracle_ibs"] - merged["benchmark_oracle_ibs"]
    ) / np.abs(merged["benchmark_oracle_ibs"])
    merged.to_csv(output_root / "ibs_crosscheck.csv", index=False)
    recovery.to_csv(output_root / "recovery_rows.csv", index=False)

    failed = merged[merged["relative_deviation"] > RELATIVE_TOLERANCE]
    if not failed.empty:
        raise ValueError(
            f"{len(failed)} recovery cells did not reproduce the benchmark "
            f"oracle IBS within {RELATIVE_TOLERANCE:g}; the refit cohorts "
            f"differ from the reported ones:\n"
            f"{failed[JOIN_KEYS + ['refit_oracle_ibs', 'benchmark_oracle_ibs']]}"
        )
    print(
        f"Cross-check passed for {len(merged)} cells "
        f"(max relative deviation {merged['relative_deviation'].max():.2e})."
    )


if __name__ == "__main__":
    main()
