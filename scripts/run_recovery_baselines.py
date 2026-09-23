"""Score the scalar-frailty recovery baselines for one dataset/seed."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from experiments.config import expand_seed_streams, load_config, validate_config
from experiments.recovery_baselines import run_recovery_baselines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed-index", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    validate_config(config)
    datasets = config["data"].get("datasets", [])
    selected = [
        item for item in datasets
        if str(item["name"]).lower() == args.dataset.lower()
    ]
    if len(selected) != 1:
        raise ValueError(f"Expected one configured dataset named {args.dataset!r}")
    configured = [
        str(name).lower()
        for name in config["evaluation"]["recovery_baseline_datasets"]
    ]
    if args.dataset.lower() not in configured:
        raise ValueError(
            f"{args.dataset!r} is not in evaluation.recovery_baseline_datasets"
        )
    # Both baselines are refit here and cross-checked against the benchmark
    # fits, so the study must actually run them.
    enabled = {str(name).lower() for name in config["models"]["enabled"]}
    missing = {"coxph", "bayesian_cox_gamma_frailty"} - enabled
    if missing:
        raise ValueError(
            f"models.enabled must include {sorted(missing)} for the recovery "
            "baselines to be cross-checked against the benchmark fits"
        )

    seed_streams = expand_seed_streams(config["seeds"])
    if not 0 <= args.seed_index < len(seed_streams):
        raise ValueError(
            f"seed-index must be between 0 and {len(seed_streams) - 1}"
        )
    if args.validate_only:
        return

    rows, crosscheck = run_recovery_baselines(
        config, selected[0], args.seed_index, seed_streams[args.seed_index]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output_dir / "recovery_rows.csv", index=False)
    crosscheck.to_csv(args.output_dir / "ibs_crosscheck.csv", index=False)
    resolved = deepcopy(config)
    resolved["data"]["datasets"] = selected
    (args.output_dir / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2, default=str), encoding="utf-8"
    )
    print(f"Saved recovery baselines to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
