"""Aggregate independent GWF shards from the canonical synthetic pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import expand_scenarios, load_config


def _read_all(shard_root: Path, filename: str) -> pd.DataFrame:
    paths = sorted(shard_root.glob(f"scenario_*/{filename}"))
    if not paths:
        raise FileNotFoundError(f"No shard files named {filename} below {shard_root}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def aggregate(config_path: Path, shard_root: Path, output_dir: Path) -> None:
    cfg = load_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_shards = len(expand_scenarios(cfg["data"])) * len(cfg["seeds"]["sampling"])
    actual_shards = len(list(shard_root.glob("scenario_*/run_manifest.csv")))
    if actual_shards != expected_shards:
        raise RuntimeError(
            f"Expected {expected_shards} successful shards, found {actual_shards}"
        )
    results = _read_all(shard_root, "results_raw.csv")
    histories = _read_all(shard_root, "training_history.csv.gz")
    diagnostics = _read_all(shard_root, "dvfm_diagnostics.csv")
    calibration = _read_all(shard_root, "calibration_curves.csv.gz")
    manifest = _read_all(shard_root, "run_manifest.csv")

    if (manifest["status"] != "success").any():
        raise RuntimeError("At least one synthetic shard contains a failed fit")

    results.to_csv(output_dir / "results_raw.csv", index=False)
    histories.to_csv(
        output_dir / "training_history.csv.gz", index=False, compression="gzip"
    )
    diagnostics.to_csv(output_dir / "dvfm_diagnostics.csv", index=False)
    calibration.to_csv(
        output_dir / "calibration_curves.csv.gz", index=False, compression="gzip"
    )
    manifest.to_csv(output_dir / "run_manifest.csv", index=False)

    groups = [
        "study", "scenario", "n_samples", "target_kendall_tau",
        "target_censoring_rate", "model", "latent_dim", "checkpoint",
        "is_primary_checkpoint", "prediction_mode",
    ]
    metrics = [
        "oracle_ibs", "oracle_ci", "oracle_mae",
        "oracle_mae_censored", "oracle_mae_uncensored",
    ]
    results.groupby(groups, dropna=False)[metrics].mean().reset_index().to_csv(
        output_dir / "results_mean.csv", index=False
    )
    results.groupby(groups, dropna=False)[metrics].std().reset_index().to_csv(
        output_dir / "results_std.csv", index=False
    )
    with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=2, default=str)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    aggregate(args.config, args.shard_root, args.output_dir)


if __name__ == "__main__":
    main()
