"""Merge per-model outputs into one semi-synthetic dataset directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from experiments.config import load_config


FILES = ("results_raw.csv", "results_mean.csv", "results_std.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--result-root", required=True, type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    datasets = [
        item for item in config["data"].get("datasets", [])
        if str(item["name"]).lower() == args.dataset.lower()
    ]
    if len(datasets) != 1:
        raise ValueError(f"Expected one configured dataset named {args.dataset!r}")
    models = [str(model) for model in config["models"]["enabled"]]
    dataset_root = args.result_root / str(datasets[0]["name"])
    dataset_root.mkdir(parents=True, exist_ok=True)

    for filename in FILES:
        frames = [
            pd.read_csv(dataset_root / model.lower() / filename)
            for model in models
        ]
        pd.concat(frames, ignore_index=True).to_csv(
            dataset_root / filename, index=False
        )

    # DGP diagnostics are identical across model jobs because generation and
    # splitting use separate deterministic seed streams. Retain one copy.
    diagnostics = pd.read_csv(
        dataset_root / models[0].lower() / "dgp_diagnostics.csv"
    )
    diagnostics.to_csv(dataset_root / "dgp_diagnostics.csv", index=False)

    resolved = dict(config)
    resolved["data"] = dict(config["data"])
    resolved["data"]["datasets"] = datasets
    (dataset_root / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
