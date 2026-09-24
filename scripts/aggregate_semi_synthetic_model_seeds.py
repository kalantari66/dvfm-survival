"""Merge seed jobs and recompute summaries for one dataset/model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.config import expand_seed_streams, load_config


GROUP_COLUMNS = (
    "Study", "Stage", "Dataset", "Scenario", "Copula", "Dependence", "Theta",
    "Model", "study", "scenario", "n_samples", "target_kendall_tau",
    "target_censoring_rate", "mechanism", "model", "latent_dim",
    "hyperparameter_variant", "comparison_parent", "epochs", "batch_size",
    "dropout", "weight_decay", "encoder_hidden", "decoder_hidden",
    "scale_link", "latent_path", "shape_mode", "latent_loading_l1",
    "latent_group_lasso", "latent_gate", "gate_l1", "gate_initial_value",
    "gate_temperature", "checkpoint", "is_primary_checkpoint",
    "prediction_mode", "partition",
)
# Keep these exclusions aligned with the summary logic in experiments.runner.
SEED_COLUMNS = {
    "Repeat", "Fold", "Seed", "Sampling Seed", "Split Seed", "Model Seed",
    "repeat", "dgp_seed", "sampling_seed", "split_seed", "model_seed",
}


def _write_summaries(results: pd.DataFrame, output_dir: Path) -> None:
    group_columns = [
        column for column in GROUP_COLUMNS
        if column in results and not results[column].isna().all()
    ]
    numeric = results.select_dtypes(include=[np.number]).columns.tolist()
    metric_columns = [
        column for column in numeric
        if column not in SEED_COLUMNS and column not in group_columns
    ]
    grouped = results.groupby(group_columns, dropna=False)[metric_columns]
    grouped.mean().reset_index().to_csv(output_dir / "results_mean.csv", index=False)
    grouped.std().reset_index().to_csv(output_dir / "results_std.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
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
    selected_models = [
        model for model in models if model.lower() == args.model.lower()
    ]
    if len(selected_models) != 1:
        raise ValueError(f"Expected one enabled model named {args.model!r}")

    model_root = args.result_root / str(datasets[0]["name"]) / args.model.lower()
    seed_roots = [
        model_root / f"seed_{index}"
        for index, _ in enumerate(expand_seed_streams(config["seeds"]))
    ]
    results = pd.concat(
        [pd.read_csv(root / "results_raw.csv") for root in seed_roots],
        ignore_index=True,
    )
    results.to_csv(model_root / "results_raw.csv", index=False)
    _write_summaries(results, model_root)
    diagnostics = pd.concat(
        [pd.read_csv(root / "dgp_diagnostics.csv") for root in seed_roots],
        ignore_index=True,
    )
    diagnostics.to_csv(model_root / "dgp_diagnostics.csv", index=False)

    resolved = dict(config)
    resolved["data"] = dict(config["data"])
    resolved["data"]["datasets"] = datasets
    resolved["models"] = dict(config["models"])
    resolved["models"]["enabled"] = selected_models
    (model_root / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
