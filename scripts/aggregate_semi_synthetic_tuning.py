"""Combine the independent per-dataset semi-synthetic tuning artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from experiments.config import load_config, semisynthetic_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--result-root", required=True, type=Path)
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        tuning = yaml.safe_load(handle) or {}
    base_path = Path(tuning["experiment_config"])
    if not base_path.is_absolute():
        base_path = args.config.parent.parent / base_path
    base = load_config(base_path)
    names = [item["name"] for item in semisynthetic_datasets(base["data"])]
    trial_paths = [args.result_root / name / "tuning_trials.csv" for name in names]
    missing = [str(path) for path in trial_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing tuning outputs: " + ", ".join(missing))
    trials = pd.concat([pd.read_csv(path) for path in trial_paths], ignore_index=True)
    trials.to_csv(args.result_root / "tuning_trials.csv", index=False)
    winners = []
    for (dataset, model), group in trials.query("status == 'ok'").groupby(["dataset", "model"]):
        best = group.loc[group["selection_score"].idxmin()]
        winners.append({
            "dataset": dataset, "model": model, "trial": int(best["trial"]),
            "parameters": json.loads(best["parameters"]),
            "validation_ibs_ipcw": float(best["validation_ibs_ipcw"]),
        })
    (args.result_root / "selected_hyperparameters.json").write_text(
        json.dumps(winners, indent=2) + "\n", encoding="utf-8"
    )
    (args.result_root / "resolved_tuning_config.json").write_text(
        json.dumps(tuning, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
