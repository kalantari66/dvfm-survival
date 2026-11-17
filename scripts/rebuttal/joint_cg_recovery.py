"""Joint shared Gamma-frailty baseline: frailty recovery on every dataset.

Runs ``experiments.recovery_baselines.run_recovery_baselines`` for one dataset
and seed of ``configs/semi_synthetic.yaml`` with two models:

* ``cox_gamma``: the paper's event-only Cox--Gamma baseline, refitted so each
  job checks it regenerated the same cohort and split (its rows must equal the
  Task 3 rows, and its oracle IBS the benchmark's);
* ``joint_cox_gamma``: the same model with the frailty also multiplying a
  censoring hazard that has its own piecewise-exponential baseline and linear
  predictor, fitted with the event-only model's hyperparameters, no retuning.

The only in-memory config change is widening
``evaluation.recovery_baseline_datasets`` to all twelve datasets.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from experiments.config import expand_seed_streams, load_config, semisynthetic_datasets, validate_config
from experiments.recovery_baselines import run_recovery_baselines

from rebuttal_common import PROJECT_ROOT, environment_record, select_dataset, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "semi_synthetic.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    cfg = deepcopy(load_config(args.config))
    cfg["evaluation"]["recovery_baseline_datasets"] = [item["name"] for item in semisynthetic_datasets(cfg["data"])]
    validate_config(cfg)
    dataset = select_dataset(cfg, args.dataset)
    seeds = expand_seed_streams(cfg["seeds"])[args.seed_index]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "environment.json", environment_record("cpu"))
    rows, crosscheck = run_recovery_baselines(
        cfg, dataset, args.seed_index, seeds, nonfinite="record",
        models=("cox_gamma", "joint_cox_gamma"),
    )
    rows.to_csv(args.output_dir / "recovery_rows.csv", index=False)
    crosscheck.to_csv(args.output_dir / "ibs_crosscheck.csv", index=False)
    (args.output_dir / "resolved_config.json").write_text(
        json.dumps({**cfg, "data": {**cfg["data"], "datasets": [dataset]}}, indent=2, default=str),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
