"""Task 3: CoxPH martingale and Cox--Gamma posterior recovery on every dataset.

Runs the paper's ``experiments.recovery_baselines.run_recovery_baselines``
unchanged for one dataset and seed of ``configs/semi_synthetic.yaml``. The only
in-memory change is ``evaluation.recovery_baseline_datasets``, widened from the
three SEER cohorts to all twelve datasets. A baseline whose statistic is
non-finite on a cell (CoxPH's cumulative hazard can diverge beyond the training
support, as on MIMIC-IV) is recorded as a failure row instead of aborting the job. Each refit is cross-checked against
the benchmark's CoxPH and Cox--Gamma oracle IBS (``ibs_crosscheck.csv``).
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
    # The paper's runs raise on a non-finite statistic; here one baseline's
    # failure is recorded so the other is still scored on that cell.
    rows, crosscheck = run_recovery_baselines(cfg, dataset, args.seed_index, seeds, nonfinite="record")
    rows.to_csv(args.output_dir / "recovery_rows.csv", index=False)
    crosscheck.to_csv(args.output_dir / "ibs_crosscheck.csv", index=False)
    (args.output_dir / "resolved_config.json").write_text(
        json.dumps({**cfg, "data": {**cfg["data"], "datasets": [dataset]}}, indent=2, default=str),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
