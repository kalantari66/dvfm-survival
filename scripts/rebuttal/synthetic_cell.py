"""Task 1 (synthetic): one cell of the primary synthetic grid, DVFM d_z = 1.

Runs ``configs/synthetic.yaml`` through ``run_synthetic_pilot`` with exactly
three in-memory changes, all recorded in the written ``resolved_config.json``:

* the grid is restricted to one scenario (``--scenario-index``), so the 12
  cells run as separate jobs and one failure cannot lose the grid;
* ``models.dvfm.latent_dims = [1]`` (d_z = 0 has no latent to compare);
* ``evaluation.joint_survival_aggregate_posterior = true``, adding joint ISE
  under the aggregate posterior next to the existing prior column.

Prior and aggregate-posterior prediction rows are already part of the study.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import pandas as pd
import torch

from experiments.config import expand_scenarios, load_config, validate_config
from experiments.synthetic import run_synthetic_pilot
from utility.runtime import resolve_device

from rebuttal_common import PROJECT_ROOT, environment_record, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "synthetic.yaml")
    parser.add_argument("--scenario-index", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--list", action="store_true", help="Print the scenario indices and exit")
    args = parser.parse_args()

    cfg = load_config(args.config)
    scenarios = expand_scenarios(cfg["data"])
    if args.list:
        for index, scenario in enumerate(scenarios):
            print(index, scenario)
        return
    if args.scenario_index is None or args.output_dir is None:
        parser.error("--scenario-index and --output-dir are required unless --list is given")
    if not 0 <= args.scenario_index < len(scenarios):
        raise ValueError(f"scenario-index must be in [0, {len(scenarios) - 1}]")
    cfg = deepcopy(cfg)
    cfg["data"].pop("grid", None)
    cfg["data"]["scenarios"] = [scenarios[args.scenario_index]]
    cfg["models"]["dvfm"]["latent_dims"] = [1]
    cfg["evaluation"]["joint_survival_aggregate_posterior"] = True
    cfg["study"]["output_dir"] = str(args.output_dir)
    validate_config(cfg)

    device = resolve_device(args.device or str(cfg["compute"].get("device", "auto")))
    torch.set_num_threads(max(1, int(cfg["compute"].get("torch_num_threads", 1))))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "environment.json", environment_record(device))
    write_json(args.output_dir / "resolved_config.json", cfg)
    results = run_synthetic_pilot(cfg, args.output_dir, device)
    results.to_csv(args.output_dir / "results_raw.csv", index=False)
    manifest = pd.read_csv(args.output_dir / "run_manifest.csv")
    failed = manifest.loc[manifest["status"].ne("success")]
    if not failed.empty:
        # run_synthetic_pilot records failures instead of raising; a non-zero exit
        # keeps GWF from marking the cell complete.
        raise SystemExit(f"{len(failed)} fit(s) failed; see {args.output_dir / 'failure_summary.csv'}")


if __name__ == "__main__":
    main()
