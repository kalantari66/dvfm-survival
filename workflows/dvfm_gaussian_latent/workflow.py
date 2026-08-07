"""Single-job GWF workflow for the DVFM Gaussian-latent benchmark.

From the repository root, assuming GWF is configured:

    gwf -f workflows/dvfm_gaussian_latent.py status
    gwf -f workflows/dvfm_gaussian_latent.py run

The GWF JSON contains scheduler-only settings. The experiment YAML remains the
single source of truth for the scientific grid and result output directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from gwf import Workflow

from src.gwf.dvfm_template import run_dvfm_experiment

gwf = Workflow()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GWF_CONFIG = PROJECT_ROOT / "configs" / "gwf" / "dvfm_gaussian_latent.json"

def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)

def required_dict(
    parent: dict[str, Any],
    key: str,
    where: str,
) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise KeyError(f"Missing required config block {where}.{key}")
    return value

gwf_cfg = load_json(GWF_CONFIG)
workflow_cfg = required_dict(gwf_cfg, "workflow", "gwf")
resources = required_dict(gwf_cfg, "resources", "gwf")
train_resources = required_dict(resources, "train", "gwf.resources")

experiment_config = PROJECT_ROOT / str(workflow_cfg["experiment_config"])
if not experiment_config.exists():
    raise FileNotFoundError(
        f"DVFM experiment config not found: {experiment_config}"
    )

with experiment_config.open(encoding="utf-8") as handle:
    experiment_cfg = yaml.safe_load(handle)

if not isinstance(experiment_cfg, dict):
    raise ValueError(
        f"Expected mapping in experiment config: {experiment_config}"
    )

output_dir_value = experiment_cfg.get("output_dir")
if not output_dir_value:
    raise KeyError(
        f"Missing output_dir in experiment config: {experiment_config}"
    )

result_dir = Path(str(output_dir_value))
if not result_dir.is_absolute():
    result_dir = PROJECT_ROOT / result_dir

target_name = str(
    workflow_cfg.get(
        "target_name",
        "dvfm_gaussian_latent_mechanistic",
    )
)

gwf.target_from_template(
    name=target_name,
    template=run_dvfm_experiment(
        experiment_config=str(experiment_config),
        result_dir=str(result_dir),
        cores=str(train_resources["cores"]),
        memory=str(train_resources["memory"]),
        walltime=str(train_resources["walltime"]),
        partition=str(train_resources["partition"]),
        account=str(resources["account"]),
    ),
)
