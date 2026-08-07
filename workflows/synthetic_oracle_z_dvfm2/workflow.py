"""GWF workflow for the new DVFM2 Oracle-Z Gaussian benchmark.

This file intentionally contains BOTH the workflow and its GWF template.
Scheduler settings live in the experiment YAML.

Expected repository placement:
    workflows/dvfm2_oracle_gaussian.py

Experiment config:
    configs/experiments/synthetic_oracle_z_dvfm2.yaml

Run:
    gwf -f workflows/dvfm2_oracle_gaussian.py status
    gwf -f workflows/dvfm2_oracle_gaussian.py run
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from gwf import AnonymousTarget, Workflow

gwf = Workflow()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_CONFIG = (
    PROJECT_ROOT / "configs" / "synthetic_oracle_z_dvfm2.yaml"
)

def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return cfg

def _require(cfg: dict[str, Any], key: str, where: str):
    if key not in cfg:
        raise KeyError(f"Missing required key {where}.{key}")
    return cfg[key]

def run_oracle_experiment(
    experiment_config: Path,
    result_dir: Path,
    gwf_cfg: dict[str, Any],
) -> AnonymousTarget:
    """Embedded GWF template for one full Oracle-Z/DVFM2 experiment job."""

    account = str(_require(gwf_cfg, "account", "gwf"))
    cores = int(_require(gwf_cfg, "cores", "gwf"))
    memory = str(_require(gwf_cfg, "memory", "gwf"))
    walltime = str(_require(gwf_cfg, "walltime", "gwf"))
    gpus = int(gwf_cfg.get("gpus", 1))
    conda_env = str(gwf_cfg.get("conda_env", "dvfm"))
    module = str(
        gwf_cfg.get(
            "experiment_module",
            "experiments.synthetic_oracle_z_dvfm2",
        )
    )

    outputs = [
        str(result_dir / "results_raw.csv"),
        str(result_dir / "results_mean.csv"),
        str(result_dir / "results_std.csv"),
        str(result_dir / "resolved_config.yaml"),
        str(result_dir / "dgp_calibration.json"),
    ]

    options = {
        "account": account,
        "cores": cores,
        "memory": memory,
        "walltime": walltime,
        "gres": f"gpu:1 -p gpu-h200",
    }

    spec = f"""
    set -euo pipefail

    cd "{PROJECT_ROOT}"

    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate "{conda_env}"

    echo "[GWF] host=$(hostname)"
    echo "[GWF] experiment={experiment_config}"
    echo "[GWF] output={result_dir}"
    echo "[GWF] CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-unset}}"

    python -u -m "{module}" \
        --config "{experiment_config}"
    """

    return AnonymousTarget(
        inputs=[
            str(experiment_config),
            str(PROJECT_ROOT / "src" / "dvfm2" / "model.py"),
            str(PROJECT_ROOT / "src" / "dvfm2" / "training.py"),
            str(PROJECT_ROOT / "src" / "dvfm2" / "prediction.py"),
            str(PROJECT_ROOT / "src" / "dvfm2" / "diagnostics.py"),
            str(PROJECT_ROOT / "src" / "experiments" / "synthetic_oracle_z_dvfm2.py"),
        ],
        outputs=outputs,
        options=options,
        spec=spec,
    )

cfg = _load_yaml(EXPERIMENT_CONFIG)
gwf_cfg = _require(cfg, "gwf", "config")

output_dir = Path(str(_require(cfg, "output_dir", "config")))
if not output_dir.is_absolute():
    output_dir = PROJECT_ROOT / output_dir

target_name = str(gwf_cfg.get("target_name", "dvfm2_oracle_gaussian_d1"))

gwf.target_from_template(
    name=target_name,
    template=run_oracle_experiment(
        experiment_config=EXPERIMENT_CONFIG,
        result_dir=output_dir,
        gwf_cfg=gwf_cfg,
    ),
)
