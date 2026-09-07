"""One-job GWF workflow for the DVFM synthetic hyperparameter sweep.

From the repository root:

    gwf -f workflows/synthetic_hyperparameter/workflow.py status
    gwf -f workflows/synthetic_hyperparameter/workflow.py run
"""

from __future__ import annotations

from pathlib import Path

import yaml
from gwf import AnonymousTarget, Workflow


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_CONFIG = PROJECT_ROOT / "configs" / "synthetic_hyperparameter_sweep.yaml"


def run_sweep(config_path: Path, result_dir: Path, resources: dict) -> AnonymousTarget:
    """Run every scenario and variant sequentially inside one GPU target."""
    outputs = [
        result_dir / "results_raw.csv",
        result_dir / "results_mean.csv",
        result_dir / "results_std.csv",
        result_dir / "training_history.csv.gz",
        result_dir / "dvfm_diagnostics.csv",
        result_dir / "calibration_curves.csv.gz",
        result_dir / "run_manifest.csv",
        result_dir / "hyperparameter_ranking.csv",
        result_dir / "hyperparameter_scenario_summary.csv",
        result_dir / "hyperparameter_paired_deltas.csv",
        result_dir / "resolved_config.json",
        result_dir / "_SUCCESS",
    ]
    options = dict(
        cores=str(resources["cores"]), memory=str(resources["memory"]),
        walltime=str(resources["walltime"]), account=str(resources["account"]),
        gres=f"gpu:1 -p {resources['partition']}",
    )
    output_checks = "\n    ".join(f'test -s "{path}"' for path in outputs[:-1])
    spec = f"""
    set -euo pipefail
    cd "{PROJECT_ROOT}"
    mkdir -p "{result_dir}"
    rm -f "{result_dir / '_SUCCESS'}"

    echo "[GWF] $(date) starting DVFM synthetic hyperparameter sweep"
    echo "[GWF] config={config_path}"
    echo "[GWF] result_dir={result_dir}"
    echo "[GWF] host=$(hostname) CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-unset}}"

    export PYTHONUNBUFFERED=1
    export OMP_NUM_THREADS="{resources['cores']}"
    export MKL_NUM_THREADS="{resources['cores']}"
    export OPENBLAS_NUM_THREADS="{resources['cores']}"
    export NUMEXPR_NUM_THREADS="{resources['cores']}"

    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate dvfm
    python -m experiments.cli --config "{config_path}" --validate-only
    python -m experiments.cli --config "{config_path}"

    {output_checks}
    touch "{outputs[-1]}"
    echo "[GWF] $(date) completed DVFM synthetic hyperparameter sweep"
    """
    inputs = [
        config_path, PROJECT_ROOT / "environment.yml", PROJECT_ROOT / "pyproject.toml",
        *sorted(
            path
            for package in ("dvfm", "experiments", "sota", "utility")
            for path in (PROJECT_ROOT / "src" / package).glob("*.py")
        ),
    ]
    return AnonymousTarget(
        inputs=[str(path) for path in inputs], outputs=[str(path) for path in outputs],
        options=options, spec=spec,
    )


with EXPERIMENT_CONFIG.open(encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

result_dir = Path(config["study"]["output_dir"])
if not result_dir.is_absolute():
    result_dir = PROJECT_ROOT / result_dir

gwf = Workflow()
gwf.target_from_template(
    name=str(config["workflow"]["target_name"]),
    template=run_sweep(EXPERIMENT_CONFIG, result_dir, config["resources"]),
)
