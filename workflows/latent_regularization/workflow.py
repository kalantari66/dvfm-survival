"""One-job GWF workflow for the DVFM latent-regularization ablation.

    gwf -f workflows/latent_regularization/workflow.py status
    gwf -f workflows/latent_regularization/workflow.py run
"""

from __future__ import annotations

from pathlib import Path

import yaml
from gwf import AnonymousTarget, Workflow


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "synthetic_latent_regularization.yaml"


def run_ablation(config_path: Path, result_dir: Path, resources: dict) -> AnonymousTarget:
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
    checks = "\n    ".join(f'test -s "{path}"' for path in outputs[:-1])
    spec = f"""
    set -euo pipefail
    cd "{PROJECT_ROOT}"
    mkdir -p "{result_dir}"
    rm -f "{result_dir / '_SUCCESS'}"

    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate dvfm
    export PYTHONUNBUFFERED=1
    export OMP_NUM_THREADS="{resources['cores']}"
    export MKL_NUM_THREADS="{resources['cores']}"
    export OPENBLAS_NUM_THREADS="{resources['cores']}"
    export NUMEXPR_NUM_THREADS="{resources['cores']}"

    echo "[GWF] $(date) starting DVFM latent-regularization ablation"
    echo "[GWF] host=$(hostname) CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-unset}}"
    python -m experiments.cli --config "{config_path}" --validate-only
    python -m experiments.cli --config "{config_path}"
    {checks}
    touch "{outputs[-1]}"
    echo "[GWF] $(date) completed DVFM latent-regularization ablation"
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
        inputs=[str(path) for path in inputs],
        outputs=[str(path) for path in outputs],
        options=dict(
            cores=str(resources["cores"]), memory=str(resources["memory"]),
            walltime=str(resources["walltime"]), account=str(resources["account"]),
            gres=f"gpu:1 -p {resources['partition']}",
        ),
        spec=spec,
    )


with CONFIG_PATH.open(encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

result_dir = Path(config["study"]["output_dir"])
if not result_dir.is_absolute():
    result_dir = PROJECT_ROOT / result_dir

gwf = Workflow()
gwf.target_from_template(
    name=str(config["workflow"]["target_name"]),
    template=run_ablation(CONFIG_PATH, result_dir, config["resources"]),
)
