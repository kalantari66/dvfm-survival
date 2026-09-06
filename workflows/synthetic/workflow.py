"""Single-job GWF workflow for the canonical DVFM synthetic pilot.

The experiment YAML is the single source of truth for scientific settings and
SLURM resources. From the repository root:

    gwf -f workflows/synthetic/workflow.py status
    gwf -f workflows/synthetic/workflow.py run
"""

from __future__ import annotations

from pathlib import Path

import yaml
from gwf import AnonymousTarget, Workflow


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_CONFIG = PROJECT_ROOT / "configs" / "synthetic_pilot.yaml"


def run_synthetic_experiment(
    experiment_config: Path,
    result_dir: Path,
    cores: int,
    memory: str,
    walltime: str,
    partition: str,
    account: str,
) -> AnonymousTarget:
    """Run the complete synthetic grid in one GWF/SLURM target."""
    inputs = [
        str(experiment_config),
        str(PROJECT_ROOT / "environment.yml"),
        str(PROJECT_ROOT / "pyproject.toml"),
        *sorted(str(path) for path in (PROJECT_ROOT / "src" / "dvfm").glob("*.py")),
    ]
    outputs = [
        str(result_dir / "results_raw.csv"),
        str(result_dir / "results_mean.csv"),
        str(result_dir / "results_std.csv"),
        str(result_dir / "training_history.csv.gz"),
        str(result_dir / "dvfm_diagnostics.csv"),
        str(result_dir / "calibration_curves.csv.gz"),
        str(result_dir / "run_manifest.csv"),
        str(result_dir / "resolved_config.json"),
        str(result_dir / "_SUCCESS"),
    ]
    options = dict(
        cores=str(cores),
        memory=memory,
        walltime=walltime,
        account=account,
        gres=f"gpu:1 -p {partition}",
    )
    spec = f"""
    set -euo pipefail

    cd "{PROJECT_ROOT}"
    mkdir -p "{result_dir}"
    rm -f "{result_dir / '_SUCCESS'}"

    echo "[GWF] $(date) starting complete DVFM Gaussian-frailty grid pilot"
    echo "[GWF] experiment_config={experiment_config}"
    echo "[GWF] result_dir={result_dir}"
    echo "[GWF] resources={cores} cores, {memory}, {walltime}, {partition}, {account}"
    echo "[GWF] host=$(hostname)"
    echo "[GWF] CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-unset}}"

    export PYTHONUNBUFFERED=1
    export OMP_NUM_THREADS="{cores}"
    export MKL_NUM_THREADS="{cores}"
    export OPENBLAS_NUM_THREADS="{cores}"
    export NUMEXPR_NUM_THREADS="{cores}"

    test -s "{experiment_config}"
    test -s "{PROJECT_ROOT / 'environment.yml'}"
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate dvfm

    python -m dvfm.cli --config "{experiment_config}" --validate-only
    python -m dvfm.cli --config "{experiment_config}"

    test -s "{result_dir / 'results_raw.csv'}"
    test -s "{result_dir / 'results_mean.csv'}"
    test -s "{result_dir / 'results_std.csv'}"
    test -s "{result_dir / 'training_history.csv.gz'}"
    test -s "{result_dir / 'dvfm_diagnostics.csv'}"
    test -s "{result_dir / 'calibration_curves.csv.gz'}"
    test -s "{result_dir / 'run_manifest.csv'}"
    test -s "{result_dir / 'resolved_config.json'}"

    touch "{result_dir / '_SUCCESS'}"
    echo "[GWF] $(date) completed complete DVFM Gaussian-frailty grid pilot"
    """
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)


with EXPERIMENT_CONFIG.open(encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

workflow_config = config["workflow"]
resources = config["resources"]
result_dir = Path(config["study"]["output_dir"])
if not result_dir.is_absolute():
    result_dir = PROJECT_ROOT / result_dir

gwf = Workflow()
gwf.target_from_template(
    name=str(workflow_config["target_name"]),
    template=run_synthetic_experiment(
        experiment_config=EXPERIMENT_CONFIG,
        result_dir=result_dir,
        cores=int(resources["cores"]),
        memory=str(resources["memory"]),
        walltime=str(resources["walltime"]),
        partition=str(resources["partition"]),
        account=str(resources["account"]),
    ),
)
