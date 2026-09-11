"""Single-job GWF workflow for the SUPPORT semi-synthetic pilot.

From the repository root:

    gwf -f workflows/semi_synthetic/workflow.py status
    gwf -f workflows/semi_synthetic/workflow.py run
"""

from __future__ import annotations

from pathlib import Path

import yaml
from gwf import AnonymousTarget, Workflow


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_CONFIG = PROJECT_ROOT / "configs" / "semi_synthetic.yaml"


def run_semi_synthetic_experiment(
    experiment_config: Path, result_dir: Path, cores: int, memory: str,
    walltime: str, partition: str, account: str,
) -> AnonymousTarget:
    """Generate and fit all configured datasets in one SLURM allocation."""
    with experiment_config.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)["data"]
    datasets = data.get("datasets", [data])
    dataset_paths = [Path(spec["path"]) for spec in datasets]
    inputs = [
        str(experiment_config),
        *(str(path if path.is_absolute() else PROJECT_ROOT / path) for path in dataset_paths),
        str(PROJECT_ROOT / "environment.yml"),
        str(PROJECT_ROOT / "pyproject.toml"),
        *sorted(
            str(path)
            for package in ("dvfm", "experiments", "sota", "utility")
            for path in (PROJECT_ROOT / "src" / package).glob("*.py")
        ),
    ]
    outputs = [
        str(result_dir / "results_raw.csv"),
        str(result_dir / "results_mean.csv"),
        str(result_dir / "results_std.csv"),
        str(result_dir / "dgp_diagnostics.csv"),
        str(result_dir / "resolved_config.json"),
        str(result_dir / "_SUCCESS"),
    ]
    options = dict(
        cores=str(cores), memory=memory, walltime=walltime, account=account,
        gres=f"gpu:1 -p {partition}",
    )
    spec = f"""
    set -euo pipefail

    cd "{PROJECT_ROOT}"
    mkdir -p "{result_dir}"
    rm -f "{result_dir / '_SUCCESS'}"

    echo "[GWF] $(date) starting SUPPORT semi-synthetic pilot"
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

    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate dvfm
    python -m experiments.cli --config "{experiment_config}" --validate-only
    python -m experiments.cli --config "{experiment_config}"

    test -s "{result_dir / 'results_raw.csv'}"
    test -s "{result_dir / 'results_mean.csv'}"
    test -s "{result_dir / 'results_std.csv'}"
    test -s "{result_dir / 'dgp_diagnostics.csv'}"
    test -s "{result_dir / 'resolved_config.json'}"
    touch "{result_dir / '_SUCCESS'}"
    echo "[GWF] $(date) completed SUPPORT semi-synthetic pilot"
    """
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)


with EXPERIMENT_CONFIG.open(encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

resources = config["resources"]
result_dir = Path(config["study"]["output_dir"])
if not result_dir.is_absolute():
    result_dir = PROJECT_ROOT / result_dir

gwf = Workflow()
gwf.target_from_template(
    name=str(config["workflow"]["target_name"]),
    template=run_semi_synthetic_experiment(
        experiment_config=EXPERIMENT_CONFIG, result_dir=result_dir,
        cores=int(resources["cores"]), memory=str(resources["memory"]),
        walltime=str(resources["walltime"]), partition=str(resources["partition"]),
        account=str(resources["account"]),
    ),
)
