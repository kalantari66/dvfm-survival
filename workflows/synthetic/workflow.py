"""Sharded GWF workflow for the canonical DVFM synthetic pilot.

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
    scenario_index: int,
    repeat_index: int,
    cores: int,
    memory: str,
    walltime: str,
    partition: str,
    account: str,
) -> AnonymousTarget:
    """Run all models for one expanded scenario and one paired seed repeat."""
    inputs = [
        str(experiment_config),
        str(PROJECT_ROOT / "environment.yml"),
        str(PROJECT_ROOT / "pyproject.toml"),
        *sorted(str(path) for path in (PROJECT_ROOT / "src" / "dvfm").glob("*.py")),
    ]

    outputs = [
        str(result_dir / "results_raw.csv"),
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
        walltime=str(walltime),
        account=account,
        gres=f"gpu:1 -p {partition}",
    )

    spec = f"""
    set -euo pipefail

    cd "{PROJECT_ROOT}"
    mkdir -p "{result_dir}"
    rm -f "{result_dir / '_SUCCESS'}"

    echo "[GWF] $(date) starting DVFM Gaussian-frailty scenario {scenario_index}, repeat {repeat_index}"
    echo "[GWF] experiment_config={experiment_config}"
    echo "[GWF] result_dir={result_dir}"
    echo "[GWF] account={account}"
    echo "[GWF] cores={cores}"
    echo "[GWF] memory={memory}"
    echo "[GWF] walltime={walltime}"
    echo "[GWF] partition={partition}"
    echo "[GWF] host=$(hostname)"
    echo "[GWF] cwd=$(pwd)"
    echo "[GWF] CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-unset}}"

    export PYTHONUNBUFFERED=1
    export OMP_NUM_THREADS="{cores}"
    export MKL_NUM_THREADS="{cores}"
    export OPENBLAS_NUM_THREADS="{cores}"
    export NUMEXPR_NUM_THREADS="{cores}"

    test -s "{experiment_config}"
    test -s "{PROJECT_ROOT / 'pyproject.toml'}"
    test -s "{PROJECT_ROOT / 'environment.yml'}"

    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate dvfm

    python -m dvfm.cli --config "{experiment_config}" --scenario-index {scenario_index} --repeat-index {repeat_index} --output-dir "{result_dir}" --validate-only
    python -m dvfm.cli --config "{experiment_config}" --scenario-index {scenario_index} --repeat-index {repeat_index} --output-dir "{result_dir}"

    test -s "{result_dir / 'results_raw.csv'}"
    test -s "{result_dir / 'training_history.csv.gz'}"
    test -s "{result_dir / 'dvfm_diagnostics.csv'}"
    test -s "{result_dir / 'calibration_curves.csv.gz'}"
    test -s "{result_dir / 'run_manifest.csv'}"
    test -s "{result_dir / 'resolved_config.json'}"

    touch "{result_dir / '_SUCCESS'}"
    echo "[GWF] $(date) completed DVFM Gaussian-frailty scenario {scenario_index}, repeat {repeat_index}"
    """

    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)


def aggregate_synthetic_experiment(
    experiment_config: Path,
    shard_root: Path,
    result_dir: Path,
    shard_outputs: list[str],
    account: str,
) -> AnonymousTarget:
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
    spec = f"""
    set -euo pipefail
    cd "{PROJECT_ROOT}"
    rm -f "{result_dir / '_SUCCESS'}"
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate dvfm
    python -m dvfm.aggregate_synthetic \
        --config "{experiment_config}" \
        --shard-root "{shard_root}" \
        --output-dir "{result_dir}"
    touch "{result_dir / '_SUCCESS'}"
    """
    return AnonymousTarget(
        inputs=[str(experiment_config), *shard_outputs], outputs=outputs,
        options=dict(cores="1", memory="8g", walltime="00:30:00", account=account),
        spec=spec,
    )


with EXPERIMENT_CONFIG.open(encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

workflow_config = config["workflow"]
resources = config["resources"]
result_dir = Path(config["study"]["output_dir"])
if not result_dir.is_absolute():
    result_dir = PROJECT_ROOT / result_dir

gwf = Workflow()
grid = config["data"]["grid"]
scenario_count = 1
for values in grid.values():
    scenario_count *= len(values if isinstance(values, list) else [values])

shard_root = result_dir / "shards"
shard_outputs = []
repeat_count = len(config["seeds"]["sampling"])
for scenario_index in range(scenario_count):
    for repeat_index in range(repeat_count):
        shard_dir = shard_root / f"scenario_{scenario_index:03d}_repeat_{repeat_index:03d}"
        success = str(shard_dir / "_SUCCESS")
        shard_outputs.append(success)
        gwf.target_from_template(
            name=f"{workflow_config['target_name']}_{scenario_index:03d}_{repeat_index:03d}",
            template=run_synthetic_experiment(
                experiment_config=EXPERIMENT_CONFIG,
                result_dir=shard_dir,
                scenario_index=scenario_index, repeat_index=repeat_index,
                cores=int(resources["cores"]), memory=str(resources["memory"]),
                walltime=str(resources["walltime"]),
                partition=str(resources["partition"]), account=str(resources["account"]),
            ),
        )

gwf.target_from_template(
    name=f"{workflow_config['target_name']}_aggregate",
    template=aggregate_synthetic_experiment(
        experiment_config=EXPERIMENT_CONFIG, shard_root=shard_root,
        result_dir=result_dir, shard_outputs=shard_outputs,
        account=str(resources["account"]),
    ),
)
