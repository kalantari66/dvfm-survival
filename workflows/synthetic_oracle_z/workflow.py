"""Single-job GWF workflow + template for synthetic_oracle_z.

From the repository root:

    gwf -f workflows/synthetic_oracle_z/workflow.py status
    gwf -f workflows/synthetic_oracle_z/workflow.py run
"""

from __future__ import annotations

from pathlib import Path

import yaml
from gwf import AnonymousTarget, Workflow

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_CONFIG = PROJECT_ROOT / "configs" / "synthetic_oracle_z.yaml"

def synthetic_oracle_z_template(
    experiment_config: Path,
    result_dir: Path,
    cores: int,
    memory: str,
    walltime: str,
    partition: str,
    account: str,
) -> AnonymousTarget:
    """Run the complete oracle-Z sanity experiment as one SLURM job."""

    experiment_script = (
        PROJECT_ROOT / "src" / "experiments" / "synthetic_oracle_z.py"
    )

    inputs = [
        str(experiment_config),
    ]

    outputs = [
        str(result_dir / "results_raw.csv"),
        str(result_dir / "results_mean.csv"),
        str(result_dir / "results_std.csv"),
        str(result_dir / "latent_effect_profiles.csv"),
        str(result_dir / "dgp_calibration.json"),
        str(result_dir / "resolved_config.yaml"),
        str(result_dir / "_SUCCESS"),
    ]

    options = {
        "cores": str(cores),
        "memory": str(memory),
        "walltime": str(walltime),
        "account": str(account),
        "gres": f"gpu:1 -p {partition}",
    }

    spec = f"""
    set -euo pipefail

    cd "{PROJECT_ROOT}"
    mkdir -p "{result_dir}"

    echo "[GWF] $(date) starting synthetic_oracle_z"
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
    test -s "{experiment_script}"

    python -u -m experiments.synthetic_oracle_z \
        --config "{experiment_config}"

    test -s "{result_dir / 'results_raw.csv'}"
    test -s "{result_dir / 'results_mean.csv'}"
    test -s "{result_dir / 'results_std.csv'}"
    test -s "{result_dir / 'latent_effect_profiles.csv'}"
    test -s "{result_dir / 'dgp_calibration.json'}"
    test -s "{result_dir / 'resolved_config.yaml'}"

    touch "{result_dir / '_SUCCESS'}"
    echo "[GWF] $(date) completed synthetic_oracle_z"
    """

    return AnonymousTarget(
        inputs=inputs,
        outputs=outputs,
        options=options,
        spec=spec,
    )

with EXPERIMENT_CONFIG.open(encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

gwf_cfg = config["gwf"]
result_dir = Path(config["output_dir"])
if not result_dir.is_absolute():
    result_dir = PROJECT_ROOT / result_dir

gwf = Workflow()
gwf.target_from_template(
    name=str(gwf_cfg["target_name"]),
    template=synthetic_oracle_z_template(
        experiment_config=EXPERIMENT_CONFIG,
        result_dir=result_dir,
        cores=int(gwf_cfg["cores"]),
        memory=str(gwf_cfg["memory"]),
        walltime=str(gwf_cfg["walltime"]),
        partition=str(gwf_cfg["partition"]),
        account=str(gwf_cfg["account"]),
    ),
)
