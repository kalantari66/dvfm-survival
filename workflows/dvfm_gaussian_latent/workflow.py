"""Single-job GWF workflow for the DVFM Gaussian-latent benchmark.

The experiment YAML is the single source of truth for both scientific settings
and scheduler resources. The GWF template is intentionally kept in this file.

From the repository root:

    gwf -f workflows/dvfm_gaussian_latent/workflow.py status
    gwf -f workflows/dvfm_gaussian_latent/workflow.py run
"""

from __future__ import annotations

from pathlib import Path

import yaml
from gwf import AnonymousTarget, Workflow

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_CONFIG = (
    PROJECT_ROOT / "configs" / "synthetic_gaussian_latent.yaml"
)

def run_dvfm_experiment(
    experiment_config: Path,
    result_dir: Path,
    cores: int,
    memory: str,
    walltime: str,
    partition: str,
    account: str,
) -> AnonymousTarget:
    """Run the full Gaussian-latent benchmark as one GWF/SLURM target."""

    inputs = [
        str(experiment_config),
        str(PROJECT_ROOT / "src" / "experiments" / "synthetic_gaussian_latent.py"),
        str(PROJECT_ROOT / "src" / "dvfm" / "model_variants.py"),
        str(PROJECT_ROOT / "src" / "dvfm" / "reference_core.py"),
    ]

    outputs = [
        str(result_dir / "results_raw.csv"),
        str(result_dir / "results_mean.csv"),
        str(result_dir / "results_std.csv"),
        str(result_dir / "dgp_tau_calibration.csv"),
        str(result_dir / "resolved_config.yaml"),
        str(result_dir / "_SUCCESS"),
    ]

    options = {
        "cores": str(cores),
        "memory": memory,
        "walltime": walltime,
        "account": account,
        "gres": f"gpu:1 -p {partition}",
    }

    spec = f"""
    set -euo pipefail

    cd "{PROJECT_ROOT}"
    mkdir -p "{result_dir}"

    echo "[GWF] $(date) starting DVFM Gaussian-latent mechanistic benchmark"
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
    test -s "{PROJECT_ROOT / 'src' / 'experiments' / 'synthetic_gaussian_latent.py'}"
    test -s "{PROJECT_ROOT / 'src' / 'dvfm' / 'model_variants.py'}"

    python -u -m experiments.synthetic_gaussian_latent \
        --config "{experiment_config}"

    test -s "{result_dir / 'results_raw.csv'}"
    test -s "{result_dir / 'results_mean.csv'}"
    test -s "{result_dir / 'results_std.csv'}"
    test -s "{result_dir / 'dgp_tau_calibration.csv'}"
    test -s "{result_dir / 'resolved_config.yaml'}"

    touch "{result_dir / '_SUCCESS'}"

    echo "[GWF] $(date) completed DVFM Gaussian-latent mechanistic benchmark"
    """

    return AnonymousTarget(
        inputs=inputs,
        outputs=outputs,
        options=options,
        spec=spec,
    )

with EXPERIMENT_CONFIG.open(encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

gwf_config = config["gwf"]

result_dir = Path(config["output_dir"])
if not result_dir.is_absolute():
    result_dir = PROJECT_ROOT / result_dir

gwf = Workflow()
gwf.target_from_template(
    name=str(gwf_config["target_name"]),
    template=run_dvfm_experiment(
        experiment_config=EXPERIMENT_CONFIG,
        result_dir=result_dir,
        cores=int(gwf_config["cores"]),
        memory=str(gwf_config["memory"]),
        walltime=str(gwf_config["walltime"]),
        partition=str(gwf_config["partition"]),
        account=str(gwf_config["account"]),
    ),
)
