"""GWF templates for DVFM experiment jobs."""

from __future__ import annotations

from pathlib import Path

from gwf import AnonymousTarget

# This file is intended for <repo>/src/gwf/dvfm_template.py.
REPO_ROOT = Path(__file__).resolve().parents[2]

def run_dvfm_experiment(
    experiment_config: str,
    result_dir: str,
    cores: str = "4",
    memory: str = "25g",
    walltime: str = "02:00:00",
    partition: str = "gpu-short",
    account: str = "c2i-colon",
) -> AnonymousTarget:
    """Run the full mechanistic DVFM benchmark as one GWF/SLURM target.

    The experiment itself owns the scientific grid and writes incrementally to
    results_raw.csv. With `resume: true` in the YAML, resubmitting after a
    SLURM timeout continues from successful fits.

    Expected output files are the top-level benchmark summaries. Training
    histories and per-latent diagnostics are written as side products but are
    intentionally not declared individually as GWF outputs.
    """
    experiment_config_p = Path(experiment_config)
    result_dir_p = Path(result_dir)

    inputs = [
        str(experiment_config_p),
        str(REPO_ROOT / "src" / "experiments" / "synthetic_gaussian_latent.py"),
        str(REPO_ROOT / "src" / "dvfm" / "model_variants.py"),
        str(REPO_ROOT / "src" / "dvfm" / "reference_core.py"),
    ]

    outputs = [
        str(result_dir_p / "results_raw.csv"),
        str(result_dir_p / "results_mean.csv"),
        str(result_dir_p / "results_std.csv"),
        str(result_dir_p / "dgp_tau_calibration.csv"),
        str(result_dir_p / "resolved_config.yaml"),
        str(result_dir_p / "_SUCCESS"),
    ]

    options = dict(
        cores=str(cores),
        memory=str(memory),
        walltime=str(walltime),
        account=str(account),
        gres=f"gpu:1 -p {partition}",
    )

    spec = f"""
    set -euo pipefail

    cd "{REPO_ROOT}"
    mkdir -p "{result_dir_p}"

    echo "[GWF] $(date) starting DVFM Gaussian-latent mechanistic benchmark"
    echo "[GWF] experiment_config={experiment_config_p}"
    echo "[GWF] result_dir={result_dir_p}"
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

    test -s "{experiment_config_p}"
    test -s "{REPO_ROOT / 'src' / 'experiments' / 'synthetic_gaussian_latent.py'}"
    test -s "{REPO_ROOT / 'src' / 'dvfm' / 'model_variants.py'}"

    poetry run python -u -m experiments.synthetic_gaussian_latent \\
        --config "{experiment_config_p}"

    test -s "{result_dir_p / 'results_raw.csv'}"
    test -s "{result_dir_p / 'results_mean.csv'}"
    test -s "{result_dir_p / 'results_std.csv'}"
    test -s "{result_dir_p / 'dgp_tau_calibration.csv'}"
    test -s "{result_dir_p / 'resolved_config.yaml'}"

    touch "{result_dir_p / '_SUCCESS'}"

    echo "[GWF] $(date) completed DVFM Gaussian-latent mechanistic benchmark"
    """

    return AnonymousTarget(
        inputs=inputs,
        outputs=outputs,
        options=options,
        spec=spec,
    )
