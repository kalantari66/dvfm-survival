"""GWF workflow for the PRO-ACT real-data latent ablation.

Two targets: a CPU job builds the death-versus-censoring cohort from the raw
PRO-ACT forms, then one GPU job fits DVFM with every configured latent
dimension on every seed. The experiment YAML is the single source of truth
for scientific settings, SLURM resources and the raw-data location. PRO-ACT is
access-controlled: copy the raw forms to ``workflow.raw_dir`` (under the
git-ignored ``data/``) before running. From the repository root:

    gwf -f workflows/proact_latent_ablation/workflow.py status
    gwf -f workflows/proact_latent_ablation/workflow.py run
"""

from __future__ import annotations

from pathlib import Path

import yaml
from gwf import AnonymousTarget, Workflow


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_CONFIG = PROJECT_ROOT / "configs" / "proact_latent_ablation.yaml"
# Only the forms the cohort builder reads; see utility.proact.
RAW_FORMS = (
    "ALSFRS", "ALSHISTORY", "DEMOGRAPHICS", "RILUZOLE", "FVC", "DEATHDATA",
    "SVC", "VITALSIGNS", "HANDGRIPSTRENGTH", "MUSCLESTRENGTH",
)


def _source_files() -> list[str]:
    return sorted(
        str(path)
        for package in ("dvfm", "experiments", "sota", "utility")
        for path in (PROJECT_ROOT / "src" / package).glob("*.py")
    )


def _preamble(cores: int) -> str:
    return f"""
    set -euo pipefail
    cd "{PROJECT_ROOT}"
    echo "[GWF] host=$(hostname)"
    export PYTHONUNBUFFERED=1
    export OMP_NUM_THREADS="{cores}"
    export MKL_NUM_THREADS="{cores}"
    export OPENBLAS_NUM_THREADS="{cores}"
    export NUMEXPR_NUM_THREADS="{cores}"
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate dvfm
    """


def build_cohort(raw_dir: Path, cohort_path: Path, resources: dict) -> AnonymousTarget:
    """Build the analysis cohort from the raw forms on CPU."""
    flow_path = cohort_path.with_name(f"{cohort_path.stem}_cohort_flow.json")
    inputs = [
        *(str(raw_dir / f"PROACT_{form}.csv") for form in RAW_FORMS),
        str(PROJECT_ROOT / "scripts" / "build_proact_death_cohort.py"),
        str(PROJECT_ROOT / "src" / "utility" / "proact.py"),
    ]
    options = dict(
        cores=str(resources["cores"]), memory=str(resources["memory"]),
        walltime="00:30:00", account=str(resources["account"]),
    )
    spec = _preamble(int(resources["cores"])) + f"""
    echo "[GWF] $(date) building PRO-ACT cohort from {raw_dir}"
    export PYTHONPATH="{PROJECT_ROOT / 'src'}"
    python scripts/build_proact_death_cohort.py --raw-dir "{raw_dir}" --output "{cohort_path}"
    test -s "{cohort_path}"
    test -s "{flow_path}"
    echo "[GWF] $(date) cohort written to {cohort_path}"
    """
    return AnonymousTarget(
        inputs=inputs, outputs=[str(cohort_path), str(flow_path)],
        options=options, spec=spec,
    )


def run_ablation(cohort_path: Path, result_dir: Path, resources: dict) -> AnonymousTarget:
    """Fit every latent dimension on every seed in one GPU job."""
    artifacts = (
        "results_raw.csv", "results_mean.csv", "results_std.csv",
        "run_manifest.csv", "paired_differences.csv", "paired_summary.csv",
        "latent_external_validation.csv", "feature_missingness.csv",
        "resolved_config.json",
    )
    inputs = [
        str(EXPERIMENT_CONFIG), str(cohort_path),
        str(PROJECT_ROOT / "environment.yml"), str(PROJECT_ROOT / "pyproject.toml"),
        *_source_files(),
    ]
    options = dict(
        cores=str(resources["cores"]), memory=str(resources["memory"]),
        walltime=str(resources["walltime"]), account=str(resources["account"]),
        gres=f"gpu:1 -p {resources['partition']}",
    )
    checks = "\n".join(f'    test -s "{result_dir / name}"' for name in artifacts)
    spec = _preamble(int(resources["cores"])) + f"""
    mkdir -p "{result_dir}"
    rm -f "{result_dir / '_SUCCESS'}"
    echo "[GWF] $(date) starting PRO-ACT latent ablation"
    echo "[GWF] result_dir={result_dir}"
    echo "[GWF] CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-unset}}"
    python -m experiments.cli --config "{EXPERIMENT_CONFIG}" --validate-only
    python -m experiments.cli --config "{EXPERIMENT_CONFIG}"
{checks}
    # The runner writes _SUCCESS only when every paired fit completed.
    test -e "{result_dir / '_SUCCESS'}"
    echo "[GWF] $(date) completed PRO-ACT latent ablation"
    """
    return AnonymousTarget(
        inputs=inputs,
        outputs=[*(str(result_dir / name) for name in artifacts), str(result_dir / "_SUCCESS")],
        options=options, spec=spec,
    )


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


with EXPERIMENT_CONFIG.open(encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

workflow_config = config["workflow"]
resources = config["resources"]
datasets = config["data"]["datasets"]
if len(datasets) != 1:
    raise ValueError("The PRO-ACT workflow expects exactly one dataset")
cohort_path = _project_path(datasets[0]["path"])
result_dir = _project_path(config["study"]["output_dir"])
target_name = str(workflow_config["target_name"])

gwf = Workflow()
gwf.target_from_template(
    name=f"{target_name}_cohort",
    template=build_cohort(_project_path(workflow_config["raw_dir"]), cohort_path, resources),
)
gwf.target_from_template(
    name=target_name,
    template=run_ablation(cohort_path, result_dir, resources),
)
