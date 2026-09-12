"""One GWF target per semi-synthetic dataset plus a cross-dataset aggregator."""

from __future__ import annotations

from pathlib import Path

import yaml
from gwf import AnonymousTarget, Workflow


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_CONFIG = PROJECT_ROOT / "configs" / "semi_synthetic.yaml"
RUNNER = PROJECT_ROOT / "scripts" / "run_semi_synthetic_dataset.py"
AGGREGATOR = PROJECT_ROOT / "scripts" / "aggregate_semi_synthetic_results.py"
RESULT_FILES = ("results_raw.csv", "results_mean.csv", "results_std.csv", "dgp_diagnostics.csv")
CROSS_DATASET_FILES = (*RESULT_FILES, "dataset_characteristics.csv")


def _options(resources: dict) -> dict[str, str]:
    return {
        "cores": str(resources["cores"]), "memory": str(resources["memory"]),
        "walltime": str(resources["walltime"]), "account": str(resources["account"]),
        "gres": f"gpu:1 -p {resources['partition']}",
    }


def run_dataset(dataset: dict, result_root: Path, resources: dict) -> AnonymousTarget:
    """Run exactly one configured dataset with the YAML's resource allocation."""
    name = str(dataset["name"])
    result_dir = result_root / name
    source_path = None
    if dataset.get("path"):
        source_path = Path(dataset["path"])
        if not source_path.is_absolute():
            source_path = PROJECT_ROOT / source_path
    outputs = [result_dir / filename for filename in RESULT_FILES]
    outputs.extend([result_dir / "resolved_config.json", result_dir / "_SUCCESS"])
    checks = "\n    ".join(f'test -s "{path}"' for path in outputs[:-1])
    spec = f"""
    set -euo pipefail
    cd "{PROJECT_ROOT}"
    mkdir -p "{result_dir}"
    rm -f "{result_dir / '_SUCCESS'}"
    echo "[GWF] $(date) starting semi-synthetic dataset: {name}"
    export PYTHONUNBUFFERED=1
    export OMP_NUM_THREADS="{resources['cores']}"
    export MKL_NUM_THREADS="{resources['cores']}"
    export OPENBLAS_NUM_THREADS="{resources['cores']}"
    export NUMEXPR_NUM_THREADS="{resources['cores']}"
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate dvfm
    python "{RUNNER}" --config "{EXPERIMENT_CONFIG}" --dataset "{name}" --output-dir "{result_dir}" --validate-only
    python "{RUNNER}" --config "{EXPERIMENT_CONFIG}" --dataset "{name}" --output-dir "{result_dir}"
    {checks}
    touch "{result_dir / '_SUCCESS'}"
    echo "[GWF] $(date) completed semi-synthetic dataset: {name}"
    """
    inputs = [
        EXPERIMENT_CONFIG, RUNNER, PROJECT_ROOT / "environment.yml",
        PROJECT_ROOT / "pyproject.toml",
        *sorted(path for package in ("dvfm", "experiments", "sota", "utility")
                for path in (PROJECT_ROOT / "src" / package).glob("*.py")),
    ]
    if source_path is not None:
        inputs.append(source_path)
    return AnonymousTarget(
        inputs=[str(path) for path in inputs],
        outputs=[str(path) for path in outputs],
        options=_options(resources), spec=spec,
    )


def aggregate(result_root: Path, datasets: list[dict], resources: dict) -> AnonymousTarget:
    """Merge cross-dataset files after every dataset target has completed."""
    names = [str(dataset["name"]) for dataset in datasets]
    inputs = [AGGREGATOR, EXPERIMENT_CONFIG, *(
        result_root / name / filename for name in names for filename in RESULT_FILES
    )]
    outputs = [result_root / filename for filename in CROSS_DATASET_FILES]
    outputs.extend([result_root / "resolved_config.json", result_root / "_SUCCESS"])
    checks = "\n    ".join(f'test -s "{path}"' for path in outputs[:-1])
    spec = f"""
    set -euo pipefail
    cd "{PROJECT_ROOT}"
    rm -f "{result_root / '_SUCCESS'}"
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate dvfm
    python "{AGGREGATOR}" --config "{EXPERIMENT_CONFIG}" --result-root "{result_root}"
    {checks}
    touch "{result_root / '_SUCCESS'}"
    """
    options = _options(resources)
    options.pop("gres")
    return AnonymousTarget(
        inputs=[str(path) for path in inputs],
        outputs=[str(path) for path in outputs], options=options, spec=spec,
    )


with EXPERIMENT_CONFIG.open(encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

result_root = Path(config["study"]["output_dir"])
if not result_root.is_absolute():
    result_root = PROJECT_ROOT / result_root
datasets = config["data"]["datasets"]

gwf = Workflow()
for dataset in datasets:
    gwf.target_from_template(
        f"{config['workflow']['target_name']}_{dataset['name']}",
        run_dataset(dataset, result_root, config["resources"]),
    )
gwf.target_from_template(
    f"{config['workflow']['target_name']}_aggregate",
    aggregate(result_root, datasets, config["resources"]),
)
