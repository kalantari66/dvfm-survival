"""One resource-aware GWF target per dataset/model/seed plus aggregation."""

from __future__ import annotations

from pathlib import Path

import yaml
from gwf import AnonymousTarget, Workflow


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_CONFIG = PROJECT_ROOT / "configs" / "semi_synthetic.yaml"
RUNNER = PROJECT_ROOT / "scripts" / "run_semi_synthetic_dataset.py"
SEED_AGGREGATOR = (
    PROJECT_ROOT / "scripts" / "aggregate_semi_synthetic_model_seeds.py"
)
MODEL_AGGREGATOR = (
    PROJECT_ROOT / "scripts" / "aggregate_semi_synthetic_dataset_models.py"
)
AGGREGATOR = PROJECT_ROOT / "scripts" / "aggregate_semi_synthetic_results.py"
RESULT_FILES = ("results_raw.csv", "results_mean.csv", "results_std.csv", "dgp_diagnostics.csv")
CROSS_DATASET_FILES = (
    *RESULT_FILES, "dataset_characteristics.csv",
    "accuracy_primary_and_sensitivity.csv",
)
CPU_MODELS = {
    "coxph", "rsf", "mtlr", "deepsurv",
    "bayesian_cox_gamma_frailty", "clayton_aft",
}
GPU_MODELS = {"hacsurv_2d", "dvfm"}


def _options(resources: dict, model: str | None = None) -> dict[str, str]:
    options = {
        "cores": str(resources["cores"]), "memory": str(resources["memory"]),
        "walltime": str(resources["walltime"]), "account": str(resources["account"]),
    }
    if model is not None:
        model = str(model).lower()
        if model in GPU_MODELS:
            options["gres"] = f"gpu:1 -p {resources['partition']}"
        elif model not in CPU_MODELS:
            raise ValueError(f"No semi-synthetic resource policy for model {model!r}")
    return options


def run_dataset_model_seed(
    dataset: dict, model: str, seed_index: int, result_root: Path, resources: dict
) -> AnonymousTarget:
    """Run exactly one configured dataset/model/seed combination."""
    name = str(dataset["name"])
    model = str(model).lower()
    result_dir = result_root / name / model / f"seed_{seed_index}"
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
    echo "[GWF] $(date) starting semi-synthetic dataset/model/seed: {name}/{model}/{seed_index}"
    export PYTHONUNBUFFERED=1
    export OMP_NUM_THREADS="{resources['cores']}"
    export MKL_NUM_THREADS="{resources['cores']}"
    export OPENBLAS_NUM_THREADS="{resources['cores']}"
    export NUMEXPR_NUM_THREADS="{resources['cores']}"
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate dvfm
    python "{RUNNER}" --config "{EXPERIMENT_CONFIG}" --dataset "{name}" --model "{model}" --seed-index "{seed_index}" --output-dir "{result_dir}" --validate-only
    python "{RUNNER}" --config "{EXPERIMENT_CONFIG}" --dataset "{name}" --model "{model}" --seed-index "{seed_index}" --output-dir "{result_dir}"
    {checks}
    touch "{result_dir / '_SUCCESS'}"
    echo "[GWF] $(date) completed semi-synthetic dataset/model/seed: {name}/{model}/{seed_index}"
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
        options=_options(resources, model), spec=spec,
    )


def aggregate_seeds(
    dataset: dict, model: str, seed_count: int, result_root: Path, resources: dict
) -> AnonymousTarget:
    """Merge all seed jobs for one dataset/model and recompute summaries."""
    name = str(dataset["name"])
    model = str(model).lower()
    result_dir = result_root / name / model
    inputs = [SEED_AGGREGATOR, EXPERIMENT_CONFIG, *(
        result_dir / f"seed_{seed_index}" / filename
        for seed_index in range(seed_count) for filename in RESULT_FILES
    )]
    outputs = [result_dir / filename for filename in RESULT_FILES]
    outputs.extend([result_dir / "resolved_config.json", result_dir / "_SUCCESS"])
    checks = "\n    ".join(f'test -s "{path}"' for path in outputs[:-1])
    spec = f"""
    set -euo pipefail
    cd "{PROJECT_ROOT}"
    rm -f "{result_dir / '_SUCCESS'}"
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate dvfm
    python "{SEED_AGGREGATOR}" --config "{EXPERIMENT_CONFIG}" --dataset "{name}" --model "{model}" --result-root "{result_root}"
    {checks}
    touch "{result_dir / '_SUCCESS'}"
    """
    return AnonymousTarget(
        inputs=[str(path) for path in inputs],
        outputs=[str(path) for path in outputs],
        options=_options(resources),
        spec=spec,
    )


def aggregate_models(
    dataset: dict, models: list[str], result_root: Path, resources: dict
) -> AnonymousTarget:
    """Merge every model job for one dataset before cross-dataset aggregation."""
    name = str(dataset["name"])
    result_dir = result_root / name
    inputs = [MODEL_AGGREGATOR, EXPERIMENT_CONFIG, *(
        result_dir / str(model).lower() / filename
        for model in models for filename in RESULT_FILES
    )]
    outputs = [result_dir / filename for filename in RESULT_FILES]
    outputs.extend([result_dir / "resolved_config.json", result_dir / "_SUCCESS"])
    checks = "\n    ".join(f'test -s "{path}"' for path in outputs[:-1])
    spec = f"""
    set -euo pipefail
    cd "{PROJECT_ROOT}"
    mkdir -p "{result_dir}"
    rm -f "{result_dir / '_SUCCESS'}"
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate dvfm
    python "{MODEL_AGGREGATOR}" --config "{EXPERIMENT_CONFIG}" --dataset "{name}" --result-root "{result_root}"
    {checks}
    touch "{result_dir / '_SUCCESS'}"
    """
    return AnonymousTarget(
        inputs=[str(path) for path in inputs],
        outputs=[str(path) for path in outputs],
        options=_options(resources),
        spec=spec,
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
    return AnonymousTarget(
        inputs=[str(path) for path in inputs],
        outputs=[str(path) for path in outputs],
        options=_options(resources), spec=spec,
    )


with EXPERIMENT_CONFIG.open(encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

result_root = Path(config["study"]["output_dir"])
if not result_root.is_absolute():
    result_root = PROJECT_ROOT / result_root
datasets = config["data"]["datasets"]
models = [str(model).lower() for model in config["models"]["enabled"]]
seed_count = (
    len(config["seeds"])
    if isinstance(config["seeds"], list)
    else len(config["seeds"]["model"])
)

gwf = Workflow()
for dataset in datasets:
    for model in models:
        for seed_index in range(seed_count):
            gwf.target_from_template(
                f"{config['workflow']['target_name']}_{dataset['name']}_{model}_seed_{seed_index}",
                run_dataset_model_seed(
                    dataset, model, seed_index, result_root, config["resources"]
                ),
            )
        gwf.target_from_template(
            f"{config['workflow']['target_name']}_{dataset['name']}_{model}_aggregate_seeds",
            aggregate_seeds(
                dataset, model, seed_count, result_root, config["resources"]
            ),
        )
    gwf.target_from_template(
        f"{config['workflow']['target_name']}_{dataset['name']}_aggregate_models",
        aggregate_models(dataset, models, result_root, config["resources"]),
    )
gwf.target_from_template(
    f"{config['workflow']['target_name']}_aggregate",
    aggregate(result_root, datasets, config["resources"]),
)
