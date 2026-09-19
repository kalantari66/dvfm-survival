"""One 2-hour CPU GWF target per dataset, tuning DeepSurv only.

DeepSurv is a small MLP trained on the full cohort (the Cox partial likelihood's
risk sets must span it), so these targets request no GPU.

Run from the repository root:
    gwf -f workflows/semi_synthetic_tuning_deepsurv/workflow.py status
    gwf -f workflows/semi_synthetic_tuning_deepsurv/workflow.py run
"""

from __future__ import annotations

from pathlib import Path

import yaml
from gwf import AnonymousTarget, Workflow


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "semi_synthetic_tuning_deepsurv.yaml"
RUNNER = PROJECT_ROOT / "scripts" / "run_semi_synthetic_tuning.py"
AGGREGATOR = PROJECT_ROOT / "scripts" / "aggregate_semi_synthetic_tuning.py"
SUMMARY = PROJECT_ROOT / "scripts" / "print_semi_synthetic_tuning_summary.py"
MODEL = "deepsurv"


def _options(resources: dict) -> dict[str, str]:
    # No ``gres``/partition entry: these targets are CPU-only by design.
    return {
        "cores": str(resources["cores"]), "memory": str(resources["memory"]),
        "walltime": str(resources["walltime"]), "account": str(resources["account"]),
    }


def dataset_target(name: str, result_dir: Path, resources: dict) -> AnonymousTarget:
    outputs = [
        result_dir / "trials.csv", result_dir / "tuning_trials.csv",
        result_dir / "selected_hyperparameters.json",
        result_dir / "resolved_tuning_config.json", result_dir / "_SUCCESS",
    ]
    checks = "\n    ".join(f'test -s "{path}"' for path in outputs[:-1])
    spec = f"""
    set -euo pipefail
    cd "{PROJECT_ROOT}"
    mkdir -p "{result_dir}"
    rm -f "{result_dir / '_SUCCESS'}"
    echo "[GWF] $(date) starting DeepSurv tuning: {name}"
    export PYTHONUNBUFFERED=1
    export CUDA_VISIBLE_DEVICES=""
    export OMP_NUM_THREADS="{resources['cores']}"
    export MKL_NUM_THREADS="{resources['cores']}"
    export OPENBLAS_NUM_THREADS="{resources['cores']}"
    export NUMEXPR_NUM_THREADS="{resources['cores']}"
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate dvfm
    python "{RUNNER}" --config "{CONFIG_PATH}" --dataset "{name}" --validate-only
    python "{RUNNER}" --config "{CONFIG_PATH}" --dataset "{name}" --output-dir "{result_dir}"
    {checks}
    python "{SUMMARY}" --result-root "{result_dir}" --model "{MODEL}"
    touch "{result_dir / '_SUCCESS'}"
    echo "[GWF] $(date) completed DeepSurv tuning: {name}"
    """
    inputs = [
        CONFIG_PATH, RUNNER, PROJECT_ROOT / "configs" / "semi_synthetic.yaml",
        PROJECT_ROOT / "environment.yml", PROJECT_ROOT / "pyproject.toml",
        *sorted(path for package in ("dvfm", "experiments", "sota", "utility")
                for path in (PROJECT_ROOT / "src" / package).glob("*.py")),
    ]
    return AnonymousTarget(
        inputs=[str(path) for path in inputs], outputs=[str(path) for path in outputs],
        options=_options(resources), spec=spec,
    )


def aggregate_target(result_root: Path, names: list[str], resources: dict) -> AnonymousTarget:
    outputs = [
        result_root / "tuning_trials.csv", result_root / "selected_hyperparameters.json",
        result_root / "resolved_tuning_config.json", result_root / "_SUCCESS",
    ]
    inputs = [CONFIG_PATH, AGGREGATOR, SUMMARY, *(
        result_root / name / "tuning_trials.csv" for name in names
    )]
    checks = "\n    ".join(f'test -s "{path}"' for path in outputs[:-1])
    spec = f"""
    set -euo pipefail
    cd "{PROJECT_ROOT}"
    rm -f "{result_root / '_SUCCESS'}"
    export PYTHONUNBUFFERED=1
    export CUDA_VISIBLE_DEVICES=""
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate dvfm
    python "{AGGREGATOR}" --config "{CONFIG_PATH}" --result-root "{result_root}"
    {checks}
    python "{SUMMARY}" --result-root "{result_root}" --model "{MODEL}"
    touch "{result_root / '_SUCCESS'}"
    """
    return AnonymousTarget(
        inputs=[str(path) for path in inputs], outputs=[str(path) for path in outputs],
        options=_options(resources), spec=spec,
    )


with CONFIG_PATH.open(encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
if set(config["tuning"]["search_spaces"]) != {MODEL}:
    raise ValueError(
        f"{CONFIG_PATH.name} must search {MODEL} only; found "
        f"{sorted(config['tuning']['search_spaces'])}"
    )
result_root = Path(config["study"]["output_dir"])
if not result_root.is_absolute():
    result_root = PROJECT_ROOT / result_root
names = [item["name"] for item in yaml.safe_load(
    (PROJECT_ROOT / "configs" / "semi_synthetic.yaml").read_text(encoding="utf-8")
)["data"]["datasets"]]

gwf = Workflow()
for name in names:
    gwf.target_from_template(
        f"{config['workflow']['target_name']}_{name}",
        dataset_target(name, result_root / name, config["resources"]),
    )
gwf.target_from_template(
    f"{config['workflow']['target_name']}_aggregate",
    aggregate_target(result_root, names, config["resources"]),
)
