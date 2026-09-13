"""One-job GWF workflow for semi-synthetic development-set tuning.

Run from the repository root:
    gwf -f workflows/semi_synthetic_tuning/workflow.py status
    gwf -f workflows/semi_synthetic_tuning/workflow.py run
"""

from __future__ import annotations

from pathlib import Path

import yaml
from gwf import AnonymousTarget, Workflow


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "semi_synthetic_tuning.yaml"
RUNNER = PROJECT_ROOT / "scripts" / "run_semi_synthetic_tuning.py"


def _options(resources: dict) -> dict[str, str]:
    return {
        "cores": str(resources["cores"]), "memory": str(resources["memory"]),
        "walltime": str(resources["walltime"]), "account": str(resources["account"]),
        "gres": f"gpu:1 -p {resources['partition']}",
    }


with CONFIG_PATH.open(encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
resources = config["resources"]
result_dir = Path(config["study"]["output_dir"])
if not result_dir.is_absolute():
    result_dir = PROJECT_ROOT / result_dir
outputs = [
    result_dir / "tuning_trials.csv",
    result_dir / "selected_hyperparameters.json",
    result_dir / "resolved_tuning_config.json",
    result_dir / "_SUCCESS",
]
checks = "\n    ".join(f'test -s "{path}"' for path in outputs[:-1])
spec = f"""
set -euo pipefail
cd "{PROJECT_ROOT}"
mkdir -p "{result_dir}"
rm -f "{result_dir / '_SUCCESS'}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="{resources['cores']}"
export MKL_NUM_THREADS="{resources['cores']}"
export OPENBLAS_NUM_THREADS="{resources['cores']}"
export NUMEXPR_NUM_THREADS="{resources['cores']}"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate dvfm
python "{RUNNER}" --config "{CONFIG_PATH}" --validate-only
python "{RUNNER}" --config "{CONFIG_PATH}" --output-dir "{result_dir}"
{checks}
touch "{result_dir / '_SUCCESS'}"
"""
inputs = [
    CONFIG_PATH, RUNNER, PROJECT_ROOT / "configs" / "semi_synthetic.yaml",
    PROJECT_ROOT / "environment.yml", PROJECT_ROOT / "pyproject.toml",
    *sorted(path for package in ("dvfm", "experiments", "sota", "utility")
            for path in (PROJECT_ROOT / "src" / package).glob("*.py")),
]
gwf = Workflow()
gwf.target_from_template(
    str(config["workflow"]["target_name"]),
    AnonymousTarget(
        inputs=[str(path) for path in inputs], outputs=[str(path) for path in outputs],
        options=_options(resources), spec=spec,
    ),
)
