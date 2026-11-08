"""GWF targets for the rebuttal experiments. Outputs go under rebuttal/.

From the repository root:

    gwf -f workflows/rebuttal/workflow.py status
    gwf -f workflows/rebuttal/workflow.py run task1_*          # one task at a time
    gwf -f workflows/rebuttal/workflow.py run

Target names start with the task number. Every job writes _SUCCESS last, so
GWF reruns only what is missing or failed. Resources: configs/rebuttal/workflow.yaml.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from gwf import AnonymousTarget, Workflow

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT_ROOT / "scripts" / "rebuttal"
OUT = PROJECT_ROOT / "rebuttal"
SETTINGS = yaml.safe_load((PROJECT_ROOT / "configs" / "rebuttal" / "workflow.yaml").read_text(encoding="utf-8"))
SEMI_CONFIG = PROJECT_ROOT / "configs" / "semi_synthetic.yaml"


def _semi_datasets() -> list[str]:
    config = yaml.safe_load(SEMI_CONFIG.read_text(encoding="utf-8"))
    return [str(item["name"]) for item in config["data"]["datasets"]]


def _sources(*extra: Path) -> list[str]:
    files = [
        *(path for package in ("dvfm", "experiments", "sota", "utility")
          for path in (PROJECT_ROOT / "src" / package).glob("*.py")),
        SCRIPTS / "rebuttal_common.py", *extra,
    ]
    return sorted(str(path) for path in files)


def _options(kind: str) -> dict[str, str]:
    resources = SETTINGS["resources"][kind]
    options = {
        "cores": str(resources["cores"]), "memory": str(resources["memory"]),
        "walltime": str(resources["walltime"]), "account": str(SETTINGS["account"]),
    }
    if kind.startswith("gpu"):
        # Same partition hand-off as workflows/semi_synthetic/workflow.py.
        options["gres"] = f"gpu:1 -p {resources['partition']}"
    return options


def _job(command: str, result_dir: Path, outputs: list[str], inputs: list[str], kind: str) -> AnonymousTarget:
    cores = SETTINGS["resources"][kind]["cores"]
    checks = "\n    ".join(f'test -s "{result_dir / name}"' for name in outputs)
    spec = f"""
    set -euo pipefail
    cd "{PROJECT_ROOT}"
    mkdir -p "{result_dir}"
    echo "[GWF] $(date) host=$(hostname) {result_dir}"
    export PYTHONUNBUFFERED=1
    export OMP_NUM_THREADS="{cores}" MKL_NUM_THREADS="{cores}" OPENBLAS_NUM_THREADS="{cores}" NUMEXPR_NUM_THREADS="{cores}"
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate dvfm
    {command}
    {checks}
    test -e "{result_dir / '_SUCCESS'}"
    echo "[GWF] $(date) done"
    """
    return AnonymousTarget(
        inputs=inputs,
        outputs=[str(result_dir / name) for name in (*outputs, "_SUCCESS")],
        options=_options(kind), spec=spec,
    )


gwf = Workflow()
seeds = range(int(SETTINGS["seed_count"]))

# Task 1: prior vs aggregate posterior. Synthetic grid (d_z = 1), one job per cell.
script = SCRIPTS / "synthetic_cell.py"
for index in range(int(SETTINGS["synthetic_scenarios"])):
    result_dir = OUT / "task1_prior_vs_aggregate" / "synthetic" / f"scenario_{index:02d}"
    gwf.target_from_template(f"task1_synthetic_{index:02d}", _job(
        f'python "{script}" --scenario-index {index} --output-dir "{result_dir}"',
        result_dir, ["results_raw.csv", "run_manifest.csv", "dvfm_diagnostics.csv"],
        _sources(script, PROJECT_ROOT / "configs" / "synthetic.yaml"), "gpu",
    ))

# Task 1: semi-synthetic DVFM d_z = 1 refits (also the DVFM side of Task 3 and
# the d_z = 1 reference of Task 5). One job per dataset x seed, 5 scenarios each.
script = SCRIPTS / "semi_synthetic_refit.py"
for dataset in _semi_datasets():
    for seed in seeds:
        result_dir = OUT / "task1_prior_vs_aggregate" / "semi_synthetic" / dataset / f"seed_{seed}"
        gwf.target_from_template(f"task1_semi_{dataset}_{seed}", _job(
            f'python "{script}" --variant dvfm --dataset {dataset} --seed-index {seed} --output-dir "{result_dir}"',
            result_dir, ["results_raw.csv"], _sources(script, SEMI_CONFIG), "gpu",
        ))

# Task 2: timings on one GPU node (pipeline devices) and its CPU (like-for-like).
script = SCRIPTS / "timings.py"
result_dir = OUT / "task2_timings"
gwf.target_from_template("task2_timings", _job(
    f'python "{script}" --datasets {" ".join(SETTINGS["task2_datasets"])} --output-dir "{result_dir}" '
    f'--threads {SETTINGS["resources"]["gpu_long"]["cores"]} && touch "{result_dir / "_SUCCESS"}"',
    result_dir, ["timing_rows.csv", "environment.json"], _sources(script, SEMI_CONFIG), "gpu_long",
))

# Task 3: CoxPH and Cox--Gamma recovery baselines on all 12 datasets (CPU).
script = SCRIPTS / "recovery_baselines_all.py"
for dataset in _semi_datasets():
    for seed in seeds:
        result_dir = OUT / "task3_recovery_all_datasets" / "baselines" / dataset / f"seed_{seed}"
        gwf.target_from_template(f"task3_baselines_{dataset}_{seed}", _job(
            f'python "{script}" --dataset {dataset} --seed-index {seed} --output-dir "{result_dir}"',
            result_dir, ["recovery_rows.csv", "ibs_crosscheck.csv"], _sources(script, SEMI_CONFIG), "cpu",
        ))

# Task 4: PRO-ACT D- and KM-calibration, 10 seeds x d_z in {0, 1}.
script = SCRIPTS / "proact_calibration.py"
result_dir = OUT / "task4_proact_calibration"
gwf.target_from_template("task4_proact_calibration", _job(
    f'python "{script}" --output-dir "{result_dir}" && touch "{result_dir / "_SUCCESS"}"',
    result_dir, ["calibration_rows.csv", "d_calibration_histograms.csv"],
    _sources(script, PROJECT_ROOT / "configs" / "proact_latent_ablation.yaml",
             PROJECT_ROOT / "data" / "proact_death.csv"), "gpu",
))

# Task 5: independent-latent control and shared two-trunk reference, d_z = 2.
script = SCRIPTS / "semi_synthetic_refit.py"
for variant in SETTINGS["task5_variants"]:
    for dataset in SETTINGS["task5_datasets"]:
        for seed in seeds:
            result_dir = OUT / "task5_independent_latent" / variant / dataset / f"seed_{seed}"
            gwf.target_from_template(f"task5_{variant}_{dataset}_{seed}", _job(
                f'python "{script}" --variant {variant} --dataset {dataset} --seed-index {seed} --output-dir "{result_dir}"',
                result_dir, ["results_raw.csv"], _sources(script, SEMI_CONFIG), "gpu",
            ))

# Task 6: covariate-varying dependence, DVFM and untuned HACSurv, 10 repeats.
script = SCRIPTS / "stratified_dependence.py"
result_dir = OUT / "task6_stratified_dependence"
gwf.target_from_template("task6_stratified_dependence", _job(
    f'python "{script}" --output-dir "{result_dir}" && touch "{result_dir / "_SUCCESS"}"',
    result_dir, ["tau_rows.csv", "recovery_rows.csv", "prediction_rows.csv"],
    _sources(script, PROJECT_ROOT / "configs" / "rebuttal" / "stratified_dependence.yaml",
             PROJECT_ROOT / "configs" / "synthetic.yaml"), "gpu_long",
))
