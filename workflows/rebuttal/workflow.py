"""GWF targets for the rebuttal experiments. Outputs go under rebuttal/.

From the repository root:

    gwf -f workflows/rebuttal/workflow.py status
    gwf -f workflows/rebuttal/workflow.py run 'task1_*'      # one task at a time
    gwf -f workflows/rebuttal/workflow.py run

Target names start with the task number. A job removes its _SUCCESS marker
first and writes it only after its command and output checks pass, so GWF
reruns exactly what is missing or failed. Resources: configs/rebuttal/workflow.yaml.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from gwf import AnonymousTarget, Workflow

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT_ROOT / "scripts" / "rebuttal"
CONFIGS = PROJECT_ROOT / "configs"
OUT = PROJECT_ROOT / "rebuttal"
SETTINGS = yaml.safe_load((CONFIGS / "rebuttal" / "workflow.yaml").read_text(encoding="utf-8"))
SEMI_CONFIG = CONFIGS / "semi_synthetic.yaml"
SEMI_DATASETS = [str(item["name"]) for item in yaml.safe_load(SEMI_CONFIG.read_text(encoding="utf-8"))["data"]["datasets"]]
SOURCES = [
    *sorted(str(path) for package in ("dvfm", "experiments", "sota", "utility")
            for path in (PROJECT_ROOT / "src" / package).glob("*.py")),
    str(SCRIPTS / "rebuttal_common.py"),
]


def _job(script: str, args: str, result_dir: Path, outputs: list[str], inputs: list[Path], kind: str) -> AnonymousTarget:
    resources = SETTINGS["resources"][kind]
    cores = resources["cores"]
    checks = "\n    ".join(f'test -s "{result_dir / name}"' for name in outputs)
    spec = f"""
    set -euo pipefail
    cd "{PROJECT_ROOT}"
    mkdir -p "{result_dir}"
    rm -f "{result_dir / '_SUCCESS'}"
    echo "[GWF] $(date) host=$(hostname) {result_dir}"
    export PYTHONUNBUFFERED=1
    export OMP_NUM_THREADS="{cores}" MKL_NUM_THREADS="{cores}" OPENBLAS_NUM_THREADS="{cores}" NUMEXPR_NUM_THREADS="{cores}"
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate dvfm
    python "{SCRIPTS / script}" {args} --output-dir "{result_dir}"
    {checks}
    touch "{result_dir / '_SUCCESS'}"
    echo "[GWF] $(date) done"
    """
    options = {
        "cores": str(cores), "memory": str(resources["memory"]),
        "walltime": str(resources["walltime"]), "account": str(SETTINGS["account"]),
    }
    if kind == "gpu":
        # Same partition hand-off as workflows/semi_synthetic/workflow.py.
        options["gres"] = f"gpu:1 -p {resources['partition']}"
    return AnonymousTarget(
        inputs=[*SOURCES, str(SCRIPTS / script), *(str(path) for path in inputs)],
        outputs=[str(result_dir / name) for name in (*outputs, "_SUCCESS")],
        options=options, spec=spec,
    )


def _placement(group: str) -> tuple[str, str]:
    """(resource kind, extra script args) for a GPU-capable target group."""
    if group in SETTINGS.get("run_on_cpu", []):
        return "cpu", " --device cpu"
    return "gpu", ""


gwf = Workflow()
seeds = range(int(SETTINGS["seed_count"]))

# Task 1: prior vs aggregate posterior. Synthetic grid (d_z = 1), one job per cell.
kind, device = _placement("task1_synthetic")
for index in range(int(SETTINGS["synthetic_scenarios"])):
    gwf.target_from_template(f"task1_synthetic_{index:02d}", _job(
        "synthetic_cell.py", f"--scenario-index {index}{device}",
        OUT / "task1_prior_vs_aggregate" / "synthetic" / f"scenario_{index:02d}",
        ["results_raw.csv", "run_manifest.csv", "dvfm_diagnostics.csv"], [CONFIGS / "synthetic.yaml"], kind,
    ))

# Task 1: semi-synthetic DVFM d_z = 1 refits, also the DVFM side of Task 3 and
# the d_z = 1 reference of Task 5. One job per dataset x seed, 5 scenarios each.
kind, device = _placement("task1_semi")
for dataset in SEMI_DATASETS:
    for seed in seeds:
        gwf.target_from_template(f"task1_semi_{dataset}_{seed}", _job(
            "semi_synthetic_refit.py", f"--variant dvfm --dataset {dataset} --seed-index {seed}{device}",
            OUT / "task1_prior_vs_aggregate" / "semi_synthetic" / dataset / f"seed_{seed}",
            ["results_raw.csv"], [SEMI_CONFIG], kind,
        ))

# Task 2: timings on an equal footing. All three models on one GPU (GPU job) and
# all three on one CPU node (CPU job), same cells, seeds and thread count.
for setting, kind in (("gpu", "gpu"), ("cpu", "cpu")):
    gwf.target_from_template(f"task2_timings_{setting}", _job(
        "timings.py", f"--setting {setting} --datasets {' '.join(SETTINGS['task2_datasets'])} "
                      f"--threads {SETTINGS['resources'][kind]['cores']}",
        OUT / "task2_timings" / setting, ["timing_rows.csv", "environment.json"], [SEMI_CONFIG], kind,
    ))

# Task 3: CoxPH and Cox--Gamma recovery baselines on all 12 datasets (CPU).
for dataset in SEMI_DATASETS:
    for seed in seeds:
        gwf.target_from_template(f"task3_baselines_{dataset}_{seed}", _job(
            "recovery_baselines_all.py", f"--dataset {dataset} --seed-index {seed}",
            OUT / "task3_recovery_all_datasets" / "baselines" / dataset / f"seed_{seed}",
            ["recovery_rows.csv", "ibs_crosscheck.csv"], [SEMI_CONFIG], "cpu",
        ))

# Task 4: PRO-ACT D- and KM-calibration, 10 seeds x d_z in {0, 1}.
kind, device = _placement("task4")
gwf.target_from_template("task4_proact_calibration", _job(
    "proact_calibration.py", device.strip(), OUT / "task4_proact_calibration",
    ["calibration_rows.csv", "d_calibration_histograms.csv"],
    [CONFIGS / "proact_latent_ablation.yaml", PROJECT_ROOT / "data" / "proact_death.csv"], kind,
))

# Task 5: independent-latent control and shared two-trunk reference, d_z = 2.
kind, device = _placement("task5")
for variant in SETTINGS["task5_variants"]:
    for dataset in SETTINGS["task5_datasets"]:
        for seed in seeds:
            gwf.target_from_template(f"task5_{variant}_{dataset}_{seed}", _job(
                "semi_synthetic_refit.py", f"--variant {variant} --dataset {dataset} --seed-index {seed}{device}",
                OUT / "task5_independent_latent" / variant / dataset / f"seed_{seed}",
                ["results_raw.csv"], [SEMI_CONFIG], kind,
            ))

# Task 6: covariate-varying dependence (step and smooth loading), DVFM and
# untuned HACSurv, 10 repeats each.
TASK6_OUTPUTS = ["tau_rows.csv", "recovery_rows.csv", "prediction_rows.csv", "posterior_rows.csv"]
for profile in ("stratified", "smooth"):
    config = CONFIGS / "rebuttal" / f"{profile}_dependence.yaml"
    kind, device = _placement("task6")
    gwf.target_from_template(f"task6_{profile}_dependence", _job(
        "stratified_dependence.py", f'--config "{config}"{device}', OUT / f"task6_{profile}_dependence",
        TASK6_OUTPUTS, [config, CONFIGS / "synthetic.yaml"], kind,
    ))
    # Diagnostics of DVFM's tau, DVFM only; the paper variant is the job above.
    for variant in yaml.safe_load(config.read_text(encoding="utf-8"))["diagnostic_variants"]:
        kind, device = _placement("task6diag")
        gwf.target_from_template(f"task6diag_{profile}_{variant}", _job(
            "stratified_dependence.py", f'--config "{config}" --models dvfm --dvfm-variant {variant}{device}',
            OUT / "task6_diagnostics" / profile / variant, TASK6_OUTPUTS, [config, CONFIGS / "synthetic.yaml"], kind,
        ))
