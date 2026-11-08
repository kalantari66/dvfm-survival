"""Helpers shared by the rebuttal scripts. Nothing here changes a paper study."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REBUTTAL_ROOT = PROJECT_ROOT / "rebuttal"

DATASET_LABEL = {
    "whas": "WHAS", "metabric": "METABRIC", "churn": "Churn", "gbsg": "GBSG",
    "nacd": "NACD", "flchain": "FLChain", "support": "SUPPORT", "employee": "Employee",
    "mimic_all": "MIMIC-IV", "mimic_iv": "MIMIC-IV", "seer_brain": "SEER (brain)",
    "seer_liver": "SEER (liver)", "seer_stomach": "SEER (stomach)",
}


def _cpu_name() -> str:
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def environment_record(device) -> dict:
    """Hardware and software actually used, for every SUMMARY.md."""
    import SurvivalEVAL

    device = torch.device(device)
    record = {
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_cpus_on_node": os.environ.get("SLURM_CPUS_ON_NODE"),
        "cpu": _cpu_name(),
        "logical_cpus": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "numpy": np.__version__,
        "survivaleval": getattr(SurvivalEVAL, "__version__", "unknown"),
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
    }
    return record


def _git(*args) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=20,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def select_dataset(cfg: dict, name: str) -> dict:
    from experiments.config import semisynthetic_datasets

    matches = [item for item in semisynthetic_datasets(cfg["data"]) if str(item["name"]).lower() == name.lower()]
    if len(matches) != 1:
        raise ValueError(f"Expected one configured dataset named {name!r}")
    return matches[0]


def semisynthetic_cells(cfg: dict, dataset_name: str, seed_index: int, scenarios=None):
    """Yield the benchmark's cohorts and splits for one dataset and seed.

    Mirrors the semi-synthetic loop of ``experiments.runner.run`` step for step
    (DGP fit, generation, split, preprocessing, oracle joint evaluation), and
    calls ``seed_everything(model_seed)`` at the same point, so a model fitted
    immediately afterwards starts from the benchmark's RNG state. ``scenarios``
    optionally restricts to (copula, tau) pairs.
    """
    from experiments.config import expand_seed_streams
    from experiments.runner import _preprocess_three, _subset
    from utility.metrics import JointSurvivalEvaluation
    from utility.runtime import seed_everything
    from utility.semisynthetic import (
        fit_semisynthetic_dgp, generate_semisynthetic, semi_synthetic_joint_survival,
    )
    from utility.splitting import time_event_stratified_split_indices
    from itertools import product

    data_cfg, evaluation = cfg["data"], cfg["evaluation"]
    dataset = select_dataset(cfg, dataset_name)
    dgp = fit_semisynthetic_dgp(dataset, cox_penalizer=float(data_cfg.get("cox_penalizer", 0.01)))
    seeds = expand_seed_streams(cfg["seeds"])[seed_index]
    tau_values = data_cfg["kendall_tau"] if isinstance(data_cfg["kendall_tau"], list) else [data_cfg["kendall_tau"]]
    copulas = data_cfg.get("copulas", [data_cfg.get("copula", "clayton")])
    rates = data_cfg["censoring_rates"]
    target_rates = [1.0 - float(dgp.source_event_rate)] if isinstance(rates, str) and rates.lower() == "original" else rates
    conditions = (
        (copula, tau, rate)
        for tau, rate in product(tau_values, target_rates)
        for copula in (["independence"] if float(tau) == 0.0 else copulas)
    )
    for copula, tau, rate in conditions:
        if scenarios is not None and (str(copula), float(tau)) not in {(str(c), float(t)) for c, t in scenarios}:
            continue
        generated = generate_semisynthetic(
            dgp, kendall_tau=float(tau), censoring_rate=float(rate),
            sampling_seed=int(seeds["sampling"]), copula=str(copula),
        )
        train_idx, validation_idx, test_idx = time_event_stratified_split_indices(
            generated.data, cfg["split"], int(seeds["split"])
        )
        train, validation, test = _preprocess_three(
            _subset(generated.data, train_idx), _subset(generated.data, validation_idx),
            _subset(generated.data, test_idx),
            {**cfg["preprocessing"], "numeric_features": dataset["numeric_features"]},
        )
        seed_everything(int(seeds["model"]))
        scenario = f"{dataset['name']}_{str(copula).lower()}_tau_{float(tau):g}_censor_{float(rate):g}"
        joint_evaluation = None
        if evaluation.get("compute_oracle_joint_survival_ise", False):
            count = min(int(evaluation["joint_n_subjects"]), len(test.X))
            chosen = np.random.default_rng(int(seeds["split"]) + 70_000).choice(len(test.X), size=count, replace=False)
            event_grid = np.linspace(0.0, np.quantile(train.true_event_time, evaluation["joint_grid_max_quantile"]), int(evaluation["joint_n_time_points"]))
            censor_grid = np.linspace(0.0, np.quantile(train.true_censor_time, evaluation["joint_grid_max_quantile"]), int(evaluation["joint_n_time_points"]))
            joint_evaluation = JointSurvivalEvaluation(
                test.X[chosen], event_grid, censor_grid,
                semi_synthetic_joint_survival(dgp, generated, test.X[chosen], event_grid, censor_grid),
            )
        yield {
            "dataset": dataset["name"], "scenario": scenario, "copula": str(copula).lower(),
            "kendall_tau": float(tau), "repeat": int(seed_index), "seeds": seeds,
            "train": train, "validation": validation, "test": test,
            "train_idx": train_idx, "validation_idx": validation_idx, "test_idx": test_idx,
            "joint_evaluation": joint_evaluation, "generated": generated, "dgp": dgp,
        }
