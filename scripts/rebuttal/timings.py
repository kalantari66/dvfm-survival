"""Task 2 (and Task 1d): wall-clock training and prediction times.

DVFM (fp32), HACSurv-2D (fp64) and the Cox--Gamma frailty model (fp64) are
fitted with each dataset's tuned settings on the benchmark's cohorts and
splits (Clayton copula, tau = 0.5) and timed on whatever node runs the script.

Two settings are timed:

* ``pipeline``: each model on the device the benchmark used (DVFM and HACSurv
  on the GPU, Cox--Gamma on the CPU);
* ``cpu``: all three on the same CPU with the same thread count, the
  like-for-like comparison.

Prediction is timed on 1,000 test subjects (resampled with replacement when
the test split is smaller) on the shared 100-point grid. For DVFM, building
the aggregate posterior (encoding the training set) is timed separately from
prediction given that cache, and per-patient prediction is timed with and
without the cache (Task 1d). GPU timings synchronize before reading the clock.
"""

from __future__ import annotations

import argparse
import time as timer
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dvfm.prediction import aggregate_posterior_mixture, predict_survival_curves, predict_survival_from_mixture
from experiments.config import load_config
from experiments.runner import (
    _models_for_dataset, evaluation_time_grid, fit_dvfm, scale_observed_time, training_time_scale,
)
from sota.bayesian_cox_gamma_frailty import fit_bayesian_cox_gamma_frailty
from sota.hacsurv import fit_hacsurv_2d
from utility.runtime import seed_everything

from rebuttal_common import PROJECT_ROOT, REBUTTAL_ROOT, environment_record, semisynthetic_cells, write_json

PREDICTION_SUBJECTS = 1000


class Clock:
    def __init__(self, device):
        self.cuda = torch.device(device).type == "cuda"

    def __enter__(self):
        if self.cuda:
            torch.cuda.synchronize()
        self.started = timer.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.cuda:
            torch.cuda.synchronize()
        self.seconds = timer.perf_counter() - self.started


def hacsurv_predict(model, X, time_points, dtype, device):
    """The prediction loop of ``fit_hacsurv_2d``, on new subjects."""
    x = torch.as_tensor(X, dtype=dtype, device=device)
    curves = np.empty((len(x), len(time_points)))
    with torch.no_grad():
        for index, point in enumerate(np.asarray(time_points, dtype=float)):
            horizon = torch.full((len(x),), float(point), dtype=dtype, device=device)
            curves[:, index] = model.event_survival(x, horizon).cpu().numpy()
    return np.minimum.accumulate(np.clip(curves, 0.0, 1.0), axis=1)


def time_cell(cell, model_cfg, setting, devices, per_patient):
    train, validation, test = cell["train"], cell["validation"], cell["test"]
    time_points = evaluation_time_grid(train, cell["evaluation"])
    scale = training_time_scale(train)
    model_train, model_validation, model_test = (scale_observed_time(p, scale) for p in (train, validation, test))
    grid = time_points / scale
    rng = np.random.default_rng(0)
    X_predict = model_test.X[rng.choice(len(model_test.X), PREDICTION_SUBJECTS, replace=len(model_test.X) < PREDICTION_SUBJECTS)]
    base = {"dataset": cell["dataset"], "scenario": cell["scenario"], "repeat": cell["repeat"],
            "setting": setting, "n_cohort": len(train.time) + len(validation.time) + len(test.time),
            "n_train": len(train.time)}
    rows = []

    device = devices["dvfm"]
    c = model_cfg["dvfm"]
    cell["reseed"]()
    with Clock(device) as fit:
        model, artifacts, train_loader, _ = fit_dvfm(model_train, model_validation, c, device)
    with Clock(device) as build:
        mixture = aggregate_posterior_mixture(model, train_loader, device)
    with Clock(device) as cached:
        predict_survival_from_mixture(model, X_predict, grid, *mixture, int(c["mc_samples"]), device)
    with Clock(device) as uncached:
        predict_survival_curves(model, X_predict, grid, train_loader, int(c["mc_samples"]), device)
    row = {**base, "model": "DVFM", "device": str(device), "precision": "fp32",
           "fit_seconds": fit.seconds, "epochs_run": len(artifacts["history"]),
           "q_agg_build_seconds": build.seconds,
           "predict_seconds_per_1000_cached": cached.seconds,
           "predict_seconds_per_1000_uncached": uncached.seconds}
    if per_patient:
        single = []
        for index in range(per_patient):
            with Clock(device) as one:
                predict_survival_from_mixture(model, X_predict[index:index + 1], grid, *mixture, int(c["mc_samples"]), device)
            single.append(one.seconds)
        single_uncached = []
        for index in range(min(per_patient, 20)):
            with Clock(device) as one:
                predict_survival_curves(model, X_predict[index:index + 1], grid, train_loader, int(c["mc_samples"]), device)
            single_uncached.append(one.seconds)
        row.update(per_patient_seconds_cached_median=float(np.median(single)),
                   per_patient_seconds_uncached_median=float(np.median(single_uncached)),
                   per_patient_calls_cached=len(single), per_patient_calls_uncached=len(single_uncached))
    rows.append(row)

    device = devices["hacsurv_2d"]
    settings = model_cfg["hacsurv_2d"]
    dtype = torch.float64 if str(settings.get("dtype", "float64")) == "float64" else torch.float32
    cell["reseed"]()
    with Clock(device) as fit:
        _, info, history, hacsurv = fit_hacsurv_2d(
            model_train.X, model_train.time, model_train.event,
            model_validation.X, model_validation.time, model_validation.event,
            model_test.X[:2], grid, device=device, seed=int(cell["seeds"]["model"]), **settings,
        )
    with Clock(device) as predict:
        hacsurv_predict(hacsurv, X_predict, grid, dtype, device)
    rows.append({**base, "model": "HACSurv", "device": str(device),
                 "precision": "fp64" if dtype == torch.float64 else "fp32",
                 "fit_seconds": fit.seconds, "epochs_run": len(history),
                 "predict_seconds_per_1000_cached": predict.seconds})

    device = devices["bayesian_cox_gamma_frailty"]
    settings = model_cfg["bayesian_cox_gamma_frailty"]
    cell["reseed"]()
    with Clock(device) as fit:
        _, _, history, gamma = fit_bayesian_cox_gamma_frailty(
            model_train.X, model_train.time, model_train.event,
            model_validation.X, model_validation.time, model_validation.event,
            model_test.X[:2], grid, settings, device=device,
        )
    with Clock(device) as predict:
        gamma.predict_marginal_survival(X_predict, grid)
    rows.append({**base, "model": "CG Frailty", "device": str(device),
                 "precision": "fp64" if str(settings.get("dtype", "float64")) == "float64" else "fp32",
                 "fit_seconds": fit.seconds, "epochs_run": len(history),
                 "predict_seconds_per_1000_cached": predict.seconds})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "semi_synthetic.yaml")
    parser.add_argument("--datasets", nargs="+", default=["gbsg", "support", "seer_liver"])
    parser.add_argument("--copula", default="clayton")
    parser.add_argument("--kendall-tau", type=float, default=0.5)
    parser.add_argument("--pipeline-seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--cpu-seeds", type=int, nargs="*", default=[0])
    parser.add_argument("--per-patient", type=int, default=100,
                        help="Single-subject DVFM predictions to time (pipeline setting, largest dataset)")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=REBUTTAL_ROOT / "task2_timings")
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.set_num_threads(args.threads)
    gpu = "cuda" if torch.cuda.is_available() else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "environment.json", {
        "pipeline_gpu": environment_record(gpu) if gpu else None, "cpu": environment_record("cpu"),
        "threads": args.threads, "note": "fp32 for DVFM; fp64 for HACSurv and CG as configured",
    })
    settings = {"pipeline": (args.pipeline_seeds, {"dvfm": gpu, "hacsurv_2d": gpu, "bayesian_cox_gamma_frailty": "cpu"}),
                "cpu": (args.cpu_seeds, {"dvfm": "cpu", "hacsurv_2d": "cpu", "bayesian_cox_gamma_frailty": "cpu"})}
    if gpu is None:
        print("No GPU visible: only the like-for-like CPU setting is timed.")
        settings.pop("pipeline")
    largest = max(args.datasets, key=lambda name: ["gbsg", "metabric", "support", "seer_liver"].index(name)
                  if name in ["gbsg", "metabric", "support", "seer_liver"] else -1)
    rows = []
    for setting, (seed_indices, devices) in settings.items():
        for name in args.datasets:
            model_cfg = _models_for_dataset(cfg, name)
            for seed_index in seed_indices:
                for cell in semisynthetic_cells(cfg, name, seed_index, scenarios=[(args.copula, args.kendall_tau)]):
                    model_seed = int(cell["seeds"]["model"])
                    cell["reseed"] = lambda seed=model_seed: seed_everything(seed)
                    cell["evaluation"] = cfg["evaluation"]
                    per_patient = args.per_patient if (setting == "pipeline" and name == largest and seed_index == seed_indices[0]) else 0
                    new_rows = time_cell(cell, model_cfg, setting, devices, per_patient)
                    rows.extend(new_rows)
                    for row in new_rows:
                        print(f"{setting:>8} {name:>11} seed {seed_index} {row['model']:<10} {row['device']:<5} "
                              f"fit {row['fit_seconds']:8.1f}s  predict/1000 {row['predict_seconds_per_1000_cached']:.3f}s", flush=True)
                    pd.DataFrame(rows).to_csv(args.output_dir / "timing_rows.csv", index=False)
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
