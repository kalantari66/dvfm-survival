"""Task 2 (and Task 1d): wall-clock training and prediction times.

DVFM (fp32), HACSurv-2D (fp64) and the Cox--Gamma frailty model (fp64) are
fitted with each dataset's tuned settings on the benchmark's cohorts and
splits (Clayton copula, tau = 0.5) and timed on whatever node runs the script.

Every comparison is on an equal footing: one setting per call (``--setting``),
each its own job, and within a job all three models run on the same device of
the same node with the same thread count, on the same cells and seeds:

* ``gpu``: all three on one GPU;
* ``cpu``: all three on the CPU.

Each model keeps its configured precision (recorded on every row), and its
configured training budget: DVFM trains a fixed number of epochs while HACSurv
and Cox--Gamma stop early, so seconds per epoch are reported next to the total.
Before anything is timed on a GPU, an untimed warm-up initializes the CUDA
context and the fp32 and fp64 matrix-multiply kernels, so that start-up cost
does not land on whichever model happens to be timed first.

Prediction is timed on 1,000 test subjects (resampled with replacement when
the test split is smaller) on the shared 100-point grid. For DVFM, building
the aggregate posterior (encoding the training split) is timed separately from
prediction given that cache, both on the actual training split and on exactly
10,000 encoded subjects, and per-patient prediction is timed with and without
the cache (Task 1d). GPU timings synchronize before reading the clock.
"""

from __future__ import annotations

import argparse
import time as timer
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dvfm.prediction import aggregate_posterior_mixture, predict_survival_curves, predict_survival_from_mixture
from experiments.config import load_config
from experiments.runner import (
    _models_for_dataset, evaluation_time_grid, fit_dvfm, scale_observed_time, training_time_scale,
)
from sota.bayesian_cox_gamma_frailty import fit_bayesian_cox_gamma_frailty
from sota.hacsurv import fit_hacsurv_2d
from utility.data import SurvivalDataset
from utility.runtime import seed_everything

from rebuttal_common import PROJECT_ROOT, environment_record, semisynthetic_cells, write_json

PREDICTION_SUBJECTS = 1000
Q_AGG_SUBJECTS = 10_000


def warm_up(device) -> None:
    """Pay one-off CUDA start-up costs before any model is timed."""
    if torch.device(device).type != "cuda":
        return
    for dtype in (torch.float32, torch.float64):
        a = torch.randn(512, 512, device=device, dtype=dtype, requires_grad=True)
        (a @ a).sum().backward()
        torch.nn.functional.softplus(a).exp().log1p().sum().item()
    torch.cuda.synchronize()


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


def time_cell(cell, model_cfg, evaluation, setting, devices, per_patient):
    train, validation, test = cell["train"], cell["validation"], cell["test"]
    time_points = evaluation_time_grid(train, evaluation)
    model_seed = int(cell["seeds"]["model"])
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
    seed_everything(model_seed)
    with Clock(device) as fit:
        model, artifacts, train_loader, _ = fit_dvfm(model_train, model_validation, c, device)
    with Clock(device) as build:
        mixture = aggregate_posterior_mixture(model, train_loader, device)
    with Clock(device) as cached:
        predict_survival_from_mixture(model, X_predict, grid, *mixture, int(c["mc_samples"]), device)
    with Clock(device) as uncached:
        predict_survival_curves(model, X_predict, grid, train_loader, int(c["mc_samples"]), device)
    # Building q_agg at exactly N = 10,000 encoded subjects (training rows
    # resampled with replacement), same batch size as the training loader.
    big = rng.choice(len(model_train.X), Q_AGG_SUBJECTS, replace=len(model_train.X) < Q_AGG_SUBJECTS)
    big_loader = DataLoader(
        SurvivalDataset(model_train.X[big], model_train.time[big], model_train.event[big]),
        batch_size=int(c["batch_size"]), shuffle=False,
    )
    with Clock(device) as build_big:
        mixture_big = aggregate_posterior_mixture(model, big_loader, device)
    with Clock(device) as cached_big:
        predict_survival_from_mixture(model, X_predict, grid, *mixture_big, int(c["mc_samples"]), device)
    row = {**base, "model": "DVFM", "device": str(device), "precision": "fp32",
           "fit_seconds": fit.seconds, "epochs_run": len(artifacts["history"]),
           "q_agg_build_seconds": build.seconds, "q_agg_build_subjects": len(model_train.X),
           "q_agg_build_seconds_n10000": build_big.seconds,
           "predict_seconds_per_1000_cached": cached.seconds,
           "predict_seconds_per_1000_cached_n10000": cached_big.seconds,
           "predict_seconds_per_1000_uncached": uncached.seconds,
           "encoder_batch_size": int(c["batch_size"])}
    if per_patient:
        single = []
        for index in range(per_patient):
            with Clock(device) as one:
                predict_survival_from_mixture(model, X_predict[index:index + 1], grid, *mixture, int(c["mc_samples"]), device)
            single.append(one.seconds)
        single_uncached = []
        for index in range(max(1, per_patient // 5)):
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
    seed_everything(model_seed)
    with Clock(device) as fit:
        _, info, history, hacsurv = fit_hacsurv_2d(
            model_train.X, model_train.time, model_train.event,
            model_validation.X, model_validation.time, model_validation.event,
            model_test.X[:2], grid, device=device, seed=model_seed, **settings,
        )
    with Clock(device) as predict:
        hacsurv_predict(hacsurv, X_predict, grid, dtype, device)
    rows.append({**base, "model": "HACSurv", "device": str(device),
                 "precision": "fp64" if dtype == torch.float64 else "fp32",
                 "fit_seconds": fit.seconds, "epochs_run": len(history),
                 "predict_seconds_per_1000_cached": predict.seconds})

    device = devices["bayesian_cox_gamma_frailty"]
    settings = model_cfg["bayesian_cox_gamma_frailty"]
    seed_everything(model_seed)
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
    parser.add_argument("--setting", choices=("gpu", "cpu"), required=True,
                        help="Device for all three models")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "semi_synthetic.yaml")
    parser.add_argument("--datasets", nargs="+", default=["gbsg", "support", "seer_liver"])
    parser.add_argument("--copula", default="clayton")
    parser.add_argument("--kendall-tau", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Seed indices")
    parser.add_argument("--per-patient", type=int, default=50,
                        help="Single-subject DVFM predictions timed per cell with the cache (a fifth without)")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.setting == "gpu" and not torch.cuda.is_available():
        raise SystemExit("The gpu setting needs a visible GPU")
    device = "cuda" if args.setting == "gpu" else "cpu"
    devices = {"dvfm": device, "hacsurv_2d": device, "bayesian_cox_gamma_frailty": device}
    seeds = args.seeds
    cfg = load_config(args.config)
    torch.set_num_threads(args.threads)
    warm_up(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "environment.json", {
        **environment_record(device), "setting": args.setting, "devices": devices, "threads": args.threads,
        "precision": {"DVFM": "fp32", "HACSurv": "fp64", "CG Frailty": "fp64"},
    })
    rows = []
    for name in args.datasets:
        model_cfg = _models_for_dataset(cfg, name)
        for seed_index in seeds:
            for cell in semisynthetic_cells(cfg, name, seed_index, scenarios=[(args.copula, args.kendall_tau)]):
                new_rows = time_cell(cell, model_cfg, cfg["evaluation"], args.setting, devices, args.per_patient)
                for row in new_rows:
                    row["fit_seconds_per_epoch"] = row["fit_seconds"] / max(row["epochs_run"], 1)
                rows.extend(new_rows)
                for row in new_rows:
                    print(f"{name:>11} seed {seed_index} {row['model']:<10} {row['device']:<5} "
                          f"fit {row['fit_seconds']:8.1f}s  predict/1000 {row['predict_seconds_per_1000_cached']:.3f}s",
                          flush=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "timing_rows.csv", index=False)
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
