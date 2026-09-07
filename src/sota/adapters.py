"""Adapters from SOTA estimators to the DVFM experiment prediction contract."""

from __future__ import annotations

import numpy as np

from .deephit import make_deephit_single, train_deephit_model
from .sksurv import make_gbsa_model, make_rsf_model
from .weibull_aft import make_weibull_aft_model


def _structured_outcome(time, event):
    outcome = np.empty(len(time), dtype=[("event", "?"), ("time", "<f8")])
    outcome["event"] = np.asarray(event, dtype=bool)
    outcome["time"] = np.asarray(time, dtype=float)
    return outcome


def _median_from_curves(curves, time_points, fallback):
    curves = np.asarray(curves, dtype=float)
    grid = np.asarray(time_points, dtype=float)
    medians = np.full(curves.shape[0], float(fallback), dtype=float)
    for index, curve in enumerate(curves):
        crossing = np.flatnonzero(curve <= 0.5)
        if crossing.size:
            medians[index] = grid[crossing[0]]
    return medians


def _step_functions_on_grid(functions, time_points):
    grid = np.asarray(time_points, dtype=float)
    curves = np.empty((len(functions), len(grid)), dtype=float)
    for row, function in enumerate(functions):
        # Interpolation gives defined behavior at zero and beyond the final
        # training event, where sksurv's StepFunction otherwise raises.
        curves[row] = np.interp(
            grid,
            np.asarray(function.x, dtype=float),
            np.asarray(function.y, dtype=float),
            left=1.0,
            right=float(function.y[-1]),
        )
    return np.minimum.accumulate(np.clip(curves, 0.0, 1.0), axis=1)


def fit_sksurv_ensemble(name, X_train, time_train, event_train, X_test,
                         time_points, config):
    model = make_gbsa_model(config) if name == "gbsa" else make_rsf_model(config)
    model.fit(np.asarray(X_train), _structured_outcome(time_train, event_train))
    functions = model.predict_survival_function(np.asarray(X_test))
    curves = _step_functions_on_grid(functions, time_points)
    medians = _median_from_curves(curves, time_points, np.max(time_train))
    return medians, curves


def fit_weibull_aft(X_train, time_train, event_train, X_test, time_points, config):
    model = make_weibull_aft_model(config)
    model.fit(X_train, _structured_outcome(time_train, event_train))
    curve_list, _ = model.predict_survival_function(X_test, times=time_points)
    curves = np.minimum.accumulate(
        np.clip(np.vstack(curve_list), 0.0, 1.0), axis=1
    )
    medians = np.asarray(model.predict_median(X_test), dtype=float)
    medians[~np.isfinite(medians)] = float(np.max(time_train))
    return medians, curves


def fit_deephit(X_train, time_train, event_train, X_validation,
                time_validation, event_validation, X_test, time_points,
                config, device):
    from pycox.models import DeepHitSingle

    time_bins = int(config.get("time_bins", 100))
    label_transform = DeepHitSingle.label_transform(time_bins)
    y_train = label_transform.fit_transform(
        np.asarray(time_train, dtype=float), np.asarray(event_train, dtype=int)
    )
    y_validation = label_transform.transform(
        np.asarray(time_validation, dtype=float),
        np.asarray(event_validation, dtype=int),
    )
    model = make_deephit_single(
        np.asarray(X_train).shape[1], time_bins, device, config,
        label_transform=label_transform,
    )
    model = train_deephit_model(
        model,
        np.asarray(X_train, dtype=np.float32),
        y_train,
        (np.asarray(X_validation, dtype=np.float32), y_validation),
        config,
    )
    frame = model.predict_surv_df(np.asarray(X_test, dtype=np.float32))
    source_time = frame.index.to_numpy(dtype=float)
    source_curves = frame.to_numpy(dtype=float).T
    grid = np.asarray(time_points, dtype=float)
    curves = np.vstack([
        np.interp(grid, source_time, curve, left=1.0, right=curve[-1])
        for curve in source_curves
    ])
    curves = np.minimum.accumulate(np.clip(curves, 0.0, 1.0), axis=1)
    medians = _median_from_curves(curves, grid, np.max(time_train))
    return medians, curves


__all__ = ["fit_deephit", "fit_sksurv_ensemble", "fit_weibull_aft"]
