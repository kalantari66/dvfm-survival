import numpy as np
import pytest
import torch

from sota.adapters import fit_deephit, fit_sksurv_ensemble, fit_weibull_aft


@pytest.fixture
def survival_data():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 4)).astype(np.float32)
    latent_time = np.exp(0.3 * X[:, 0]) * rng.exponential(size=len(X))
    censor_time = rng.exponential(scale=2.0, size=len(X))
    return X, np.minimum(latent_time, censor_time), latent_time <= censor_time


@pytest.mark.parametrize("name", ["gbsa", "rsf"])
def test_sksurv_ensembles_return_valid_curves(survival_data, name):
    X, time, event = survival_data
    grid = np.linspace(0.0, np.quantile(time, 0.9), 15)
    median, curves = fit_sksurv_ensemble(
        name, X[:60], time[:60], event[:60], X[60:], grid,
        {"n_estimators": 2, "max_depth": 2, "random_state": 7, "n_jobs": 1},
    )
    assert median.shape == (20,)
    assert curves.shape == (20, 15)
    assert np.isfinite(curves).all()
    assert ((curves >= 0.0) & (curves <= 1.0)).all()
    assert (np.diff(curves, axis=1) <= 1e-10).all()


def test_weibull_aft_returns_valid_curves(survival_data):
    X, time, event = survival_data
    grid = np.linspace(0.0, np.quantile(time, 0.9), 15)
    median, curves = fit_weibull_aft(
        X[:60], time[:60], event[:60], X[60:], grid,
        {"penalizer": 0.0, "l1_ratio": 0.0},
    )
    assert median.shape == (20,)
    assert curves.shape == (20, 15)
    assert np.isfinite(curves).all()
    assert (np.diff(curves, axis=1) <= 1e-10).all()


def test_deephit_returns_valid_curves(survival_data):
    X, time, event = survival_data
    grid = np.linspace(0.0, np.quantile(time, 0.9), 15)
    median, curves = fit_deephit(
        X[:50], time[:50], event[:50], X[50:65], time[50:65], event[50:65],
        X[65:], grid,
        {
            "epochs": 2, "batch_size": 16, "learning_rate": 0.001,
            "time_bins": 10, "num_nodes_shared": [8], "batch_norm": False,
            "dropout": 0.0, "alpha": 0.2, "sigma": 0.1,
            "early_stop": False, "verbose": False,
        },
        torch.device("cpu"),
    )
    assert median.shape == (15,)
    assert curves.shape == (15, 15)
    assert np.isfinite(curves).all()
    assert (np.diff(curves, axis=1) <= 1e-10).all()
