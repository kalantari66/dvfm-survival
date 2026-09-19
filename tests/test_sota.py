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



def test_deepsurv_partial_likelihood_uses_the_event_indicator():
    """Censored rows must not contribute as events to the Cox partial likelihood.

    The risk scores arrive from the network as (N, 1) while the event mask is
    (N,). If they are multiplied before flattening, the product broadcasts to
    (N, N) and its sum factorizes as sum(event) * sum_i(r_i - logsumexp_i), so
    dividing by sum(event) cancels the indicator and every observation trains
    as an event. Two cohorts that differ only in *which* rows are censored then
    yield identical fits.
    """
    from sota.baselines import train_deepsurv

    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 3)).astype(np.float64)
    time = np.exp(0.5 * X[:, 0]) * rng.exponential(size=len(X)) + 0.1
    early = np.zeros(len(X), dtype=int)
    early[np.argsort(time)[:60]] = 1          # only the earliest half are events
    late = 1 - early                           # only the latest half are events

    fits = []
    for event in (early, late):
        torch.manual_seed(0)
        risk, _, _ = train_deepsurv(
            X, time, event, X, n_epochs=60, batch_size=None, lr=0.05,
            hidden_dims=[], dropout=0.0, weight_decay=0.0,
        )
        fits.append(np.asarray(risk, dtype=float))

    assert not np.allclose(fits[0], fits[1], atol=1e-6), (
        "DeepSurv fit is insensitive to which observations are censored; the "
        "event indicator is being cancelled out of the partial likelihood."
    )
