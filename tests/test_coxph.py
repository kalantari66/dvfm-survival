import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sksurv")
from sksurv.util import Surv

import sota.coxph as coxph_module
from sota.coxph import RobustCoxPHSurvivalAnalysis, _fit_cox_and_predict_survival


def _outcome(n):
    return Surv.from_arrays(
        event=np.arange(n) % 3 != 0,
        time=np.linspace(1.0, 20.0, n),
    )


def test_robust_coxph_cleans_and_drops_constant_columns():
    rng = np.random.default_rng(7)
    X = pd.DataFrame({
        "signal": rng.normal(size=60),
        "constant": np.ones(60),
        "dirty": rng.normal(size=60),
    })
    X.loc[0, "signal"] = np.inf
    X.loc[1, "dirty"] = np.nan

    model = RobustCoxPHSurvivalAnalysis().fit(X, _outcome(len(X)))

    assert model.keep_columns_ == ["signal", "dirty"]
    assert model.alpha_ is not None
    assert np.isfinite(model.predict(X.iloc[:5])).all()


def test_robust_coxph_retries_with_stronger_ridge(monkeypatch):
    attempted = []
    real_estimator = coxph_module.CoxPHSurvivalAnalysis

    class RetryEstimator:
        def __init__(self, alpha, **kwargs):
            attempted.append(alpha)
            if alpha < 0.1:
                raise ValueError("synthetic convergence failure")
            self.delegate = real_estimator(alpha=alpha, **kwargs)

        def fit(self, X, y):
            self.delegate.fit(X, y)
            return self

        def __getattr__(self, name):
            return getattr(self.delegate, name)

    monkeypatch.setattr(coxph_module, "CoxPHSurvivalAnalysis", RetryEstimator)
    X = np.column_stack([np.linspace(-1.0, 1.0, 60), np.arange(60) % 2])
    with pytest.warns(RuntimeWarning, match="stronger ridge alpha=0.1"):
        model = RobustCoxPHSurvivalAnalysis(alpha=1e-4).fit(X, _outcome(60))

    assert attempted == [1e-4, 1e-3, 1e-2, 1e-1]
    assert model.alpha_ == 0.1


def test_coxph_adapter_returns_finite_monotone_survival_curves():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(80, 3))
    event = rng.binomial(1, 0.7, size=80)
    time = rng.lognormal(mean=2.0 - 0.4 * X[:, 0], sigma=0.4)
    grid = np.linspace(0.0, np.quantile(time, 0.95), 30)

    median, survival = _fit_cox_and_predict_survival(
        X[:60], time[:60], event[:60], X[60:], grid
    )

    assert median.shape == (20,)
    assert survival.shape == (20, 30)
    assert np.isfinite(median).all()
    assert np.isfinite(survival).all()
    assert np.all((survival >= 0.0) & (survival <= 1.0))
    assert np.all(np.diff(survival, axis=1) <= 1e-12)
