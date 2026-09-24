"""Robust scikit-survival Cox proportional-hazards baseline."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

try:
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.util import Surv
except ImportError:  # pragma: no cover - exercised in minimal local environments
    CoxPHSurvivalAnalysis = None
    Surv = None


class RobustCoxPHSurvivalAnalysis:
    """CoxPH wrapper with split-safe cleaning and escalating ridge penalties."""

    def __init__(self, alpha=1e-4, ties="breslow", n_iter=100, tol=1e-9):
        self.alpha = float(alpha)
        self.ties = str(ties)
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.model_ = None
        self.alpha_ = None
        self.columns_ = None
        self.keep_columns_ = None
        self.median_ = None
        self.mean_ = None
        self.std_ = None

    def _prepare_fit_X(self, X):
        X = pd.DataFrame(X).copy()
        self.columns_ = list(X.columns)

        X = X.replace([np.inf, -np.inf], np.nan).astype(float)
        self.median_ = X.median(axis=0).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)
        X = X.fillna(self.median_)

        variance = X.var(axis=0)
        self.keep_columns_ = variance.index[
            np.asarray(variance > 1e-12)
        ].tolist()
        if not self.keep_columns_:
            raise ValueError("No non-constant features left for CoxPH.")
        X = X[self.keep_columns_]

        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0).replace(0.0, 1.0).fillna(1.0)
        return ((X - self.mean_) / self.std_).replace(
            [np.inf, -np.inf], 0.0
        ).fillna(0.0)

    def _prepare_predict_X(self, X):
        if self.model_ is None:
            raise RuntimeError("Robust CoxPH must be fitted before prediction.")
        X = pd.DataFrame(X).copy()
        if self.columns_ is not None and set(self.columns_).issubset(X.columns):
            X = X[self.columns_]

        X = X.replace([np.inf, -np.inf], np.nan).astype(float)
        X = X.fillna(self.median_)
        X = X[self.keep_columns_]
        X = (X - self.mean_) / self.std_
        return X.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    def fit(self, X, y):
        if CoxPHSurvivalAnalysis is None:
            raise ImportError(
                "Robust CoxPH requires scikit-survival. Install the project "
                "baseline dependencies before fitting CoxPH."
            )
        X_fit = self._prepare_fit_X(X)
        alphas = []
        for alpha in (self.alpha, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0):
            if float(alpha) not in alphas:
                alphas.append(float(alpha))

        last_error = None
        for alpha in alphas:
            try:
                model = CoxPHSurvivalAnalysis(
                    alpha=alpha,
                    ties=self.ties,
                    n_iter=self.n_iter,
                    tol=self.tol,
                )
                model.fit(X_fit, y)
                self.model_ = model
                self.alpha_ = alpha
                if alpha != self.alpha:
                    warnings.warn(
                        f"CoxPH fit required stronger ridge alpha={alpha} "
                        f"(requested alpha={self.alpha}).",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                return self
            except Exception as exc:
                last_error = exc

        raise RuntimeError(
            "Robust CoxPH failed for all ridge penalties. "
            f"Last error: {last_error}"
        ) from last_error

    def predict(self, X):
        return self.model_.predict(self._prepare_predict_X(X))

    def score(self, X, y):
        return self.model_.score(self._prepare_predict_X(X), y)

    def predict_survival_function(self, X, *args, **kwargs):
        return self.model_.predict_survival_function(
            self._prepare_predict_X(X), *args, **kwargs
        )

    def predict_cumulative_hazard_function(self, X, *args, **kwargs):
        return self.model_.predict_cumulative_hazard_function(
            self._prepare_predict_X(X), *args, **kwargs
        )


def make_cox_model(config):
    """Build the robust CoxPH estimator from canonical model settings."""
    return RobustCoxPHSurvivalAnalysis(
        alpha=float(config.get("alpha", 1e-4)),
        ties=config.get("ties", "breslow"),
        n_iter=int(config["n_iter"]),
        tol=float(config["tol"]),
    )


def _fit_cox_and_predict_survival(
    X_train, t_train, e_train, X_test, time_points, config=None
):
    """Fit robust CoxPH and adapt its predictions to the common contract."""
    settings = {
        "alpha": 1e-4,
        "ties": "breslow",
        "n_iter": 100,
        "tol": 1e-9,
        **(config or {}),
    }
    if Surv is None:
        raise ImportError(
            "Robust CoxPH requires scikit-survival. Install the project "
            "baseline dependencies before fitting CoxPH."
        )
    model = make_cox_model(settings)
    outcome = Surv.from_arrays(
        event=np.asarray(e_train, dtype=bool),
        time=np.asarray(t_train, dtype=float),
    )
    model.fit(X_train, outcome)
    functions = model.predict_survival_function(X_test)
    grid = np.asarray(time_points, dtype=float)
    survival = np.vstack([
        np.interp(grid, function.x, function.y, left=1.0, right=function.y[-1])
        for function in functions
    ])

    max_train_time = float(np.max(t_train))
    medians = np.asarray([
        function.x[np.flatnonzero(function.y <= 0.5)[0]]
        if np.any(function.y <= 0.5) else max_train_time
        for function in functions
    ], dtype=float)
    return medians, survival


def train_coxph(X_train, time_train, event_train, X_test, config=None):
    """Fit robust CoxPH and return predicted median survival times."""
    time_points = np.unique(np.asarray(time_train, dtype=float))
    medians, _ = _fit_cox_and_predict_survival(
        X_train, time_train, event_train, X_test, time_points, config=config
    )
    return medians


__all__ = [
    "RobustCoxPHSurvivalAnalysis",
    "make_cox_model",
    "_fit_cox_and_predict_survival",
    "train_coxph",
]
