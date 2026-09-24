"""Sklearn-style Weibull AFT wrapper ported from survival-copula."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import WeibullAFTFitter
from utility.metrics import concordance_index


class WeibullAFTWrapper:
    """Wrap ``lifelines.WeibullAFTFitter`` in the common baseline interface."""

    def __init__(self, penalizer: float = 0.0, l1_ratio: float = 0.0):
        self.penalizer = float(penalizer)
        self.l1_ratio = float(l1_ratio)
        self.model = WeibullAFTFitter(
            penalizer=self.penalizer, l1_ratio=self.l1_ratio
        )
        self.feature_names_: list[str] | None = None
        self.is_fitted_ = False

    @staticmethod
    def _y_to_arrays(y):
        if getattr(getattr(y, "dtype", None), "names", None):
            names = y.dtype.names
            time_name = next(
                (name for name in names if name.lower() in {"time", "duration", "t"}),
                None,
            )
            event_name = next(
                (name for name in names if name.lower() in {"event", "status", "e", "delta"}),
                None,
            )
            if time_name is None or event_name is None:
                raise ValueError("Structured y must contain time and event fields")
            return (
                np.asarray(y[time_name], dtype=float),
                np.asarray(y[event_name], dtype=bool).astype(int),
            )
        array = np.asarray(y)
        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError("y must be structured or an [n, 2] (time, event) array")
        return array[:, 0].astype(float), array[:, 1].astype(bool).astype(int)

    def _features(self, X) -> pd.DataFrame:
        array = np.asarray(X, dtype=float)
        if self.feature_names_ is None:
            self.feature_names_ = [f"x{i}" for i in range(array.shape[1])]
        return pd.DataFrame(array, columns=self.feature_names_)

    def fit(self, X, y):
        time, event = self._y_to_arrays(y)
        frame = self._features(X)
        frame["time"] = time
        frame["event"] = event
        self.model.fit(frame, duration_col="time", event_col="event")
        self.is_fitted_ = True
        return self

    def predict_survival_function(self, X, times=None):
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before prediction")
        frame = self._features(X)
        if times is None:
            durations = self.model.durations.to_numpy()
            lower, upper = np.percentile(durations, [1, 99])
            times = np.linspace(max(lower, 1e-6), upper, 100)
        timeline = np.asarray(times, dtype=float)
        curves = self.model.predict_survival_function(frame, times=timeline)
        return [curves.iloc[:, i].to_numpy() for i in range(curves.shape[1])], timeline

    def predict_median(self, X):
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before prediction")
        return self.model.predict_median(self._features(X)).to_numpy(dtype=float)

    def score(self, X, y):
        time, event = self._y_to_arrays(y)
        return concordance_index(time, self.predict_median(X), event)


def make_weibull_aft_model(config: dict) -> WeibullAFTWrapper:
    return WeibullAFTWrapper(
        penalizer=float(config.get("penalizer", 0.0)),
        l1_ratio=float(config.get("l1_ratio", 0.0)),
    )


__all__ = ["WeibullAFTWrapper", "make_weibull_aft_model"]
