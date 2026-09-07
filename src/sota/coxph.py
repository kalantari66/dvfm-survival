"""Cox proportional-hazards baselines."""

import pandas as pd
from lifelines import CoxPHFitter

from .baselines import _fit_cox_and_predict_survival


def train_coxph(X_train, time_train, event_train, X_test):
    """Fit CoxPH and return predicted median survival times."""
    frame = pd.DataFrame(X_train, columns=[f"X{i}" for i in range(X_train.shape[1])])
    frame["time"] = time_train
    frame["event"] = event_train
    model = CoxPHFitter()
    model.fit(frame, duration_col="time", event_col="event")
    test = pd.DataFrame(X_test, columns=[f"X{i}" for i in range(X_test.shape[1])])
    return model.predict_median(test).to_numpy()

__all__ = ["_fit_cox_and_predict_survival", "train_coxph"]
