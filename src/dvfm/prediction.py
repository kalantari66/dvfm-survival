"""Prediction functions re-exported from the supplied reference implementation."""

from .reference_core import get_median_survival_time, predict_survival_curves

__all__ = ["predict_survival_curves", "get_median_survival_time"]
