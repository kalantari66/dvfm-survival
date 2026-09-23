from types import SimpleNamespace

import numpy as np

from experiments.recovery_baselines import (
    aligned_recovery_rows,
    martingale_residuals,
)


class _ConstantHazardModel:
    """CoxPH stand-in whose cumulative hazard is ``rate_i * t`` per subject."""

    def __init__(self, rates, horizon=10.0):
        self.rates = np.asarray(rates, dtype=float)
        self.horizon = float(horizon)

    def predict_cumulative_hazard_function(self, X):
        grid = np.linspace(0.0, self.horizon, 1001)
        return [
            SimpleNamespace(x=grid, y=rate * grid) for rate in self.rates
        ]


def test_martingale_residual_matches_delta_minus_cumulative_hazard():
    model = _ConstantHazardModel([0.5, 2.0, 1.0])
    residuals = martingale_residuals(
        model, np.zeros((3, 1)), [2.0, 1.0, 4.0], [1, 0, 1]
    )
    np.testing.assert_allclose(residuals, [1 - 1.0, 0 - 2.0, 1 - 4.0], atol=1e-9)


def test_martingale_residual_clamps_beyond_the_training_support():
    """An observed time past the last fitted point must not raise or diverge."""
    model = _ConstantHazardModel([1.0], horizon=5.0)
    residuals = martingale_residuals(model, np.zeros((1, 1)), [500.0], [0])
    assert residuals[0] == -5.0


def test_alignment_sign_is_estimated_on_validation_only():
    rng = np.random.default_rng(0)
    truth = rng.normal(size=400)
    statistic = -3.0 * truth + 1.0
    rows = aligned_recovery_rows(
        statistic, truth, statistic, truth, np.ones(400, dtype=int)
    )
    assert {row["subgroup"] for row in rows} == {
        "All", "Event observed", "Censored"
    }
    calibrated = next(
        row for row in rows
        if row["subgroup"] == "All"
        and row["representation"] == "Validation-calibrated"
    )
    assert calibrated["alignment_sign"] == -1.0
    assert calibrated["spearman"] > 0.999
    assert calibrated["r2"] > 0.999
    assert calibrated["rmse"] < 1e-8


def test_recovery_rows_split_subgroups_by_censoring_status():
    rng = np.random.default_rng(1)
    truth = rng.normal(size=200)
    event = (np.arange(200) % 2).astype(int)
    rows = aligned_recovery_rows(truth, truth, truth, truth, event)
    counts = {
        row["subgroup"]: row["n"] for row in rows
        if row["representation"] == "Validation-calibrated"
    }
    assert counts == {"All": 200, "Event observed": 100, "Censored": 100}
    assert all(
        row["split"] == "test" for row in rows
    )


def test_uninformative_statistic_reports_no_recovery():
    rng = np.random.default_rng(2)
    truth = rng.normal(size=500)
    statistic = rng.normal(size=500)
    rows = aligned_recovery_rows(
        statistic, truth, statistic, truth, np.ones(500, dtype=int)
    )
    calibrated = next(
        row for row in rows
        if row["subgroup"] == "All"
        and row["representation"] == "Validation-calibrated"
    )
    assert abs(calibrated["spearman"]) < 0.2
