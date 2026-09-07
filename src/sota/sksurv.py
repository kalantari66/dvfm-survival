"""Scikit-survival tree ensembles ported from survival-copula.

Imports are lazy so the core DVFM package can still be inspected without the
optional SOTA dependencies installed.
"""

from __future__ import annotations


def make_gbsa_model(config: dict):
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis

    return GradientBoostingSurvivalAnalysis(
        n_estimators=int(config.get("n_estimators", 100)),
        learning_rate=float(config.get("learning_rate", 0.1)),
        max_depth=int(config.get("max_depth", 3)),
        loss=str(config.get("loss", "coxph")),
        min_samples_split=int(config.get("min_samples_split", 2)),
        min_samples_leaf=int(config.get("min_samples_leaf", 1)),
        max_features=config.get("max_features", "sqrt"),
        subsample=float(config.get("subsample", 0.8)),
        random_state=int(config.get("random_state", config.get("seed", 0))),
    )


def make_rsf_model(config: dict):
    from sksurv.ensemble import RandomSurvivalForest

    return RandomSurvivalForest(
        n_estimators=int(config.get("n_estimators", 100)),
        max_depth=int(config.get("max_depth", 3)),
        min_samples_split=int(config.get("min_samples_split", 2)),
        min_samples_leaf=int(config.get("min_samples_leaf", 1)),
        max_features=config.get("max_features", "sqrt"),
        random_state=int(config.get("random_state", 0)),
        n_jobs=int(config.get("n_jobs", 1)),
    )


__all__ = ["make_gbsa_model", "make_rsf_model"]
