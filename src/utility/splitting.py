"""Reusable stratified survival-data splitting and preprocessing."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from .data import SurvivalData


def subset_survival_data(data: SurvivalData, indices) -> SurvivalData:
    indices = np.asarray(indices)
    return SurvivalData(
        X=data.X[indices].copy(),
        time=data.time[indices].copy(),
        event=data.event[indices].copy(),
        feature_names=list(data.feature_names),
        true_event_time=None if data.true_event_time is None else data.true_event_time[indices].copy(),
        true_censor_time=None if data.true_censor_time is None else data.true_censor_time[indices].copy(),
        true_z=None if data.true_z is None else data.true_z[indices].copy(),
    )


def three_way_split_indices(data: SurvivalData, split_cfg: dict, seed: int):
    indices = np.arange(len(data.time))
    test_fraction = float(split_cfg["test_fraction"])
    validation_fraction = float(split_cfg["validation_fraction"])
    train_validation, test = train_test_split(
        indices, test_size=test_fraction, random_state=seed, stratify=data.event
    )
    relative_validation = validation_fraction / (1.0 - test_fraction)
    train, validation = train_test_split(
        train_validation,
        test_size=relative_validation,
        random_state=seed + 1,
        stratify=data.event[train_validation],
    )
    return train, validation, test


def time_event_stratified_split_indices(data: SurvivalData, split_cfg: dict, seed: int):
    """70/10/20-style split stratified jointly by time bands and event status.

    The referenced semi-synthetic implementation uses 20 time bands plus the
    event indicator. Quantile bands retain that intent while avoiding nearly
    empty tail bands on heavily right-skewed survival times.
    """
    indices = np.arange(len(data.time))
    n_bins = min(int(split_cfg.get("time_bins", 20)), max(2, len(indices) // 20))
    event = np.asarray(data.event, dtype=int)
    time_band = np.zeros(len(indices), dtype=int)
    # Form bands within event status. Thus every stratum has support and the
    # observed-time distribution is preserved conditionally on event status.
    for status in np.unique(event):
        mask = event == status
        ranked = pd.Series(np.asarray(data.time)[mask]).rank(method="average")
        time_band[mask] = pd.qcut(
            ranked, q=min(n_bins, int(mask.sum())), labels=False, duplicates="drop"
        ).to_numpy(dtype=int)
    strata = event * n_bins + time_band
    test_fraction = float(split_cfg["test_fraction"])
    validation_fraction = float(split_cfg["validation_fraction"])
    train_validation, test = train_test_split(
        indices, test_size=test_fraction, random_state=seed, stratify=strata
    )
    relative_validation = validation_fraction / (1.0 - test_fraction)
    train, validation = train_test_split(
        train_validation, test_size=relative_validation, random_state=seed + 1,
        stratify=strata[train_validation],
    )
    return train, validation, test


def split_survival_data(data: SurvivalData, split_cfg: dict, seed: int):
    return tuple(
        subset_survival_data(data, indices)
        for indices in three_way_split_indices(data, split_cfg, seed)
    )


def iter_split_indices(data: SurvivalData, split_cfg: dict, seed: int):
    strategy = str(split_cfg.get("strategy", "holdout")).lower()
    if strategy == "holdout":
        yield three_way_split_indices(data, split_cfg, seed)
        return
    if strategy != "kfold":
        raise ValueError(f"Unknown split strategy: {strategy}")
    indices = np.arange(len(data.time))
    splitter = StratifiedKFold(
        n_splits=int(split_cfg.get("folds", 5)), shuffle=True, random_state=seed
    )
    for fold, (train_validation, test) in enumerate(splitter.split(indices, data.event)):
        train, validation = train_test_split(
            train_validation,
            test_size=float(split_cfg["validation_fraction"]),
            random_state=seed + fold + 1,
            stratify=data.event[train_validation],
        )
        yield train, validation, test


def preprocess_covariates(train, validation, test, cfg):
    if bool(cfg.get("zscore_x", False)):
        numeric_features = cfg.get("numeric_features")
        columns = (list(range(train.X.shape[1])) if numeric_features is None else
                   [train.feature_names.index(name) for name in numeric_features])
        if columns:
            scaler = StandardScaler().fit(train.X[:, columns])
            for part in (train, validation, test):
                part.X = np.asarray(part.X, dtype=float).copy()
                part.X[:, columns] = scaler.transform(part.X[:, columns])
    return train, validation, test


__all__ = [
    "iter_split_indices",
    "preprocess_covariates",
    "split_survival_data",
    "subset_survival_data",
    "three_way_split_indices",
    "time_event_stratified_split_indices",
]
