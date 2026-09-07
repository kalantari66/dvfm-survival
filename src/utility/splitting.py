"""Reusable stratified survival-data splitting and preprocessing."""

import numpy as np
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
        scaler = StandardScaler().fit(train.X)
        train.X = scaler.transform(train.X)
        validation.X = scaler.transform(validation.X)
        test.X = scaler.transform(test.X)
    return train, validation, test


__all__ = [
    "iter_split_indices",
    "preprocess_covariates",
    "split_survival_data",
    "subset_survival_data",
    "three_way_split_indices",
]
