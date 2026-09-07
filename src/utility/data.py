"""Generic CSV/XLSX loaders and survival-data container."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class SurvivalData:
    X: np.ndarray
    time: np.ndarray
    event: np.ndarray
    feature_names: list[str]
    true_event_time: np.ndarray | None = None
    true_censor_time: np.ndarray | None = None
    true_z: np.ndarray | None = None


class SurvivalDataset(Dataset):
    """Torch dataset for covariates, observed time, and event indicator."""

    def __init__(self, X, time, event):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.time = torch.as_tensor(time, dtype=torch.float32)
        self.event = torch.as_tensor(event, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.time[index], self.event[index]


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def to_binary_event(values: Iterable) -> np.ndarray:
    s = pd.Series(values)
    if pd.api.types.is_bool_dtype(s):
        return s.astype(int).to_numpy()
    if pd.api.types.is_numeric_dtype(s):
        return (pd.to_numeric(s, errors="coerce").fillna(0) > 0).astype(int).to_numpy()
    mapping = {
        "1": 1, "true": 1, "yes": 1, "y": 1, "dead": 1, "event": 1,
        "0": 0, "false": 0, "no": 0, "n": 0, "alive": 0, "censored": 0,
    }
    return s.astype(str).str.strip().str.lower().map(mapping).fillna(0).astype(int).to_numpy()


def _encode_features(df: pd.DataFrame, excluded: set[str], feature_cols: list[str] | None) -> tuple[np.ndarray, list[str]]:
    if feature_cols:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        x_df = df[feature_cols].copy()
    else:
        x_df = df.drop(columns=[c for c in excluded if c in df.columns]).copy()
    x_df = pd.get_dummies(x_df, drop_first=True)
    x_df = x_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return x_df.to_numpy(dtype=float), list(x_df.columns)


def load_real_data(
    path: str | Path,
    time_col: str,
    event_col: str,
    feature_cols: list[str] | None = None,
) -> SurvivalData:
    df = _read_table(path)
    missing = [c for c in (time_col, event_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")
    t = pd.to_numeric(df[time_col], errors="coerce")
    e = pd.Series(to_binary_event(df[event_col]), index=df.index)
    valid = t.notna() & (t > 0) & e.notna()
    clean = df.loc[valid].copy()
    X, names = _encode_features(clean, {time_col, event_col}, feature_cols)
    return SurvivalData(X, t.loc[valid].to_numpy(float), e.loc[valid].to_numpy(int), names)


def load_semi_synthetic_data(
    path: str | Path,
    true_event_time_col: str,
    true_censor_time_col: str,
    feature_cols: list[str] | None = None,
) -> SurvivalData:
    df = _read_table(path)
    required = (true_event_time_col, true_censor_time_col)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")
    true_t = pd.to_numeric(df[true_event_time_col], errors="coerce")
    true_c = pd.to_numeric(df[true_censor_time_col], errors="coerce")
    valid = true_t.notna() & true_c.notna() & (true_t > 0) & (true_c > 0)
    clean = df.loc[valid].copy()
    true_t_arr = true_t.loc[valid].to_numpy(float)
    true_c_arr = true_c.loc[valid].to_numpy(float)
    observed = np.minimum(true_t_arr, true_c_arr)
    event = (true_t_arr <= true_c_arr).astype(int)
    X, names = _encode_features(clean, set(required), feature_cols)
    return SurvivalData(X, observed, event, names, true_t_arr, true_c_arr)
