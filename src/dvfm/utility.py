import numpy as np
import pandas as pd

import numpy as np
import pandas as pd
import math
import torch
from typing import List, Tuple, Optional, Union
from sklearn.utils import shuffle
from dataclasses import InitVar, dataclass, field
from sklearn.utils import shuffle
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.model_selection import train_test_split

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

Numeric = Union[float, int, bool]
NumericArrayLike = Union[List[Numeric], Tuple[Numeric], np.ndarray, pd.Series, pd.DataFrame, torch.Tensor]

def multilabel_train_test_split(X, y, test_size, random_state=None):
    """Iteratively stratified train/test split
    (Add random_state to scikit-multilearn iterative_train_test_split function)
    See this paper for details: https://link.springer.com/chapter/10.1007/978-3-642-23808-6_10
    """
    X, y = shuffle(X, y, random_state=random_state)
    X_train, y_train, X_test, y_test = iterative_train_test_split(X, y, test_size=test_size)
    return X_train, y_train, X_test, y_test

def make_stratified_split(
        df: pd.DataFrame,
        stratify_colname: str = 'event',
        frac_train: float = 0.5,
        frac_valid: float = 0.0,
        frac_test: float = 0.5,
        random_state: int = None
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame): # type: ignore
    '''Courtesy of https://github.com/shi-ang/BNN-ISD/tree/main'''
    assert frac_train >= 0 and frac_valid >= 0 and frac_test >= 0, "Check train validation test fraction."
    frac_sum = frac_train + frac_valid + frac_test
    frac_train = frac_train / frac_sum
    frac_valid = frac_valid / frac_sum
    frac_test = frac_test / frac_sum

    X = df.values  # Contains all columns.
    columns = df.columns
    if stratify_colname == 'Event':
        stra_lab = df[stratify_colname]
    elif stratify_colname == 'Survival_time':
        stra_lab = df[stratify_colname]
        bins = np.linspace(start=stra_lab.min(), stop=stra_lab.max(), num=20)
        stra_lab = np.digitize(stra_lab, bins, right=True)
    elif stratify_colname == "both":
        t = df["Survival_time"]
        bins = np.linspace(start=t.min(), stop=t.max(), num=20)
        t = np.digitize(t, bins, right=True)
        e = df["Event"]
        stra_lab = np.stack([t, e], axis=1)
    else:
        raise ValueError("unrecognized stratify policy")

    x_train, _, x_temp, y_temp = multilabel_train_test_split(X, y=stra_lab, test_size=(1.0 - frac_train),
                                                             random_state=random_state)
    if frac_valid == 0:
        x_val, x_test = [], x_temp
    else:
        x_val, _, x_test, _ = multilabel_train_test_split(x_temp, y=y_temp,
                                                          test_size=frac_test / (frac_valid + frac_test),
                                                          random_state=random_state)
    df_train = pd.DataFrame(data=x_train, columns=columns)
    df_val = pd.DataFrame(data=x_val, columns=columns)
    df_test = pd.DataFrame(data=x_test, columns=columns)
    assert len(df) == len(df_train) + len(df_val) + len(df_test)
    return df_train, df_val, df_test

def make_stratification_label(df):
    t = df["Survival_time"]
    bins = np.linspace(start=t.min(), stop=t.max(), num=20)
    t = np.digitize(t, bins, right=True)
    e = df["Event"]
    stra_lab = np.stack([t, e], axis=1)
    return stra_lab

def convert_to_structured (T, E):
    default_dtypes = {"names": ("event", "time"), "formats": ("bool", "f8")}
    concat = list(zip(E, T))
    return np.array(concat, dtype=default_dtypes)

def fix_types(df_train, df_valid, df_test):
    df_train = df_train.astype({col: float for col in df_train.columns if col not in ["time", "true_time", "event"]})
    df_train["time"] = df_train["time"].astype(int)
    df_train["true_time"] = df_train["true_time"].astype(int)
    df_train["event"] = df_train["event"].astype(bool)
    df_valid = df_valid.astype({col: float for col in df_valid.columns if col not in ["time", "true_time", "event"]})
    df_valid["time"] = df_valid["time"].astype(int)
    df_valid["true_time"] = df_valid["true_time"].astype(int)
    df_valid["event"] = df_valid["event"].astype(bool)
    df_test = df_test.astype({col: float for col in df_train.columns if col not in ["time", "true_time", "event"]})
    df_test["time"] = df_test["time"].astype(int)
    df_test["true_time"] = df_test["true_time"].astype(int)
    df_test["event"] = df_test["event"].astype(bool)
    return df_train, df_valid, df_test

def map_dataset_name(dataset_name):
    return {
        "gbsg": "GBSG",
        "aids": "AIDS",
        "metabric": "METABRIC",
        "mimic_all": "MIMIC-IV (all)",
        "mimic_hospital": "MIMIC-IV (hospital)",
        "nacd": "NACD",
        "support": "SUPPORT",
        "whas": "WHAS",
        "seer_brain": "SEER (brain)",
        "seer_breast": "SEER (breast)",
        "seer_liver": "SEER (liver)",
        "seer_prostate": "SEER (prostate)",
        "seer_stomach": "SEER (stomach)",
    }.get(dataset_name, dataset_name)
    
def subsample_dataset(df, name, time_col="time", event_col="event",
                      n_bins=10, censor_ratio=5, target_size=None, random_state=42):
    """
    Downsample survival datasets according to predefined rules.
    Dataset names supported:
    - "metabric"
    - "mimic_all"
    - "mimic_hospital"
    - "seer_brain", "seer_liver", "seer_stomach"
    """
    df = df.copy()
    df["time_bin"] = pd.qcut(df[time_col], q=n_bins, duplicates="drop")

    if name == "metabric":
        out = df

    elif name == "mimic_hospital":
        # Keep all events, sample censored up to ratio
        events = df[df[event_col] == 1]
        cens   = df[df[event_col] == 0]

        n_events = len(events)
        n_censor_target = min(len(cens), censor_ratio * n_events)

        cens_keep = cens.groupby("time_bin", group_keys=False).apply(
            lambda x: x.sample(
                n=max(1, int(len(x) * n_censor_target / len(cens))),
                random_state=random_state
            )
        )
        out = pd.concat([events, cens_keep], axis=0)

        # Enforce target_size if given
        if target_size is not None and len(out) > target_size:
            grouped = out.groupby([event_col, "time_bin"], group_keys=False)
            out = grouped.apply(
                lambda x: x.sample(
                    n=max(1, int(len(x) * target_size / len(out))),
                    random_state=random_state
                )
            )

    elif name in ["employee", "mimic_all"]:
        if target_size is None:
            out = df
        else:
            grouped = df.groupby([event_col, "time_bin"], group_keys=False)
            out = grouped.apply(
                lambda x: x.sample(
                    n=max(1, int(len(x) * target_size / len(df))),
                    random_state=random_state
                )
            )

    elif name in ["seer_brain", "seer_liver", "seer_stomach"]:
        if target_size is None:
            target_size = 20000
        grouped = df.groupby([event_col, "time_bin"], group_keys=False)
        out = grouped.apply(
            lambda x: x.sample(
                n=max(1, int(len(x) * target_size / len(df))),
                random_state=random_state
            )
        )

    else:
        raise ValueError(f"Unknown dataset name: {name}")

    # Drop helper column and shuffle
    out = out.drop(columns=["time_bin"]).sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    return out
