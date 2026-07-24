"""
End-to-end semi-synthetic DVFM experiment.

Usage
-----
Run from the repository root:

    python -m dvfm.semi_synthetic --config configs/semi_synthetic.yaml

The script:
1. loads a real GBSG CSV or SUPPORT Feather dataset;
2. fits event and censoring CoxPH marginals;
3. samples either a population-level copula or an explicit Clayton Gamma-frailty mechanism;
4. creates complete E/C, observed T/delta, and (for the frailty mechanism) the true subject-level frailty;
5. writes the semi-synthetic CSV;
6. splits and preprocesses the generated data;
7. trains DVFM from dvfm.reference_core;
8. compares the encoder posterior mean with either a dependence proxy or the known subject-level frailty;
9. compares oracle event-time prediction using posterior z against z=0;
10. writes latent and survival-prediction outputs.

Interpretation
--------------
For copula generation, the uniform coordinates are transformed with Phi^{-1}
to z_event and z_censor. For the Gaussian copula these are the actual
Gaussian copula coordinates. For Clayton they are only normal-score rank
coordinates. The scalar diagnostic target

    z_shared_direction = (z_event + z_censor) / sqrt(2 * (1 + rho))

is a normalized shared rank direction. For Gaussian it is standard normal;
for Clayton it is a convenient dependence proxy rather than a true
generating frailty. It is not uniquely identified from right-censored
observations. This is an intentionally favorable diagnostic of
whether DVFM learns dependence-relevant latent structure.

For generation.mechanism=clayton_frailty, Clayton samples are generated with
the Marshall--Olkin frailty representation:

    W_i ~ Gamma(1/theta, 1)
    A_i, B_i ~ Exp(1) independently
    U_Ei = (1 + A_i/W_i)^(-1/theta)
    U_Ci = (1 + B_i/W_i)^(-1/theta)

Conditional on W_i, U_Ei and U_Ci are independent. The standardized log W_i
is stored as true_frailty_z and is the ground-truth subject-level target.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from lifelines import CoxPHFitter
from pycop import simulation
from scipy.stats import kendalltau, norm, pearsonr, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader

from .reference_core import DVFM, SurvivalDataset
from .survival_prediction import evaluate_survival_prediction

EPS = 1e-10
ORACLE_COLUMNS = {
    "time",
    "event",
    "true_event_time",
    "true_censor_time",
    "u_event",
    "u_censor",
    "z_event",
    "z_censor",
    "z_shared_direction",
    "true_frailty_raw",
    "true_frailty_log",
    "true_frailty_z",
}

@dataclass
class CoxPHMarginal:
    penalizer: float = 1e-3
    l1_ratio: float = 0.0
    tail_points: int = 10

    def fit(
        self,
        X: np.ndarray,
        time: np.ndarray,
        event: np.ndarray,
        feature_names: Sequence[str],
    ) -> "CoxPHMarginal":
        frame = pd.DataFrame(X, columns=_safe_feature_names(feature_names))
        frame["_duration"] = np.asarray(time, dtype=float)
        frame["_event"] = np.asarray(event, dtype=int)

        if frame["_event"].sum() == 0:
            raise ValueError("CoxPH marginal requires at least one endpoint.")

        self.model_ = CoxPHFitter(
            penalizer=float(self.penalizer),
            l1_ratio=float(self.l1_ratio),
        )
        self.model_.fit(
            frame,
            duration_col="_duration",
            event_col="_event",
            show_progress=False,
        )
        self.feature_names_ = list(frame.columns[:-2])

        baseline = self.model_.baseline_cumulative_hazard_.iloc[:, 0]
        times = np.concatenate(([0.0], baseline.index.to_numpy(dtype=float)))
        hazards = np.concatenate(([0.0], baseline.to_numpy(dtype=float)))

        keep = np.concatenate(([True], np.diff(times) > 0))
        self.baseline_times_ = times[keep]
        self.baseline_hazards_ = np.maximum.accumulate(hazards[keep])
        self.tail_slope_ = _tail_slope(
            self.baseline_times_,
            self.baseline_hazards_,
            int(self.tail_points),
        )
        return self

    def partial_hazard(self, X: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(X, columns=self.feature_names_)
        values = self.model_.predict_partial_hazard(frame).to_numpy(dtype=float)
        return np.clip(values.reshape(-1), EPS, 1.0 / EPS)

    def inverse_survival(self, u: np.ndarray, X: np.ndarray) -> np.ndarray:
        u = np.clip(np.asarray(u, dtype=float), EPS, 1.0 - EPS)
        target_h0 = -np.log(u) / self.partial_hazard(X)

        unique_h, indices = np.unique(
            self.baseline_hazards_,
            return_index=True,
        )
        unique_t = self.baseline_times_[indices]
        result = np.interp(target_h0, unique_h, unique_t)

        beyond = target_h0 > unique_h[-1]
        if np.any(beyond):
            result[beyond] = (
                unique_t[-1]
                + (target_h0[beyond] - unique_h[-1]) / self.tail_slope_
            )
        return np.maximum(result, EPS)

def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping.")
    return config

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_source_data(config: dict[str, Any]) -> pd.DataFrame:
    """
    Load a real source dataset from CSV or Feather.

    Supported examples
    ------------------
    GBSG CSV:
        source_path: data/gbsg.csv
        source_time_col: rfstime
        source_event_col: status
        id_col: pid

    SUPPORT Feather:
        source_path: data/support.feather
        source_time_col: duration
        source_event_col: event

    `id_col` is optional. Numerical and categorical feature lists are read
    from the YAML and validated against the loaded dataframe.
    """
    data_cfg = config["data"]
    raw_path = data_cfg.get("source_path", data_cfg.get("source_csv"))
    if raw_path is None:
        raise ValueError(
            "data.source_path is required "
            "(data.source_csv is accepted for backward compatibility)."
        )

    path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(f"Source dataset not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".feather", ".ftr"}:
        df = pd.read_feather(path)
    else:
        raise ValueError(
            f"Unsupported source file type '{suffix}'. "
            "Use CSV or Feather."
        )

    time_col = data_cfg["source_time_col"]
    event_col = data_cfg["source_event_col"]
    numerical = list(data_cfg["numerical_features"])
    categorical = list(data_cfg["categorical_features"])
    id_col = data_cfg.get("id_col")

    required = {
        time_col,
        event_col,
        *numerical,
        *categorical,
    }
    if id_col is not None:
        required.add(id_col)

    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing source columns: {missing}")

    if id_col is not None:
        df = df.drop(columns=[id_col])

    df = df.rename(
        columns={
            time_col: "time",
            event_col: "event",
        }
    )
    df["time"] = pd.to_numeric(df["time"], errors="raise")
    df["event"] = pd.to_numeric(df["event"], errors="raise").astype(int)

    df = df.loc[df["time"] > 0].reset_index(drop=True)

    if not np.isin(df["event"], [0, 1]).all():
        raise ValueError("The event indicator must contain only 0/1.")

    keep_columns = numerical + categorical + ["time", "event"]
    return df.loc[:, keep_columns].copy()

def make_preprocessor(
    numerical: Sequence[str],
    categorical: Sequence[str],
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, list(numerical)),
            ("categorical", categorical_pipeline, list(categorical)),
        ],
        remainder="drop",
    )


def kendall_to_gaussian_rho(tau: float) -> float:
    """Gaussian-copula relation rho = sin(pi*tau/2)."""
    if not -1.0 < tau < 1.0:
        raise ValueError("Gaussian-copula Kendall's tau must be in (-1, 1).")
    return float(np.sin(np.pi * tau / 2.0))


def kendall_to_clayton_theta(tau: float) -> float:
    """Clayton-copula relation theta = 2*tau/(1-tau)."""
    if not 0.0 <= tau < 1.0:
        raise ValueError("Clayton Kendall's tau must be in [0, 1).")
    return 0.0 if tau == 0.0 else float(2.0 * tau / (1.0 - tau))


def sample_clayton_frailty_quantiles(
    n_samples: int,
    kendall_tau: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Sample Clayton quantiles together with their true Gamma frailty.

    For theta > 0, the Marshall--Olkin representation is used. Conditional
    on W, the event and censoring copula coordinates are independent.
    The returned ``frailty_z`` is standardized log(W), which is easier to
    compare with DVFM's approximately Gaussian scalar latent than raw W.
    """
    theta = kendall_to_clayton_theta(kendall_tau)
    rng = np.random.default_rng(seed)

    if theta == 0.0:
        # Independence has no non-degenerate shared Clayton frailty. Keep a
        # random diagnostic target, but mark it as undefined in diagnostics.
        u_event = rng.uniform(EPS, 1.0 - EPS, size=n_samples)
        u_censor = rng.uniform(EPS, 1.0 - EPS, size=n_samples)
        frailty_raw = np.ones(n_samples, dtype=float)
        frailty_z = np.zeros(n_samples, dtype=float)
        frailty_defined = False
    else:
        frailty_raw = rng.gamma(
            shape=1.0 / theta,
            scale=1.0,
            size=n_samples,
        )
        event_noise = rng.exponential(scale=1.0, size=n_samples)
        censor_noise = rng.exponential(scale=1.0, size=n_samples)
        u_event = (1.0 + event_noise / frailty_raw) ** (-1.0 / theta)
        u_censor = (1.0 + censor_noise / frailty_raw) ** (-1.0 / theta)

        log_w = np.log(np.clip(frailty_raw, EPS, None))
        frailty_z = (log_w - log_w.mean()) / max(log_w.std(ddof=0), EPS)
        frailty_defined = True

    return (
        np.clip(u_event, EPS, 1.0 - EPS),
        np.clip(u_censor, EPS, 1.0 - EPS),
        frailty_raw,
        {
            "clayton_theta": float(theta),
            "frailty_defined": bool(frailty_defined),
            "frailty_target": "standardized log Gamma frailty W",
        },
    )


def sample_copula_quantiles(
    copula: str,
    n_samples: int,
    kendall_tau: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """
    Sample paired uniform quantiles from the requested copula.

    Supported copulas
    -----------------
    gaussian:
        Uses pycop.simulation.simu_gaussian and converts Kendall's tau to rho.

    clayton:
        Uses pycop.simulation.simu_archimedean with
        theta = 2*tau/(1-tau).

    Each returned marginal is Uniform(0,1). The copula controls only their
    pairing/dependence.
    """
    name = str(copula).strip().lower()
    np.random.seed(seed)  # pycop uses NumPy's global RNG.

    if name == "gaussian":
        rho = kendall_to_gaussian_rho(kendall_tau)
        corr = np.array([[1.0, rho], [rho, 1.0]], dtype=float)
        u_event, u_censor = simulation.simu_gaussian(
            2,
            int(n_samples),
            corr,
        )
        parameters = {
            "gaussian_rho": float(rho),
        }

    elif name == "clayton":
        theta = kendall_to_clayton_theta(kendall_tau)
        if theta == 0.0:
            rng = np.random.default_rng(seed)
            u_event = rng.uniform(EPS, 1.0 - EPS, size=n_samples)
            u_censor = rng.uniform(EPS, 1.0 - EPS, size=n_samples)
        else:
            u_event, u_censor = simulation.simu_archimedean(
                "clayton",
                2,
                int(n_samples),
                theta=theta,
            )
        parameters = {
            "clayton_theta": float(theta),
        }

    else:
        raise ValueError(
            f"Unsupported generation.copula='{copula}'. "
            "Choose 'gaussian' or 'clayton'."
        )

    u_event = np.clip(
        np.asarray(u_event, dtype=float).reshape(-1),
        EPS,
        1.0 - EPS,
    )
    u_censor = np.clip(
        np.asarray(u_censor, dtype=float).reshape(-1),
        EPS,
        1.0 - EPS,
    )
    return u_event, u_censor, parameters


def generate_semi_synthetic(
    source: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, float]]:
    data_cfg = config["data"]
    gen_cfg = config["generation"]
    seed = int(config["experiment"]["seed"])

    numerical = list(data_cfg["numerical_features"])
    categorical = list(data_cfg["categorical_features"])
    features = numerical + categorical

    preprocessor = make_preprocessor(numerical, categorical)
    X = np.asarray(
        preprocessor.fit_transform(source[features]),
        dtype=float,
    )
    encoded_names = preprocessor.get_feature_names_out().tolist()

    event_model = CoxPHMarginal(
        penalizer=gen_cfg["coxph_penalizer"],
        l1_ratio=gen_cfg["coxph_l1_ratio"],
        tail_points=gen_cfg["coxph_tail_points"],
    ).fit(
        X,
        source["time"].to_numpy(),
        source["event"].to_numpy(),
        encoded_names,
    )
    censor_model = CoxPHMarginal(
        penalizer=gen_cfg["coxph_penalizer"],
        l1_ratio=gen_cfg["coxph_l1_ratio"],
        tail_points=gen_cfg["coxph_tail_points"],
    ).fit(
        X,
        source["time"].to_numpy(),
        1 - source["event"].to_numpy(),
        encoded_names,
    )

    mechanism = str(gen_cfg.get("mechanism", "copula")).strip().lower()
    copula_name = str(gen_cfg.get("copula", "gaussian")).lower()
    tau = float(gen_cfg["kendall_tau"])

    true_frailty_raw = np.full(len(source), np.nan, dtype=float)
    true_frailty_log = np.full(len(source), np.nan, dtype=float)
    true_frailty_z = np.full(len(source), np.nan, dtype=float)

    if mechanism == "copula":
        u_event, u_censor, copula_parameters = sample_copula_quantiles(
            copula=copula_name,
            n_samples=len(source),
            kendall_tau=tau,
            seed=seed,
        )
        latent_target_interpretation = (
            "true Gaussian shared direction"
            if copula_name == "gaussian"
            else "normal-score shared dependence proxy; not true Clayton frailty"
        )
    elif mechanism == "clayton_frailty":
        if copula_name != "clayton":
            raise ValueError(
                "generation.mechanism=clayton_frailty requires "
                "generation.copula=clayton."
            )
        (
            u_event,
            u_censor,
            true_frailty_raw,
            copula_parameters,
        ) = sample_clayton_frailty_quantiles(
            n_samples=len(source),
            kendall_tau=tau,
            seed=seed,
        )
        true_frailty_log = np.log(np.clip(true_frailty_raw, EPS, None))
        if tau > 0.0:
            true_frailty_z = (
                true_frailty_log - true_frailty_log.mean()
            ) / max(true_frailty_log.std(ddof=0), EPS)
        else:
            true_frailty_z = np.zeros(len(source), dtype=float)
        latent_target_interpretation = (
            "true subject-level standardized log Gamma frailty"
        )
    else:
        raise ValueError(
            "generation.mechanism must be 'copula' or 'clayton_frailty'."
        )

    # Normal-score coordinates are exact Gaussian copula coordinates only
    # when copula_name == "gaussian". For Clayton they provide a common
    # rank-scale diagnostic, not a true latent frailty.
    z_event = norm.ppf(u_event)
    z_censor = norm.ppf(u_censor)

    empirical_z_corr = float(np.corrcoef(z_event, z_censor)[0, 1])
    shared_denominator = np.sqrt(
        max(2.0 * (1.0 + empirical_z_corr), EPS)
    )
    z_shared = (z_event + z_censor) / shared_denominator

    true_event = event_model.inverse_survival(u_event, X)
    raw_censor = censor_model.inverse_survival(u_censor, X)
    multiplier = calibrate_censor_multiplier(
        true_event,
        raw_censor,
        float(gen_cfg["target_censoring_rate"]),
        float(gen_cfg["censoring_rate_tolerance"]),
    )
    true_censor = multiplier * raw_censor

    observed = np.minimum(true_event, true_censor)
    event = (true_event <= true_censor).astype(int)

    result = source[features].copy()
    result["time"] = observed
    result["event"] = event
    result["true_event_time"] = true_event
    result["true_censor_time"] = true_censor
    result["u_event"] = u_event
    result["u_censor"] = u_censor
    result["z_event"] = z_event
    result["z_censor"] = z_censor
    result["z_shared_direction"] = z_shared
    result["true_frailty_raw"] = true_frailty_raw
    result["true_frailty_log"] = true_frailty_log
    result["true_frailty_z"] = true_frailty_z
    result["copula_name"] = copula_name
    result["generation_mechanism"] = mechanism

    diagnostics = {
        "mechanism": mechanism,
        "copula": copula_name,
        **copula_parameters,
        "target_kendall_tau": tau,
        "empirical_quantile_tau": float(
            kendalltau(u_event, u_censor).statistic
        ),
        "empirical_time_tau": float(
            kendalltau(true_event, true_censor).statistic
        ),
        "normal_score_pearson": empirical_z_corr,
        "latent_target_interpretation": latent_target_interpretation,
        "target_censoring_rate": float(gen_cfg["target_censoring_rate"]),
        "achieved_censoring_rate": float(1.0 - event.mean()),
        "censor_time_multiplier": float(multiplier),
    }
    return result.reset_index(drop=True), diagnostics


def calibrate_censor_multiplier(
    event_time: np.ndarray,
    censor_time: np.ndarray,
    target: float,
    tolerance: float,
) -> float:
    if not 0.0 < target < 1.0:
        raise ValueError("target_censoring_rate must be in (0,1).")

    def rate(log_multiplier: float) -> float:
        return float(
            np.mean(np.exp(log_multiplier) * censor_time < event_time)
        )

    low, high = -12.0, 12.0
    for _ in range(100):
        middle = 0.5 * (low + high)
        current = rate(middle)
        if abs(current - target) <= tolerance:
            break
        if current > target:
            low = middle
        else:
            high = middle
    return float(np.exp(middle))


def split_indices(
    data: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split_cfg = config["split"]
    seed = int(config["experiment"]["seed"])
    indices = np.arange(len(data))
    stratify = data["event"] if split_cfg.get("stratify_by_event", True) else None

    train_fraction = float(split_cfg["train_fraction"])
    validation_fraction = float(split_cfg["validation_fraction"])
    test_fraction = float(split_cfg["test_fraction"])
    if not np.isclose(
        train_fraction + validation_fraction + test_fraction,
        1.0,
    ):
        raise ValueError("Train/validation/test fractions must sum to one.")

    train_idx, remaining_idx = train_test_split(
        indices,
        test_size=validation_fraction + test_fraction,
        random_state=seed,
        stratify=stratify,
    )
    remaining_events = data.iloc[remaining_idx]["event"]
    relative_test = test_fraction / (validation_fraction + test_fraction)
    valid_idx, test_idx = train_test_split(
        remaining_idx,
        test_size=relative_test,
        random_state=seed,
        stratify=remaining_events,
    )
    return train_idx, valid_idx, test_idx


def prepare_model_data(
    data: pd.DataFrame,
    indices: tuple[np.ndarray, np.ndarray, np.ndarray],
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], ColumnTransformer, float]:
    data_cfg = config["data"]
    numerical = list(data_cfg["numerical_features"])
    categorical = list(data_cfg["categorical_features"])
    features = numerical + categorical
    train_idx, valid_idx, test_idx = indices

    preprocessor = make_preprocessor(numerical, categorical)
    X_train = np.asarray(
        preprocessor.fit_transform(data.iloc[train_idx][features]),
        dtype=np.float32,
    )
    X_valid = np.asarray(
        preprocessor.transform(data.iloc[valid_idx][features]),
        dtype=np.float32,
    )
    X_test = np.asarray(
        preprocessor.transform(data.iloc[test_idx][features]),
        dtype=np.float32,
    )

    normalization = config["preprocessing"]["time_normalization"]
    if normalization != "train_max":
        raise NotImplementedError("Only time_normalization=train_max is supported.")

    time_scale = float(data.iloc[train_idx]["time"].max())
    if time_scale <= 0:
        raise ValueError("Invalid training time scale.")

    def build(split_idx: np.ndarray, X: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "X": X,
            "time": (
                data.iloc[split_idx]["time"].to_numpy(dtype=np.float32)
                / time_scale
            ),
            "event": data.iloc[split_idx]["event"].to_numpy(dtype=np.float32),
            "target_z": data.iloc[split_idx][
                config["latent_recovery"]["target_column"]
            ].to_numpy(dtype=np.float32),
            "true_event_time": (
                data.iloc[split_idx]["true_event_time"].to_numpy(dtype=np.float32)
                / time_scale
            ),
            "row_index": split_idx,
        }

    return (
        build(train_idx, X_train),
        build(valid_idx, X_valid),
        build(test_idx, X_test),
        preprocessor,
        time_scale,
    )


def train_and_measure_latent(
    train: dict[str, np.ndarray],
    valid: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    config: dict[str, Any],
) -> tuple[DVFM, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Train the unchanged DVFM architecture and evaluate latent recovery.

    Improvements over reference_core.train_dvfm:
    - corrected beta schedule:
          beta = beta_max * min(1, epoch / warmup_epochs)
    - best-checkpoint restoration;
    - checkpoint selection and early stopping by validation reconstruction NLL;
    - early stopping;
    - latent recovery metrics remain diagnostics only and never select the checkpoint;
- sign alignment determined on validation data, never on test data.
    """
    dvfm_cfg = config["dvfm"]
    training_cfg = config.get("training", {})
    recovery_cfg = config["latent_recovery"]
    device = str(dvfm_cfg.get("device", "cpu"))

    latent_dim = int(dvfm_cfg["latent_dim"])
    if latent_dim != 1:
        raise ValueError(
            "Direct scalar latent recovery currently requires latent_dim=1."
        )

    train_dataset = SurvivalDataset(
        train["X"], train["time"], train["event"]
    )
    valid_dataset = SurvivalDataset(
        valid["X"], valid["time"], valid["event"]
    )
    test_dataset = SurvivalDataset(
        test["X"], test["time"], test["event"]
    )

    batch_size = int(dvfm_cfg["batch_size"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=(len(train_dataset) % batch_size == 1),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    model = DVFM(
        input_dim=train["X"].shape[1],
        latent_dim=latent_dim,
        encoder_hidden=list(dvfm_cfg["encoder_hidden"]),
        decoder_hidden=list(dvfm_cfg["decoder_hidden"]),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(dvfm_cfg["learning_rate"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=int(training_cfg.get("lr_patience", 12)),
    )

    epochs = int(dvfm_cfg["epochs"])
    beta_max = float(dvfm_cfg["beta_max"])
    warmup_epochs = int(dvfm_cfg["warmup_epochs"])
    free_bits = float(dvfm_cfg["free_bits"])
    min_epochs = int(training_cfg.get("minimum_epochs", 0))
    patience = int(training_cfg.get("early_stopping_patience", epochs))
    selection_metric = str(
        recovery_cfg.get("selection_metric", "validation_likelihood")
    )
    if selection_metric != "validation_likelihood":
        raise ValueError(
            "latent_recovery.selection_metric must be "
            "'validation_likelihood'. Checkpoint selection may not use "
            "the oracle frailty target."
        )

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_score = -np.inf
    best_val_loss = np.inf
    best_val_nll = np.inf
    no_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        if warmup_epochs > 0:
            beta = beta_max * min(1.0, (epoch + 1) / warmup_epochs)
        else:
            beta = beta_max

        model.train()
        train_loss = 0.0
        train_recon = 0.0
        train_kl = 0.0

        for x, time, event in train_loader:
            x = x.to(device)
            time = time.to(device)
            event = event.to(device)

            optimizer.zero_grad()
            outputs = model(x, time, event)
            loss, recon, kl = model.loss_function(
                *outputs[:4],
                outputs[4],
                outputs[5],
                time,
                event,
                beta,
                free_bits,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += float(loss.item())
            train_recon += float(recon.item())
            train_kl += float(kl.item())

        train_loss /= len(train_loader)
        train_recon /= len(train_loader)
        train_kl /= len(train_loader)

        model.eval()
        val_loss = 0.0
        val_recon_nll = 0.0
        val_kl = 0.0
        with torch.no_grad():
            for x, time, event in valid_loader:
                x = x.to(device)
                time = time.to(device)
                event = event.to(device)
                outputs = model(x, time, event)
                loss, recon, kl = model.loss_function(
                    *outputs[:4],
                    outputs[4],
                    outputs[5],
                    time,
                    event,
                    beta,
                    free_bits,
                )
                val_loss += float(loss.item())
                val_recon_nll += float(recon.item())
                val_kl += float(kl.item())

        val_loss /= len(valid_loader)
        val_recon_nll /= len(valid_loader)
        val_kl /= len(valid_loader)

        # Use the observed-data reconstruction NLL as the validation
        # likelihood criterion. Unlike the beta-weighted ELBO, this
        # criterion is comparable across warm-up epochs.
        scheduler.step(val_recon_nll)
        score = -val_recon_nll

        # Oracle frailty correlation is retained only as a diagnostic.
        valid_mu, _ = encode_loader(model, valid_loader, device)
        valid_target = valid["target_z"].reshape(-1)
        valid_pearson = _safe_corr(pearsonr, valid_mu, valid_target)
        valid_abs_pearson = abs(valid_pearson)

        history.append(
            {
                "epoch": float(epoch + 1),
                "beta": float(beta),
                "train_loss": train_loss,
                "train_reconstruction": train_recon,
                "train_kl": train_kl,
                "validation_loss": val_loss,
                "validation_reconstruction_nll": val_recon_nll,
                "validation_kl": val_kl,
                "validation_pearson": valid_pearson,
                "validation_abs_pearson": valid_abs_pearson,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )

        if score > best_score + 1e-6:
            best_score = score
            best_val_loss = val_loss
            best_val_nll = val_recon_nll
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            no_improvement = 0
        else:
            no_improvement += 1

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1}/{epochs}, "
                f"Beta: {beta:.4f}, "
                f"Train Loss: {train_loss:.4f} "
                f"(Recon: {train_recon:.4f}, KL: {train_kl:.4f}), "
                f"Val ELBO: {val_loss:.4f}, "
                f"Val NLL: {val_recon_nll:.4f}, "
                f"Val |r|: {valid_abs_pearson:.4f}"
            )

        if epoch + 1 >= min_epochs and no_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch + 1}; "
                f"best epoch was {best_epoch}."
            )
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    model.to(device)

    valid_mu, valid_std = encode_loader(model, valid_loader, device)
    valid_target = valid["target_z"].reshape(-1)
    validation_pearson_raw = _safe_corr(
        pearsonr,
        valid_mu,
        valid_target,
    )
    sign = 1.0 if validation_pearson_raw >= 0 else -1.0
    valid_aligned_mu = sign * valid_mu

    validation_latent_frame = pd.DataFrame(
        {
            "row_index": valid["row_index"],
            "event": valid["event"].astype(int),
            "true_z": valid_target,
            "learned_mu_raw": valid_mu,
            "learned_mu_aligned": valid_aligned_mu,
            "learned_std": valid_std,
        }
    )

    learned_mu, learned_std = encode_loader(model, test_loader, device)
    target = test["target_z"].reshape(-1)
    aligned_mu = sign * learned_mu

    test_latent_frame = pd.DataFrame(
        {
            "row_index": test["row_index"],
            "event": test["event"].astype(int),
            "true_z": target,
            "learned_mu_raw": learned_mu,
            "learned_mu_aligned": aligned_mu,
            "learned_std": learned_std,
        }
    )

    metrics = latent_metrics(aligned_mu, target)
    metrics.update(
        {
            "raw_test_pearson": _safe_corr(
                pearsonr,
                learned_mu,
                target,
            ),
            "validation_pearson_raw": validation_pearson_raw,
            "validation_abs_pearson": abs(validation_pearson_raw),
            "sign_alignment_from_validation": sign,
            "posterior_std_mean": float(np.mean(learned_std)),
            "posterior_std_median": float(np.median(learned_std)),
            "best_epoch": int(best_epoch),
            "checkpoint_selection_metric": "validation_reconstruction_nll",
            "best_selection_score": float(best_score),
            "best_validation_reconstruction_nll": float(best_val_nll),
            "best_validation_loss": float(best_val_loss),
            "epochs_completed": int(len(history)),
        }
    )

    if recovery_cfg.get("report_subgroups", True):
        for subgroup_name, mask in {
            "uncensored": test_latent_frame["event"].to_numpy() == 1,
            "censored": test_latent_frame["event"].to_numpy() == 0,
        }.items():
            if mask.sum() >= 3:
                subgroup = latent_metrics(
                    aligned_mu[mask],
                    target[mask],
                )
                for key, value in subgroup.items():
                    metrics[f"{subgroup_name}_{key}"] = value

    metrics["training_history"] = history
    return model, validation_latent_frame, test_latent_frame, metrics


def encode_loader(
    model: DVFM,
    loader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    mus: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    with torch.no_grad():
        for X, time, event in loader:
            X = X.to(device)
            time = time.to(device)
            event = event.to(device)
            mu, logvar = model.encoder(X, time, event)
            mus.append(mu.cpu().numpy().reshape(-1))
            stds.append(torch.exp(0.5 * logvar).cpu().numpy().reshape(-1))
    return np.concatenate(mus), np.concatenate(stds)


def latent_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=float)
    target = np.asarray(target, dtype=float)

    linear = LinearRegression().fit(prediction.reshape(-1, 1), target)
    calibrated = linear.predict(prediction.reshape(-1, 1))
    return {
        "pearson": _safe_corr(pearsonr, prediction, target),
        "spearman": _safe_corr(spearmanr, prediction, target),
        "linear_alignment_r2": float(r2_score(target, calibrated)),
        "linear_alignment_rmse": float(
            mean_squared_error(target, calibrated) ** 0.5
        ),
    }


def _safe_corr(function: Any, x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) < EPS or np.std(y) < EPS:
        return float("nan")
    result = function(x, y)
    return float(result.statistic)


def _tail_slope(
    times: np.ndarray,
    hazards: np.ndarray,
    tail_points: int,
) -> float:
    start = max(0, len(times) - tail_points - 1)
    dt = np.diff(times[start:])
    dh = np.diff(hazards[start:])
    valid = (dt > 0) & (dh > 0)
    if np.any(valid):
        slope = float(np.median(dh[valid] / dt[valid]))
    else:
        slope = float(
            max((hazards[-1] - hazards[0]) / max(times[-1], EPS), EPS)
        )
    return max(slope, EPS)


def _safe_feature_names(names: Sequence[str]) -> list[str]:
    output: list[str] = []
    counts: dict[str, int] = {}
    for index, value in enumerate(names):
        base = "".join(
            character if character.isalnum() or character == "_" else "_"
            for character in str(value)
        )
        if not base:
            base = f"x{index}"
        count = counts.get(base, 0)
        counts[base] = count + 1
        output.append(base if count == 0 else f"{base}_{count}")
    return output


def run(config_path: Path) -> None:
    config = load_config(config_path)
    seed = int(config["experiment"]["seed"])
    set_seed(seed)

    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading real source dataset")
    source = load_source_data(config)

    print("[2/5] Fitting CoxPH marginals and generating copula-dependent data")
    generated, generation_metrics = generate_semi_synthetic(source, config)
    generated_path = Path(config["data"]["generated_csv"])
    generated_path.parent.mkdir(parents=True, exist_ok=True)
    generated.to_csv(generated_path, index=False)

    print("[3/5] Splitting and preprocessing generated data")
    indices = split_indices(generated, config)
    train, valid, test, _, time_scale = prepare_model_data(
        generated,
        indices,
        config,
    )

    print("[4/5] Training DVFM")
    model, validation_latent_predictions, test_latent_predictions, latent_results = train_and_measure_latent(
        train,
        valid,
        test,
        config,
    )

    print("[5/6] Evaluating oracle survival prediction with z and z=0")
    test_loader = DataLoader(
        SurvivalDataset(test["X"], test["time"], test["event"]),
        batch_size=int(config["dvfm"]["batch_size"]),
        shuffle=False,
    )
    survival_results, survival_predictions = evaluate_survival_prediction(
        model=model,
        loader=test_loader,
        test_split=test,
        train_split=train,
        time_scale=time_scale,
        config=config,
    )

    print("[6/6] Saving and reporting results")
    torch.save(model.state_dict(), output_dir / "dvfm_state_dict.pt")
    survival_predictions.to_csv(
        output_dir / "survival_prediction_test.csv",
        index=False,
    )

    if config["latent_recovery"].get("save_latent_predictions", True):
        validation_latent_predictions.to_csv(
            output_dir / "latent_recovery_validation.csv",
            index=False,
        )
        test_latent_predictions.to_csv(
            output_dir / "latent_recovery_test.csv",
            index=False,
        )

    training_history = latent_results.pop("training_history", [])
    if training_history:
        pd.DataFrame(training_history).to_csv(
            output_dir / "training_history.csv",
            index=False,
        )

    results = {
        "generation": generation_metrics,
        "latent_recovery": latent_results,
        "survival_prediction": survival_results,
        "time_scale": time_scale,
        "n_source": len(source),
        "n_train": len(train["time"]),
        "n_validation": len(valid["time"]),
        "n_test": len(test["time"]),
    }
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    print()
    print("Generation diagnostics")
    print("----------------------")
    print(
        f"Mechanism:                 {generation_metrics['mechanism']}"
    )
    print(
        f"Copula:                    {generation_metrics['copula']}"
    )
    if "gaussian_rho" in generation_metrics:
        print(
            f"Gaussian rho:              "
            f"{generation_metrics['gaussian_rho']:.4f}"
        )
    if "clayton_theta" in generation_metrics:
        print(
            f"Clayton theta:             "
            f"{generation_metrics['clayton_theta']:.4f}"
        )
    print(
        f"Target Kendall tau:        "
        f"{generation_metrics['target_kendall_tau']:.4f}"
    )
    print(
        f"Empirical quantile tau:    {generation_metrics['empirical_quantile_tau']:.4f}"
    )
    print(
        f"Achieved censoring rate:   {generation_metrics['achieved_censoring_rate']:.4f}"
    )
    print()
    print("DVFM latent recovery on held-out test data")
    print("------------------------------------------")
    print(
        f"Pearson (sign aligned):    {latent_results['pearson']:.4f}"
    )
    print(
        f"Spearman (sign aligned):   {latent_results['spearman']:.4f}"
    )
    print(
        f"Linear-alignment R^2:      {latent_results['linear_alignment_r2']:.4f}"
    )
    print(
        f"Linear-alignment RMSE:     {latent_results['linear_alignment_rmse']:.4f}"
    )
    print(
        f"Mean posterior std:        {latent_results['posterior_std_mean']:.4f}"
    )
    print(
        f"Best validation epoch:     {latent_results['best_epoch']}"
    )
    print(
        f"Best validation NLL:       "
        f"{latent_results['best_validation_reconstruction_nll']:.4f}"
    )
    print()
    print("Oracle survival prediction (censored test subjects; primary)")
    print("-----------------------------------------------------------")
    primary = survival_results.get("censored_test_primary", {})
    if primary:
        print(f"N censored:                 {primary['n']}")
        print(f"Oracle CI, posterior z:     {primary['posterior_oracle_ci']:.4f}")
        print(f"Oracle CI, z=0:             {primary['zero_oracle_ci']:.4f}")
        print(f"Delta CI (z - zero):        {primary['delta_oracle_ci']:+.4f}")
        print(f"Oracle IBS, posterior z:    {primary['posterior_oracle_ibs']:.4f}")
        print(f"Oracle IBS, z=0:            {primary['zero_oracle_ibs']:.4f}")
        print(f"Delta IBS (z - zero):       {primary['delta_oracle_ibs']:+.4f}")
    print()
    print(f"Generated dataset: {generated_path}")
    print(f"Experiment outputs: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate semi-synthetic GBSG data, train DVFM, and assess latent recovery."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the end-to-end semi-synthetic YAML configuration.",
    )
    args = parser.parse_args()
    run(args.config)

if __name__ == "__main__":
    main()
