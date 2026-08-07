"""Oracle-Z Gaussian benchmark comparing original DVFM and DVFM2.

Models fitted on the exact same generated dataset/split for every seed:

1. no_latent
2. inferred_dvfm_d1
3. dvfm2_d1
   - scalar shared z_s
   - scalar event-private z_e
   - scalar censor-private z_c
4. oracle_z

The true synthetic frailty is scalar.  The primary DVFM2 recovery target is
therefore z_s.  z_e and z_c are retained as scalar nuisance/private latents so
we can explicitly test whether the architecture routes shared dependence into
z_s rather than duplicating it in the private branches.

All models train for the full configured number of epochs (200 by default).
Best validation reconstruction NLL is recorded only as a diagnostic; final
epoch weights are evaluated.

Run from repository root:
    python -u -m experiments.synthetic_oracle_z_dvfm2 \
        --config configs/experiments/synthetic_oracle_z_dvfm2.yaml
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import torch
import torch.nn as nn
import yaml
from lifelines.utils import concordance_index
from scipy.integrate import trapezoid
from scipy.special import gamma as gamma_fn
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from dvfm.model_variants import NoLatentJointWeibull
from dvfm.reference_core import DVFM, Decoder, SurvivalDataset

from dvfm2.model import SharedPrivateDVFM
from dvfm2.training import train_dvfm2
from dvfm2.prediction import (
    predict_event_survival_prior as predict_dvfm2_prior,
    predict_event_survival_qagg as predict_dvfm2_qagg,
)
from dvfm2.diagnostics import learned_conditional_tau as learned_dvfm2_tau

EPS = 1e-8

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


# ---------------------------------------------------------------------------
# DGP: intentionally the same mechanism as synthetic_gaussian_latent
# ---------------------------------------------------------------------------

def _fixed_beta_vectors(n_features: int, seed: int, scale: float):
    rng = np.random.default_rng(seed)
    beta_t = rng.normal(size=n_features)
    beta_t /= np.linalg.norm(beta_t) + 1e-12

    beta_c = rng.normal(size=n_features)
    beta_c = beta_c - beta_t * np.dot(beta_t, beta_c)
    beta_c /= np.linalg.norm(beta_c) + 1e-12
    return scale * beta_t, scale * beta_c


def _conditional_tau_for_loading(
    loading: float,
    latent_effect: str,
    n_samples: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n_samples)
    e_t = rng.exponential(size=n_samples)
    e_c = rng.exponential(size=n_samples)

    if latent_effect == "tanh":
        g = loading * np.tanh(z)
    elif latent_effect == "linear":
        g = loading * z
    else:
        raise ValueError(f"Unknown latent_effect={latent_effect!r}")

    return float(
        kendalltau(-g + np.log(e_t), -g + np.log(e_c)).statistic
    )


def calibrate_loading(target_tau: float, cfg: dict[str, Any]) -> float:
    if target_tau <= 0:
        return 0.0

    latent_effect = str(cfg.get("latent_effect", "tanh")).lower()
    n = int(cfg.get("tau_calibration_samples", 100_000))
    seed = int(cfg.get("tau_calibration_seed", 12345))
    lo = 0.0
    hi = float(cfg.get("tau_loading_upper", 16.0))

    if _conditional_tau_for_loading(hi, latent_effect, n, seed) < target_tau:
        raise ValueError("tau_loading_upper is too small for requested target_tau.")

    for _ in range(int(cfg.get("tau_calibration_steps", 24))):
        mid = 0.5 * (lo + hi)
        if _conditional_tau_for_loading(mid, latent_effect, n, seed) < target_tau:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)



def calibrate_censor_multiplier(
    target_censoring: float,
    n_features: int,
    loading: float,
    cfg: dict[str, Any],
) -> float:
    """Calibrate censoring on an independent Monte Carlo population.

    Unlike the old benchmark, this does not use the realized train/validation/
    test sample (or its oracle T/C values) to choose the censoring multiplier.
    """
    rng = np.random.default_rng(int(cfg.get("censor_calibration_seed", 54321)))
    n = int(cfg.get("censor_calibration_samples", 100_000))

    beta_t, beta_c = _fixed_beta_vectors(
        n_features,
        int(cfg.get("dgp_parameter_seed", 2026)),
        float(cfg.get("covariate_effect_scale", 0.60)),
    )
    shape_t = float(cfg.get("event_shape", 1.5))
    shape_c = float(cfg.get("censor_shape", 1.3))
    base_t = float(cfg.get("event_base_scale", 1.0))
    base_c = float(cfg.get("censor_base_scale", 1.0))
    latent_effect = str(cfg.get("latent_effect", "tanh")).lower()

    X = rng.normal(size=(n, n_features))
    z = rng.normal(size=n)
    u_t = rng.uniform(1e-8, 1 - 1e-8, size=n)
    u_c = rng.uniform(1e-8, 1 - 1e-8, size=n)

    if latent_effect == "tanh":
        g = loading * np.tanh(z)
    else:
        g = loading * z

    eta_t = X @ beta_t + g
    eta_c = X @ beta_c + g
    T = base_t * ((-np.log(u_t)) / np.exp(eta_t)) ** (1.0 / shape_t)
    C0 = base_c * ((-np.log(u_c)) / np.exp(eta_c)) ** (1.0 / shape_c)

    return float(
        np.quantile(T / np.maximum(C0, 1e-12), 1.0 - target_censoring)
    )


def generate_data(
    n_samples: int,
    n_features: int,
    target_tau: float,
    target_censoring: float,
    seed: int,
    cfg: dict[str, Any],
    loading: float,
    censor_multiplier: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)

    beta_t, beta_c = _fixed_beta_vectors(
        n_features,
        int(cfg.get("dgp_parameter_seed", 2026)),
        float(cfg.get("covariate_effect_scale", 0.60)),
    )

    shape_t = float(cfg.get("event_shape", 1.5))
    shape_c = float(cfg.get("censor_shape", 1.3))
    base_t = float(cfg.get("event_base_scale", 1.0))
    base_c = float(cfg.get("censor_base_scale", 1.0))
    latent_effect = str(cfg.get("latent_effect", "tanh")).lower()

    X = rng.normal(size=(n_samples, n_features))
    z = rng.normal(size=n_samples)
    u_t = rng.uniform(1e-8, 1.0 - 1e-8, size=n_samples)
    u_c = rng.uniform(1e-8, 1.0 - 1e-8, size=n_samples)

    if latent_effect == "tanh":
        g = loading * np.tanh(z)
    elif latent_effect == "linear":
        g = loading * z
    else:
        raise ValueError(latent_effect)

    eta_t = np.clip(X @ beta_t + g, -30.0, 30.0)
    eta_c = np.clip(X @ beta_c + g, -30.0, 30.0)

    T = base_t * ((-np.log(u_t)) / np.exp(eta_t)) ** (1.0 / shape_t)
    C0 = base_c * ((-np.log(u_c)) / np.exp(eta_c)) ** (1.0 / shape_c)

    # The multiplier is calibrated on an independent Monte Carlo population,
    # never on the realized experiment subjects.
    C = C0 * float(censor_multiplier)

    observed = np.minimum(T, C)
    event = (T <= C).astype(np.float32)

    return {
        "X": X.astype(np.float32),
        "time": observed.astype(np.float32),
        "event": event,
        "true_event_time": T.astype(np.float64),
        "true_censor_time": C.astype(np.float64),
        "true_z": z.astype(np.float32),
        "loading": float(loading),
        "beta_t": beta_t,
        "beta_c": beta_c,
        "censor_multiplier": censor_multiplier,
        "achieved_censoring": float(np.mean(event == 0)),
        "overall_time_tau": float(kendalltau(T, C).statistic),
    }


def split_indices(data: dict[str, Any], cfg: dict[str, Any], seed: int):
    idx = np.arange(len(data["time"]))
    test_fraction = float(cfg.get("test_fraction", 0.15))
    valid_fraction = float(cfg.get("validation_fraction", 0.15))
    train_fraction = 1.0 - valid_fraction - test_fraction
    if train_fraction <= 0:
        raise ValueError("Split fractions leave no training data.")

    train_valid, test = train_test_split(
        idx,
        test_size=test_fraction,
        random_state=seed,
        stratify=data["event"],
    )
    valid_rel = valid_fraction / (train_fraction + valid_fraction)
    train, valid = train_test_split(
        train_valid,
        test_size=valid_rel,
        random_state=seed + 1,
        stratify=data["event"][train_valid],
    )
    return train, valid, test


def prepare_data(data, train_idx, valid_idx, test_idx, cfg):
    X = np.asarray(data["X"], dtype=np.float32).copy()
    scaler = None
    if bool(cfg.get("standardize_x", True)):
        scaler = StandardScaler()
        X[train_idx] = scaler.fit_transform(X[train_idx])
        X[valid_idx] = scaler.transform(X[valid_idx])
        X[test_idx] = scaler.transform(X[test_idx])

    method = str(cfg.get("time_normalization", "train_max")).lower()
    if method == "train_max":
        time_scale = max(float(np.max(data["time"][train_idx])), EPS)
    elif method == "none":
        time_scale = 1.0
    else:
        raise ValueError(f"Unsupported time_normalization={method!r}")

    def take(indices):
        return {
            "X": X[indices].astype(np.float32),
            "X_raw": np.asarray(data["X"])[indices].astype(np.float32),
            "time": (
                np.asarray(data["time"])[indices] / time_scale
            ).astype(np.float32),
            "event": np.asarray(data["event"])[indices].astype(np.float32),
            "true_event_time": (
                np.asarray(data["true_event_time"])[indices] / time_scale
            ).astype(np.float64),
            "true_censor_time": (
                np.asarray(data["true_censor_time"])[indices] / time_scale
            ).astype(np.float64),
            "true_z": np.asarray(data["true_z"])[indices].astype(np.float32),
        }

    return take(train_idx), take(valid_idx), take(test_idx), time_scale, scaler


# ---------------------------------------------------------------------------
# Models and losses
# ---------------------------------------------------------------------------

def weibull_log_pdf(t, shape, scale):
    t = torch.clamp(t, min=EPS)
    shape = torch.clamp(shape, min=EPS)
    scale = torch.clamp(scale, min=EPS)
    return (
        torch.log(shape)
        - torch.log(scale)
        + (shape - 1.0) * (torch.log(t) - torch.log(scale))
        - (t / scale).pow(shape)
    )


def weibull_log_survival(t, shape, scale):
    t = torch.clamp(t, min=EPS)
    shape = torch.clamp(shape, min=EPS)
    scale = torch.clamp(scale, min=EPS)
    return -(t / scale).pow(shape)


def observed_reconstruction_nll(
    shape_t, scale_t, shape_c, scale_c, time_, event
):
    ll = event * (
        weibull_log_pdf(time_, shape_t, scale_t)
        + weibull_log_survival(time_, shape_c, scale_c)
    ) + (1.0 - event) * (
        weibull_log_survival(time_, shape_t, scale_t)
        + weibull_log_pdf(time_, shape_c, scale_c)
    )
    return -ll.mean()


class OracleDataset(Dataset):
    def __init__(self, split: dict[str, np.ndarray]):
        self.x = torch.as_tensor(split["X"], dtype=torch.float32)
        self.time = torch.as_tensor(split["time"], dtype=torch.float32)
        self.event = torch.as_tensor(split["event"], dtype=torch.float32)
        self.z = torch.as_tensor(split["true_z"], dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.time[index], self.event[index], self.z[index]


class OracleZModel(nn.Module):
    """Reference DVFM decoder trained with the true scalar Z."""

    def __init__(self, input_dim: int, decoder_hidden: list[int]):
        super().__init__()
        self.latent_dim = 1
        self.decoder = Decoder(
            input_dim=input_dim,
            latent_dim=1,
            hidden_dims=list(decoder_hidden),
        )

    def forward(self, x, z):
        return self.decoder(x, z)


def make_standard_loader(split, batch_size, shuffle):
    ds = SurvivalDataset(split["X"], split["time"], split["event"])
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=bool(shuffle and len(ds) % batch_size == 1),
    )


def make_oracle_loader(split, batch_size, shuffle):
    ds = OracleDataset(split)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=bool(shuffle and len(ds) % batch_size == 1),
    )


def inferred_loss(model, batch, beta, device):
    x, time_, event = [v.to(device) for v in batch]
    shape_t, scale_t, shape_c, scale_c, mu, logvar = model(x, time_, event)
    recon = observed_reconstruction_nll(
        shape_t, scale_t, shape_c, scale_c, time_, event
    )
    kl_per_dim = -0.5 * (
        1.0 + logvar - mu.pow(2) - logvar.exp()
    )
    kl = kl_per_dim.sum(dim=1).mean()
    return recon + beta * kl, recon, kl


def no_latent_loss(model, batch, device):
    x, time_, event = [v.to(device) for v in batch]
    shape_t, scale_t, shape_c, scale_c = model(x)
    recon = observed_reconstruction_nll(
        shape_t, scale_t, shape_c, scale_c, time_, event
    )
    return recon, recon, x.new_tensor(0.0)


def oracle_loss(model, batch, device):
    x, time_, event, z = [v.to(device) for v in batch]
    shape_t, scale_t, shape_c, scale_c = model(x, z)
    recon = observed_reconstruction_nll(
        shape_t, scale_t, shape_c, scale_c, time_, event
    )
    return recon, recon, x.new_tensor(0.0)


def _validation_objective(
    model,
    architecture,
    loader,
    beta,
    device,
    validation_mc_samples: int,
):
    """Use MC averaging for inferred DVFM validation to reduce checkpoint noise."""
    architecture_family = _architecture_family(architecture)
    if architecture_family != "inferred_dvfm":
        total = 0.0
        recon_total = 0.0
        kl_total = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in loader:
                if architecture_family == "no_latent":
                    loss, recon, kl = no_latent_loss(model, batch, device)
                else:
                    loss, recon, kl = oracle_loss(model, batch, device)
                total += float(loss.item())
                recon_total += float(recon.item())
                kl_total += float(kl.item())
                n_batches += 1
        return total / n_batches, recon_total / n_batches, kl_total / n_batches

    # q(z|x,t,delta) is stochastic. Average several samples at validation time.
    total = recon_total = kl_total = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            batch_loss = batch_recon = batch_kl = 0.0
            for _ in range(validation_mc_samples):
                loss, recon, kl = inferred_loss(model, batch, beta, device)
                batch_loss += float(loss.item())
                batch_recon += float(recon.item())
                batch_kl += float(kl.item())
            total += batch_loss / validation_mc_samples
            recon_total += batch_recon / validation_mc_samples
            kl_total += batch_kl / validation_mc_samples
            n_batches += 1
    return total / n_batches, recon_total / n_batches, kl_total / n_batches


def _architecture_family(architecture: str) -> str:
    if architecture.startswith("inferred_dvfm"):
        return "inferred_dvfm"
    if architecture.startswith("dvfm2"):
        return "dvfm2"
    return architecture


def train_model(model, architecture, train, valid, cfg, device):
    architecture_family = _architecture_family(architecture)
    batch_size = int(cfg["batch_size"])
    if architecture_family == "oracle_z":
        train_loader = make_oracle_loader(train, batch_size, True)
        valid_loader = make_oracle_loader(valid, batch_size, False)
    else:
        train_loader = make_standard_loader(train, batch_size, True)
        valid_loader = make_standard_loader(valid, batch_size, False)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(cfg.get("lr_factor", 0.5)),
        patience=int(cfg.get("lr_patience", 10)),
    )

    # Full-training diagnostic: always run the complete configured schedule.
    # Set training.epochs: 200 in the YAML for the colleague-matched run.
    epochs = int(cfg.get("epochs", 200))
    warmup = int(cfg.get("warmup_epochs", 50))
    beta_max = float(cfg.get("beta_max", 1.0))
    validation_mc = int(cfg.get("validation_mc_samples", 5))

    best_val = math.inf
    best_epoch = -1
    history = []

    model.to(device)
    for epoch in range(epochs):
        beta = (
            min(beta_max, beta_max * (epoch + 1) / max(warmup, 1))
            if architecture_family == "inferred_dvfm"
            else 0.0
        )

        model.train()
        train_loss = train_recon = train_kl = 0.0
        n_train_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()
            if architecture_family == "no_latent":
                loss, recon, kl = no_latent_loss(model, batch, device)
            elif architecture_family == "oracle_z":
                loss, recon, kl = oracle_loss(model, batch, device)
            else:
                loss, recon, kl = inferred_loss(model, batch, beta, device)

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite {architecture} loss at epoch {epoch + 1}."
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(cfg.get("grad_clip", 1.0))
            )
            optimizer.step()

            train_loss += float(loss.item())
            train_recon += float(recon.item())
            train_kl += float(kl.item())
            n_train_batches += 1

        train_loss /= n_train_batches
        train_recon /= n_train_batches
        train_kl /= n_train_batches

        model.eval()
        val_loss, val_recon, val_kl = _validation_objective(
            model,
            architecture,
            valid_loader,
            beta,
            device,
            validation_mc,
        )
        # The beta-weighted ELBO changes during KL warmup and is therefore
        # not comparable across epochs. Use the observed-data reconstruction
        # NLL for LR scheduling and checkpoint selection.
        scheduler.step(val_recon)

        history.append(
            {
                "epoch": epoch + 1,
                "beta": beta,
                "train_loss": train_loss,
                "train_reconstruction_nll": train_recon,
                "train_kl": train_kl,
                "validation_loss": val_loss,
                "validation_reconstruction_nll": val_recon,
                "validation_kl": val_kl,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        # Track the best reconstruction-NLL epoch only as a diagnostic.
        # Do NOT restore this checkpoint: the experiment evaluates the final
        # model after the complete training schedule.
        if val_recon < best_val - float(cfg.get("minimum_improvement", 1e-5)):
            best_val = val_recon
            best_epoch = epoch + 1

        if epoch == 0 or (epoch + 1) % int(cfg.get("log_every", 25)) == 0:
            print(
                f"    {architecture:<14} epoch={epoch+1:03d} beta={beta:.3f} "
                f"train={train_loss:.4f} val={val_loss:.4f} "
                f"recon={val_recon:.4f} kl={val_kl:.4f}"
            )

    if best_epoch < 0:
        raise RuntimeError(f"{architecture} produced no valid validation objective.")

    # Important: leave the model at the FINAL epoch. We still report the best
    # validation-reconstruction epoch for diagnostics, but it is not used for
    # model selection in this experiment.
    model.to(device)

    return pd.DataFrame(history), {
        "best_epoch": int(best_epoch),
        "best_validation_reconstruction_nll": float(best_val),
        "best_validation_loss": float(best_val),  # legacy alias
        "checkpoint_selection_metric": "none_final_epoch_evaluated",
        "evaluated_epoch": int(epochs),
        "epochs_completed": len(history),
    }



def train_dvfm2_model(model, train, valid, cfg, device):
    """Train DVFM2 with the same full-epoch policy as the Oracle experiment."""
    batch_size = int(cfg["batch_size"])
    train_loader = make_standard_loader(train, batch_size, True)
    valid_loader = make_standard_loader(valid, batch_size, False)

    history, info = train_dvfm2(
        model,
        train_loader,
        valid_loader,
        epochs=int(cfg.get("epochs", 200)),
        lr=float(cfg["learning_rate"]),
        warmup_epochs=int(cfg.get("warmup_epochs", 50)),
        beta_shared_max=float(cfg.get("beta_shared_max", cfg.get("beta_max", 1.0))),
        private_kl_multiplier=float(cfg.get("private_kl_multiplier", 2.0)),
        free_bits_shared=float(cfg.get("free_bits_shared", 0.0)),
        free_bits_private=float(cfg.get("free_bits_private", 0.0)),
        validation_mc_samples=int(cfg.get("validation_mc_samples", 5)),
        grad_clip=float(cfg.get("grad_clip", 1.0)),
        lr_factor=float(cfg.get("lr_factor", 0.5)),
        lr_patience=int(cfg.get("lr_patience", 10)),
        device=device,
        log_every=int(cfg.get("log_every", 25)),
    )

    # Harmonize key names with the original experiment output.
    info = {
        **info,
        "best_epoch": int(info["best_validation_reconstruction_epoch"]),
        "best_validation_loss": float(info["best_validation_reconstruction_nll"]),
    }
    return history, info


def _encode_dvfm2_posterior_means(model, split, batch_size, device):
    loader = make_standard_loader(split, batch_size, False)
    collected = {"shared": [], "event_private": [], "censor_private": []}
    model.eval()
    with torch.no_grad():
        for x, time_, event in loader:
            q = model.encode(x.to(device), time_.to(device), event.to(device))
            collected["shared"].append(q["mu_s"].cpu().numpy())
            collected["event_private"].append(q["mu_e"].cpu().numpy())
            collected["censor_private"].append(q["mu_c"].cpu().numpy())
    return {
        key: np.concatenate(parts, axis=0)
        for key, parts in collected.items()
    }


def _dvfm2_latent_recovery_metrics(
    model,
    train,
    test,
    batch_size,
    device,
    latent_effect,
):
    """Oracle-only diagnostics for where the true frailty is routed in DVFM2."""
    train_mu = _encode_dvfm2_posterior_means(
        model, train, batch_size, device
    )
    test_mu = _encode_dvfm2_posterior_means(
        model, test, batch_size, device
    )

    out = {}
    for component in ("shared", "event_private", "censor_private"):
        metrics = _aligned_latent_recovery_metrics(
            train_mu=train_mu[component],
            test_mu=test_mu[component],
            train_z=train["true_z"],
            test_z=test["true_z"],
            test_event=test["event"],
            latent_effect=latent_effect,
        )
        for key, value in metrics.items():
            # latent_all_z_spearman -> shared_latent_all_z_spearman
            out[f"{component}_{key}"] = value
    return out


def _dvfm2_private_posterior_stats(model, train, batch_size, device):
    """Deployment-available posterior diagnostics: no use of true z."""
    mus = _encode_dvfm2_posterior_means(model, train, batch_size, device)
    out = {}
    for component, arr in mus.items():
        out[f"{component}_posterior_mu_mean"] = float(np.mean(arr))
        out[f"{component}_posterior_mu_std"] = float(np.std(arr))
    return out


# ---------------------------------------------------------------------------
# Latent recovery diagnostics
# ---------------------------------------------------------------------------

def _safe_corr(fn, x, y):
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(fn(x, y).statistic)


def _encode_inferred_mu(model, split, batch_size, device):
    loader = make_standard_loader(split, batch_size, False)
    mus = []
    model.eval()
    with torch.no_grad():
        for x, time_, event in loader:
            mu, _ = model.encoder(
                x.to(device), time_.to(device), event.to(device)
            )
            mus.append(mu.cpu().numpy())
    return np.concatenate(mus, axis=0)


def _aligned_latent_recovery_metrics(
    train_mu,
    test_mu,
    train_z,
    test_z,
    test_event,
    latent_effect,
):
    """Report recovery for all / uncensored / censored test subjects.

    For d_z=1, sign is identified from training data only. For higher-dimensional
    representations a linear projection is fit on training data, then frozen for
    test evaluation.
    """
    train_mu = np.asarray(train_mu, dtype=float)
    test_mu = np.asarray(test_mu, dtype=float)
    train_z = np.asarray(train_z, dtype=float).reshape(-1)
    test_z = np.asarray(test_z, dtype=float).reshape(-1)
    test_event = np.asarray(test_event, dtype=int).reshape(-1)

    if train_mu.ndim == 1:
        train_mu = train_mu[:, None]
    if test_mu.ndim == 1:
        test_mu = test_mu[:, None]

    if train_mu.shape[1] == 1:
        raw_train = train_mu[:, 0]
        r_train = _safe_corr(spearmanr, raw_train, train_z)
        sign = 1.0 if (np.isnan(r_train) or r_train >= 0) else -1.0
        score = sign * test_mu[:, 0]
    else:
        # Training-only linear alignment for dz>1.
        ridge = Ridge(alpha=1e-6)
        ridge.fit(train_mu, train_z)
        score = ridge.predict(test_mu)

    target_tanh = np.tanh(test_z)

    groups = {
        "all": np.ones(len(test_z), dtype=bool),
        "uncensored": test_event == 1,
        "censored": test_event == 0,
    }
    out = {}
    for group, mask in groups.items():
        if np.sum(mask) < 3:
            continue
        prefix = f"latent_{group}"
        out[f"{prefix}_n"] = int(np.sum(mask))
        out[f"{prefix}_z_pearson"] = _safe_corr(pearsonr, score[mask], test_z[mask])
        out[f"{prefix}_z_spearman"] = _safe_corr(spearmanr, score[mask], test_z[mask])
        out[f"{prefix}_z_kendall"] = _safe_corr(kendalltau, score[mask], test_z[mask])
        out[f"{prefix}_tanhz_pearson"] = _safe_corr(
            pearsonr, score[mask], target_tanh[mask]
        )
        out[f"{prefix}_tanhz_spearman"] = _safe_corr(
            spearmanr, score[mask], target_tanh[mask]
        )
        out[f"{prefix}_tanhz_kendall"] = _safe_corr(
            kendalltau, score[mask], target_tanh[mask]
        )
    return out

# ---------------------------------------------------------------------------
# Prediction and scoring
# ---------------------------------------------------------------------------

def weibull_survival(grid, shape, scale):
    grid = np.asarray(grid, dtype=float).reshape(1, -1)
    shape = np.asarray(shape, dtype=float).reshape(-1, 1)
    scale = np.asarray(scale, dtype=float).reshape(-1, 1)
    return np.exp(-np.power(grid / np.maximum(scale, EPS), shape))


def predict_no_latent(model, X, grid, batch_size, device):
    pieces = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            x = torch.as_tensor(
                X[start:start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            shape_t, scale_t, _, _ = model(x)
            pieces.append(
                weibull_survival(
                    grid,
                    shape_t.cpu().numpy(),
                    scale_t.cpu().numpy(),
                )
            )
    return np.concatenate(pieces)


def predict_prior_marginal(
    model,
    X,
    grid,
    batch_size,
    mc_samples,
    device,
    seed,
):
    rng = np.random.default_rng(seed)
    # Common random numbers across subjects make the MC comparison cleaner.
    latent_dim = int(model.latent_dim)
    z_draws = rng.normal(size=(mc_samples, latent_dim)).astype(np.float32)

    pieces = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            x = torch.as_tensor(
                X[start:start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            acc = np.zeros((len(x), len(grid)), dtype=np.float64)
            for z_np in z_draws:
                z = torch.as_tensor(
                    z_np, dtype=torch.float32, device=device
                ).reshape(1, latent_dim).expand(len(x), latent_dim)
                shape_t, scale_t, _, _ = model.decoder(x, z)
                acc += weibull_survival(
                    grid,
                    shape_t.cpu().numpy(),
                    scale_t.cpu().numpy(),
                )
            pieces.append(acc / mc_samples)
    return np.concatenate(pieces)



def _aggregate_posterior_parameters(model, train, batch_size, device):
    """Collect q_phi(z|x,t,delta) parameters over the training population."""
    loader = make_standard_loader(train, batch_size, False)
    mus = []
    logvars = []
    model.eval()
    with torch.no_grad():
        for x, time_, event in loader:
            mu, logvar = model.encoder(
                x.to(device), time_.to(device), event.to(device)
            )
            mus.append(mu.cpu().numpy())
            logvars.append(logvar.cpu().numpy())
    return np.concatenate(mus, axis=0), np.concatenate(logvars, axis=0)


def predict_aggregate_posterior_marginal(
    model,
    train,
    X,
    grid,
    batch_size,
    mc_samples,
    device,
    seed,
):
    """Predict S_T(t|x) by sampling from the aggregate training posterior.

    This matches the current DVFM prediction rule used in the colleague/reference
    implementation: draw a training posterior component uniformly, then sample
    z from that Gaussian component.
    """
    rng = np.random.default_rng(seed)
    all_mu, all_logvar = _aggregate_posterior_parameters(
        model, train, batch_size, device
    )
    all_std = np.exp(0.5 * all_logvar)
    n_train = len(all_mu)

    pieces = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            x_np = X[start:start + batch_size]
            x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
            n_batch = len(x_np)
            acc = np.zeros((n_batch, len(grid)), dtype=np.float64)

            for _ in range(mc_samples):
                component_idx = rng.integers(0, n_train, size=n_batch)
                eps = rng.normal(size=(n_batch, model.latent_dim)).astype(np.float32)
                z_np = (
                    all_mu[component_idx]
                    + all_std[component_idx] * eps
                ).astype(np.float32)
                z = torch.as_tensor(z_np, dtype=torch.float32, device=device)
                shape_t, scale_t, _, _ = model.decoder(x, z)
                acc += weibull_survival(
                    grid, shape_t.cpu().numpy(), scale_t.cpu().numpy()
                )

            pieces.append(acc / mc_samples)
    return np.concatenate(pieces, axis=0)


def predict_empirical_true_z_marginal(
    model,
    train_true_z,
    X,
    grid,
    batch_size,
    mc_samples,
    device,
    seed,
):
    """Oracle analogue of q_agg: marginalize over empirical training true-z.

    The oracle-Z model has no encoder/posterior, so q_agg is undefined. This
    empirical distribution is the closest matched population-level analogue.
    """
    rng = np.random.default_rng(seed)
    train_true_z = np.asarray(train_true_z, dtype=np.float32).reshape(-1, 1)

    pieces = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            x_np = X[start:start + batch_size]
            x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
            n_batch = len(x_np)
            acc = np.zeros((n_batch, len(grid)), dtype=np.float64)

            for _ in range(mc_samples):
                idx = rng.integers(0, len(train_true_z), size=n_batch)
                z = torch.as_tensor(
                    train_true_z[idx], dtype=torch.float32, device=device
                )
                shape_t, scale_t, _, _ = model.decoder(x, z)
                acc += weibull_survival(
                    grid, shape_t.cpu().numpy(), scale_t.cpu().numpy()
                )

            pieces.append(acc / mc_samples)
    return np.concatenate(pieces, axis=0)

def true_dgp_event_survival(
    X_raw,
    grid_normalized,
    time_scale,
    dgp,
    cfg,
    mc_samples,
    seed,
):
    """Monte Carlo true S_T(t|x), using the known DGP only for evaluation."""
    rng = np.random.default_rng(seed)
    z_draws = rng.normal(size=mc_samples)
    beta_t = np.asarray(dgp["beta_t"])
    loading = float(dgp["loading"])
    shape_t = float(cfg.get("event_shape", 1.5))
    base_t = float(cfg.get("event_base_scale", 1.0))
    latent_effect = str(cfg.get("latent_effect", "tanh")).lower()

    t_original = np.asarray(grid_normalized) * float(time_scale)
    acc = np.zeros((len(X_raw), len(t_original)), dtype=np.float64)

    x_eta = np.asarray(X_raw) @ beta_t
    for z in z_draws:
        if latent_effect == "tanh":
            g = loading * np.tanh(z)
        else:
            g = loading * z
        eta = x_eta + g
        # S(t|x,z)=exp[-exp(eta)*(t/base)^shape].
        acc += np.exp(
            -np.exp(eta)[:, None]
            * (t_original[None, :] / base_t) ** shape_t
        )
    return acc / mc_samples


def evaluate_survival(survival, true_t, grid):
    truth = (true_t[:, None] > grid[None, :]).astype(float)
    brier = np.mean((truth - survival) ** 2, axis=0)
    ibs = float(
        trapezoid(brier, grid) / max(float(grid[-1] - grid[0]), EPS)
    )
    rmst = trapezoid(survival, grid, axis=1)
    ci = float(concordance_index(true_t, rmst))
    return {
        "oracle_ibs": ibs,
        "oracle_ci_rmst": ci,
        "mean_rmst": float(np.mean(rmst)),
    }


def learned_tau(model, architecture, x_fixed, n_samples, device, seed):
    architecture_family = _architecture_family(architecture)
    rng = np.random.default_rng(seed)
    x = torch.as_tensor(
        x_fixed, dtype=torch.float32, device=device
    ).reshape(1, -1).expand(n_samples, -1)

    model.eval()
    with torch.no_grad():
        if architecture_family == "no_latent":
            shape_t, scale_t, shape_c, scale_c = model(x)
        else:
            z = torch.as_tensor(
                rng.normal(size=(n_samples, 1)).astype(np.float32),
                device=device,
            )
            shape_t, scale_t, shape_c, scale_c = model.decoder(x, z)

        u_t = torch.as_tensor(
            rng.uniform(1e-8, 1 - 1e-8, n_samples).astype(np.float32),
            device=device,
        )
        u_c = torch.as_tensor(
            rng.uniform(1e-8, 1 - 1e-8, n_samples).astype(np.float32),
            device=device,
        )
        T = scale_t * (-torch.log(u_t)).pow(1.0 / shape_t)
        C = scale_c * (-torch.log(u_c)).pow(1.0 / shape_c)

    return float(kendalltau(T.cpu().numpy(), C.cpu().numpy()).statistic)


def latent_effect_profile(
    model,
    architecture,
    x_fixed,
    z_values,
    device,
    seed,
):
    architecture_family = _architecture_family(architecture)
    rows = []
    if architecture_family == "no_latent":
        return rows

    model.eval()
    x = torch.as_tensor(
        x_fixed, dtype=torch.float32, device=device
    ).reshape(1, -1)

    with torch.no_grad():
        for z_value in z_values:
            z = torch.tensor([[z_value]], dtype=torch.float32, device=device)
            shape_t, scale_t, shape_c, scale_c = model.decoder(x, z)

            k_t = float(shape_t.item())
            l_t = float(scale_t.item())
            k_c = float(shape_c.item())
            l_c = float(scale_c.item())

            rows.append(
                {
                    "seed": seed,
                    "architecture": architecture,
                    "z": float(z_value),
                    "event_shape": k_t,
                    "event_scale": l_t,
                    "event_mean": l_t * float(gamma_fn(1.0 + 1.0 / k_t)),
                    "censor_shape": k_c,
                    "censor_scale": l_c,
                    "censor_mean": l_c * float(gamma_fn(1.0 + 1.0 / k_c)),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def build_models(input_dim, cfg, device):
    model_cfg = cfg["model"]
    decoder_hidden = list(model_cfg["decoder_hidden"])
    encoder_hidden = list(model_cfg["encoder_hidden"])

    # Original DVFM remains scalar: this experiment is explicitly d_z=1.
    inferred = DVFM(
        input_dim=input_dim,
        latent_dim=1,
        encoder_hidden=encoder_hidden,
        decoder_hidden=decoder_hidden,
    ).to(device)

    dvfm2_cfg = model_cfg["dvfm2"]
    # All three DVFM2 factors are scalar. z_s is the shared dependence latent.
    dims = {
        "shared_dim": int(dvfm2_cfg.get("shared_dim", 1)),
        "event_dim": int(dvfm2_cfg.get("event_dim", 1)),
        "censor_dim": int(dvfm2_cfg.get("censor_dim", 1)),
    }
    if dims != {"shared_dim": 1, "event_dim": 1, "censor_dim": 1}:
        raise ValueError(
            "This Oracle workflow is intentionally scalar-only: "
            "shared_dim=event_dim=censor_dim=1."
        )

    dvfm2 = SharedPrivateDVFM(
        input_dim=input_dim,
        shared_dim=1,
        event_dim=1,
        censor_dim=1,
        encoder_hidden=list(dvfm2_cfg.get("encoder_hidden", encoder_hidden)),
        decoder_hidden=list(dvfm2_cfg.get("decoder_hidden", decoder_hidden)),
        latent_hidden=list(dvfm2_cfg.get("latent_hidden", [])),
        shared_same_sign=bool(dvfm2_cfg.get("shared_same_sign", True)),
    ).to(device)

    return {
        "no_latent": NoLatentJointWeibull(
            input_dim=input_dim,
            hidden_dims=decoder_hidden,
        ).to(device),
        "inferred_dvfm_d1": inferred,
        "dvfm2_d1": dvfm2,
        "oracle_z": OracleZModel(
            input_dim=input_dim,
            decoder_hidden=decoder_hidden,
        ).to(device),
    }


def save_summary(results: pd.DataFrame, out_dir: Path):
    numeric_cols = [
        c for c in results.select_dtypes(include=[np.number]).columns
        if c != "seed"
    ]
    group = ["architecture", "prediction_distribution"]
    results.groupby(group, dropna=False)[numeric_cols].mean().reset_index().to_csv(
        out_dir / "results_mean.csv", index=False
    )
    results.groupby(group, dropna=False)[numeric_cols].std().reset_index().to_csv(
        out_dir / "results_std.csv", index=False
    )


def run(cfg: dict[str, Any]):
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    history_dir = out_dir / "training_history"
    history_dir.mkdir(exist_ok=True)

    dgp_cfg = cfg["synthetic"]
    target_tau = float(dgp_cfg["target_tau"])
    target_censoring = float(dgp_cfg["target_censoring"])
    loading = calibrate_loading(target_tau, dgp_cfg)
    censor_multiplier = calibrate_censor_multiplier(
        target_censoring=target_censoring,
        n_features=int(dgp_cfg["n_features"]),
        loading=loading,
        cfg=dgp_cfg,
    )

    calibration = {
        "target_tau": target_tau,
        "loading": loading,
        "censor_multiplier": censor_multiplier,
        "calibrated_tau": _conditional_tau_for_loading(
            loading,
            str(dgp_cfg.get("latent_effect", "tanh")),
            int(dgp_cfg.get("tau_calibration_samples", 100_000)),
            int(dgp_cfg.get("tau_calibration_seed", 12345)),
        ),
    }
    (out_dir / "dgp_calibration.json").write_text(
        json.dumps(calibration, indent=2),
        encoding="utf-8",
    )

    device = resolve_device(str(cfg.get("device", "auto")))
    print(f"Device: {device}")
    print(
        f"Oracle-Z + DVFM2 benchmark: tau={target_tau:.2f}, "
        f"censoring={target_censoring:.0%}, loading={loading:.4f}, "
        "original_dz=1, dvfm2=(z_s=1,z_e=1,z_c=1), oracle_dz=1"
    )

    result_rows = []
    profile_rows = []
    seeds = [int(x) for x in cfg["seeds"]]

    for seed in seeds:
        print(f"\n=== seed {seed} ===")
        seed_everything(seed)
        dgp = generate_data(
            n_samples=int(dgp_cfg["n_samples"]),
            n_features=int(dgp_cfg["n_features"]),
            target_tau=target_tau,
            target_censoring=target_censoring,
            seed=seed,
            cfg=dgp_cfg,
            loading=loading,
            censor_multiplier=censor_multiplier,
        )
        train_idx, valid_idx, test_idx = split_indices(
            dgp, cfg["split"], seed
        )
        train, valid, test, time_scale, scaler = prepare_data(
            dgp,
            train_idx,
            valid_idx,
            test_idx,
            cfg["preprocessing"],
        )

        # Build models deterministically. The inferred model uses the configured
        # latent dimension (20 for this diagnostic); Oracle-Z remains scalar.
        seed_everything(seed + 60_000)
        models = build_models(
            train["X"].shape[1],
            cfg,
            device,
        )

        grid_max = float(
            np.quantile(
                train["true_event_time"],
                float(cfg["evaluation"].get("grid_max_quantile", 0.95)),
            )
        )
        grid = np.linspace(
            0.0,
            grid_max,
            int(cfg["evaluation"]["n_time_points"]),
        )

        true_survival = true_dgp_event_survival(
            X_raw=test["X_raw"],
            grid_normalized=grid,
            time_scale=time_scale,
            dgp=dgp,
            cfg=dgp_cfg,
            mc_samples=int(cfg["evaluation"]["mc_samples"]),
            seed=seed + 700_000,
        )
        true_metrics = evaluate_survival(
            true_survival, test["true_event_time"], grid
        )
        result_rows.append(
            {
                "seed": seed,
                "architecture": "true_dgp",
                "prediction_distribution": "true_dgp",
                "latent_dim": 1,
                "target_tau": target_tau,
                "target_censoring": target_censoring,
                "achieved_censoring": dgp["achieved_censoring"],
                "overall_time_tau": dgp["overall_time_tau"],
                "learned_conditional_tau": target_tau,
                "n_parameters": 0,
                "best_epoch": np.nan,
                "best_validation_loss": np.nan,
                **true_metrics,
            }
        )

        architecture_order = [
            "no_latent",
            "inferred_dvfm_d1",
            "dvfm2_d1",
            "oracle_z",
        ]

        for architecture in architecture_order:
            print(f"  Training {architecture}")
            model = models[architecture]
            architecture_family = _architecture_family(architecture)

            # Re-seed each model deterministically. The inferred d=1 and d=20
            # variants see the exact same data split but have independent
            # optimization streams.
            if architecture_family == "no_latent":
                train_seed_offset = 100_000
            elif architecture_family == "inferred_dvfm":
                train_seed_offset = 111_000
            elif architecture_family == "dvfm2":
                train_seed_offset = 115_000
            elif architecture_family == "oracle_z":
                train_seed_offset = 120_000
            else:
                raise ValueError(architecture)
            seed_everything(seed + train_seed_offset)

            start = time.time()
            if architecture_family == "dvfm2":
                history, info = train_dvfm2_model(
                    model,
                    train,
                    valid,
                    cfg["training"],
                    device,
                )
            else:
                history, info = train_model(
                    model,
                    architecture,
                    train,
                    valid,
                    cfg["training"],
                    device,
                )
            wall = time.time() - start
            history.to_csv(
                history_dir / f"seed{seed}_{architecture}.csv",
                index=False,
            )

            batch_size = int(cfg["training"]["batch_size"])
            mc_samples = int(cfg["evaluation"]["mc_samples"])

            if architecture_family == "no_latent":
                predictions = {
                    "none": predict_no_latent(
                        model, test["X"], grid, batch_size, device
                    )
                }
            elif architecture_family == "inferred_dvfm":
                predictions = {
                    "prior_N01": predict_prior_marginal(
                        model, test["X"], grid, batch_size, mc_samples,
                        device, seed + 800_000,
                    ),
                    "qagg": predict_aggregate_posterior_marginal(
                        model, train, test["X"], grid, batch_size, mc_samples,
                        device, seed + 810_000,
                    ),
                }
            elif architecture_family == "dvfm2":
                dvfm2_train_loader = make_standard_loader(
                    train, batch_size, False
                )
                predictions = {
                    "prior_N01": predict_dvfm2_prior(
                        model,
                        test["X"],
                        grid,
                        n_samples=mc_samples,
                        batch_size=batch_size,
                        device=device,
                        seed=seed + 800_000,
                    ),
                    "qagg": predict_dvfm2_qagg(
                        model,
                        test["X"],
                        grid,
                        dvfm2_train_loader,
                        n_samples=mc_samples,
                        batch_size=batch_size,
                        device=device,
                        seed=seed + 810_000,
                    ),
                }
            else:
                predictions = {
                    "prior_N01": predict_prior_marginal(
                        model, test["X"], grid, batch_size, mc_samples,
                        device, seed + 800_000,
                    ),
                    "empirical_train_true_z": predict_empirical_true_z_marginal(
                        model, train["true_z"], test["X"], grid, batch_size,
                        mc_samples, device, seed + 820_000,
                    ),
                }

            if architecture_family == "dvfm2":
                tau_hat = learned_dvfm2_tau(
                    model,
                    np.zeros(train["X"].shape[1], dtype=np.float32),
                    n_samples=int(cfg["evaluation"]["dependence_mc_samples"]),
                    device=device,
                    seed=seed + 900_000,
                )
            else:
                tau_hat = learned_tau(
                    model,
                    architecture,
                    np.zeros(train["X"].shape[1], dtype=np.float32),
                    int(cfg["evaluation"]["dependence_mc_samples"]),
                    device,
                    seed + 900_000,
                )

            latent_metrics = {}
            posterior_stats = {}
            if architecture_family == "inferred_dvfm":
                train_mu = _encode_inferred_mu(model, train, batch_size, device)
                test_mu = _encode_inferred_mu(model, test, batch_size, device)
                latent_metrics = _aligned_latent_recovery_metrics(
                    train_mu=train_mu,
                    test_mu=test_mu,
                    train_z=train["true_z"],
                    test_z=test["true_z"],
                    test_event=test["event"],
                    latent_effect=str(dgp_cfg.get("latent_effect", "tanh")),
                )
            elif architecture_family == "dvfm2":
                latent_metrics = _dvfm2_latent_recovery_metrics(
                    model=model,
                    train=train,
                    test=test,
                    batch_size=batch_size,
                    device=device,
                    latent_effect=str(dgp_cfg.get("latent_effect", "tanh")),
                )
                posterior_stats = _dvfm2_private_posterior_stats(
                    model, train, batch_size, device
                )

            for prediction_distribution, survival in predictions.items():
                metrics = evaluate_survival(
                    survival, test["true_event_time"], grid
                )
                result_rows.append(
                    {
                        "seed": seed,
                        "architecture": architecture,
                        "prediction_distribution": prediction_distribution,
                        "latent_dim": (
                            1 if architecture_family in {"inferred_dvfm", "oracle_z"}
                            else int(getattr(model, "shared_dim", 0))
                            if architecture_family == "dvfm2"
                            else 0
                        ),
                        "shared_dim": int(getattr(model, "shared_dim", 0)),
                        "event_private_dim": int(getattr(model, "event_dim", 0)),
                        "censor_private_dim": int(getattr(model, "censor_dim", 0)),
                        "target_tau": target_tau,
                        "target_censoring": target_censoring,
                        "achieved_censoring": dgp["achieved_censoring"],
                        "overall_time_tau": dgp["overall_time_tau"],
                        "learned_conditional_tau": tau_hat,
                        "tau_absolute_error": abs(tau_hat - target_tau),
                        "n_parameters": sum(
                            p.numel() for p in model.parameters()
                        ),
                        "training_wall_seconds": wall,
                        **info,
                        **metrics,
                        **latent_metrics,
                        **posterior_stats,
                    }
                )

            # The existing latent-effect profile varies a single scalar z.
            # Keep it for Oracle-Z; skip it for a multi-dimensional inferred model.
            # Existing profile helper applies to the original scalar joint decoder
            # and Oracle-Z. DVFM2 has explicit z_s/z_e/z_c paths and is diagnosed
            # through its dedicated recovery/tau/posterior metrics above.
            if architecture_family in {"inferred_dvfm", "oracle_z"}:
                profile_rows.extend(
                    latent_effect_profile(
                        model,
                        architecture,
                        np.zeros(train["X"].shape[1], dtype=np.float32),
                        np.asarray(cfg["evaluation"]["latent_profile_z"], dtype=float),
                        device,
                        seed,
                    )
                )

    results = pd.DataFrame(result_rows)
    profiles = pd.DataFrame(profile_rows)

    results.to_csv(out_dir / "results_raw.csv", index=False)
    save_summary(results, out_dir)
    profiles.to_csv(out_dir / "latent_effect_profiles.csv", index=False)

    (out_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False),
        encoding="utf-8",
    )

    print("\nOracle-Z sanity experiment")
    print("--------------------------")
    display_cols = [
        "architecture",
        "oracle_ibs",
        "oracle_ci_rmst",
        "learned_conditional_tau",
    ]
    print(
        results.groupby(["architecture", "prediction_distribution"], dropna=False)[display_cols[1:]]
        .mean(numeric_only=True)
        .round(4)
        .to_string()
    )
    print(f"\nResults: {out_dir.resolve()}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run(cfg)


if __name__ == "__main__":
    main()
