"""Oracle-Z sanity experiment for the Gaussian-latent DVFM benchmark.

Purpose
-------
Diagnose whether failure in the synthetic Gaussian-latent benchmark comes from
the DGP / decoder / likelihood / prediction pipeline, or specifically from
variational latent inference.

For each seed, the exact same synthetic dataset and split are used to train:

1. no_latent:
   x -> Weibull event/censoring parameters.

2. inferred_dvfm:
   q(z | x,t,delta), standard Gaussian prior, shared decoder, C-ELBO.

3. oracle_z:
   the same shared decoder class as DVFM, but the *true simulated scalar z* is
   supplied during training and validation. There is no encoder and no KL term.

Both latent models are evaluated at baseline by marginalizing the known prior
z ~ N(0,1). The oracle model therefore tests whether the decoder, observed-data
likelihood, prior marginalization, and evaluation machinery can recover the
correct event marginal when the latent variable itself is known.

Run from repository root:
    python -u -m experiments.synthetic_oracle_z \
        --config configs/synthetic_oracle_z.yaml
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
import torch
import torch.nn as nn
import yaml
from lifelines.utils import concordance_index
from scipy.integrate import trapezoid
from scipy.special import gamma as gamma_fn
from scipy.stats import kendalltau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from dvfm.model_variants import NoLatentJointWeibull
from dvfm.reference_core import DVFM, Decoder, SurvivalDataset

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
    if architecture != "inferred_dvfm":
        total = 0.0
        recon_total = 0.0
        kl_total = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in loader:
                if architecture == "no_latent":
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


def train_model(model, architecture, train, valid, cfg, device):
    batch_size = int(cfg["batch_size"])
    if architecture == "oracle_z":
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

    epochs = int(cfg["epochs"])
    minimum_epochs = int(cfg.get("minimum_epochs", 50))
    patience = int(cfg.get("early_stopping_patience", 20))
    warmup = int(cfg.get("warmup_epochs", 50))
    beta_max = float(cfg.get("beta_max", 1.0))
    validation_mc = int(cfg.get("validation_mc_samples", 5))

    best_state = None
    best_val = math.inf
    best_epoch = -1
    stale = 0
    history = []

    model.to(device)
    for epoch in range(epochs):
        beta = (
            min(beta_max, beta_max * (epoch + 1) / max(warmup, 1))
            if architecture == "inferred_dvfm"
            else 0.0
        )

        model.train()
        train_loss = train_recon = train_kl = 0.0
        n_train_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()
            if architecture == "no_latent":
                loss, recon, kl = no_latent_loss(model, batch, device)
            elif architecture == "oracle_z":
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
        scheduler.step(val_loss)

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

        if val_loss < best_val - float(cfg.get("minimum_improvement", 1e-5)):
            best_val = val_loss
            best_epoch = epoch + 1
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1

        if epoch == 0 or (epoch + 1) % int(cfg.get("log_every", 25)) == 0:
            print(
                f"    {architecture:<14} epoch={epoch+1:03d} beta={beta:.3f} "
                f"train={train_loss:.4f} val={val_loss:.4f} "
                f"recon={val_recon:.4f} kl={val_kl:.4f}"
            )

        if epoch + 1 >= minimum_epochs and stale >= patience:
            break

    if best_state is None:
        raise RuntimeError(f"{architecture} produced no valid checkpoint.")
    model.load_state_dict(best_state)
    model.to(device)

    return pd.DataFrame(history), {
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_val),
        "epochs_completed": len(history),
    }


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
    z_draws = rng.normal(size=(mc_samples, 1)).astype(np.float32)

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
                ).reshape(1, 1).expand(len(x), 1)
                shape_t, scale_t, _, _ = model.decoder(x, z)
                acc += weibull_survival(
                    grid,
                    shape_t.cpu().numpy(),
                    scale_t.cpu().numpy(),
                )
            pieces.append(acc / mc_samples)
    return np.concatenate(pieces)


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
    rng = np.random.default_rng(seed)
    x = torch.as_tensor(
        x_fixed, dtype=torch.float32, device=device
    ).reshape(1, -1).expand(n_samples, -1)

    model.eval()
    with torch.no_grad():
        if architecture == "no_latent":
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
    rows = []
    if architecture == "no_latent":
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

def build_models(input_dim, cfg, decoder_init_state, device):
    model_cfg = cfg["model"]
    decoder_hidden = list(model_cfg["decoder_hidden"])
    encoder_hidden = list(model_cfg["encoder_hidden"])

    no_latent = NoLatentJointWeibull(
        input_dim=input_dim,
        hidden_dims=decoder_hidden,
    ).to(device)

    inferred = DVFM(
        input_dim=input_dim,
        latent_dim=1,
        encoder_hidden=encoder_hidden,
        decoder_hidden=decoder_hidden,
    ).to(device)
    inferred.decoder.load_state_dict(copy.deepcopy(decoder_init_state))

    oracle = OracleZModel(
        input_dim=input_dim,
        decoder_hidden=decoder_hidden,
    ).to(device)
    oracle.decoder.load_state_dict(copy.deepcopy(decoder_init_state))

    return {
        "no_latent": no_latent,
        "inferred_dvfm": inferred,
        "oracle_z": oracle,
    }


def save_summary(results: pd.DataFrame, out_dir: Path):
    numeric_cols = [
        c for c in results.select_dtypes(include=[np.number]).columns
        if c != "seed"
    ]
    group = ["architecture"]
    results.groupby(group)[numeric_cols].mean().reset_index().to_csv(
        out_dir / "results_mean.csv", index=False
    )
    results.groupby(group)[numeric_cols].std().reset_index().to_csv(
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
        f"Oracle-Z benchmark: tau={target_tau:.2f}, "
        f"censoring={target_censoring:.0%}, loading={loading:.4f}"
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

        # One shared initialization for the two dz=1 decoders. This removes an
        # initialization confound between inferred_dvfm and oracle_z.
        decoder_seed = seed + 50_000
        seed_everything(decoder_seed)
        init_decoder = Decoder(
            input_dim=train["X"].shape[1],
            latent_dim=1,
            hidden_dims=list(cfg["model"]["decoder_hidden"]),
        )
        decoder_init_state = copy.deepcopy(init_decoder.state_dict())

        # Build each model deterministically. The inferred/oracle decoders start
        # from exactly the same weights.
        seed_everything(seed + 60_000)
        models = build_models(
            train["X"].shape[1],
            cfg,
            decoder_init_state,
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

        for architecture in ("no_latent", "inferred_dvfm", "oracle_z"):
            print(f"  Training {architecture}")
            model = models[architecture]

            # Re-seed training streams. Inferred and oracle use distinct
            # stochastic streams, while their decoder initialization is matched.
            seed_everything(seed + {
                "no_latent": 100_000,
                "inferred_dvfm": 110_000,
                "oracle_z": 120_000,
            }[architecture])

            start = time.time()
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

            if architecture == "no_latent":
                survival = predict_no_latent(
                    model,
                    test["X"],
                    grid,
                    int(cfg["training"]["batch_size"]),
                    device,
                )
            else:
                survival = predict_prior_marginal(
                    model,
                    test["X"],
                    grid,
                    int(cfg["training"]["batch_size"]),
                    int(cfg["evaluation"]["mc_samples"]),
                    device,
                    seed + 800_000,
                )

            metrics = evaluate_survival(
                survival, test["true_event_time"], grid
            )
            tau_hat = learned_tau(
                model,
                architecture,
                np.zeros(train["X"].shape[1], dtype=np.float32),
                int(cfg["evaluation"]["dependence_mc_samples"]),
                device,
                seed + 900_000,
            )

            result_rows.append(
                {
                    "seed": seed,
                    "architecture": architecture,
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
                }
            )

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
        results.groupby("architecture")[display_cols[1:]]
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
