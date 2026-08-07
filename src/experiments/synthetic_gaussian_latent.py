"""Fully synthetic Gaussian-latent mechanistic DVFM benchmark.

Run from repository root:
    python -m experiments.synthetic_gaussian_latent \
        --config configs/experiments/gaussian_latent_mechanistic.yaml

Design
------
True data-generating mechanism:
    Z ~ N(0,1), independent of X
    T | X,Z ~ Weibull(k_T, lambda_T(X,Z))
    C | X,Z ~ Weibull(k_C, lambda_C(X,Z))
    T independent of C conditional on (X,Z)

The latent effect is a *bounded* Gaussian frailty g(Z)=a*tanh(Z). This keeps the
prior exactly N(0,1), keeps T|X,Z and C|X,Z exactly Weibull, and avoids
pathological time ranges at target Kendall tau 0.75.

For each target conditional Kendall tau and censoring rate, the benchmark fits:
  - no_latent, d_z=0
  - shared DVFM for d_z in configured positive latent dimensions
  - separate-latent negative control for the same dimensions

The separate-latent model has independent event and censoring latents and
therefore cannot induce T-C dependence at fixed X after marginalization.

Primary baseline prediction uses prior marginalization z~N(0,I), deliberately
matching the true generative prior.
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
import yaml
from lifelines.utils import concordance_index
from scipy.integrate import trapezoid
from scipy.stats import kendalltau, pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from dvfm.model_variants import NoLatentJointWeibull, SeparateLatentDVFM
from dvfm.reference_core import DVFM, SurvivalDataset

# Reproducibility and DGP
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

def _fixed_beta_vectors(n_features: int, seed: int, scale: float):
    rng = np.random.default_rng(seed)
    beta_t = rng.normal(size=n_features)
    beta_t /= np.linalg.norm(beta_t) + 1e-12

    beta_c = rng.normal(size=n_features)
    # Orthogonalize censoring covariate effect to the event covariate effect so
    # marginal X structure does not itself create a large T-C association.
    beta_c = beta_c - beta_t * np.dot(beta_t, beta_c)
    beta_c /= np.linalg.norm(beta_c) + 1e-12
    return scale * beta_t, scale * beta_c

def _conditional_tau_for_loading(
    loading: float,
    latent_effect: str,
    calibration_samples: int,
    calibration_seed: int,
) -> float:
    rng = np.random.default_rng(calibration_seed)
    z = rng.normal(size=calibration_samples)
    e_t = rng.exponential(size=calibration_samples)
    e_c = rng.exponential(size=calibration_samples)

    if latent_effect == "tanh":
        g = loading * np.tanh(z)
    elif latent_effect == "linear":
        g = loading * z
    else:
        raise ValueError(f"Unknown latent_effect: {latent_effect}")

    # Weibull shapes and fixed scales are monotone transformations, so the
    # conditional Kendall tau is the same as for these log-time coordinates.
    log_t_rank = -g + np.log(e_t)
    log_c_rank = -g + np.log(e_c)
    return float(kendalltau(log_t_rank, log_c_rank).statistic)

def calibrate_loading_for_tau(target_tau: float, cfg: dict[str, Any]) -> float:
    if target_tau <= 0:
        return 0.0
    latent_effect = str(cfg.get("latent_effect", "tanh")).lower()
    n = int(cfg.get("tau_calibration_samples", 100_000))
    seed = int(cfg.get("tau_calibration_seed", 12345))
    upper = float(cfg.get("tau_loading_upper", 16.0))

    lo, hi = 0.0, upper
    tau_hi = _conditional_tau_for_loading(hi, latent_effect, n, seed)
    if tau_hi < target_tau:
        raise ValueError(
            f"Cannot reach target tau={target_tau} with loading upper={upper}; "
            f"achieved only {tau_hi:.4f}."
        )
    for _ in range(int(cfg.get("tau_calibration_steps", 24))):
        mid = 0.5 * (lo + hi)
        tau_mid = _conditional_tau_for_loading(mid, latent_effect, n, seed)
        if tau_mid < target_tau:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

def generate_gaussian_latent_data(
    n_samples: int,
    n_features: int,
    target_tau: float,
    target_censoring: float,
    seed: int,
    cfg: dict[str, Any],
) -> dict[str, np.ndarray | float]:
    """Generate exactly one scalar Gaussian shared latent.

    T|X,Z and C|X,Z are Weibull with log-scale shifts induced by the same Z.
    """
    rng = np.random.default_rng(seed)
    dgp_seed = int(cfg.get("dgp_parameter_seed", 2026))
    beta_scale = float(cfg.get("covariate_effect_scale", 0.6))
    beta_t, beta_c = _fixed_beta_vectors(n_features, dgp_seed, beta_scale)

    shape_t = float(cfg.get("event_shape", 1.5))
    shape_c = float(cfg.get("censor_shape", 1.3))
    base_scale_t = float(cfg.get("event_base_scale", 1.0))
    base_scale_c = float(cfg.get("censor_base_scale", 1.0))
    latent_effect = str(cfg.get("latent_effect", "tanh")).lower()
    precomputed = cfg.get("_calibrated_loadings", {})
    loading = float(
        precomputed.get(
            str(float(target_tau)),
            calibrate_loading_for_tau(target_tau, cfg),
        )
    )

    X = rng.normal(size=(n_samples, n_features))
    z = rng.normal(size=n_samples)
    eps_t = rng.uniform(1e-8, 1.0 - 1e-8, size=n_samples)
    eps_c = rng.uniform(1e-8, 1.0 - 1e-8, size=n_samples)

    if latent_effect == "tanh":
        g = loading * np.tanh(z)
    elif latent_effect == "linear":
        g = loading * z
    else:
        raise ValueError(f"Unknown latent_effect: {latent_effect}")

    eta_t = np.clip(X @ beta_t + g, -30.0, 30.0)
    eta_c = np.clip(X @ beta_c + g, -30.0, 30.0)

    # Weibull proportional-hazards form:
    # S(t|x,z)=exp[-exp(eta)*(t/base_scale)^k].
    T = base_scale_t * (
        (-np.log(eps_t)) / np.exp(eta_t)
    ) ** (1.0 / shape_t)
    C_base = base_scale_c * (
        (-np.log(eps_c)) / np.exp(eta_c)
    ) ** (1.0 / shape_c)

    # Multiplying every censoring time by one positive constant preserves its
    # ranks/dependence while calibrating P(C<T) very accurately.
    ratio = T / np.maximum(C_base, 1e-12)
    censor_multiplier = float(
        np.quantile(ratio, 1.0 - float(target_censoring))
    )
    C = C_base * censor_multiplier

    observed = np.minimum(T, C)
    event = (T <= C).astype(np.float32)
    achieved_censoring = float(np.mean(event == 0))
    overall_tau = float(kendalltau(T, C).statistic)

    return {
        "X": X.astype(np.float32),
        "time": observed.astype(np.float32),
        "event": event,
        "true_event_time": T.astype(np.float64),
        "true_censor_time": C.astype(np.float64),
        "true_z": z.astype(np.float32),
        "loading": float(loading),
        "target_tau": float(target_tau),
        "conditional_tau_calibrated": _conditional_tau_for_loading(
            loading,
            latent_effect,
            int(cfg.get("tau_calibration_samples", 100_000)),
            int(cfg.get("tau_calibration_seed", 12345)),
        ),
        "overall_time_tau": overall_tau,
        "target_censoring": float(target_censoring),
        "achieved_censoring": achieved_censoring,
        "censor_multiplier": censor_multiplier,
    }

# Splits and preprocessing
def make_splits(data: dict[str, Any], cfg: dict[str, Any], seed: int):
    n = len(data["time"])
    idx = np.arange(n)
    test_fraction = float(cfg.get("test_fraction", 0.15))
    valid_fraction = float(cfg.get("validation_fraction", 0.15))
    train_fraction = 1.0 - test_fraction - valid_fraction
    if train_fraction <= 0:
        raise ValueError("train fraction must be positive.")

    train_valid, test = train_test_split(
        idx,
        test_size=test_fraction,
        random_state=seed,
        stratify=data["event"],
    )
    valid_relative = valid_fraction / (train_fraction + valid_fraction)
    train, valid = train_test_split(
        train_valid,
        test_size=valid_relative,
        random_state=seed + 1,
        stratify=data["event"][train_valid],
    )
    return train, valid, test

def prepare_split(
    data: dict[str, Any],
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    test_idx: np.ndarray,
    cfg: dict[str, Any],
):
    X = np.asarray(data["X"], dtype=np.float32).copy()
    if bool(cfg.get("standardize_x", True)):
        scaler = StandardScaler()
        X[train_idx] = scaler.fit_transform(X[train_idx])
        X[valid_idx] = scaler.transform(X[valid_idx])
        X[test_idx] = scaler.transform(X[test_idx])

    method = str(cfg.get("time_normalization", "train_max")).lower()
    if method == "train_max":
        time_scale = max(float(np.max(data["time"][train_idx])), 1e-8)
    elif method == "train_quantile":
        q = float(cfg.get("time_normalization_quantile", 0.99))
        time_scale = max(float(np.quantile(data["time"][train_idx], q)), 1e-8)
    elif method == "none":
        time_scale = 1.0
    else:
        raise ValueError(f"Unknown time_normalization: {method}")

    def take(indices):
        return {
            "X": X[indices].astype(np.float32),
            "time": (np.asarray(data["time"])[indices] / time_scale).astype(np.float32),
            "event": np.asarray(data["event"])[indices].astype(np.float32),
            "true_event_time": (
                np.asarray(data["true_event_time"])[indices] / time_scale
            ).astype(np.float64),
            "true_censor_time": (
                np.asarray(data["true_censor_time"])[indices] / time_scale
            ).astype(np.float64),
            "true_z": np.asarray(data["true_z"])[indices].astype(np.float32),
        }

    return take(train_idx), take(valid_idx), take(test_idx), float(time_scale)

def make_loader(split, batch_size, shuffle, drop_last=False):
    return DataLoader(
        SurvivalDataset(split["X"], split["time"], split["event"]),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )

# Training
def _shared_loss(model, batch, beta, free_bits, device):
    """Original DVFM C-ELBO, written device-safely for experiment runs."""
    x, time_, event = [v.to(device) for v in batch]
    shape_t, scale_t, shape_c, scale_c, mu, logvar = model(x, time_, event)

    eps = 1e-8
    time_safe = torch.clamp(time_, min=eps)
    shape_t = torch.clamp(shape_t, min=eps)
    scale_t = torch.clamp(scale_t, min=eps)
    shape_c = torch.clamp(shape_c, min=eps)
    scale_c = torch.clamp(scale_c, min=eps)

    def log_pdf(t, shape, scale):
        return (
            torch.log(shape) - torch.log(scale)
            + (shape - 1.0) * (torch.log(t) - torch.log(scale))
            - (t / scale).pow(shape)
        )

    def log_surv(t, shape, scale):
        return -(t / scale).pow(shape)

    ll = event * (
        log_pdf(time_safe, shape_t, scale_t)
        + log_surv(time_safe, shape_c, scale_c)
    ) + (1.0 - event) * (
        log_surv(time_safe, shape_t, scale_t)
        + log_pdf(time_safe, shape_c, scale_c)
    )
    recon_nll = -ll.mean()

    kl_per_dim = -0.5 * (
        1.0 + logvar - mu.pow(2) - logvar.exp()
    )
    kl_raw = kl_per_dim.sum(dim=1).mean()
    if free_bits > 0:
        kl_for_loss = torch.clamp(
            kl_per_dim, min=float(free_bits)
        ).sum(dim=1).mean()
    else:
        kl_for_loss = kl_raw
    return recon_nll + beta * kl_for_loss, recon_nll, kl_raw

def _separate_loss(model, batch, beta, free_bits, device):
    x, time_, event = [v.to(device) for v in batch]
    outputs = model(x, time_, event)
    loss, recon, kl = model.loss_function(
        *outputs, time_, event, beta=beta, free_bits=free_bits
    )
    return loss, recon, kl

def _no_latent_loss(model, batch, beta, free_bits, device):
    x, time_, event = [v.to(device) for v in batch]
    return model.loss(x, time_, event)

def train_model(model, architecture, train, valid, cfg, device):
    batch_size = int(cfg["batch_size"])
    train_loader = make_loader(
        train,
        batch_size,
        True,
        drop_last=(len(train["time"]) % batch_size == 1),
    )
    valid_loader = make_loader(valid, batch_size, False)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(cfg.get("lr_factor", 0.5)),
        patience=int(cfg.get("lr_patience", 10)),
    )
    epochs = int(cfg["epochs"])
    warmup = int(cfg.get("warmup_epochs", 50))
    beta_max = float(cfg.get("beta_max", 1.0))
    free_bits = float(cfg.get("free_bits", 0.0))
    patience = int(cfg.get("early_stopping_patience", 30))
    min_epochs = int(cfg.get("minimum_epochs", 50))

    loss_fn = {
        "shared": _shared_loss,
        "separate": _separate_loss,
        "no_latent": _no_latent_loss,
    }[architecture]

    best_state = None
    best_epoch = -1
    best_val = float("inf")
    stale = 0
    history = []

    model.to(device)
    for epoch in range(epochs):
        beta = (
            min(beta_max, beta_max * (epoch + 1) / max(warmup, 1))
            if architecture != "no_latent"
            else 0.0
        )
        model.train()
        train_loss = train_recon = train_kl = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            loss, recon, kl = loss_fn(model, batch, beta, free_bits, device)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss in {architecture}, epoch {epoch+1}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(cfg.get("grad_clip", 1.0)),
            )
            optimizer.step()
            train_loss += float(loss.item())
            train_recon += float(recon.item())
            train_kl += float(kl.item())

        denom = max(len(train_loader), 1)
        train_loss /= denom
        train_recon /= denom
        train_kl /= denom

        model.eval()
        val_loss = val_recon = val_kl = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                loss, recon, kl = loss_fn(model, batch, beta, free_bits, device)
                val_loss += float(loss.item())
                val_recon += float(recon.item())
                val_kl += float(kl.item())
        denom_v = max(len(valid_loader), 1)
        val_loss /= denom_v
        val_recon /= denom_v
        val_kl /= denom_v
        scheduler.step(val_loss)

        history.append({
            "epoch": epoch + 1,
            "beta": beta,
            "train_loss": train_loss,
            "train_reconstruction_nll": train_recon,
            "train_kl": train_kl,
            "validation_loss": val_loss,
            "validation_reconstruction_nll": val_recon,
            "validation_kl": val_kl,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })

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
                f"    epoch={epoch+1:03d} beta={beta:.3f} "
                f"train={train_loss:.4f} val={val_loss:.4f} "
                f"recon={val_recon:.4f} kl={val_kl:.4f}"
            )

        if epoch + 1 >= min_epochs and stale >= patience:
            break

    if best_state is None:
        raise RuntimeError("No valid checkpoint was produced.")
    model.load_state_dict(best_state)
    model.to(device)
    return pd.DataFrame(history), {
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_val),
        "epochs_completed": int(len(history)),
    }

# Prediction and evaluation
def weibull_survival(grid, shape, scale):
    grid = np.asarray(grid, dtype=float).reshape(1, -1)
    shape = np.asarray(shape, dtype=float).reshape(-1, 1)
    scale = np.asarray(scale, dtype=float).reshape(-1, 1)
    return np.exp(-np.power(grid / np.maximum(scale, 1e-10), shape))

def _predict_no_latent(model, X, grid, batch_size, device):
    out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            x = torch.as_tensor(
                X[start:start+batch_size], dtype=torch.float32, device=device
            )
            shape, scale, _, _ = model(x)
            out.append(
                weibull_survival(grid, shape.cpu().numpy(), scale.cpu().numpy())
            )
    return np.concatenate(out, axis=0)

def _predict_shared_prior(model, X, grid, batch_size, mc_samples, device, seed):
    rng = np.random.default_rng(seed)
    draws = rng.normal(size=(mc_samples, model.latent_dim)).astype(np.float32)
    pieces = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            x = torch.as_tensor(
                X[start:start+batch_size], dtype=torch.float32, device=device
            )
            acc = np.zeros((len(x), len(grid)), dtype=np.float64)
            for draw in draws:
                z = torch.as_tensor(draw, device=device).reshape(1, -1).expand(len(x), -1)
                shape, scale, _, _ = model.decoder(x, z)
                acc += weibull_survival(
                    grid, shape.cpu().numpy(), scale.cpu().numpy()
                )
            pieces.append(acc / mc_samples)
    return np.concatenate(pieces, axis=0)

def _predict_separate_prior(model, X, grid, batch_size, mc_samples, device, seed):
    # Only event latent is relevant for the marginal event distribution.
    rng = np.random.default_rng(seed)
    draws = rng.normal(size=(mc_samples, model.latent_dim)).astype(np.float32)
    pieces = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            x = torch.as_tensor(
                X[start:start+batch_size], dtype=torch.float32, device=device
            )
            acc = np.zeros((len(x), len(grid)), dtype=np.float64)
            for draw in draws:
                z = torch.as_tensor(draw, device=device).reshape(1, -1).expand(len(x), -1)
                shape, scale = model.event_decoder(x, z)
                acc += weibull_survival(
                    grid, shape.cpu().numpy(), scale.cpu().numpy()
                )
            pieces.append(acc / mc_samples)
    return np.concatenate(pieces, axis=0)

def _median_from_curve(survival, grid):
    medians = np.full(len(survival), np.nan, dtype=float)
    crossed = np.zeros(len(survival), dtype=bool)
    for i, curve in enumerate(survival):
        idx = np.flatnonzero(curve <= 0.5)
        if not len(idx):
            continue
        j = int(idx[0])
        crossed[i] = True
        if j == 0:
            medians[i] = grid[0]
        else:
            s0, s1 = curve[j-1], curve[j]
            t0, t1 = grid[j-1], grid[j]
            if abs(s1 - s0) < 1e-12:
                medians[i] = t1
            else:
                medians[i] = t0 + (0.5 - s0) * (t1 - t0) / (s1 - s0)
    return medians, crossed

def evaluate_survival(survival, true_t, grid):
    truth = (true_t[:, None] > grid[None, :]).astype(float)
    brier = np.mean((truth - survival) ** 2, axis=0)
    ibs = float(trapezoid(brier, grid) / max(grid[-1] - grid[0], 1e-12))

    rmst = trapezoid(survival, grid, axis=1)
    ci_rmst = float(concordance_index(true_t, rmst))

    medians, crossed = _median_from_curve(survival, grid)
    # Finite fallback only for comparable C-index; retain crossing fraction.
    median_for_ci = np.where(crossed, medians, grid[-1])
    ci_median = float(concordance_index(true_t, median_for_ci))
    mae_crossed = (
        float(np.mean(np.abs(true_t[crossed] - medians[crossed])))
        if np.any(crossed)
        else float("nan")
    )
    return {
        "oracle_ibs": ibs,
        "oracle_ci_rmst": ci_rmst,
        "oracle_ci_median": ci_median,
        "median_mae_crossed": mae_crossed,
        "median_crossing_fraction": float(np.mean(crossed)),
        "mean_rmst": float(np.mean(rmst)),
    }

# Latent diagnostics
def _posterior_arrays(model, architecture, split, batch_size, device):
    loader = make_loader(split, batch_size, False)
    if architecture == "no_latent":
        return {}
    result = {}
    model.eval()
    with torch.no_grad():
        if architecture == "shared":
            mus, logvars = [], []
            for x, t, e in loader:
                mu, logvar = model.encoder(x.to(device), t.to(device), e.to(device))
                mus.append(mu.cpu().numpy())
                logvars.append(logvar.cpu().numpy())
            result["shared_mu"] = np.concatenate(mus)
            result["shared_logvar"] = np.concatenate(logvars)
        else:
            me, le, mc, lc = [], [], [], []
            for x, t, e in loader:
                x, t, e = x.to(device), t.to(device), e.to(device)
                mu_e, lv_e = model.event_encoder(x, t, e)
                mu_c, lv_c = model.censor_encoder(x, t, e)
                me.append(mu_e.cpu().numpy()); le.append(lv_e.cpu().numpy())
                mc.append(mu_c.cpu().numpy()); lc.append(lv_c.cpu().numpy())
            result["event_mu"] = np.concatenate(me)
            result["event_logvar"] = np.concatenate(le)
            result["censor_mu"] = np.concatenate(mc)
            result["censor_logvar"] = np.concatenate(lc)
    return result

def _probe_true_z(train_mu, test_mu, train_z, test_z):
    ridge = Ridge(alpha=1e-6)
    ridge.fit(train_mu, train_z)
    pred = ridge.predict(test_mu)
    r = float(pearsonr(pred, test_z).statistic)
    r2 = float(r2_score(test_z, pred))
    dim_corrs = []
    for j in range(test_mu.shape[1]):
        if np.std(test_mu[:, j]) < 1e-12:
            dim_corrs.append(0.0)
        else:
            dim_corrs.append(abs(float(pearsonr(test_mu[:, j], test_z).statistic)))
    return {
        "latent_probe_pearson": r,
        "latent_probe_r2": r2,
        "best_single_dim_abs_pearson": float(max(dim_corrs)),
    }

def _kl_dimension_stats(mu, logvar, threshold):
    per_dim = 0.5 * np.mean(
        mu**2 + np.exp(logvar) - 1.0 - logvar,
        axis=0,
    )
    return {
        "mean_kl": float(np.sum(per_dim)),
        "active_dims": int(np.sum(per_dim > threshold)),
        "max_dim_kl": float(np.max(per_dim)),
        "kl_per_dim": per_dim.tolist(),
    }

def learned_conditional_tau(model, architecture, x_fixed, n_samples, device, seed):
    rng = np.random.default_rng(seed)
    x = torch.as_tensor(x_fixed, dtype=torch.float32, device=device).reshape(1, -1)
    x = x.expand(n_samples, -1)
    model.eval()
    with torch.no_grad():
        if architecture == "no_latent":
            shape_t, scale_t, shape_c, scale_c = model(x)
        elif architecture == "shared":
            z = torch.as_tensor(
                rng.normal(size=(n_samples, model.latent_dim)).astype(np.float32),
                device=device,
            )
            shape_t, scale_t, shape_c, scale_c = model.decoder(x, z)
        else:
            z_e = torch.as_tensor(
                rng.normal(size=(n_samples, model.latent_dim)).astype(np.float32),
                device=device,
            )
            z_c = torch.as_tensor(
                rng.normal(size=(n_samples, model.latent_dim)).astype(np.float32),
                device=device,
            )
            shape_t, scale_t = model.event_decoder(x, z_e)
            shape_c, scale_c = model.censor_decoder(x, z_c)

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

# Single fit and grid runner
def build_model(architecture, latent_dim, input_dim, cfg):
    encoder_hidden = list(cfg["encoder_hidden"])
    decoder_hidden = list(cfg["decoder_hidden"])
    if architecture == "no_latent":
        return NoLatentJointWeibull(input_dim, decoder_hidden)
    if architecture == "shared":
        return DVFM(
            input_dim=input_dim,
            latent_dim=latent_dim,
            encoder_hidden=encoder_hidden,
            decoder_hidden=decoder_hidden,
        )
    if architecture == "separate":
        return SeparateLatentDVFM(
            input_dim=input_dim,
            latent_dim=latent_dim,
            encoder_hidden=encoder_hidden,
            decoder_hidden=decoder_hidden,
        )
    raise ValueError(architecture)

def run_one(
    cfg: dict[str, Any],
    target_tau: float,
    censoring: float,
    seed: int,
    architecture: str,
    latent_dim: int,
    device: torch.device,
):
    dgp_cfg = cfg["synthetic"]
    train_cfg = cfg["training"]
    eval_cfg = cfg["evaluation"]

    data = generate_gaussian_latent_data(
        n_samples=int(dgp_cfg["n_samples"]),
        n_features=int(dgp_cfg["n_features"]),
        target_tau=target_tau,
        target_censoring=censoring,
        seed=seed,
        cfg=dgp_cfg,
    )
    train_idx, valid_idx, test_idx = make_splits(data, cfg["split"], seed)
    train, valid, test, time_scale = prepare_split(
        data, train_idx, valid_idx, test_idx, cfg["preprocessing"]
    )

    # Re-seed model independently from generation.
    model_seed = seed + 100_000 + 1000 * latent_dim + {
        "no_latent": 0, "shared": 100, "separate": 200
    }[architecture]
    seed_everything(model_seed)
    model = build_model(
        architecture, latent_dim, train["X"].shape[1], cfg["model"]
    )
    start = time.time()
    history, train_info = train_model(
        model, architecture, train, valid, train_cfg, device
    )
    wall = time.time() - start

    grid_max = float(
        np.quantile(
            train["true_event_time"],
            float(eval_cfg.get("grid_max_quantile", 0.95)),
        )
    )
    grid = np.linspace(0.0, grid_max, int(eval_cfg["n_time_points"]))
    batch_size = int(train_cfg["batch_size"])
    mc = int(eval_cfg["mc_samples"])
    pred_seed = seed + 900_000 + 1000 * latent_dim

    if architecture == "no_latent":
        survival = _predict_no_latent(
            model, test["X"], grid, batch_size, device
        )
    elif architecture == "shared":
        survival = _predict_shared_prior(
            model, test["X"], grid, batch_size, mc, device, pred_seed
        )
    else:
        survival = _predict_separate_prior(
            model, test["X"], grid, batch_size, mc, device, pred_seed
        )

    metrics = evaluate_survival(survival, test["true_event_time"], grid)
    tau_hat = learned_conditional_tau(
        model,
        architecture,
        np.zeros(train["X"].shape[1], dtype=np.float32),
        int(eval_cfg.get("dependence_mc_samples", 3000)),
        device,
        seed + 700_000,
    )

    row = {
        "seed": seed,
        "target_tau": target_tau,
        "target_censoring": censoring,
        "architecture": architecture,
        "latent_dim": latent_dim,
        "n_samples": int(dgp_cfg["n_samples"]),
        "n_features": int(dgp_cfg["n_features"]),
        "train_size": len(train["time"]),
        "validation_size": len(valid["time"]),
        "test_size": len(test["time"]),
        "achieved_censoring": data["achieved_censoring"],
        "conditional_tau_calibrated": data["conditional_tau_calibrated"],
        "overall_time_tau": data["overall_time_tau"],
        "true_latent_loading": data["loading"],
        "censor_multiplier": data["censor_multiplier"],
        "time_scale": time_scale,
        "learned_conditional_tau": tau_hat,
        "tau_absolute_error": abs(tau_hat - target_tau),
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "training_wall_seconds": wall,
        **train_info,
        **metrics,
    }

    latent_dump = {}
    if architecture != "no_latent":
        train_post = _posterior_arrays(
            model, architecture, train, batch_size, device
        )
        test_post = _posterior_arrays(
            model, architecture, test, batch_size, device
        )
        threshold = float(eval_cfg.get("active_dim_kl_threshold", 0.01))
        if architecture == "shared":
            row.update(
                _probe_true_z(
                    train_post["shared_mu"],
                    test_post["shared_mu"],
                    train["true_z"],
                    test["true_z"],
                )
            )
            kl = _kl_dimension_stats(
                test_post["shared_mu"],
                test_post["shared_logvar"],
                threshold,
            )
            row.update({
                "posterior_total_kl": kl["mean_kl"],
                "active_dims": kl["active_dims"],
                "max_dim_kl": kl["max_dim_kl"],
            })
            latent_dump["shared_kl_per_dim"] = kl["kl_per_dim"]
        else:
            event_probe = _probe_true_z(
                train_post["event_mu"], test_post["event_mu"],
                train["true_z"], test["true_z"]
            )
            censor_probe = _probe_true_z(
                train_post["censor_mu"], test_post["censor_mu"],
                train["true_z"], test["true_z"]
            )
            for key, value in event_probe.items():
                row[f"event_{key}"] = value
            for key, value in censor_probe.items():
                row[f"censor_{key}"] = value
            kl_e = _kl_dimension_stats(
                test_post["event_mu"], test_post["event_logvar"], threshold
            )
            kl_c = _kl_dimension_stats(
                test_post["censor_mu"], test_post["censor_logvar"], threshold
            )
            row.update({
                "event_posterior_total_kl": kl_e["mean_kl"],
                "event_active_dims": kl_e["active_dims"],
                "censor_posterior_total_kl": kl_c["mean_kl"],
                "censor_active_dims": kl_c["active_dims"],
            })
            latent_dump["event_kl_per_dim"] = kl_e["kl_per_dim"]
            latent_dump["censor_kl_per_dim"] = kl_c["kl_per_dim"]

    return row, history, latent_dump

def _job_key(row_or_job):
    return (
        int(row_or_job["seed"]),
        float(row_or_job["target_tau"]),
        float(row_or_job["target_censoring"]),
        str(row_or_job["architecture"]),
        int(row_or_job["latent_dim"]),
    )

def build_jobs(cfg):
    seeds = [int(s) for s in cfg["grid"]["seeds"]]
    taus = [float(v) for v in cfg["grid"]["kendall_tau"]]
    censoring = [float(v) for v in cfg["grid"]["censoring_rates"]]
    dims = [int(v) for v in cfg["grid"]["latent_dims"]]

    jobs = []
    for seed in seeds:
        for tau in taus:
            for censor in censoring:
                jobs.append({
                    "seed": seed,
                    "target_tau": tau,
                    "target_censoring": censor,
                    "architecture": "no_latent",
                    "latent_dim": 0,
                })
                for dim in dims:
                    if dim <= 0:
                        continue
                    jobs.append({
                        "seed": seed, "target_tau": tau,
                        "target_censoring": censor,
                        "architecture": "shared", "latent_dim": dim,
                    })
                    if bool(cfg["grid"].get("include_separate_latent", True)):
                        jobs.append({
                            "seed": seed, "target_tau": tau,
                            "target_censoring": censor,
                            "architecture": "separate", "latent_dim": dim,
                        })
    return jobs

def save_summaries(raw: pd.DataFrame, out_dir: Path):
    group_cols = [
        "target_tau", "target_censoring", "architecture", "latent_dim"
    ]
    numeric = raw.select_dtypes(include=[np.number]).columns
    metrics = [
        c for c in numeric
        if c not in {"seed"} and c not in group_cols
    ]
    raw.groupby(group_cols, dropna=False)[metrics].mean().reset_index().to_csv(
        out_dir / "results_mean.csv", index=False
    )
    raw.groupby(group_cols, dropna=False)[metrics].std().reset_index().to_csv(
        out_dir / "results_std.csv", index=False
    )

def run(cfg: dict[str, Any], dry_run: bool = False):
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "training_history").mkdir(exist_ok=True)
    (out_dir / "latent_diagnostics").mkdir(exist_ok=True)

    # Calibrate the DGP loading once per tau, not once per model fit.
    calibrated = {}
    for tau in [float(v) for v in cfg["grid"]["kendall_tau"]]:
        calibrated[str(tau)] = calibrate_loading_for_tau(
            tau, cfg["synthetic"]
        )
    cfg["synthetic"]["_calibrated_loadings"] = calibrated
    calibration_rows = []
    for tau, loading in calibrated.items():
        calibration_rows.append({
            "target_tau": float(tau),
            "loading": float(loading),
            "calibrated_tau": _conditional_tau_for_loading(
                float(loading),
                str(cfg["synthetic"].get("latent_effect", "tanh")),
                int(cfg["synthetic"].get("tau_calibration_samples", 100_000)),
                int(cfg["synthetic"].get("tau_calibration_seed", 12345)),
            ),
        })
    pd.DataFrame(calibration_rows).to_csv(
        out_dir / "dgp_tau_calibration.csv", index=False
    )

    jobs = build_jobs(cfg)
    print(f"Planned model fits: {len(jobs)}")
    if dry_run:
        print(pd.DataFrame(jobs).groupby(
            ["architecture", "latent_dim"]
        ).size().to_string())
        print("\nDGP tau calibration:")
        print(pd.DataFrame(calibration_rows).to_string(index=False))
        return pd.DataFrame(jobs)

    raw_path = out_dir / "results_raw.csv"
    completed = set()
    rows = []
    if bool(cfg.get("resume", True)) and raw_path.exists():
        existing = pd.read_csv(raw_path)
        if "status" in existing.columns:
            existing = existing[existing["status"] == "ok"].copy()
        rows = existing.to_dict("records")
        completed = {_job_key(row) for row in rows}
        print(f"Resuming with {len(completed)} successful fits.")

    device = resolve_device(str(cfg.get("device", "auto")))
    if device.type == "cpu":
        torch.set_num_threads(max(1, int(cfg.get("torch_num_threads", 1))))
    print(f"Device: {device}")

    for i, job in enumerate(jobs, start=1):
        key = _job_key(job)
        if key in completed:
            continue
        print(
            f"[{i}/{len(jobs)}] seed={job['seed']} "
            f"tau={job['target_tau']:.2f} censor={job['target_censoring']:.2f} "
            f"{job['architecture']} dz={job['latent_dim']}"
        )
        try:
            row, history, latent_dump = run_one(
                cfg=cfg,
                target_tau=job["target_tau"],
                censoring=job["target_censoring"],
                seed=job["seed"],
                architecture=job["architecture"],
                latent_dim=job["latent_dim"],
                device=device,
            )
            row["status"] = "ok"
            row["error"] = ""
        except Exception as exc:
            row = {**job, "status": "failed", "error": repr(exc)}
            history = pd.DataFrame()
            latent_dump = {}
            print(f"  FAILED: {exc}")

        rows.append(row)
        pd.DataFrame(rows).to_csv(raw_path, index=False)

        stem = (
            f"seed{job['seed']}_tau{job['target_tau']:.2f}_"
            f"cens{job['target_censoring']:.2f}_"
            f"{job['architecture']}_dz{job['latent_dim']}"
        ).replace(".", "p")
        if not history.empty:
            history.to_csv(
                out_dir / "training_history" / f"{stem}.csv",
                index=False,
            )
        if latent_dump:
            (out_dir / "latent_diagnostics" / f"{stem}.json").write_text(
                json.dumps(latent_dump, indent=2),
                encoding="utf-8",
            )

        ok = pd.DataFrame(rows)
        ok = ok[ok["status"] == "ok"] if "status" in ok else ok
        if not ok.empty:
            save_summaries(ok, out_dir)

    final = pd.DataFrame(rows)
    with (out_dir / "resolved_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"Saved benchmark to {out_dir.resolve()}")
    return final

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned grid without training.",
    )
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run(cfg, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
