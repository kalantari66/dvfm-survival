"""Baseline event-survival prediction for DVFM2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .model import SharedPrivateDVFM

EPS = 1e-8


def _weibull_survival(grid, shape, scale):
    grid = np.asarray(grid, dtype=float).reshape(1, -1)
    shape = np.asarray(shape, dtype=float).reshape(-1, 1)
    scale = np.asarray(scale, dtype=float).reshape(-1, 1)
    return np.exp(-np.power(grid / np.maximum(scale, EPS), shape))


def predict_event_survival_prior(
    model: SharedPrivateDVFM,
    X,
    time_points,
    *,
    n_samples: int = 100,
    batch_size: int = 256,
    device: str | torch.device = "cpu",
    seed: int = 0,
):
    """Marginalize the generative priors for z_s and z_e.

    z_c is not sampled because, by construction, it cannot affect event-time
    prediction.
    """
    device = torch.device(device)
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float32)
    time_points = np.asarray(time_points, dtype=float)

    z_s_draws = rng.normal(
        size=(n_samples, model.shared_dim)
    ).astype(np.float32)
    z_e_draws = rng.normal(
        size=(n_samples, model.event_dim)
    ).astype(np.float32)

    out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            x = torch.as_tensor(
                X[start:start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            acc = np.zeros((len(x), len(time_points)), dtype=float)
            for z_s_np, z_e_np in zip(z_s_draws, z_e_draws):
                z_s = torch.as_tensor(z_s_np, device=device).reshape(
                    1, model.shared_dim
                ).expand(len(x), -1)
                z_e = torch.as_tensor(z_e_np, device=device).reshape(
                    1, model.event_dim
                ).expand(len(x), -1)
                shape_t, scale_t = model.decoder.event_parameters(x, z_s, z_e)
                acc += _weibull_survival(
                    time_points,
                    shape_t.cpu().numpy(),
                    scale_t.cpu().numpy(),
                )
            out.append(acc / float(n_samples))
    return np.concatenate(out, axis=0)


@dataclass
class AggregatePosterior:
    mu_s: np.ndarray
    logvar_s: np.ndarray
    mu_e: np.ndarray
    logvar_e: np.ndarray
    mu_c: np.ndarray
    logvar_c: np.ndarray


def collect_aggregate_posterior(
    model: SharedPrivateDVFM,
    train_loader,
    *,
    device: str | torch.device = "cpu",
) -> AggregatePosterior:
    device = torch.device(device)
    storage = {
        "mu_s": [], "logvar_s": [],
        "mu_e": [], "logvar_e": [],
        "mu_c": [], "logvar_c": [],
    }
    model.eval()
    with torch.no_grad():
        for x, time, event in train_loader:
            q = model.encode(x.to(device), time.to(device), event.to(device))
            for key in storage:
                storage[key].append(q[key].cpu().numpy())

    return AggregatePosterior(
        **{k: np.concatenate(v, axis=0) for k, v in storage.items()}
    )


def predict_event_survival_qagg(
    model: SharedPrivateDVFM,
    X,
    time_points,
    train_loader,
    *,
    n_samples: int = 100,
    batch_size: int = 256,
    device: str | torch.device = "cpu",
    seed: int = 0,
):
    """Aggregate-posterior event prediction.

    A training posterior component is sampled and its z_s and z_e are drawn
    jointly from that subject's posterior. z_c remains irrelevant to T by
    architecture.
    """
    device = torch.device(device)
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float32)
    time_points = np.asarray(time_points, dtype=float)
    qagg = collect_aggregate_posterior(model, train_loader, device=device)

    std_s = np.exp(0.5 * qagg.logvar_s)
    std_e = np.exp(0.5 * qagg.logvar_e)
    n_train = len(qagg.mu_s)

    out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            x_np = X[start:start + batch_size]
            x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
            acc = np.zeros((len(x), len(time_points)), dtype=float)

            for _ in range(int(n_samples)):
                idx = rng.integers(0, n_train, size=len(x))
                z_s_np = qagg.mu_s[idx] + std_s[idx] * rng.normal(
                    size=(len(x), model.shared_dim)
                )
                z_e_np = qagg.mu_e[idx] + std_e[idx] * rng.normal(
                    size=(len(x), model.event_dim)
                )
                z_s = torch.as_tensor(
                    z_s_np.astype(np.float32), device=device
                )
                z_e = torch.as_tensor(
                    z_e_np.astype(np.float32), device=device
                )
                shape_t, scale_t = model.decoder.event_parameters(x, z_s, z_e)
                acc += _weibull_survival(
                    time_points,
                    shape_t.cpu().numpy(),
                    scale_t.cpu().numpy(),
                )

            out.append(acc / float(n_samples))
    return np.concatenate(out, axis=0)
