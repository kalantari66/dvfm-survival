"""DVFM survival prediction utilities."""
from __future__ import annotations
import numpy as np
import torch
from utility.metrics import median_survival_time as get_median_survival_time


def predict_survival_curves(model, X, time_points, train_loader, n_samples=100, device="cpu"):
    """Predict by Monte Carlo integration over the aggregate training posterior."""
    model.eval().to(device)
    mus, logvars = [], []
    with torch.no_grad():
        for x_batch, t_batch, event_batch in train_loader:
            x_batch = x_batch.to(device)
            t_batch = t_batch.to(device)
            event_batch = event_batch.to(device)
            if model.encoder is None:
                mu = x_batch.new_empty((x_batch.shape[0], 0))
                logvar = x_batch.new_empty((x_batch.shape[0], 0))
            else:
                mu, logvar = model.encoder(x_batch, t_batch, event_batch)
            mus.append(mu)
            logvars.append(logvar)
    all_mus = torch.cat(mus)
    all_stds = torch.exp(0.5 * torch.cat(logvars))
    x_tensor = torch.as_tensor(X, dtype=torch.float32, device=device)
    grid = torch.as_tensor(time_points, dtype=torch.float32, device=device)
    result = torch.zeros((len(X), len(time_points)), device=device)
    with torch.no_grad():
        for _ in range(n_samples):
            indices = torch.randint(0, len(all_mus), (len(X),), device=device)
            z = all_mus[indices] + all_stds[indices] * torch.randn_like(all_mus[indices])
            shape, scale, _, _ = model.decoder(x_tensor, z)
            result += torch.exp(-((grid[None, :] / scale[:, None]) ** shape[:, None]))
    return (result / n_samples).cpu().numpy()


def predict_survival_from_prior(model, X, time_points, n_samples=100, device="cpu"):
    model.eval().to(device)
    x = torch.as_tensor(X, dtype=torch.float32, device=device)
    grid = torch.as_tensor(time_points, dtype=torch.float32, device=device)
    result = torch.zeros((len(X), len(time_points)), device=device)
    with torch.no_grad():
        for _ in range(n_samples):
            z = torch.randn((len(X), model.latent_dim), device=device)
            shape, scale, _, _ = model.decoder(x, z)
            result += torch.exp(-((grid[None, :] / scale[:, None]) ** shape[:, None]))
    return (result / n_samples).cpu().numpy()


__all__ = [
    "predict_survival_curves", "predict_survival_from_prior",
    "get_median_survival_time",
]
