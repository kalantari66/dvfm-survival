"""DVFM survival prediction utilities."""
from __future__ import annotations
import numpy as np
import torch
from scipy.stats import kendalltau
from .reference_core import get_median_survival_time, predict_survival_curves


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


def posterior_diagnostics(model, loader, true_z, device="cpu", active_kl_threshold=0.01):
    if model.latent_dim == 0:
        return {"active_latent_dimensions": 0, "mean_kl": 0.0, "best_abs_z_pearson": np.nan}
    mus, logvars = [], []
    model.eval()
    with torch.no_grad():
        for x, t, e in loader:
            mu, logvar = model.encoder(x.to(device), t.to(device), e.to(device))
            mus.append(mu.cpu().numpy()); logvars.append(logvar.cpu().numpy())
    mu = np.concatenate(mus); lv = np.concatenate(logvars)
    per_dim_kl = np.mean(-0.5 * (1 + lv - mu ** 2 - np.exp(lv)), axis=0)
    correlations = [abs(float(np.corrcoef(mu[:, j], true_z)[0, 1])) for j in range(model.latent_dim)]
    return {"active_latent_dimensions": int(np.sum(per_dim_kl > active_kl_threshold)),
            "mean_kl": float(per_dim_kl.sum()), "best_abs_z_pearson": float(np.nanmax(correlations)),
            "kl_per_dimension": per_dim_kl, "posterior_mu": mu, "posterior_logvar": lv}


def learned_conditional_kendall_tau(model, x_reference, n_samples=2000, device="cpu", seed=0):
    rng = np.random.default_rng(seed)
    x = torch.as_tensor(np.repeat(np.asarray(x_reference)[None, :], n_samples, axis=0), dtype=torch.float32, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        z = torch.randn((n_samples, model.latent_dim), generator=generator, device=device)
        shape_e, scale_e, shape_c, scale_c = model.decoder(x, z)
        ue = torch.as_tensor(rng.uniform(size=n_samples), dtype=torch.float32, device=device)
        uc = torch.as_tensor(rng.uniform(size=n_samples), dtype=torch.float32, device=device)
        event_time = scale_e * (-torch.log(ue)) ** (1 / shape_e)
        censor_time = scale_c * (-torch.log(uc)) ** (1 / shape_c)
    return float(kendalltau(event_time.cpu().numpy(), censor_time.cpu().numpy()).statistic)


__all__ = ["predict_survival_curves", "predict_survival_from_prior", "get_median_survival_time", "posterior_diagnostics", "learned_conditional_kendall_tau"]
