"""Structural diagnostics for DVFM2."""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import kendalltau

from .model import SharedPrivateDVFM


def posterior_summary(model, loader, *, device="cpu"):
    device = torch.device(device)
    vals = {k: [] for k in ("mu_s", "mu_e", "mu_c", "logvar_s", "logvar_e", "logvar_c")}
    model.eval()
    with torch.no_grad():
        for x, time, event in loader:
            q = model.encode(x.to(device), time.to(device), event.to(device))
            for key in vals:
                vals[key].append(q[key].cpu().numpy())
    out = {}
    for key, pieces in vals.items():
        arr = np.concatenate(pieces, axis=0)
        out[f"{key}_mean"] = float(arr.mean())
        out[f"{key}_std"] = float(arr.std())
    return out


def learned_conditional_tau(
    model: SharedPrivateDVFM,
    x_fixed,
    *,
    n_samples=5000,
    device="cpu",
    seed=0,
):
    """Kendall tau(T,C|X=x), integrating all DVFM2 latent priors."""
    rng = np.random.default_rng(seed)
    device = torch.device(device)

    x = torch.as_tensor(
        np.asarray(x_fixed, dtype=np.float32),
        device=device,
    ).reshape(1, -1).expand(n_samples, -1)

    z_s = torch.as_tensor(
        rng.normal(size=(n_samples, model.shared_dim)).astype(np.float32),
        device=device,
    )
    z_e = torch.as_tensor(
        rng.normal(size=(n_samples, model.event_dim)).astype(np.float32),
        device=device,
    )
    z_c = torch.as_tensor(
        rng.normal(size=(n_samples, model.censor_dim)).astype(np.float32),
        device=device,
    )

    with torch.no_grad():
        shape_t, scale_t, shape_c, scale_c = model.decoder(x, z_s, z_e, z_c)

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
