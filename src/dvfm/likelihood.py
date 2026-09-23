"""Held-out observed-data log-likelihood estimates for a fitted DVFM.

For ``latent_dim == 0`` every quantity is the exact log-likelihood. For a
latent model the joint ELBO and IWAE are lower bounds on log p(t, delta | x),
and the Monte Carlo margin estimates are downward biased (Jensen) at finite
sample size, so a latent model's win against the exact latent-free
likelihood is conservative.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .model import DVFM


def _margin_log_terms(model, x, time, event, z, beyond=None):
    """Per-subject event and censoring log contributions given ``z``.

    ``beyond`` marks subjects still event- and censoring-free at the horizon
    (``time`` is then the horizon): both margins contribute a survival term.
    """
    shape_t, scale_t, shape_c, scale_c = model.decoder(x, z)
    # Select rather than weight by event: an overflowing unused term would
    # otherwise turn 0 * inf into NaN instead of dropping out.
    observed = event > 0.5
    log_s_t = DVFM.weibull_log_survival(time, shape_t, scale_t)
    log_s_c = DVFM.weibull_log_survival(time, shape_c, scale_c)
    event_term = torch.where(observed, DVFM.weibull_log_pdf(time, shape_t, scale_t), log_s_t)
    censor_term = torch.where(observed, log_s_c, DVFM.weibull_log_pdf(time, shape_c, scale_c))
    if beyond is not None:
        event_term = torch.where(beyond, log_s_t, event_term)
        censor_term = torch.where(beyond, log_s_c, censor_term)
    return event_term, censor_term


def _repeat(tensor, count):
    return tensor.unsqueeze(0).expand(count, *tensor.shape).reshape(-1, *tensor.shape[1:])


def encode_posterior(model, X, time, event, *, batch_size=512, device="cpu"):
    """Return posterior means and standard deviations, shape (n, latent_dim)."""
    if model.encoder is None:
        return np.empty((len(X), 0)), np.empty((len(X), 0))
    model.eval().to(device)
    means, stds = [], []
    with torch.no_grad():
        for start in range(0, len(X), int(batch_size)):
            stop = start + int(batch_size)
            mu, logvar = model.encoder(
                torch.as_tensor(X[start:stop], dtype=torch.float32, device=device),
                torch.as_tensor(time[start:stop], dtype=torch.float32, device=device),
                torch.as_tensor(event[start:stop], dtype=torch.float32, device=device),
            )
            means.append(mu.cpu().numpy())
            stds.append(torch.exp(0.5 * logvar).cpu().numpy())
    return np.concatenate(means), np.concatenate(stds)


def heldout_log_likelihood(
    model, X, time, event, *, mixing_mu, mixing_std, n_samples=1000,
    time_scale=1.0, horizon=None, max_rows=200_000, seed=0, device="cpu",
) -> dict[str, np.ndarray]:
    """Per-subject held-out log-likelihood estimates in natural time units.

    ``time`` is in model units, i.e. natural time divided by ``time_scale``.
    Every density term then carries a ``-log(time_scale)`` Jacobian, which is
    applied here so results are comparable across splits with different
    training median durations.

    ``joint_elbo`` and ``joint_iwae`` use the encoder q(z | x, t, delta) and
    the N(0, I) prior. ``event_margin`` and ``censor_margin`` integrate each
    margin over the aggregate training posterior (``mixing_mu``,
    ``mixing_std``), the same mixing distribution ``predict_survival_curves``
    uses; ``event_margin`` is therefore the likelihood of the reported
    marginal event survival curve.

    ``horizon`` (model units) administratively censors the evaluation: a
    subject with ``time > horizon`` contributes log P(T > h, C > h | x) rather
    than a density, and the encoder sees only that coarsened observation.
    """
    model.eval().to(device)
    n_subjects = len(X)
    time = np.asarray(time, dtype=float)
    beyond_np = np.zeros(n_subjects, dtype=bool) if horizon is None else time > float(horizon)
    if horizon is not None:
        time = np.minimum(time, float(horizon))
    event_np = np.where(beyond_np, 0.0, np.asarray(event, dtype=float))
    density_jacobian = math.log(float(time_scale))
    output = {key: np.empty(n_subjects) for key in (
        "joint_elbo", "joint_iwae", "event_margin", "censor_margin",
    )}
    samples = 1 if model.latent_dim == 0 else int(n_samples)
    batch_size = max(1, int(max_rows) // samples)
    mixing_mu_t = torch.as_tensor(mixing_mu, dtype=torch.float32, device=device)
    mixing_std_t = torch.as_tensor(mixing_std, dtype=torch.float32, device=device)
    # Own RNG stream: likelihood evaluation must not shift later MC draws.
    with torch.random.fork_rng(devices=[device] if torch.device(device).type == "cuda" else []):
        torch.manual_seed(int(seed))
        with torch.no_grad():
            for start in range(0, n_subjects, batch_size):
                stop = min(start + batch_size, n_subjects)
                x = torch.as_tensor(X[start:stop], dtype=torch.float32, device=device)
                t = torch.as_tensor(time[start:stop], dtype=torch.float32, device=device)
                e = torch.as_tensor(event_np[start:stop], dtype=torch.float32, device=device)
                b = torch.as_tensor(beyond_np[start:stop], device=device)
                count = stop - start
                if model.latent_dim == 0:
                    z = x.new_empty((count, 0))
                    event_term, censor_term = _margin_log_terms(model, x, t, e, z, b)
                    joint = event_term + censor_term
                    values = {
                        "joint_elbo": joint, "joint_iwae": joint,
                        "event_margin": event_term, "censor_margin": censor_term,
                    }
                else:
                    x_rep, t_rep, e_rep, b_rep = (_repeat(v, samples) for v in (x, t, e, b))
                    mu, logvar = model.encoder(x, t, e)
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn((samples, count, model.latent_dim), device=device)
                    z = mu.unsqueeze(0) + std.unsqueeze(0) * eps
                    event_term, censor_term = _margin_log_terms(
                        model, x_rep, t_rep, e_rep, z.reshape(-1, model.latent_dim), b_rep
                    )
                    reconstruction = (event_term + censor_term).reshape(samples, count)
                    kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(dim=1)
                    log_prior = torch.distributions.Normal(0.0, 1.0).log_prob(z).sum(-1)
                    log_q = torch.distributions.Normal(mu, std).log_prob(z).sum(-1)
                    log_k = math.log(samples)
                    index = torch.randint(0, len(mixing_mu_t), (samples * count,), device=device)
                    z_mix = mixing_mu_t[index] + mixing_std_t[index] * torch.randn(
                        (samples * count, model.latent_dim), device=device
                    )
                    event_mix, censor_mix = _margin_log_terms(
                        model, x_rep, t_rep, e_rep, z_mix, b_rep
                    )
                    values = {
                        "joint_elbo": reconstruction.mean(dim=0) - kl,
                        "joint_iwae": torch.logsumexp(
                            reconstruction + log_prior - log_q, dim=0
                        ) - log_k,
                        "event_margin": torch.logsumexp(
                            event_mix.reshape(samples, count), dim=0
                        ) - log_k,
                        "censor_margin": torch.logsumexp(
                            censor_mix.reshape(samples, count), dim=0
                        ) - log_k,
                    }
                for key, value in values.items():
                    output[key][start:stop] = value.cpu().numpy()
    # Each subject inside the horizon has exactly one density term; a subject
    # beyond it has only survival terms, which need no Jacobian.
    inside = (~beyond_np).astype(float)
    output["joint_elbo"] -= inside * density_jacobian
    output["joint_iwae"] -= inside * density_jacobian
    output["event_margin"] -= inside * event_np * density_jacobian
    output["censor_margin"] -= inside * (1.0 - event_np) * density_jacobian
    return output


def event_martingale_residual(model, X, time, event, *, device="cpu") -> np.ndarray:
    """delta - Lambda_T(t | x) for a latent-free model's event margin."""
    if model.latent_dim != 0:
        raise ValueError("Martingale residuals are defined here for latent_dim = 0 only")
    model.eval().to(device)
    with torch.no_grad():
        x = torch.as_tensor(X, dtype=torch.float32, device=device)
        t = torch.as_tensor(time, dtype=torch.float32, device=device)
        shape, scale, _, _ = model.decoder(x, x.new_empty((len(x), 0)))
        cumulative_hazard = (t / scale).pow(shape)
    return np.asarray(event, dtype=float) - cumulative_hazard.cpu().numpy()


__all__ = ["encode_posterior", "event_martingale_residual", "heldout_log_likelihood"]
