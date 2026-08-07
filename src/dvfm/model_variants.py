"""Model variants used by mechanistic DVFM experiments.

These are intentionally kept under dvfm because they are model definitions,
while experiment orchestration lives under src/experiments.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .reference_core import Decoder, Encoder

EPS = 1e-8

def _weibull_log_pdf(t, shape, scale):
    t = torch.clamp(t, min=EPS)
    shape = torch.clamp(shape, min=EPS)
    scale = torch.clamp(scale, min=EPS)
    return (
        torch.log(shape)
        - torch.log(scale)
        + (shape - 1.0) * (torch.log(t) - torch.log(scale))
        - (t / scale).pow(shape)
    )

def _weibull_log_survival(t, shape, scale):
    t = torch.clamp(t, min=EPS)
    shape = torch.clamp(shape, min=EPS)
    scale = torch.clamp(scale, min=EPS)
    return -(t / scale).pow(shape)

class NoLatentJointWeibull(nn.Module):
    """Matched x-only joint event/censoring Weibull model."""

    def __init__(self, input_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.decoder = Decoder(
            input_dim=input_dim,
            latent_dim=0,
            hidden_dims=hidden_dims,
        )
        self.latent_dim = 0

    def forward(self, x):
        z = x.new_empty((x.shape[0], 0))
        return self.decoder(x, z)

    def loss(self, x, time, event):
        shape_t, scale_t, shape_c, scale_c = self(x)
        ll = event * (
            _weibull_log_pdf(time, shape_t, scale_t)
            + _weibull_log_survival(time, shape_c, scale_c)
        ) + (1.0 - event) * (
            _weibull_log_survival(time, shape_t, scale_t)
            + _weibull_log_pdf(time, shape_c, scale_c)
        )
        return -ll.mean(), -ll.mean(), x.new_tensor(0.0)

class _SingleMarginDecoder(nn.Module):
    """One Weibull margin conditioned on x and its own latent."""

    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: list[int]):
        super().__init__()
        layers = []
        prev = input_dim + latent_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(prev, hidden), nn.ReLU(), nn.BatchNorm1d(hidden)])
            prev = hidden
        self.network = nn.Sequential(*layers)
        self.out = nn.Linear(prev, 2)

    def forward(self, x, z):
        h = self.network(torch.cat([x, z], dim=1))
        raw = self.out(h)
        shape = F.softplus(raw[:, 0]) + 1e-6
        scale = F.softplus(raw[:, 1]) + 1e-6
        return shape, scale

class SeparateLatentDVFM(nn.Module):
    """Negative-control DVFM with independent event and censoring latents.

    q_E(z_E | x,t,delta) and q_C(z_C | x,t,delta) are independent Gaussian
    posteriors with independent N(0,I) priors. Event parameters depend only on
    z_E and censoring parameters only on z_C. Consequently, at fixed x the
    generative model cannot induce event/censoring dependence through a shared
    latent variable.

    Each branch uses `latent_dim` dimensions. This intentionally gives the
    separate-latent control at least as much latent capacity as the shared model.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        encoder_hidden: list[int],
        decoder_hidden: list[int],
    ):
        super().__init__()
        if latent_dim < 1:
            raise ValueError("SeparateLatentDVFM requires latent_dim >= 1")
        self.latent_dim = int(latent_dim)
        self.event_encoder = Encoder(input_dim, latent_dim, encoder_hidden)
        self.censor_encoder = Encoder(input_dim, latent_dim, encoder_hidden)
        self.event_decoder = _SingleMarginDecoder(
            input_dim, latent_dim, decoder_hidden
        )
        self.censor_decoder = _SingleMarginDecoder(
            input_dim, latent_dim, decoder_hidden
        )

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def forward(self, x, time, event):
        mu_e, logvar_e = self.event_encoder(x, time, event)
        mu_c, logvar_c = self.censor_encoder(x, time, event)
        z_e = self.reparameterize(mu_e, logvar_e)
        z_c = self.reparameterize(mu_c, logvar_c)
        shape_t, scale_t = self.event_decoder(x, z_e)
        shape_c, scale_c = self.censor_decoder(x, z_c)
        return (
            shape_t, scale_t, shape_c, scale_c,
            mu_e, logvar_e, mu_c, logvar_c,
        )

    def loss_function(
        self,
        shape_t, scale_t, shape_c, scale_c,
        mu_e, logvar_e, mu_c, logvar_c,
        time, event, beta=1.0, free_bits=0.0,
    ):
        ll = event * (
            _weibull_log_pdf(time, shape_t, scale_t)
            + _weibull_log_survival(time, shape_c, scale_c)
        ) + (1.0 - event) * (
            _weibull_log_survival(time, shape_t, scale_t)
            + _weibull_log_pdf(time, shape_c, scale_c)
        )
        recon_nll = -ll.mean()

        def kl_terms(mu, logvar):
            per_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
            raw = per_dim.sum(dim=1).mean()
            if free_bits > 0:
                per_dim = torch.clamp(per_dim, min=float(free_bits))
            return per_dim.sum(dim=1).mean(), raw

        kl_e_free, kl_e = kl_terms(mu_e, logvar_e)
        kl_c_free, kl_c = kl_terms(mu_c, logvar_c)
        kl_free = kl_e_free + kl_c_free
        kl_raw = kl_e + kl_c
        return recon_nll + beta * kl_free, recon_nll, kl_raw
