"""DVFM with an explicit scalar shared-frailty pathway.

The encoder remains q_phi(z | x, y, delta).  The decoder differs from the
reference implementation: x determines the baseline Weibull shapes/scales,
while z can only multiply the event and censoring hazards through learned
non-negative loadings.

For a Weibull baseline
    S_0(t|x) = exp(-(t / lambda(x)) ** k(x)),
and hazard multiplier exp(a z),
    S(t|x,z) = exp(-exp(a z) * (t / lambda(x)) ** k(x)).
This is exactly another Weibull with
    lambda_eff(x,z) = lambda(x) * exp(-a z / k(x)).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .reference_core import Encoder


@dataclass(frozen=True)
class WeibullBounds:
    min_shape: float = 0.25
    max_shape: float = 8.0
    min_scale: float = 1e-3
    max_scale: float = 5.0

    def validate(self) -> None:
        if not 0.0 < self.min_shape < self.max_shape:
            raise ValueError("Require 0 < min_shape < max_shape.")
        if not 0.0 < self.min_scale < self.max_scale:
            raise ValueError("Require 0 < min_scale < max_scale.")


def _bounded_sigmoid(raw: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    return lower + (upper - lower) * torch.sigmoid(raw)


class SharedFrailtyDecoder(nn.Module):
    """Weibull decoder with x-only marginals and an explicit shared z pathway.

    x -> baseline k_T(x), lambda_T(x), k_C(x), lambda_C(x)
    z -> hazard multipliers exp(a_T z), exp(a_C z)

    The scalar z cannot alter Weibull shapes and cannot enter arbitrary hidden
    layers. Positive loadings make the same latent direction increase both
    hazards, inducing positive event-censoring dependence after marginalizing z.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 1,
        hidden_dims: Sequence[int] = (32, 64),
        bounds: WeibullBounds | None = None,
        min_loading: float = 0.0,
        max_loading: float = 3.0,
        loading_init: float = 0.5,
    ) -> None:
        super().__init__()
        if latent_dim != 1:
            raise ValueError("SharedFrailtyDecoder currently requires latent_dim=1.")
        if not 0.0 <= min_loading < max_loading:
            raise ValueError("Require 0 <= min_loading < max_loading.")

        self.latent_dim = latent_dim
        self.bounds = bounds or WeibullBounds()
        self.bounds.validate()
        self.min_loading = float(min_loading)
        self.max_loading = float(max_loading)

        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_dims:
            layers.extend([nn.Linear(previous, int(width)), nn.ReLU(), nn.BatchNorm1d(int(width))])
            previous = int(width)
        self.network = nn.Sequential(*layers)
        self.fc_baseline = nn.Linear(previous, 4)

        fraction = (float(loading_init) - self.min_loading) / (
            self.max_loading - self.min_loading
        )
        fraction = min(max(fraction, 1e-4), 1.0 - 1e-4)
        raw_init = torch.logit(torch.tensor(fraction, dtype=torch.float32))
        self.raw_event_loading = nn.Parameter(raw_init.clone())
        self.raw_censor_loading = nn.Parameter(raw_init.clone())

    @property
    def event_loading(self) -> torch.Tensor:
        return _bounded_sigmoid(
            self.raw_event_loading, self.min_loading, self.max_loading
        )

    @property
    def censor_loading(self) -> torch.Tensor:
        return _bounded_sigmoid(
            self.raw_censor_loading, self.min_loading, self.max_loading
        )

    def baseline_parameters(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.network(x)
        raw = self.fc_baseline(hidden)
        b = self.bounds
        shape_t = _bounded_sigmoid(raw[:, 0], b.min_shape, b.max_shape)
        scale_t = _bounded_sigmoid(raw[:, 1], b.min_scale, b.max_scale)
        shape_c = _bounded_sigmoid(raw[:, 2], b.min_shape, b.max_shape)
        scale_c = _bounded_sigmoid(raw[:, 3], b.min_scale, b.max_scale)
        return shape_t, scale_t, shape_c, scale_c

    def forward(
        self, x: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if z.ndim != 2 or z.shape[1] != 1:
            raise ValueError(f"Expected z with shape (batch, 1), got {tuple(z.shape)}")

        shape_t, base_scale_t, shape_c, base_scale_c = self.baseline_parameters(x)
        z_scalar = z[:, 0]

        # Exact Weibull PH transformation:
        # exp(a z) * (t/lambda)^k = (t/[lambda exp(-a z/k)])^k
        log_scale_t = torch.log(base_scale_t) - self.event_loading * z_scalar / shape_t
        log_scale_c = torch.log(base_scale_c) - self.censor_loading * z_scalar / shape_c

        # Effective scales may leave the baseline range because frailty is a
        # multiplicative hazard effect. Clamp only for numerical stability.
        scale_t = torch.exp(torch.clamp(log_scale_t, min=-12.0, max=12.0))
        scale_c = torch.exp(torch.clamp(log_scale_c, min=-12.0, max=12.0))
        return shape_t, scale_t, shape_c, scale_c

    def diagnostics(self) -> dict[str, float]:
        return {
            "event_frailty_loading": float(self.event_loading.detach().cpu()),
            "censor_frailty_loading": float(self.censor_loading.detach().cpu()),
            "loading_product": float(
                (self.event_loading * self.censor_loading).detach().cpu()
            ),
        }


class SharedFrailtyDVFM(nn.Module):
    """DVFM retaining the reference encoder/loss with constrained decoder."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 1,
        encoder_hidden: Sequence[int] = (64, 32),
        decoder_hidden: Sequence[int] = (32, 64),
        bounds: WeibullBounds | None = None,
        min_loading: float = 0.0,
        max_loading: float = 3.0,
        loading_init: float = 0.5,
    ) -> None:
        super().__init__()
        if latent_dim != 1:
            raise ValueError("SharedFrailtyDVFM currently requires latent_dim=1.")
        self.encoder = Encoder(input_dim, latent_dim, list(encoder_hidden))
        self.decoder = SharedFrailtyDecoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=decoder_hidden,
            bounds=bounds,
            min_loading=min_loading,
            max_loading=max_loading,
            loading_init=loading_init,
        )
        self.latent_dim = latent_dim

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    @staticmethod
    def weibull_log_pdf(
        t: torch.Tensor, shape: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        t = torch.clamp(t, min=1e-8)
        scale = torch.clamp(scale, min=1e-8)
        return (
            torch.log(shape)
            - torch.log(scale)
            + (shape - 1.0) * (torch.log(t) - torch.log(scale))
            - (t / scale).pow(shape)
        )

    @staticmethod
    def weibull_log_survival(
        t: torch.Tensor, shape: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        return -(torch.clamp(t, min=0.0) / torch.clamp(scale, min=1e-8)).pow(shape)

    def forward(
        self, x: torch.Tensor, time: torch.Tensor, event: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        mu, logvar = self.encoder(x, time, event)
        z = self.reparameterize(mu, logvar)
        shape_t, scale_t, shape_c, scale_c = self.decoder(x, z)
        return shape_t, scale_t, shape_c, scale_c, mu, logvar

    def loss_function(
        self,
        shape_t: torch.Tensor,
        scale_t: torch.Tensor,
        shape_c: torch.Tensor,
        scale_c: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        time: torch.Tensor,
        event: torch.Tensor,
        beta: float = 1.0,
        free_bits: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_f_t = self.weibull_log_pdf(time, shape_t, scale_t)
        log_s_t = self.weibull_log_survival(time, shape_t, scale_t)
        log_f_c = self.weibull_log_pdf(time, shape_c, scale_c)
        log_s_c = self.weibull_log_survival(time, shape_c, scale_c)

        observed_log_likelihood = event * (log_f_t + log_s_c) + (
            1.0 - event
        ) * (log_s_t + log_f_c)
        reconstruction_nll = -observed_log_likelihood.mean()

        kl_per_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
        raw_kl = kl_per_dim.sum(dim=1).mean()
        if free_bits > 0.0:
            kl_for_loss = torch.clamp(kl_per_dim, min=float(free_bits)).sum(dim=1).mean()
        else:
            kl_for_loss = raw_kl
        loss = reconstruction_nll + float(beta) * kl_for_loss
        return loss, reconstruction_nll, raw_kl
