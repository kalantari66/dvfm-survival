"""DVFM2 model: shared/private latent decomposition with structural semantics.

Generative structure
--------------------

    z_s ~ N(0, I)       shared latent
    z_e ~ N(0, I)       event-private latent
    z_c ~ N(0, I)       censor-private latent

    T | x,z_s,z_e ~ Weibull(k_T(x), lambda_T(x,z_s,z_e))
    C | x,z_s,z_c ~ Weibull(k_C(x), lambda_C(x,z_s,z_c))

Only z_s is allowed to enter both margins.

The latent variables do NOT arbitrarily control all Weibull parameters.
Instead, x defines the baseline Weibull shape and log-scale, while latents
provide centered additive residuals on log-scale:

    log lambda_T = log lambda_T,0(x) + r_sT(z_s) + r_e(z_e)
    log lambda_C = log lambda_C,0(x) + r_sC(z_s) + r_c(z_c)

Each residual is exactly zero at z=0 by construction.

For the shared effect, the default implementation first computes one common
scalar score h_s(z_s). Margin-specific loadings then map that score into event
and censoring log-scale effects. This is substantially more restrictive than
giving each margin an arbitrary MLP over the whole latent vector.

If shared_same_sign=True, both shared loadings are constrained positive. This
is useful for a positive shared-frailty benchmark because it rules out the
pathological opposite-sign dependence solution observed with the original
DVFM. Set it False for experiments where negative dependence must be possible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


def _mlp(
    input_dim: int,
    hidden_dims: list[int],
    output_dim: int,
    *,
    activation: type[nn.Module] = nn.ReLU,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = int(input_dim)
    for hidden in hidden_dims:
        layers.extend([nn.Linear(prev, int(hidden)), activation()])
        prev = int(hidden)
    layers.append(nn.Linear(prev, int(output_dim)))
    return nn.Sequential(*layers)


class SharedPrivateEncoder(nn.Module):
    """One observation encoder with distinct posterior heads for z_s, z_e, z_c."""

    def __init__(
        self,
        input_dim: int,
        shared_dim: int = 1,
        event_dim: int = 1,
        censor_dim: int = 1,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        hidden_dims = [64, 32] if hidden_dims is None else list(hidden_dims)
        if min(shared_dim, event_dim, censor_dim) < 1:
            raise ValueError("All DVFM2 latent dimensions must be >= 1.")

        self.shared_dim = int(shared_dim)
        self.event_dim = int(event_dim)
        self.censor_dim = int(censor_dim)

        trunk_dim = hidden_dims[-1] if hidden_dims else input_dim + 2
        self.trunk = (
            _mlp(input_dim + 2, hidden_dims[:-1], hidden_dims[-1])
            if hidden_dims
            else nn.Identity()
        )

        self.mu_s = nn.Linear(trunk_dim, self.shared_dim)
        self.logvar_s = nn.Linear(trunk_dim, self.shared_dim)
        self.mu_e = nn.Linear(trunk_dim, self.event_dim)
        self.logvar_e = nn.Linear(trunk_dim, self.event_dim)
        self.mu_c = nn.Linear(trunk_dim, self.censor_dim)
        self.logvar_c = nn.Linear(trunk_dim, self.censor_dim)

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        event: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        obs = torch.cat([x, time[:, None], event[:, None]], dim=1)
        h = self.trunk(obs)
        return {
            "mu_s": self.mu_s(h),
            "logvar_s": self.logvar_s(h),
            "mu_e": self.mu_e(h),
            "logvar_e": self.logvar_e(h),
            "mu_c": self.mu_c(h),
            "logvar_c": self.logvar_c(h),
        }


class _BaselineMargin(nn.Module):
    """x-only Weibull shape and baseline log-scale."""

    def __init__(self, input_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        self.network = _mlp(input_dim, hidden_dims, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.network(x)
        shape = F.softplus(raw[:, 0]) + 1e-4
        log_scale = raw[:, 1]
        return shape, log_scale


class _CenteredLatentEffect(nn.Module):
    """Scalar residual r(z)-r(0), hence exactly zero at z=0."""

    def __init__(self, latent_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.network = _mlp(self.latent_dim, hidden_dims, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(
                f"Expected latent shape (batch,{self.latent_dim}); got {tuple(z.shape)}."
            )
        zero = torch.zeros_like(z)
        return (self.network(z) - self.network(zero)).squeeze(-1)


class SharedPrivateDecoder(nn.Module):
    """Structurally separated event/censoring Weibull decoder."""

    def __init__(
        self,
        input_dim: int,
        shared_dim: int = 1,
        event_dim: int = 1,
        censor_dim: int = 1,
        baseline_hidden: list[int] | None = None,
        latent_hidden: list[int] | None = None,
        shared_same_sign: bool = False,
        shared_loading_init: float = 0.5,
    ) -> None:
        super().__init__()
        baseline_hidden = [32, 64] if baseline_hidden is None else list(baseline_hidden)
        latent_hidden = [] if latent_hidden is None else list(latent_hidden)

        self.shared_dim = int(shared_dim)
        self.event_dim = int(event_dim)
        self.censor_dim = int(censor_dim)
        self.shared_same_sign = bool(shared_same_sign)

        # Separate x-only margin baselines.
        self.event_base = _BaselineMargin(input_dim, baseline_hidden)
        self.censor_base = _BaselineMargin(input_dim, baseline_hidden)

        # A single common shared score is the ONLY latent path touching both margins.
        self.shared_effect = _CenteredLatentEffect(self.shared_dim, latent_hidden)

        # Private paths are margin-exclusive by construction.
        self.event_private_effect = _CenteredLatentEffect(self.event_dim, latent_hidden)
        self.censor_private_effect = _CenteredLatentEffect(self.censor_dim, latent_hidden)

        init = torch.tensor(float(shared_loading_init))
        if self.shared_same_sign:
            # Stored in inverse-softplus-ish unconstrained space; exact init is not critical.
            raw = torch.log(torch.expm1(torch.clamp(init, min=1e-4)))
            self.raw_shared_loading_event = nn.Parameter(raw.clone())
            self.raw_shared_loading_censor = nn.Parameter(raw.clone())
        else:
            self.raw_shared_loading_event = nn.Parameter(init.clone())
            self.raw_shared_loading_censor = nn.Parameter(init.clone())

        # Private effects have independent signed loadings.
        self.event_private_loading = nn.Parameter(torch.tensor(0.25))
        self.censor_private_loading = nn.Parameter(torch.tensor(0.25))

    @property
    def shared_loading_event(self) -> torch.Tensor:
        if self.shared_same_sign:
            return F.softplus(self.raw_shared_loading_event)
        return self.raw_shared_loading_event

    @property
    def shared_loading_censor(self) -> torch.Tensor:
        if self.shared_same_sign:
            return F.softplus(self.raw_shared_loading_censor)
        return self.raw_shared_loading_censor

    def event_parameters(
        self,
        x: torch.Tensor,
        z_s: torch.Tensor,
        z_e: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape, base_log_scale = self.event_base(x)
        shared = self.shared_loading_event * self.shared_effect(z_s)
        private = self.event_private_loading * self.event_private_effect(z_e)
        log_scale = torch.clamp(base_log_scale + shared + private, -20.0, 20.0)
        return shape, torch.exp(log_scale)

    def censor_parameters(
        self,
        x: torch.Tensor,
        z_s: torch.Tensor,
        z_c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape, base_log_scale = self.censor_base(x)
        shared = self.shared_loading_censor * self.shared_effect(z_s)
        private = self.censor_private_loading * self.censor_private_effect(z_c)
        log_scale = torch.clamp(base_log_scale + shared + private, -20.0, 20.0)
        return shape, torch.exp(log_scale)

    def forward(
        self,
        x: torch.Tensor,
        z_s: torch.Tensor,
        z_e: torch.Tensor,
        z_c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        shape_t, scale_t = self.event_parameters(x, z_s, z_e)
        shape_c, scale_c = self.censor_parameters(x, z_s, z_c)
        return shape_t, scale_t, shape_c, scale_c


@dataclass
class DVFM2Output:
    shape_t: torch.Tensor
    scale_t: torch.Tensor
    shape_c: torch.Tensor
    scale_c: torch.Tensor
    mu_s: torch.Tensor
    logvar_s: torch.Tensor
    mu_e: torch.Tensor
    logvar_e: torch.Tensor
    mu_c: torch.Tensor
    logvar_c: torch.Tensor
    z_s: torch.Tensor
    z_e: torch.Tensor
    z_c: torch.Tensor


class SharedPrivateDVFM(nn.Module):
    """DVFM2 with explicit shared/event-private/censor-private latent semantics."""

    def __init__(
        self,
        input_dim: int,
        shared_dim: int = 1,
        event_dim: int = 1,
        censor_dim: int = 1,
        encoder_hidden: list[int] | None = None,
        decoder_hidden: list[int] | None = None,
        latent_hidden: list[int] | None = None,
        shared_same_sign: bool = False,
    ) -> None:
        super().__init__()
        self.shared_dim = int(shared_dim)
        self.event_dim = int(event_dim)
        self.censor_dim = int(censor_dim)
        self.latent_dim = self.shared_dim + self.event_dim + self.censor_dim

        self.encoder = SharedPrivateEncoder(
            input_dim=input_dim,
            shared_dim=self.shared_dim,
            event_dim=self.event_dim,
            censor_dim=self.censor_dim,
            hidden_dims=encoder_hidden,
        )
        self.decoder = SharedPrivateDecoder(
            input_dim=input_dim,
            shared_dim=self.shared_dim,
            event_dim=self.event_dim,
            censor_dim=self.censor_dim,
            baseline_hidden=decoder_hidden,
            latent_hidden=latent_hidden,
            shared_same_sign=shared_same_sign,
        )

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def encode(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        event: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return self.encoder(x, time, event)

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        event: torch.Tensor,
    ) -> DVFM2Output:
        q = self.encode(x, time, event)
        z_s = self.reparameterize(q["mu_s"], q["logvar_s"])
        z_e = self.reparameterize(q["mu_e"], q["logvar_e"])
        z_c = self.reparameterize(q["mu_c"], q["logvar_c"])

        shape_t, scale_t, shape_c, scale_c = self.decoder(x, z_s, z_e, z_c)
        return DVFM2Output(
            shape_t=shape_t,
            scale_t=scale_t,
            shape_c=shape_c,
            scale_c=scale_c,
            z_s=z_s,
            z_e=z_e,
            z_c=z_c,
            **q,
        )

    @staticmethod
    def _weibull_log_pdf(
        t: torch.Tensor,
        shape: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        t = torch.clamp(t, min=EPS)
        shape = torch.clamp(shape, min=EPS)
        scale = torch.clamp(scale, min=EPS)
        return (
            torch.log(shape)
            - torch.log(scale)
            + (shape - 1.0) * (torch.log(t) - torch.log(scale))
            - (t / scale).pow(shape)
        )

    @staticmethod
    def _weibull_log_survival(
        t: torch.Tensor,
        shape: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        t = torch.clamp(t, min=EPS)
        shape = torch.clamp(shape, min=EPS)
        scale = torch.clamp(scale, min=EPS)
        return -(t / scale).pow(shape)

    @staticmethod
    def _kl(
        mu: torch.Tensor,
        logvar: torch.Tensor,
        free_bits: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        per_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
        raw = per_dim.sum(dim=1).mean()
        if free_bits > 0:
            per_dim = torch.clamp(per_dim, min=float(free_bits))
        penalized = per_dim.sum(dim=1).mean()
        return penalized, raw

    def loss_function(
        self,
        output: DVFM2Output,
        time: torch.Tensor,
        event: torch.Tensor,
        *,
        beta_shared: float = 1.0,
        beta_private: float = 2.0,
        free_bits_shared: float = 0.0,
        free_bits_private: float = 0.0,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """C-ELBO with private latent capacity penalized more strongly.

        beta_private > beta_shared is intentional: duplicating dependence signal
        independently in z_e and z_c should cost more than representing it once
        through z_s.
        """
        ll = event * (
            self._weibull_log_pdf(time, output.shape_t, output.scale_t)
            + self._weibull_log_survival(time, output.shape_c, output.scale_c)
        ) + (1.0 - event) * (
            self._weibull_log_survival(time, output.shape_t, output.scale_t)
            + self._weibull_log_pdf(time, output.shape_c, output.scale_c)
        )
        recon_nll = -ll.mean()

        kl_s_free, kl_s = self._kl(
            output.mu_s, output.logvar_s, free_bits_shared
        )
        kl_e_free, kl_e = self._kl(
            output.mu_e, output.logvar_e, free_bits_private
        )
        kl_c_free, kl_c = self._kl(
            output.mu_c, output.logvar_c, free_bits_private
        )

        kl_private_free = kl_e_free + kl_c_free
        loss = (
            recon_nll
            + float(beta_shared) * kl_s_free
            + float(beta_private) * kl_private_free
        )

        metrics = {
            "reconstruction_nll": recon_nll,
            "kl_shared": kl_s,
            "kl_event_private": kl_e,
            "kl_censor_private": kl_c,
            "kl_private_total": kl_e + kl_c,
            "kl_total": kl_s + kl_e + kl_c,
        }
        return loss, metrics
