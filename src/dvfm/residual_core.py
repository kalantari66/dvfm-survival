"""Residual-decoder DVFM.

This module preserves the original DVFM encoder, Gaussian posterior, Gaussian
prior, reparameterization, Weibull likelihood, and C-ELBO. Only the decoder is
changed.

Raw Weibull parameters are

    eta(x, z) = eta_base(x) + alpha * [r(x, z) - r(x, 0)],

where separate learned alpha values are used for event and censoring outputs.
The centering guarantees that decoder(x, z=0) is exactly the x-only base model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .reference_core import DVFM, Encoder


class _MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous, hidden),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden),
                ]
            )
            previous = hidden
        layers.append(nn.Linear(previous, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class ResidualDecoder(nn.Module):
    """x-only Weibull decoder with a centered latent residual correction."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: list[int] | None = None,
        residual_logit_init: float = -2.0,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 64]
        if latent_dim < 1:
            raise ValueError("ResidualDecoder requires latent_dim >= 1.")

        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)

        # Strong baseline pathway: all four raw Weibull parameters come from x.
        self.base_network = _MLP(
            input_dim=self.input_dim,
            hidden_dims=list(hidden_dims),
            output_dim=4,
        )

        # Latent correction pathway. It receives x and z, but is explicitly
        # centered by subtracting its output at z=0.
        self.residual_network = _MLP(
            input_dim=self.input_dim + self.latent_dim,
            hidden_dims=list(hidden_dims),
            output_dim=4,
        )

        # One gate for the event parameters (shape_T, scale_T) and one for the
        # censoring parameters (shape_C, scale_C).
        self.event_latent_scale = nn.Parameter(
            torch.tensor(float(residual_logit_init))
        )
        self.censor_latent_scale = nn.Parameter(
            torch.tensor(float(residual_logit_init))
        )

    @property
    def alpha_event(self) -> torch.Tensor:
        return torch.sigmoid(self.event_latent_scale)

    @property
    def alpha_censor(self) -> torch.Tensor:
        return torch.sigmoid(self.censor_latent_scale)

    def raw_parameters(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(
                f"Expected z with shape (batch, {self.latent_dim}), "
                f"received {tuple(z.shape)}."
            )

        base = self.base_network(x)
        zero_z = torch.zeros_like(z)

        residual_at_z = self.residual_network(torch.cat([x, z], dim=1))
        residual_at_zero = self.residual_network(
            torch.cat([x, zero_z], dim=1)
        )
        centered_residual = residual_at_z - residual_at_zero

        gates = torch.stack(
            [
                self.alpha_event,
                self.alpha_event,
                self.alpha_censor,
                self.alpha_censor,
            ]
        ).reshape(1, 4)

        combined = base + gates * centered_residual
        return combined, base, centered_residual

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw, _, _ = self.raw_parameters(x, z)

        shape_t = F.softplus(raw[:, 0]) + 1e-6
        scale_t = F.softplus(raw[:, 1]) + 1e-6
        shape_c = F.softplus(raw[:, 2]) + 1e-6
        scale_c = F.softplus(raw[:, 3]) + 1e-6
        return shape_t, scale_t, shape_c, scale_c


class ResidualDVFM(DVFM):
    """Original DVFM with only its decoder replaced."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 1,
        encoder_hidden: list[int] | None = None,
        decoder_hidden: list[int] | None = None,
        residual_logit_init: float = -2.0,
    ) -> None:
        nn.Module.__init__(self)
        if encoder_hidden is None:
            encoder_hidden = [64, 32]
        if decoder_hidden is None:
            decoder_hidden = [32, 64]

        self.encoder = Encoder(
            input_dim,
            latent_dim,
            list(encoder_hidden),
        )
        self.decoder = ResidualDecoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=list(decoder_hidden),
            residual_logit_init=residual_logit_init,
        )
        self.latent_dim = int(latent_dim)

    def decoder_diagnostics(self) -> dict[str, float]:
        return {
            "event_latent_alpha": float(
                self.decoder.alpha_event.detach().cpu().item()
            ),
            "censor_latent_alpha": float(
                self.decoder.alpha_censor.detach().cpu().item()
            ),
            "event_latent_logit": float(
                self.decoder.event_latent_scale.detach().cpu().item()
            ),
            "censor_latent_logit": float(
                self.decoder.censor_latent_scale.detach().cpu().item()
            ),
        }
