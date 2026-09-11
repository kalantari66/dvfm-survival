"""Neural encoder, decoder, and likelihood for DVFM."""

from __future__ import annotations

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims=(64, 32), dropout=0.0):
        super().__init__()
        layers = []
        previous = input_dim + 2
        for width in hidden_dims:
            layers.extend((nn.Linear(previous, width), nn.ReLU(), nn.BatchNorm1d(width)))
            if float(dropout) > 0:
                layers.append(nn.Dropout(float(dropout)))
            previous = width
        self.network = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(previous, latent_dim)
        self.fc_logvar = nn.Linear(previous, latent_dim)

    def forward(self, x, time, event):
        hidden = self.network(torch.cat((x, time[:, None], event[:, None]), dim=1))
        return self.fc_mu(hidden), self.fc_logvar(hidden)


class Decoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims=(32, 64), dropout=0.0,
                 scale_link="softplus", latent_path="nonlinear",
                 shape_mode="conditional", latent_gate="none",
                 gate_initial_value=0.9, gate_temperature=0.67,
                 gate_stretch=(-0.1, 1.1)):
        super().__init__()
        if scale_link not in {"softplus", "exp"}:
            raise ValueError("scale_link must be 'softplus' or 'exp'")
        if latent_path not in {"nonlinear", "additive_scale"}:
            raise ValueError("latent_path must be 'nonlinear' or 'additive_scale'")
        if shape_mode not in {"conditional", "global"}:
            raise ValueError("shape_mode must be 'conditional' or 'global'")
        if latent_gate not in {"none", "hard_concrete", "sigmoid"}:
            raise ValueError("latent_gate must be 'none', 'hard_concrete', or 'sigmoid'")
        if not 0.0 < float(gate_initial_value) < 1.0:
            raise ValueError("gate_initial_value must be in (0, 1)")
        if float(gate_temperature) <= 0.0:
            raise ValueError("gate_temperature must be positive")
        self.scale_link = scale_link
        self.latent_path = latent_path
        self.shape_mode = shape_mode
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.latent_gate = latent_gate
        self.gate_temperature = float(gate_temperature)
        self.gate_lower, self.gate_upper = map(float, gate_stretch)
        if not self.gate_lower < 0.0 < 1.0 < self.gate_upper:
            raise ValueError("gate_stretch must extend below 0 and above 1")
        if latent_gate != "none" and latent_dim > 0:
            if latent_gate == "hard_concrete":
                initial = (float(gate_initial_value) - self.gate_lower) / (
                    self.gate_upper - self.gate_lower
                )
                initial = min(max(initial, 1e-6), 1.0 - 1e-6)
                initial_log_alpha = self.gate_temperature * torch.logit(
                    torch.tensor(initial)
                )
            else:
                initial_log_alpha = torch.logit(
                    torch.tensor(float(gate_initial_value))
                )
            self.gate_log_alpha = nn.Parameter(initial_log_alpha)
        else:
            self.register_parameter("gate_log_alpha", None)
        layers = []
        previous = input_dim + latent_dim if latent_path == "nonlinear" else input_dim
        for width in hidden_dims:
            layers.extend((nn.Linear(previous, width), nn.ReLU(), nn.BatchNorm1d(width)))
            if float(dropout) > 0:
                layers.append(nn.Dropout(float(dropout)))
            previous = width
        self.network = nn.Sequential(*layers)
        self.fc_params = nn.Linear(previous, 4)
        if latent_path == "additive_scale" and latent_dim > 0:
            self.latent_scale_loadings = nn.Parameter(torch.empty(latent_dim, 2))
            nn.init.normal_(self.latent_scale_loadings, mean=0.0, std=0.05)
        else:
            self.register_parameter("latent_scale_loadings", None)
        if shape_mode == "global":
            # Softplus^{-1}(1) initializes both positive shapes at one without
            # using knowledge of the synthetic DGP's true shapes.
            initial = torch.log(torch.expm1(torch.ones(2)))
            self.global_shape_unconstrained = nn.Parameter(initial)
        else:
            self.register_parameter("global_shape_unconstrained", None)

    def _positive_scale(self, value):
        if self.scale_link == "exp":
            return torch.exp(value.clamp(min=-12.0, max=12.0)) + 1e-6
        return nn.functional.softplus(value) + 1e-6

    def gate_value(self, stochastic=False):
        """Return the single shared latent gate, including hard clamping."""
        if self.latent_dim == 0:
            return self.fc_params.weight.new_tensor(0.0)
        if self.gate_log_alpha is None:
            return self.fc_params.weight.new_tensor(1.0)
        if self.latent_gate == "sigmoid":
            return torch.sigmoid(self.gate_log_alpha)
        logit = self.gate_log_alpha
        if stochastic:
            uniform = torch.rand((), device=logit.device, dtype=logit.dtype)
            uniform = uniform.clamp(1e-6, 1.0 - 1e-6)
            logit = logit + torch.log(uniform) - torch.log1p(-uniform)
        soft = torch.sigmoid(logit / self.gate_temperature)
        stretched = soft * (self.gate_upper - self.gate_lower) + self.gate_lower
        return stretched.clamp(0.0, 1.0)

    def latent_loading_l1(self):
        """L1 magnitude of decoder weights carrying z into both margin heads."""
        if self.latent_dim == 0:
            return self.fc_params.weight.new_tensor(0.0)
        if self.latent_scale_loadings is not None:
            return self.latent_scale_loadings.abs().mean()
        first_linear = next(layer for layer in self.network if isinstance(layer, nn.Linear))
        return first_linear.weight[:, self.input_dim:].abs().mean()

    def latent_loading_group_norm(self):
        """Dimension-normalized group norm for the complete latent-input block."""
        if self.latent_dim == 0:
            return self.fc_params.weight.new_tensor(0.0)
        if self.latent_scale_loadings is not None:
            weights = self.latent_scale_loadings
        else:
            first_linear = next(
                layer for layer in self.network if isinstance(layer, nn.Linear)
            )
            weights = first_linear.weight[:, self.input_dim:]
        return weights.square().mean().sqrt()

    def forward(self, x, z):
        if self.latent_dim > 0:
            z = z * self.gate_value(stochastic=self.training)
        decoder_input = torch.cat((x, z), dim=1) if self.latent_path == "nonlinear" else x
        parameters = self.fc_params(self.network(decoder_input))
        event_scale_raw = parameters[:, 1]
        censor_scale_raw = parameters[:, 3]
        if self.latent_scale_loadings is not None:
            latent_offsets = z @ self.latent_scale_loadings
            event_scale_raw = event_scale_raw + latent_offsets[:, 0]
            censor_scale_raw = censor_scale_raw + latent_offsets[:, 1]
        if self.global_shape_unconstrained is None:
            event_shape = nn.functional.softplus(parameters[:, 0]) + 1e-6
            censor_shape = nn.functional.softplus(parameters[:, 2]) + 1e-6
        else:
            shapes = nn.functional.softplus(self.global_shape_unconstrained) + 1e-6
            event_shape = shapes[0].expand(len(x))
            censor_shape = shapes[1].expand(len(x))
        return (
            event_shape,
            self._positive_scale(event_scale_raw),
            censor_shape,
            self._positive_scale(censor_scale_raw),
        )


class DVFM(nn.Module):
    def __init__(self, input_dim, latent_dim=8, encoder_hidden=(64, 32),
                 decoder_hidden=(32, 64), dropout=0.0,
                 encoder_dropout=None, decoder_dropout=None,
                 scale_link="softplus", latent_path="nonlinear",
                 shape_mode="conditional", latent_gate="none",
                 gate_initial_value=0.9, gate_temperature=0.67,
                 latent_loading_l1=0.1):
        super().__init__()
        if latent_dim < 0:
            raise ValueError("latent_dim must be nonnegative")
        if float(latent_loading_l1) < 0.0:
            raise ValueError("latent_loading_l1 must be nonnegative")
        encoder_dropout = dropout if encoder_dropout is None else encoder_dropout
        decoder_dropout = dropout if decoder_dropout is None else decoder_dropout
        self.encoder = None if latent_dim == 0 else Encoder(
            input_dim, latent_dim, encoder_hidden, encoder_dropout
        )
        self.decoder = Decoder(
            input_dim, latent_dim, decoder_hidden, decoder_dropout,
            scale_link=scale_link, latent_path=latent_path,
            shape_mode=shape_mode,
            latent_gate=latent_gate, gate_initial_value=gate_initial_value,
            gate_temperature=gate_temperature,
        )
        self.latent_dim = latent_dim
        # Alpha for the L1 penalty on decoder weights that carry the shared
        # latent into the event and censoring margins.  It lives on the model
        # so ordinary DVFM training uses the regularized method by default;
        # callers may still override it for explicit ablations.
        self.latent_loading_l1_alpha = float(latent_loading_l1)

    def regularization_terms(
        self, latent_loading_l1=None, latent_group_lasso=0.0, gate_l1=0.0
    ):
        """Return differentiable L1 penalties and their unweighted diagnostics."""
        loading = self.decoder.latent_loading_l1()
        group_norm = self.decoder.latent_loading_group_norm()
        gate = self.decoder.gate_value(stochastic=False)
        loading_alpha = (
            self.latent_loading_l1_alpha
            if latent_loading_l1 is None else float(latent_loading_l1)
        )
        penalty = (
            loading_alpha * loading
            + float(latent_group_lasso) * group_norm
            + float(gate_l1) * gate.abs()
        )
        return penalty, loading, group_norm, gate

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    @staticmethod
    def weibull_log_pdf(time, shape, scale):
        return (
            torch.log(shape) - torch.log(scale)
            + (shape - 1) * (torch.log(time) - torch.log(scale))
            - (time / scale).pow(shape)
        )

    @staticmethod
    def weibull_log_survival(time, shape, scale):
        return -(time / scale).pow(shape)

    def forward(self, x, time, event):
        if self.encoder is None:
            mu = x.new_empty((len(x), 0))
            logvar = x.new_empty((len(x), 0))
        else:
            mu, logvar = self.encoder(x, time, event)
        shape_t, scale_t, shape_c, scale_c = self.decoder(
            x, self.reparameterize(mu, logvar)
        )
        return shape_t, scale_t, shape_c, scale_c, mu, logvar

    def loss_function(self, shape_t, scale_t, shape_c, scale_c,
                      mu, logvar, time, event, beta=1.0, free_bits=0.0):
        log_f_t = self.weibull_log_pdf(time, shape_t, scale_t)
        log_s_t = self.weibull_log_survival(time, shape_t, scale_t)
        log_f_c = self.weibull_log_pdf(time, shape_c, scale_c)
        log_s_c = self.weibull_log_survival(time, shape_c, scale_c)
        reconstruction = (
            event * (log_f_t + log_s_c)
            + (1 - event) * (log_s_t + log_f_c)
        ).mean()
        kl_per_dimension = -0.5 * (
            1 + logvar - mu.square() - logvar.exp()
        )
        raw_kl = kl_per_dimension.sum(dim=1).mean()
        free_kl = torch.maximum(
            kl_per_dimension, kl_per_dimension.new_tensor(float(free_bits))
        ).sum(dim=1).mean()
        return -reconstruction + beta * free_kl, -reconstruction, raw_kl


__all__ = ["Encoder", "Decoder", "DVFM"]
