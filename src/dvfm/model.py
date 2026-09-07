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
    def __init__(self, input_dim, latent_dim, hidden_dims=(32, 64), dropout=0.0):
        super().__init__()
        layers = []
        previous = input_dim + latent_dim
        for width in hidden_dims:
            layers.extend((nn.Linear(previous, width), nn.ReLU(), nn.BatchNorm1d(width)))
            if float(dropout) > 0:
                layers.append(nn.Dropout(float(dropout)))
            previous = width
        self.network = nn.Sequential(*layers)
        self.fc_params = nn.Linear(previous, 4)

    def forward(self, x, z):
        parameters = self.fc_params(self.network(torch.cat((x, z), dim=1)))
        positive = nn.functional.softplus(parameters) + 1e-6
        return positive[:, 0], positive[:, 1], positive[:, 2], positive[:, 3]


class DVFM(nn.Module):
    def __init__(self, input_dim, latent_dim=8, encoder_hidden=(64, 32),
                 decoder_hidden=(32, 64), dropout=0.0,
                 encoder_dropout=None, decoder_dropout=None):
        super().__init__()
        if latent_dim < 0:
            raise ValueError("latent_dim must be nonnegative")
        encoder_dropout = dropout if encoder_dropout is None else encoder_dropout
        decoder_dropout = dropout if decoder_dropout is None else decoder_dropout
        self.encoder = None if latent_dim == 0 else Encoder(
            input_dim, latent_dim, encoder_hidden, encoder_dropout
        )
        self.decoder = Decoder(input_dim, latent_dim, decoder_hidden, decoder_dropout)
        self.latent_dim = latent_dim

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
