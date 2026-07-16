"""DVFM architecture copied from the supplied implementation.

The neural architecture and C-ELBO loss are unchanged.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset

class SurvivalDataset(Dataset):
    def __init__(self, X, time, event):
        self.X = torch.FloatTensor(X)
        self.time = torch.FloatTensor(time)
        self.event = torch.FloatTensor(event)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.time[idx], self.event[idx]


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims=[64, 32]):
        super(Encoder, self).__init__()
        
        layers = []
        prev_dim = input_dim + 2  # X + time + event
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
    
    def forward(self, x, time, event):
        # Concatenate all inputs
        inputs = torch.cat([x, time.unsqueeze(1), event.unsqueeze(1)], dim=1)
        h = self.network(inputs)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims=[32, 64]):
        super(Decoder, self).__init__()
        
        layers = []
        prev_dim = input_dim + latent_dim  # X + z
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        # Output: shape and scale for T and C (4 parameters)
        self.fc_params = nn.Linear(prev_dim, 4)
    
    def forward(self, x, z):
        inputs = torch.cat([x, z], dim=1)
        h = self.network(inputs)
        params = self.fc_params(h)
        
        # Apply softplus to ensure positive parameters
        shape_T = nn.functional.softplus(params[:, 0]) + 1e-6
        scale_T = nn.functional.softplus(params[:, 1]) + 1e-6
        shape_C = nn.functional.softplus(params[:, 2]) + 1e-6
        scale_C = nn.functional.softplus(params[:, 3]) + 1e-6
        
        return shape_T, scale_T, shape_C, scale_C


class DVFM(nn.Module):
    def __init__(self, input_dim, latent_dim=8, encoder_hidden=[64, 32], 
                 decoder_hidden=[32, 64]):
        super(DVFM, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim, encoder_hidden)
        self.decoder = Decoder(input_dim, latent_dim, decoder_hidden)
        self.latent_dim = latent_dim
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def weibull_log_pdf(self, t, shape, scale):
        """Log PDF of Weibull distribution"""
        return (torch.log(shape) - torch.log(scale) + 
                (shape - 1) * (torch.log(t) - torch.log(scale)) - 
                (t / scale) ** shape)
    
    def weibull_log_survival(self, t, shape, scale):
        """Log survival function of Weibull distribution"""
        return -(t / scale) ** shape
    
    def forward(self, x, time, event):
        # Encode
        mu, logvar = self.encoder(x, time, event)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        shape_T, scale_T, shape_C, scale_C = self.decoder(x, z)
        
        return shape_T, scale_T, shape_C, scale_C, mu, logvar
    
    def loss_function(self, shape_T, scale_T, shape_C, scale_C, 
                      mu, logvar, time, event, beta=1.0, free_bits=0.0):
        """
        Compute the ELBO loss with free bits and beta annealing
        """
        # Reconstruction loss (conditional log-likelihood)
        # For observed events: log f_T(t) + log S_C(t)
        # For censored: log S_T(t) + log f_C(t)
        
        log_f_T = self.weibull_log_pdf(time, shape_T, scale_T)
        log_S_T = self.weibull_log_survival(time, shape_T, scale_T)
        log_f_C = self.weibull_log_pdf(time, shape_C, scale_C)
        log_S_C = self.weibull_log_survival(time, shape_C, scale_C)
        
        recon_loss = event * (log_f_T + log_S_C) + (1 - event) * (log_S_T + log_f_C)
        recon_loss = recon_loss.mean()
        
        # KL divergence with free bits per dimension
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        
        # Apply free bits per dimension
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_per_dim = torch.maximum(kl_per_dim, torch.tensor(free_bits))
        kl_div_free = kl_per_dim.sum(dim=1).mean()
        
        # Total loss
        loss = -recon_loss + beta * kl_div_free
        
        return loss, -recon_loss, kl_div.mean()
