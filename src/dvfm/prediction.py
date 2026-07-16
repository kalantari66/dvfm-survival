"""Aggregate-posterior Monte Carlo prediction used in the paper code."""

import numpy as np
import torch

def predict_survival_curves(model, X, time_points, train_loader, n_samples=100, device='cpu'):
    """
    Predict survival curves using Aggregate Posterior Sampling.
    
    Parameters:
    -----------
    model : DVFM
        Trained model
    X : array-like
        Test Covariates (n_patients, n_features)
    time_points : array-like
        Time points at which to evaluate survival
    train_loader : DataLoader
        The loader used for training (to sample the empirical prior)
    n_samples : int
        Number of Monte Carlo samples
        
    Returns:
    --------
    survival_curves : array (n_patients, len(time_points))
    """
    model.eval()
    model.to(device)
    
    # --- 1. Construct the Aggregate Posterior from Training Data ---
    # We collect the mu and sigma from the training set to act as our "Prior"
    mus = []
    logvars = []
    
    with torch.no_grad():
        for batch in train_loader:
            # SurvivalDataset returns (x, time, event)
            x_batch, t_batch, delta_batch = batch
            x_batch = x_batch.to(device)
            t_batch = t_batch.to(device)
            delta_batch = delta_batch.to(device)
            
            # Get posterior stats from Encoder
            mu, logvar = model.encoder(x_batch, t_batch, delta_batch)
            mus.append(mu)
            logvars.append(logvar)
    
    # Concatenate to get the pool of all training latent distributions
    all_mus = torch.cat(mus, dim=0)       # (N_train, latent_dim)
    all_stds = torch.exp(0.5 * torch.cat(logvars, dim=0)) 
    n_train = all_mus.shape[0]

    # --- 2. Predict for Test Patients ---
    X_tensor = torch.FloatTensor(X).to(device)
    n_patients = X.shape[0]
    n_times = len(time_points)
    
    survival_curves = np.zeros((n_patients, n_times))
    
    with torch.no_grad():
        for i in range(n_samples):
            # A. Randomly select 'n_patients' indices from the training set
            # This simulates drawing from the population distribution
            indices = torch.randint(low=0, high=n_train, size=(n_patients,)).to(device)
            
            # B. Sample z using the posterior stats of the selected training samples
            # z = mu_train + sigma_train * epsilon
            sampled_mus = all_mus[indices]
            sampled_stds = all_stds[indices]
            epsilon = torch.randn_like(sampled_mus)
            
            z = sampled_mus + sampled_stds * epsilon
            
            # C. Decode (Get Weibull Parameters)
            shape_T, scale_T, _, _ = model.decoder(X_tensor, z)
            
            # D. Compute Survival Function
            for t_idx, t in enumerate(time_points):
                t_tensor = torch.FloatTensor([t]).to(device)
                # Weibull Survival: S(t) = exp(-(t/scale)^shape)
                S_t = torch.exp(-(t_tensor / scale_T) ** shape_T)
                survival_curves[:, t_idx] += S_t.cpu().numpy()
        
        # Average over the Monte Carlo samples
        survival_curves /= n_samples
    
    return survival_curves


def get_median_survival_time(survival_curves, time_points):
    """
    Extract median survival time from survival curves.
    """
    median_times = np.zeros(len(survival_curves))
    
    for i, curve in enumerate(survival_curves):
        # Find where curve crosses 0.5
        idx = np.where(curve <= 0.5)[0]
        if len(idx) > 0:
            median_times[i] = time_points[idx[0]]
        else:
            median_times[i] = time_points[-1]  # Censored at max time
    
    return median_times
