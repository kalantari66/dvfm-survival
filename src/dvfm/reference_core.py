import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.stats import weibull_min
from scipy.integrate import trapezoid as trapz
from sklearn.model_selection import train_test_split
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. DATA GENERATION WITH COPULAS
# ============================================================================


def _theta_to_frailty_alpha(theta):
    """
    Map user-facing theta to shared-frailty loading alpha_f.
    Designed so theta=1.2 is low dependence and larger theta increases dependence.
    """
    theta = float(theta)
    alpha_f = (theta - 1.0) / 4.0
    return float(np.clip(alpha_f, 0.02, 2.0))


def _softplus_np(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _sample_clayton_uv_vector_theta(theta_vec, rng):
    """
    Marshall-Olkin Clayton sampling with per-sample theta.
    theta_i must be > 0 for every i.
    """
    theta_vec = np.asarray(theta_vec, dtype=float)
    if np.any(theta_vec <= 0):
        raise ValueError("All theta values must be > 0 for Clayton copula.")
    shape = 1.0 / theta_vec
    S = rng.gamma(shape=shape, scale=1.0, size=theta_vec.shape[0])
    E1 = rng.exponential(scale=1.0, size=theta_vec.shape[0])
    E2 = rng.exponential(scale=1.0, size=theta_vec.shape[0])
    U1 = (1.0 + E1 / S) ** (-1.0 / theta_vec)
    U2 = (1.0 + E2 / S) ** (-1.0 / theta_vec)
    return U1, U2


def _weibull_ph_time(U, k, lam0, linpred):
    """
    Weibull proportional hazards inversion:
    S(t|x) = exp(-lam0 * exp(linpred) * t^k)
    """
    U = np.clip(np.asarray(U, dtype=float), 1e-12, 1.0 - 1e-12)
    rate = lam0 * np.exp(np.asarray(linpred, dtype=float))
    return (-np.log(U) / np.maximum(rate, 1e-12)) ** (1.0 / k)


def _covdep_clayton_components(X, theta_min=1.2, theta_max=10.0):
    """
    Build nonlinear event/censor predictors and covariate-dependent theta(x).
    """
    X = np.asarray(X, dtype=float)
    n_features = X.shape[1]
    x0 = X[:, 0]
    x1 = X[:, 1] if n_features > 1 else 0.0
    x2 = X[:, 2] if n_features > 2 else 0.0
    x3 = X[:, 3] if n_features > 3 else 0.0
    x4 = X[:, 4] if n_features > 4 else 0.0

    eta_event = (
        0.9 * np.sin(1.5 * x0)
        + 0.7 * (x1 ** 2)
        - 0.6 * np.tanh(x2)
        + 0.8 * (x0 * x3)
        + 0.5 * (x4 > 0).astype(float)
    )
    eta_cens = (
        0.6 * np.cos(1.2 * x0)
        + 0.5 * np.abs(x2)
        - 0.4 * (x1 * x3)
        + 0.3 * (x4 < -0.5).astype(float)
    )
    if n_features > 5:
        # deterministic small linear tails for reproducibility
        w_event = np.linspace(0.05, 0.2, n_features - 5)
        w_cens = np.linspace(0.04, 0.16, n_features - 5)
        eta_event = eta_event + X[:, 5:] @ w_event
        eta_cens = eta_cens + X[:, 5:] @ w_cens

    g = (
        0.8 * np.sin(x0)
        + 0.6 * (x1 ** 2)
        + 0.5 * np.tanh(x2)
        - 0.4 * (x0 * x3)
        + 0.3 * (x4 > 0).astype(float)
    )
    # Use a pointwise mapping (sigmoid) instead of batch min-max scaling.
    # This avoids collapse when evaluating at fixed X (e.g., in conditional tau checks).
    g = (g - np.mean(g)) / (np.std(g) + 1e-8)
    raw = 1.0 / (1.0 + np.exp(-g))
    theta_vec = theta_min + raw * (theta_max - theta_min)
    return eta_event, eta_cens, theta_vec

def generate_copula_data(n_samples=1000, n_features=5, copula_type='clayton', 
                         theta=2.0, seed=42):
    """
    # Generate synthetic survival data with dependent censoring using copulas.
    # Covariates X have different means and variances.
    # """
    np.random.seed(seed)
    
    # ==========================================================
    # 1. GENERATE COVARIATES WITH DIFFERENT MEANS AND VARIANCES
    # ==========================================================
    # Randomly select a mean for each feature between -2 and 2
    feat_means = np.random.uniform(0, 1.0, size=n_features)
    
    # Randomly select a std dev for each feature between 0.5 and 2.0
    feat_stds = np.random.uniform(0.5, 2.0, size=n_features)
    
    # Generate standard normal data
    X_base = np.random.randn(n_samples, n_features)

    # Apply scaling and shifting: X = X * std + mean
    # Broadcasting allows multiplying (n_samples, n_features) by (n_features,)
    X = X_base * feat_stds + feat_means
    if copula_type == 'clayton_covdep':
        # Use a mixed covariate design (continuous + binary injection) as requested.
        rng_local = np.random.default_rng(seed)
        X = rng_local.normal(size=(n_samples, n_features))
        b = rng_local.integers(0, 2, size=(n_samples,))
        X[:, 0] = 0.7 * X[:, 0] + 0.3 * b
    
    # ==========================================================
    # 2. GENERATE COPULAS (Same as before)
    # ==========================================================
    # Initialize with safe defaults
    U1 = np.random.uniform(1e-5, 1.0 - 1e-5, n_samples)
    V = np.random.uniform(1e-5, 1.0 - 1e-5, n_samples)
    U2 = np.zeros_like(U1)

    if copula_type == 'gaussian':
        from scipy.stats import norm
        rho = (theta - 1) / (theta + 1)
        mean = [0, 0]
        cov = [[1, rho], [rho, 1]]
        U = np.random.multivariate_normal(mean, cov, n_samples)
        U1 = norm.cdf(U[:, 0])
        U2 = norm.cdf(U[:, 1])
        
    elif copula_type == 'clayton':
        val = U1**(-theta) * (V**(-theta/(1+theta)) - 1) + 1
        val = np.maximum(val, 1e-9)
        U2 = val**(-1/theta)
        
    elif copula_type == 'frank':
        if abs(theta) < 1e-6:
            U2 = V
        else:
            exp_neg_theta = np.exp(-theta)
            exp_neg_theta_u1 = np.exp(-theta * U1)
            num = V * (exp_neg_theta - 1)
            den = V * (exp_neg_theta_u1 - 1) + exp_neg_theta_u1
            den = np.sign(den) * np.maximum(np.abs(den), 1e-9)
            fraction = num / den
            log_arg = np.maximum(1 + fraction, 1e-9)
            U2 = -1/theta * np.log(log_arg)
        
    elif copula_type == 'gumbel':
        from scipy.stats import levy_stable
        alpha = 1/theta
        S = levy_stable.rvs(alpha, 1, size=n_samples, random_state=seed)
        S = np.maximum(S, 1e-9)
        E1 = np.random.exponential(1, n_samples)
        E2 = np.random.exponential(1, n_samples)
        term1 = np.power(E1/S, 1/theta)
        term2 = np.power(E2/S, 1/theta)
        U1 = np.exp(-term1)
        U2 = np.exp(-term2)

    elif copula_type == 'frailty_mixture':
        # Shared latent frailty model (non-copula construction):
        # lambda_T(t|X,U) = lambda_T0(t) * exp(X beta + U)
        # lambda_C(t|X,U) = lambda_C0(t) * exp(X gamma + alpha U)
        # T and C are sampled from Weibull PH marginals via inverse CDF.
        pass
    elif copula_type == 'clayton_covdep':
        # Clayton with per-sample theta(x) is handled in the PH-time branch below.
        pass
    
    else:
        raise ValueError(f"Unknown copula type: {copula_type}")
    
    # Clip
    epsilon = 1e-5
    U1 = np.clip(U1, epsilon, 1.0 - epsilon)
    U2 = np.clip(U2, epsilon, 1.0 - epsilon)
    
    # ==========================================================
    # 3. WEIBULL TRANSFORM
    # ==========================================================
    # Generate weights
    beta_T = np.random.uniform(-0.5, 0.5, size=n_features)
    beta_C = np.random.uniform(-0.5, 0.5, size=n_features)
    
    linear_pred_T = X @ beta_T 
    linear_pred_C = X @ beta_C 
    
    # Clip to prevent overflow
    linear_pred_T = np.clip(linear_pred_T, -100, 100)
    linear_pred_C = np.clip(linear_pred_C, -100, 100)
    
    shape_T = 2.0
    shape_C = 2.0

    if copula_type == 'frailty_mixture':
        eps = 1e-8
        alpha_f = _theta_to_frailty_alpha(theta)
        U_frailty = np.random.randn(n_samples)

        linear_pred_T = np.clip(X @ beta_T + U_frailty, -50, 50)
        linear_pred_C = np.clip(X @ beta_C + alpha_f * U_frailty, -50, 50)

        haz_mult_T = np.exp(linear_pred_T)
        haz_mult_C = np.exp(linear_pred_C)

        u_t = np.random.uniform(eps, 1.0 - eps, n_samples)
        u_c = np.random.uniform(eps, 1.0 - eps, n_samples)

        # Weibull PH inverse-CDF: T = lambda0 * [ -log(U) / exp(lp) ]^(1/k)
        lambda0_T = 1.0
        lambda0_C = 1.0
        T = lambda0_T * np.power((-np.log(u_t)) / np.maximum(haz_mult_T, eps), 1.0 / shape_T)
        C = lambda0_C * np.power((-np.log(u_c)) / np.maximum(haz_mult_C, eps), 1.0 / shape_C)
    elif copula_type == 'clayton_covdep':
        rng_local = np.random.default_rng(seed)
        theta_min = 1.2
        theta_max = max(theta_min + 1e-6, float(theta))
        eta_event, eta_cens, theta_vec = _covdep_clayton_components(
            X, theta_min=theta_min, theta_max=theta_max
        )
        Ue, Uc = _sample_clayton_uv_vector_theta(theta_vec, rng_local)

        k_event = 1.4
        k_cens = 1.2
        lam0_event = 0.01
        lam0_cens = 0.01
        censor_target = 0.50

        T = _weibull_ph_time(Ue, k=k_event, lam0=lam0_event, linpred=eta_event)
        for _ in range(25):
            C_tmp = _weibull_ph_time(Uc, k=k_cens, lam0=lam0_cens, linpred=eta_cens)
            delta_tmp = (T <= C_tmp).astype(int)
            censor_rate = 1.0 - np.mean(delta_tmp)
            ratio = (censor_target + 1e-6) / (censor_rate + 1e-6)
            ratio = np.clip(ratio, 0.85, 1.15)
            lam0_cens *= ratio
        C = _weibull_ph_time(Uc, k=k_cens, lam0=lam0_cens, linpred=eta_cens)
    else:
        scale_T = np.exp(linear_pred_T)
        scale_C = np.exp(linear_pred_C)
        T = scale_T * np.power(-np.log(1 - U1), 1/shape_T)
        C = scale_C * np.power(-np.log(1 - U2), 1/shape_C)
    
    # Handle NaNs
    if np.isnan(T).any(): T = np.nan_to_num(T, nan=np.nanmean(T))
    if np.isnan(C).any(): C = np.nan_to_num(C, nan=np.nanmean(C))
    
    # T = T / T.max()
    # C = C/ C.max()

    observed_time = np.minimum(T, C)
    event = (T <= C).astype(int)

    # X = X/X.max()
    # observed_time = observed_time/ observed_time.max()
    
    return X, observed_time, event, T, C


# ============================================================================
# 2. PYTORCH DATASET
# ============================================================================

class SurvivalDataset(Dataset):
    def __init__(self, X, time, event):
        self.X = torch.FloatTensor(X)
        self.time = torch.FloatTensor(time)
        self.event = torch.FloatTensor(event)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.time[idx], self.event[idx]


# ============================================================================
# 3. DEEP VARIATIONAL FRAILTY MODEL (DVFM)
# ============================================================================

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
        if latent_dim < 0:
            raise ValueError("latent_dim must be nonnegative")
        self.encoder = None if latent_dim == 0 else Encoder(input_dim, latent_dim, encoder_hidden)
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
        if self.encoder is None:
            mu = x.new_empty((x.shape[0], 0))
            logvar = x.new_empty((x.shape[0], 0))
        else:
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


# ============================================================================
# 4. TRAINING FUNCTION
# ============================================================================

def train_dvfm(model, train_loader, val_loader, n_epochs=200, lr=1e-3,
               beta_max=1.0, warmup_epochs=50, free_bits=0.0, device='cpu',
               return_history=False, checkpoint_min_epoch=None,
               return_artifacts=False):
    """
    Train DVFM with KL annealing
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      factor=0.5, patience=10)
    
    train_losses = []
    val_losses = []
    history = []
    best_validation_elbo = float('inf')
    best_validation_elbo_epoch = None
    best_validation_elbo_state = None
    
    model.to(device)
    
    for epoch in range(n_epochs):
        # Update beta (KL annealing)
        beta = min(beta_max, (epoch + 1) / warmup_epochs) if warmup_epochs > 0 else beta_max
        
        # Training
        model.train()
        train_loss = 0
        train_recon = 0
        train_kl = 0
        
        for x, time, event in train_loader:
            x, time, event = x.to(device), time.to(device), event.to(device)
            
            optimizer.zero_grad()
            shape_T, scale_T, shape_C, scale_C, mu, logvar = model(x, time, event)
            loss, recon, kl = model.loss_function(shape_T, scale_T, shape_C, scale_C,
                                                   mu, logvar, time, event, 
                                                   beta, free_bits)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_recon += recon.item()
            train_kl += kl.item()
        
        train_loss /= len(train_loader)
        train_recon /= len(train_loader)
        train_kl /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        val_recon = 0
        val_kl = 0
        with torch.no_grad():
            for x, time, event in val_loader:
                x, time, event = x.to(device), time.to(device), event.to(device)
                shape_T, scale_T, shape_C, scale_C, mu, logvar = model(x, time, event)
                loss, recon, kl = model.loss_function(shape_T, scale_T, shape_C, scale_C,
                                                  mu, logvar, time, event, 
                                                  beta, free_bits)
                val_loss += loss.item()
                val_recon += recon.item()
                val_kl += kl.item()
        
        val_loss /= len(val_loader)
        val_recon /= len(val_loader)
        val_kl /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        history.append({"epoch": epoch + 1, "beta": beta, "learning_rate": optimizer.param_groups[0]["lr"],
                        "train_loss": train_loss, "train_reconstruction": train_recon, "train_kl": train_kl,
                        "validation_loss": val_loss, "validation_reconstruction": val_recon, "validation_kl": val_kl})

        eligible_epoch = checkpoint_min_epoch is None or (epoch + 1) >= int(checkpoint_min_epoch)
        if eligible_epoch and np.isfinite(val_loss) and val_loss < best_validation_elbo:
            best_validation_elbo = val_loss
            best_validation_elbo_epoch = epoch + 1
            best_validation_elbo_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Beta: {beta:.3f}, "
                  f"Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f}), "
                  f"Val Loss: {val_loss:.4f}")
    
    if return_artifacts:
        if best_validation_elbo_state is None:
            raise RuntimeError("No finite validation ELBO was available for checkpointing")
        return {
            "history": history,
            "final_state": {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            },
            "best_validation_elbo_state": best_validation_elbo_state,
            "best_validation_elbo_epoch": best_validation_elbo_epoch,
            "best_validation_elbo": best_validation_elbo,
        }
    if return_history:
        return history
    return train_losses, val_losses


# ============================================================================
# 5. PREDICTION (MONTE CARLO)
# ============================================================================

# def predict_survival_curves(model, X, time_points, n_samples=100, device='cpu'):
#     """
#     Predict survival curves using Monte Carlo integration over the prior.
    
#     Parameters:
#     -----------
#     model : DVFM
#         Trained model
#     X : array-like
#         Covariates (n_patients, n_features)
#     time_points : array-like
#         Time points at which to evaluate survival
#     n_samples : int
#         Number of Monte Carlo samples from the prior
    
#     Returns:
#     --------
#     survival_curves : array (n_patients, len(time_points))
#     """
#     model.eval()
#     model.to(device)
    
#     X_tensor = torch.FloatTensor(X).to(device)
#     n_patients = X.shape[0]
#     n_times = len(time_points)
    
#     survival_curves = np.zeros((n_patients, n_times))
    
#     with torch.no_grad():
#         for i in range(n_samples):
#             # Sample from prior N(0, I)
#             z = torch.randn(n_patients, model.latent_dim).to(device)
            
#             # Decode
#             shape_T, scale_T, _, _ = model.decoder(X_tensor, z)
            
#             # Compute survival function for each time point
#             for t_idx, t in enumerate(time_points):
#                 t_tensor = torch.FloatTensor([t]).to(device)
#                 S_t = torch.exp(-(t_tensor / scale_T) ** shape_T)
#                 survival_curves[:, t_idx] += S_t.cpu().numpy()
        
#         # Average over samples
#         survival_curves /= n_samples
    
#     return survival_curves


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
            if model.encoder is None:
                mu = x_batch.new_empty((x_batch.shape[0], 0))
                logvar = x_batch.new_empty((x_batch.shape[0], 0))
            else:
                mu, logvar = model.encoder(x_batch, t_batch, delta_batch)
            mus.append(mu)
            logvars.append(logvar)
    
    # Concatenate to get the pool of all training latent distributions
    all_mus = torch.cat(mus, dim=0)       # (N_train, latent_dim)
    all_stds = torch.exp(0.5 * torch.cat(logvars, dim=0)) 
    n_train = all_mus.shape[0]

    # --- 2. Predict for Test Patients ---
    X_tensor = torch.FloatTensor(X).to(device)
    time_tensor = torch.as_tensor(time_points, dtype=torch.float32, device=device)
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
            
            # D. Compute the complete survival grid in one device operation.
            S_t = torch.exp(-((time_tensor[None, :] / scale_T[:, None]) ** shape_T[:, None]))
            survival_curves += S_t.cpu().numpy()
        
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


# ============================================================================
# 6. EVALUATION UTILITIES (DEPENDENT CENSORING DIAGNOSTICS)
# ============================================================================

def _km_step_survival(times, event_observed):
    """
    Kaplan-Meier step curve for P(T > t).
    event_observed is 1 when a failure/event is observed.
    """
    times = np.asarray(times, dtype=float)
    event_observed = np.asarray(event_observed, dtype=int)

    uniq_event_times = np.unique(times[event_observed == 1])
    surv = 1.0
    step_times = [0.0]
    step_surv = [1.0]

    for t in uniq_event_times:
        at_risk = np.sum(times >= t)
        d_t = np.sum((times == t) & (event_observed == 1))
        if at_risk > 0:
            surv *= (1.0 - d_t / at_risk)
        step_times.append(float(t))
        step_surv.append(float(surv))

    return np.asarray(step_times), np.asarray(step_surv)


def _eval_step_curve(query_t, step_t, step_s, side='right'):
    """
    Evaluate right-continuous ('right') or left-limit ('left') step survival.
    """
    query_t = np.asarray(query_t, dtype=float)
    idx = np.searchsorted(step_t, query_t, side=side) - 1
    idx = np.clip(idx, 0, len(step_s) - 1)
    return step_s[idx]


def compute_oracle_brier_ibs(survival_curves, time_points, true_event_times):
    """
    Oracle Brier/IBS using fully observed true event times from simulation.
    """
    time_points = np.asarray(time_points, dtype=float)
    true_event_times = np.asarray(true_event_times, dtype=float)
    y_true = (true_event_times[:, None] > time_points[None, :]).astype(float)

    brier = np.mean((y_true - survival_curves) ** 2, axis=0)
    if len(time_points) > 1:
        ibs = trapz(brier, time_points) / (time_points[-1] - time_points[0] + 1e-12)
    else:
        ibs = float(brier[0])

    return brier, float(ibs)


def compute_ipcw_brier_ibs(survival_curves, time_points, observed_time, event, tau=None, eps=1e-6):
    """
    IPCW Brier/IBS on observed data.
    Note: under dependent censoring this estimator is biased; comparing it
    to Oracle-IBS is useful as a diagnostic gap.
    """
    time_points = np.asarray(time_points, dtype=float)
    observed_time = np.asarray(observed_time, dtype=float)
    event = np.asarray(event, dtype=int)

    if tau is None:
        tau = np.quantile(observed_time, 0.8)
    eval_mask = time_points <= tau
    eval_times = time_points[eval_mask]
    eval_surv = survival_curves[:, eval_mask]

    # Censoring distribution G(t) = P(C > t)
    censor_event = 1 - event
    g_t, g_s = _km_step_survival(observed_time, censor_event)
    g_obs_left = np.maximum(_eval_step_curve(observed_time, g_t, g_s, side='left'), eps)

    brier = np.zeros(len(eval_times), dtype=float)
    n = len(observed_time)
    for j, t in enumerate(eval_times):
        s_hat = eval_surv[:, j]
        g_at_t = max(float(_eval_step_curve(np.array([t]), g_t, g_s, side='right')[0]), eps)

        event_before_t = (observed_time <= t) & (event == 1)
        survive_past_t = observed_time > t

        term_event = np.zeros(n, dtype=float)
        term_event[event_before_t] = ((0.0 - s_hat[event_before_t]) ** 2) / g_obs_left[event_before_t]

        term_survive = np.zeros(n, dtype=float)
        term_survive[survive_past_t] = ((1.0 - s_hat[survive_past_t]) ** 2) / g_at_t

        brier[j] = np.mean(term_event + term_survive)

    if len(eval_times) > 1:
        ibs = trapz(brier, eval_times) / (eval_times[-1] - eval_times[0] + 1e-12)
    else:
        ibs = float(brier[0]) if len(eval_times) == 1 else np.nan

    return brier, float(ibs), float(tau)


# ============================================================================
# 7. BASELINE MODELS
# ============================================================================

def train_coxph(X_train, time_train, event_train, X_test):
    """Train Cox PH model"""
    df_train = pd.DataFrame(X_train, columns=[f'X{i}' for i in range(X_train.shape[1])])
    df_train['time'] = time_train
    df_train['event'] = event_train
    
    cph = CoxPHFitter()
    cph.fit(df_train, duration_col='time', event_col='event')
    
    df_test = pd.DataFrame(X_test, columns=[f'X{i}' for i in range(X_test.shape[1])])
    
    # Predict risk scores
    risk_scores = cph.predict_partial_hazard(df_test).values
    
    # Predict median survival time
    try:
        median_survival = cph.predict_median(df_test).values
    except:
        # If median is undefined, use expectation
        median_survival = cph.predict_expectation(df_test).values
    
    return risk_scores, median_survival, cph


def train_deepsurv(X_train, time_train, event_train, X_test,
                   n_epochs=200, batch_size=64, lr=1e-3, device='cpu',
                   eval_time_points=None):
    """
    Simple DeepSurv implementation (Cox proportional hazards with neural network)
    """
    class DeepSurv(nn.Module):
        def __init__(self, input_dim, hidden_dims=[64, 32]):
            super(DeepSurv, self).__init__()
            layers = []
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.3))
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, 1))
            self.network = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.network(x)
    
    def cox_loss(risk_scores, time, event):
        """
        # # Cox Negative Log Likelihood loss.
        # Assumes inputs are NOT sorted.
        # """
        # Sort by time descending
        idx = torch.argsort(time, descending=True)
        risk_scores = risk_scores[idx]
        event = event[idx]
        
        # Compute log-cumsum-exp (log of denominator of partial likelihood)
        max_val = torch.max(risk_scores)
        # Numerical stability trick
        exp_scores = torch.exp(risk_scores - max_val)
        sum_exp_scores = torch.cumsum(exp_scores, dim=0)
        log_sum_exp_scores = torch.log(sum_exp_scores) + max_val
        
        # Calculate loss for observed events
        loss = -torch.sum(event * (risk_scores - log_sum_exp_scores))
        return loss / (torch.sum(event) + 1e-8)

    # Prepare data
    train_dataset = SurvivalDataset(X_train, time_train, event_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    input_dim = X_train.shape[1]
    model = DeepSurv(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training Loop
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0
        for x_b, t_b, e_b in train_loader:
            x_b, t_b, e_b = x_b.to(device), t_b.to(device), e_b.to(device)
            optimizer.zero_grad()
            scores = model(x_b)
            loss = cox_loss(scores, t_b, e_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
    # --- Prediction Phase (Risk Scores) ---
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        risk_scores_test = model(X_test_tensor).cpu().numpy().flatten()
        
        # For Median Survival, we need the Breslow Baseline Hazard
        # 1. Get risk scores for ALL training data
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        train_risks = model(X_train_tensor).cpu().numpy().flatten()
        
    # --- Breslow Estimator for Survival Curves ---
    # Sort training data by time
    idx = np.argsort(time_train)
    sorted_times = time_train[idx]
    sorted_events = event_train[idx]
    sorted_risks = np.exp(train_risks[idx])
    
    unique_times = np.unique(sorted_times[sorted_events == 1])
    baseline_hazard = []
    if len(unique_times) == 0:
        median_predictions = np.full(len(X_test), np.max(time_train), dtype=float)
        surv_curves_eval = None
        if eval_time_points is not None:
            surv_curves_eval = np.ones((len(X_test), len(eval_time_points)), dtype=float)
        return risk_scores_test, median_predictions, surv_curves_eval
    
    # Simple cumulative hazard calculation
    for t in unique_times:
        # Risk set: all subjects who lived at least until t
        at_risk_mask = sorted_times >= t
        denominator = np.sum(sorted_risks[at_risk_mask])
        d_i = np.sum((sorted_times == t) & (sorted_events == 1))
        baseline_hazard.append(d_i / denominator)
    
    baseline_cum_hazard = np.cumsum(baseline_hazard)
    
    # Calculate survival curves for test set: S(t|x) = exp(-H0(t) * exp(risk(x)))
    median_predictions = []
    test_risks_exp = np.exp(risk_scores_test)
    
    for r in test_risks_exp:
        surv_curve = np.exp(-baseline_cum_hazard * r)
        # Find median (crossing 0.5)
        cross_idx = np.where(surv_curve <= 0.5)[0]
        if len(cross_idx) > 0:
            median_predictions.append(unique_times[cross_idx[0]])
        else:
            median_predictions.append(unique_times[-1])
            
    surv_curves_eval = None
    if eval_time_points is not None:
        eval_time_points = np.asarray(eval_time_points, dtype=float)
        h0 = np.interp(
            eval_time_points,
            unique_times,
            baseline_cum_hazard,
            left=0.0,
            right=baseline_cum_hazard[-1]
        )
        surv_curves_eval = np.exp(-np.exp(risk_scores_test)[:, None] * h0[None, :])

    return risk_scores_test, np.array(median_predictions), surv_curves_eval

# ============================================================================
# 7. MTLR (Neural Multi-Task Logistic Regression)
# ============================================================================

def train_mtlr(X_train, time_train, event_train, X_test, num_bins=45,
               n_epochs=200, lr=0.005, device='cpu', eval_time_points=None):
    """
    Simplified Neural MTLR. Discretizes time and predicts probability of 
    event occurring in specific bins.
    """
    # 1. Discretize Time
    # Use quantiles of observed events to define bins
    events_only = time_train[event_train == 1]
    quantiles = np.linspace(0, 1, num_bins + 1)
    bins = np.quantile(events_only, quantiles)
    # Ensure unique bins and cover max range
    bins = np.unique(bins)
    bins[-1] = max(np.max(time_train), bins[-1]) + 1e-5
    bins[0] = 0
    actual_num_bins = len(bins) - 1
    
    # 2. Encode Targets
    def encode_target(times, events):
        y_class = np.zeros(len(times), dtype=int)
        for i, (t, e) in enumerate(zip(times, events)):
            # Find bin index
            bin_idx = np.digitize(t, bins) - 1
            bin_idx = min(max(0, bin_idx), actual_num_bins - 1)
            
            if e == 1:
                y_class[i] = bin_idx # Event happened in this bin
            else:
                # Censored in this bin implies it survived this bin
                # In standard N-MTLR, this is handled in loss. 
                # Here we use a simplification: Censored data creates a mask.
                y_class[i] = bin_idx 
        return torch.LongTensor(y_class)

    y_train_bins = encode_target(time_train, event_train)
    
    # 3. Model
    class N_MTLR(nn.Module):
        def __init__(self, input_dim, num_bins):
            super(N_MTLR, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, num_bins)
            )
        def forward(self, x):
            return self.net(x) # Logits
    
    model = N_MTLR(X_train.shape[1], actual_num_bins).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = y_train_bins.to(device)
    event_t = torch.FloatTensor(event_train).to(device)
    
    # 4. MTLR Loss (Simplified Log-Likelihood)
    # Ideally: P(T > t) = sum_{k>j} P(T in bin k).
    # We use a masked CrossEntropy approach for simplicity in this demo.
    # For events: CrossEntropy.
    # For censored: We want to maximize probability of sum(bins > current).
    # Standard N-MTLR loss implementation is complex; we use a simpler discrete approximation.
    
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        logits = model(X_train_t)
        
        # Standard Cross Entropy for everyone (treating censored as event for a moment)
        ce_loss = criterion(logits, y_train_t)
        
        # Reweight or modify for censored
        # For censored data, the "label" is the bin censoring occurred.
        # We know true event is > bin. 
        # A proper MTLR loss requires summation. Here we simply downweight censored loss
        # to focus learning on observed events, which is a naive heuristic but functional for a quick baseline.
        loss = (ce_loss * event_t).mean() + 0.1 * (ce_loss * (1-event_t)).mean()
        
        loss.backward()
        optimizer.step()
        
    # 5. Prediction
    model.eval()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test).to(device)
        logits = model(X_test_t)
        # Softmax to get density
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        
        # Survival Function S(t) = 1 - CDF(t)
        cdf = np.cumsum(probs, axis=1)
        survival_probs = 1.0 - cdf
        
        # Calculate Risk Score (Expected Time)
        # Midpoints of bins
        bin_mids = (bins[:-1] + bins[1:]) / 2
        predicted_means = np.sum(probs * bin_mids, axis=1)
        
        # Median Survival
        predicted_medians = np.zeros(len(X_test))
        for i in range(len(X_test)):
            idx = np.where(survival_probs[i, :] <= 0.5)[0]
            if len(idx) > 0:
                predicted_medians[i] = bin_mids[idx[0]]
            else:
                predicted_medians[i] = bin_mids[-1]
                
    # MTLR risk score is roughly -predicted_time (higher time = lower risk)
    surv_curves_eval = None
    if eval_time_points is not None:
        eval_time_points = np.asarray(eval_time_points, dtype=float)
        bin_right_edges = bins[1:]
        surv_curves_eval = np.zeros((len(X_test), len(eval_time_points)), dtype=float)
        for i in range(len(X_test)):
            surv_curves_eval[i] = np.interp(
                eval_time_points,
                bin_right_edges,
                survival_probs[i],
                left=1.0,
                right=survival_probs[i, -1]
            )

    return -predicted_means, predicted_medians, surv_curves_eval


from scipy.stats import kendalltau

def measure_learned_dependence(model, X_test, n_samples=1000, device='cpu'):
    """
    Measures the dependence between T and C learned by the DVFM model
    by simulating data from the trained Decoder.
    """
    model.eval()
    model.to(device)
    
    # Take the first patient (or mean of patients) to test structural dependence
    # We only need X to set the baseline scale
    x_sample = torch.FloatTensor(X_test[0:1]).to(device) 
    
    # 1. Sample latent space z from Prior N(0,1)
    z_samples = torch.randn(n_samples, model.latent_dim).to(device)
    
    # 2. Duplicate x to match z size
    x_repeated = x_sample.repeat(n_samples, 1)
    
    with torch.no_grad():
        # 3. Decode parameters
        shape_T, scale_T, shape_C, scale_C = model.decoder(x_repeated, z_samples)
        
        # 4. Sample Times from the predicted Weibull distributions
        # Weibull sampling: scale * (-ln(U))^(1/shape)
        u = torch.rand(n_samples).to(device)
        sim_T = scale_T.squeeze() * (-torch.log(u))**(1/shape_T.squeeze())
        
        u = torch.rand(n_samples).to(device)
        sim_C = scale_C.squeeze() * (-torch.log(u))**(1/shape_C.squeeze())
        
        # Move to numpy
        sim_T = sim_T.cpu().numpy()
        sim_C = sim_C.cpu().numpy()
        
    # 5. Calculate Kendall's Tau correlation between generated T and C
    tau, p_value = kendalltau(sim_T, sim_C)
    
    return tau


######################################################################################
# ============================================================================
# 8. ANALYZE CONDITIONAL DEPENDENCE (Including Ground Truth)
# ============================================================================

def analyze_conditional_dependence(model, X_test, copula_type, theta, device='cpu'):
    """
    Measures Conditional Kendall's Tau (Given Fixed X) for:
    1. The True Data Generation Process (Ground Truth)
    2. The Model (Marginalizing over Z)
    3. The Model (Conditioning on Z)
    """
    from scipy.stats import kendalltau, norm
    from scipy.stats import levy_stable
    
    model.eval()
    model.to(device)
    
    # Simulation settings
    n_sim = 2000
    x_patient = X_test[0] 
    X_fixed = torch.FloatTensor(x_patient).unsqueeze(0).repeat(n_sim, 1).to(device)
    
    # =====================================================
    # 1. GROUND TRUTH TAU (T ⊥ C | X)
    # =====================================================
    # We generate U1, U2 from the true Copula. 
    # Since Tau is rank-invariant, Tau(T, C | X) == Tau(U1, U2)
    
    U1 = np.random.uniform(1e-5, 1.0 - 1e-5, n_sim)
    V = np.random.uniform(1e-5, 1.0 - 1e-5, n_sim)
    U2 = np.zeros_like(U1)
    tau_true = None

    if copula_type == 'independent':
        U2 = np.random.uniform(1e-5, 1.0 - 1e-5, n_sim)
        
    elif copula_type == 'gaussian':
        rho = (theta - 1) / (theta + 1) # High correlation for clear signal
        mean = [0, 0]; cov = [[1, rho], [rho, 1]]
        U = np.random.multivariate_normal(mean, cov, n_sim)
        U1 = norm.cdf(U[:, 0]); U2 = norm.cdf(U[:, 1])
        
    elif copula_type == 'clayton':
        # Inverse conditional sampling for Clayton
        val = U1**(-theta) * (V**(-theta/(1+theta)) - 1) + 1
        val = np.maximum(val, 1e-9)
        U2 = val**(-1/theta)
        
    elif copula_type == 'frank':
        if abs(theta) < 1e-6:
            U2 = V
        else:
            exp_neg_theta = np.exp(-theta)
            exp_neg_theta_u1 = np.exp(-theta * U1)
            num = V * (exp_neg_theta - 1)
            den = V * (exp_neg_theta_u1 - 1) + exp_neg_theta_u1
            den = np.sign(den) * np.maximum(np.abs(den), 1e-9)
            fraction = num / den
            log_arg = np.maximum(1 + fraction, 1e-9)
            U2 = -1/theta * np.log(log_arg)

    # elif copula_type == 'gumbel':
    #     from scipy.stats import levy_stable
    #     alpha = 1/theta
    #     S = levy_stable.rvs(alpha, 1, size=n_sim, random_state=seed)
    #     S = np.maximum(S, 1e-9)
    #     E1 = np.random.exponential(1, n_samples)
    #     E2 = np.random.exponential(1, n_samples)
    #     term1 = np.power(E1/S, 1/theta)
    #     term2 = np.power(E2/S, 1/theta)
    #     U1 = np.exp(-term1)
    #     U2 = np.exp(-term2)
            
    elif copula_type == 'gumbel':
        # Gumbel generator
        alpha = 1/theta
        S = levy_stable.rvs(alpha, 1, size=n_sim); S = np.maximum(S, 1e-9)
        E1 = np.random.exponential(1, n_sim); E2 = np.random.exponential(1, n_sim)
        term1 = np.power(E1/S, 1/theta); term2 = np.power(E2/S, 1/theta)
        U1 = np.exp(-term1); U2 = np.exp(-term2)

    elif copula_type == 'frailty_mixture':
        # True dependence under shared frailty hazards:
        # lambda_T(t|X,U)=lambda0_T(t) exp(X beta + U),
        # lambda_C(t|X,U)=lambda0_C(t) exp(gamma X + alpha U).
        eps = 1e-8
        alpha_f = _theta_to_frailty_alpha(theta)
        shape_T = 2.0
        shape_C = 2.0

        U_frailty = np.random.randn(n_sim)
        lp_t = np.clip(U_frailty, -50, 50)
        lp_c = np.clip(alpha_f * U_frailty, -50, 50)
        hz_t = np.exp(lp_t)
        hz_c = np.exp(lp_c)

        u_t = np.random.uniform(eps, 1.0 - eps, n_sim)
        u_c = np.random.uniform(eps, 1.0 - eps, n_sim)

        true_T = np.power((-np.log(u_t)) / np.maximum(hz_t, eps), 1.0 / shape_T)
        true_C = np.power((-np.log(u_c)) / np.maximum(hz_c, eps), 1.0 / shape_C)
        tau_true, _ = kendalltau(true_T, true_C)
    elif copula_type == 'clayton_covdep':
        # Ground-truth conditional dependence for fixed X under theta(X).
        eps = 1e-8
        x_np = x_patient.reshape(1, -1)
        theta_min = 1.2
        theta_max = max(theta_min + 1e-6, float(theta))
        eta_event, eta_cens, theta_vec = _covdep_clayton_components(
            np.repeat(x_np, n_sim, axis=0),
            theta_min=theta_min,
            theta_max=theta_max,
        )
        Ue, Uc = _sample_clayton_uv_vector_theta(theta_vec, np.random.default_rng(123))
        true_T = _weibull_ph_time(Ue, k=1.4, lam0=0.01, linpred=eta_event)
        true_C = _weibull_ph_time(Uc, k=1.2, lam0=0.01, linpred=eta_cens)
        tau_true, _ = kendalltau(true_T, true_C)

    if tau_true is None:
        mask = ~np.isnan(U1) & ~np.isnan(U2)
        tau_true, _ = kendalltau(U1[mask], U2[mask])

    # =====================================================
    # 2. MODEL TAU (T ⊥ C | X) -> Should match Truth
    # =====================================================
    with torch.no_grad():
        # Sample random Z (integrating it out)
        z_random = torch.randn(n_sim, model.latent_dim).to(device)
        shape_T, scale_T, shape_C, scale_C = model.decoder(X_fixed, z_random)
        
        # Sample T/C (Independent Uniforms for noise)
        u_t = torch.rand(n_sim).to(device)
        u_c = torch.rand(n_sim).to(device)
        pred_T = scale_T.squeeze() * (-torch.log(u_t))**(1/shape_T.squeeze())
        pred_C = scale_C.squeeze() * (-torch.log(u_c))**(1/shape_C.squeeze())
        
        tau_est_X, _ = kendalltau(pred_T.cpu().numpy(), pred_C.cpu().numpy())

    # =====================================================
    # 3. MODEL TAU (T ⊥ C | X, Z) -> Should be Zero
    # =====================================================
    with torch.no_grad():
        # Fix Z (Conditioning on specific frailty)
        z_fixed_val = torch.randn(1, model.latent_dim).to(device)
        z_fixed = z_fixed_val.repeat(n_sim, 1)
        
        shape_T, scale_T, shape_C, scale_C = model.decoder(X_fixed, z_fixed)
        
        # Sample T/C
        u_t = torch.rand(n_sim).to(device)
        u_c = torch.rand(n_sim).to(device)
        pred_T = scale_T.squeeze() * (-torch.log(u_t))**(1/shape_T.squeeze())
        pred_C = scale_C.squeeze() * (-torch.log(u_c))**(1/shape_C.squeeze())
        
        tau_est_XZ, _ = kendalltau(pred_T.cpu().numpy(), pred_C.cpu().numpy())
        
    return tau_true, tau_est_X, tau_est_XZ
# ============================================================================
# 9. UPDATED RUN_EXPERIMENT FUNCTION
# ============================================================================

def run_experiment():
    Theta = 1
    Latent = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    copulas = ['gaussian']  #['gaussian', 'clayton', 'frank'] #, 'gumbel'
    results = []
    
    for copula in copulas:
        print(f"\n" + "="*50)
        print(f"Processing Copula: {copula.upper()}")
        print("="*50)
        
        # 1. Generate Data
        X, obs_time, event, true_T, true_C = generate_copula_data(
            n_samples=10000, n_features=10, copula_type=copula, theta=Theta
        )
        
        # # Split data
        # X_train, X_test, t_train, t_test, e_train, e_test, true_T_train, true_T_test = train_test_split(
        #     X, obs_time, event, true_T, test_size=0.3, random_state=42
        # )
        # 2. Split Data
        # We pass ALL arrays (including true_C) to ensure they are split identically
        (X_train, X_test, 
         t_train, t_test, 
         e_train, e_test, 
         true_T_train, true_T_test, 
         true_C_train, true_C_test) = train_test_split(
            X, obs_time, event, true_T, true_C, test_size=0.3, random_state=42
        )

        max_time = np.max(t_train) * 1.5
        time_points = np.linspace(0, max_time, 1000)
        
        # # --- Model 1: CoxPH ---
        # print("Training CoxPH...")
        # try:
        #     df_train = pd.DataFrame(X_train, columns=[f'X{i}' for i in range(X_train.shape[1])])
        #     df_train['time'] = t_train
        #     df_train['event'] = e_train
            
        #     cph = CoxPHFitter()
        #     cph.fit(df_train, duration_col='time', event_col='event')
            
        #     df_test = pd.DataFrame(X_test, columns=[f'X{i}' for i in range(X_test.shape[1])])
        #     cox_risk = cph.predict_partial_hazard(df_test).values
        #     cox_median = cph.predict_median(df_test).values.flatten()
            
        #     max_train_time = np.max(t_train)
        #     cox_median[np.isinf(cox_median)] = max_train_time
            
        #     cox_cindex = concordance_index(t_test, cox_median, e_test)
        #     cox_mae = np.mean(np.abs(true_T_test - cox_median))
            
        # except Exception as e:
        #     print(f"CoxPH failed: {e}")
        #     cox_cindex, cox_mae = 0.5, 999.0

        # --- Model 1: CoxPH ---
        print("Training CoxPH...")
        try:
            df_train = pd.DataFrame(X_train, columns=[f'X{i}' for i in range(X_train.shape[1])])
            df_train['time'] = t_train
            df_train['event'] = e_train
            
            cph = CoxPHFitter()
            cph.fit(df_train, duration_col='time', event_col='event')
            
            df_test = pd.DataFrame(X_test, columns=[f'X{i}' for i in range(X_test.shape[1])])
            cox_risk = cph.predict_partial_hazard(df_test).values.flatten()
            
            # =====================================================
            # Compute Survival Curves and Find Median (S(t) = 0.5)
            # =====================================================
            max_train_time = np.max(t_train)
            
            # Get baseline survival function
            baseline_survival = cph.baseline_survival_
            baseline_times = baseline_survival.index.values
            baseline_S = baseline_survival.values.flatten()
            baseline_interp = np.interp(time_points, baseline_times, baseline_S, left=1.0, right=baseline_S[-1])
            cox_surv_curves = np.power(baseline_interp[None, :], cox_risk[:, None])
            
            cox_median = []
            
            for i, risk_score in enumerate(cox_risk):
                # S(t | X) = S_0(t)^exp(lp) and partial_hazard = exp(lp)
                patient_survival = baseline_S ** risk_score
                
                # Find time where survival crosses 0.5
                # Search for where S(t) transitions from > 0.5 to <= 0.5
                crossing_idx = np.where(patient_survival <= 0.5)[0]
                
                if len(crossing_idx) > 0:
                    # Get the first time where S(t) <= 0.5
                    median_time = baseline_times[crossing_idx[0]]
                else:
                    # If survival never crosses 0.5, use max training time
                    median_time = max_train_time
                
                cox_median.append(median_time)
            
            cox_median = np.array(cox_median)
            
            cox_cindex = concordance_index(t_test, cox_median, e_test)
            # cox_mae = np.mean(np.abs(true_T_test - cox_median))
            uncensored_mask = e_test == 1
            cox_mae = np.mean(np.abs(true_T_test[uncensored_mask] - cox_median[uncensored_mask]))
            
        except Exception as e:
            print(f"CoxPH failed: {e}")
            cox_cindex, cox_mae = 0.5, 999.0
            cox_median = np.full_like(t_test, fill_value=np.max(t_train), dtype=float)
            cox_surv_curves = np.ones((len(X_test), len(time_points)), dtype=float)
            
        # --- Model 2: DeepSurv ---
        print("Training DeepSurv...")
        ds_risk, ds_median, ds_surv_curves = train_deepsurv(
            X_train, t_train, e_train, X_test, device=device, eval_time_points=time_points
        )
        ds_cindex = concordance_index(t_test, ds_median, e_test)
        # ds_mae = np.mean(np.abs(true_T_test - ds_median))
        uncensored_mask = e_test == 1
        ds_mae = np.mean(np.abs(true_T_test[uncensored_mask] - ds_median[uncensored_mask]))


        # --- Model 3: MTLR ---
        print("Training MTLR...")
        mtlr_risk, mtlr_median, mtlr_surv_curves = train_mtlr(
            X_train, t_train, e_train, X_test, device=device, num_bins=200, eval_time_points=time_points
        )
        mtlr_cindex = concordance_index(t_test, mtlr_median, e_test) 
        # mtlr_mae = np.mean(np.abs(true_T_test - mtlr_median))
        uncensored_mask = e_test == 1
        mtlr_mae = np.mean(np.abs(true_T_test[uncensored_mask] - mtlr_median[uncensored_mask]))

        
        # --- Model 4: DVFM ---
        print("Training DVFM...")
        train_ds = SurvivalDataset(X_train, t_train, e_train)
        val_ds = SurvivalDataset(X_test, t_test, e_test)
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
        
        dvfm = DVFM(input_dim=X.shape[1], latent_dim=Latent).to(device)
        train_dvfm(dvfm, train_loader, val_loader, n_epochs=200, 
                   beta_max=1, free_bits=0, device=device)

        # surv_curves = predict_survival_curves(dvfm, X_test, time_points, n_samples=100, device=device)
        # # --- UPDATE THIS LINE ---
        dvfm_surv_curves = predict_survival_curves(
            model=dvfm, 
            X=X_test, 
            time_points=time_points, 
            train_loader=train_loader,  # <--- Pass the loader here
            n_samples=100, 
            device=device
        )

        dvfm_median = get_median_survival_time(dvfm_surv_curves, time_points)
        
        dvfm_cindex = concordance_index(t_test, dvfm_median, e_test)
        # dvfm_mae = np.mean(np.abs(true_T_test - dvfm_median))
        uncensored_mask = e_test == 1
        dvfm_mae = np.mean(np.abs(true_T_test[uncensored_mask] - dvfm_median[uncensored_mask]))

        # --- Additional evaluation under dependent censoring ---
        cox_mae_oracle = np.mean(np.abs(true_T_test - cox_median))
        ds_mae_oracle = np.mean(np.abs(true_T_test - ds_median))
        mtlr_mae_oracle = np.mean(np.abs(true_T_test - mtlr_median))
        dvfm_mae_oracle = np.mean(np.abs(true_T_test - dvfm_median))

        _, cox_ibs_oracle = compute_oracle_brier_ibs(cox_surv_curves, time_points, true_T_test)
        _, ds_ibs_oracle = compute_oracle_brier_ibs(ds_surv_curves, time_points, true_T_test)
        _, mtlr_ibs_oracle = compute_oracle_brier_ibs(mtlr_surv_curves, time_points, true_T_test)
        _, dvfm_ibs_oracle = compute_oracle_brier_ibs(dvfm_surv_curves, time_points, true_T_test)

        _, cox_ibs_ipcw, eval_tau = compute_ipcw_brier_ibs(cox_surv_curves, time_points, t_test, e_test)
        _, ds_ibs_ipcw, _ = compute_ipcw_brier_ibs(ds_surv_curves, time_points, t_test, e_test, tau=eval_tau)
        _, mtlr_ibs_ipcw, _ = compute_ipcw_brier_ibs(mtlr_surv_curves, time_points, t_test, e_test, tau=eval_tau)
        _, dvfm_ibs_ipcw, _ = compute_ipcw_brier_ibs(dvfm_surv_curves, time_points, t_test, e_test, tau=eval_tau)
        # # --- DEPENDENCE DETECTION ---
        # print("Analyzing Dependence Structure...")
        # kl_div, cos_sim = analyze_dependence_structure(dvfm, X_test, device=device)
        # learned_tau = measure_learned_dependence(dvfm, X_test, device=device)
        
        # print(f"Copula: {copula.upper()}")
        # print(f"KL Divergence: {kl_div:.4f}")
        # print(f"Gradient Correlation (psi_T vs psi_C): {cos_sim:.4f}")
        # print(f"Kendall's Tau: {learned_tau:.4f}")
        
        # # Store Results
        # row = {
        #     'Copula': copula,
        #     'CoxPH C-Idx': cox_cindex, 'CoxPH MAE': cox_mae,
        #     'DeepSurv C-Idx': ds_cindex, 'DeepSurv MAE': ds_mae,
        #     'MTLR C-Idx': mtlr_cindex, 'MTLR MAE': mtlr_mae,
        #     'DVFM C-Idx': dvfm_cindex, 'DVFM MAE': dvfm_mae,
        #     'KL Divergence': kl_div, 'Cos Sim': cos_sim, 'Kendall Tau': learned_tau
        # }
        # results.append(row)
        # --- DEPENDENCE DETECTION ---
        print("Analyzing Conditional Dependence...")
        
        # Pass copula_type and theta=2.0 (matching your generation theta)
        tau_true, tau_x, tau_xz = analyze_conditional_dependence(
            dvfm, X, copula, theta=Theta, device=device
        )
        
        print(f"Copula: {copula.upper()}")
        print(f"  True Tau (T,C|X):    {tau_true:.4f}")
        print(f"  Model Tau(T,C|X):    {tau_x:.4f}  (Should match True)")
        print(f"  Model Tau(T,C|X,Z):  {tau_xz:.4f}  (Should be ~0)")
        
        # Get KL for context
        # (Quick calculation on test set)
        dummy_t = torch.zeros(len(X_test)).to(device)
        dummy_e = torch.zeros(len(X_test)).to(device)
        mu, logvar = dvfm.encoder(torch.FloatTensor(X_test).to(device), dummy_t, dummy_e)
        kl_div = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)).mean().item()
        
        print(f"  KL Divergence:           {kl_div:.4f}")
        print(f"  IBS Oracle (C/DS/M/D):   {cox_ibs_oracle:.4f} / {ds_ibs_oracle:.4f} / {mtlr_ibs_oracle:.4f} / {dvfm_ibs_oracle:.4f}")
        print(f"  IBS IPCW @tau={eval_tau:.2f}:    {cox_ibs_ipcw:.4f} / {ds_ibs_ipcw:.4f} / {mtlr_ibs_ipcw:.4f} / {dvfm_ibs_ipcw:.4f}")
        
        # Store Results
        
        # Store Results
        row = {
            'Copula': copula,
            # Performance (Test Set)
            'CoxPH C-Idx': cox_cindex, 'CoxPH MAE': cox_mae,
            'DeepSurv C-Idx': ds_cindex, 'DeepSurv MAE': ds_mae,
            'MTLR C-Idx': mtlr_cindex, 'MTLR MAE': mtlr_mae,
            'DVFM C-Idx': dvfm_cindex,'DVFM MAE': dvfm_mae,
            # Oracle performance (full true event times)
            'CoxPH MAE Oracle': cox_mae_oracle,
            'DeepSurv MAE Oracle': ds_mae_oracle,
            'MTLR MAE Oracle': mtlr_mae_oracle,
            'DVFM MAE Oracle': dvfm_mae_oracle,
            'CoxPH IBS Oracle': cox_ibs_oracle,
            'DeepSurv IBS Oracle': ds_ibs_oracle,
            'MTLR IBS Oracle': mtlr_ibs_oracle,
            'DVFM IBS Oracle': dvfm_ibs_oracle,
            'CoxPH IBS IPCW': cox_ibs_ipcw,
            'DeepSurv IBS IPCW': ds_ibs_ipcw,
            'MTLR IBS IPCW': mtlr_ibs_ipcw,
            'DVFM IBS IPCW': dvfm_ibs_ipcw,
            'Eval Tau': eval_tau,
            # Diagnostics (All Data)
            'True Tau': tau_true,
            'Tau(T,C|X)': tau_x,
            'Tau(T,C|X,Z)': tau_xz,
            'Tau Error |X|': abs(tau_x - tau_true),
            'KL Div': kl_div
        }
        results.append(row)

    # Display Results
    res_df = pd.DataFrame(results)
    print("\n" + "="*50)
    print("FINAL RESULTS COMPARISON")
    print("="*50)
    print(res_df.round(4).to_string())
    
    # ==========================================
    # PLOT 1: C-INDEX
    # ==========================================
    plt.figure(figsize=(14, 13))
    
    plt.subplot(3, 1, 1)
    c_cols = ['DeepSurv C-Idx', 'MTLR C-Idx', 'DVFM C-Idx']
    ax1 = res_df.plot(x='Copula', y=c_cols, kind='bar', ax=plt.gca(), width=0.8,
                      color=['#e74c3c', '#f39c12', '#2ecc71'])
    plt.title('C-Index Comparison (Higher is Better)', fontsize=12, fontweight='bold')
    plt.ylabel('Concordance Index')
    plt.xlabel('Copula Type')
    plt.xticks(rotation=0)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend(title='Models', loc='lower right', frameon=True)

    # ==========================================
    # PLOT 2: MAE
    # ==========================================
    plt.subplot(3, 1, 2)
    mae_cols = ['DeepSurv MAE', 'MTLR MAE', 'DVFM MAE']
    ax2 = res_df.plot(x='Copula', y=mae_cols, kind='bar', ax=plt.gca(), width=0.8,
                      color=['#e74c3c', '#f39c12', '#2ecc71'])
    plt.title('MAE Comparison (Lower is Better)', fontsize=12, fontweight='bold')
    plt.ylabel('Mean Absolute Error')
    plt.xlabel('Copula Type')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend(title='Models', loc='upper right', frameon=True)

    # ==========================================
    # PLOT 3: IPCW IBS (Observed Data)
    # ==========================================
    plt.subplot(3, 1, 3)
    ibs_cols = ['DeepSurv IBS IPCW', 'MTLR IBS IPCW', 'DVFM IBS IPCW']
    ax3 = res_df.plot(x='Copula', y=ibs_cols, kind='bar', ax=plt.gca(), width=0.8,
                      color=['#e74c3c', '#f39c12', '#2ecc71'])
    plt.title('IPCW IBS Comparison (Lower is Better)', fontsize=12, fontweight='bold')
    plt.ylabel('Integrated Brier Score')
    plt.xlabel('Copula Type')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend(title='Models', loc='upper right', frameon=True)

    # # ==========================================
    # # PLOT 4: GRADIENT CORRELATION (Decoder Sensitivity)
    # # ==========================================
    # plt.subplot(2, 2, 4)
    # ax4 = res_df.plot(x='Copula', y=['Cos Sim'], kind='bar', ax=plt.gca(), 
    #                   width=0.6, color=['#e67e22'], legend=False)
    # plt.title('Decoder Gradient Correlation (Higher = Shared Frailty)', fontsize=12, fontweight='bold')
    # plt.ylabel('Cos Sim between ∇psi_T and ∇psi_C')
    # plt.xlabel('Copula Type')
    # plt.xticks(rotation=0)
    # plt.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_experiment()

###########################################################################################

# def run_experiment():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
    
#     copulas = ['gaussian', 'clayton', 'frank', 'gumbel']
#     results = []
    
#     for copula in copulas:
#         print(f"\n" + "="*50)
#         print(f"Processing Copula: {copula.upper()}")
#         print("="*50)
        
#         # 1. Generate Data
#         X, obs_time, event, true_T, true_C = generate_copula_data(
#             n_samples=500, n_features=10, copula_type=copula, theta=1.0
#         )
        
#         # Split data (We need True T for MAE calculation, usually unavailable, but available here)
#         X_train, X_test, t_train, t_test, e_train, e_test, true_T_train, true_T_test = train_test_split(
#             X, obs_time, event, true_T, test_size=0.3, random_state=42
#         )
        
#         # --- Model 1: CoxPH ---
#                 # --- Model 1: CoxPH ---
#         print("Training CoxPH...")
#         try:
#             # Train
#             df_train = pd.DataFrame(X_train, columns=[f'X{i}' for i in range(X_train.shape[1])])
#             df_train['time'] = t_train
#             df_train['event'] = e_train
            
#             cph = CoxPHFitter()
#             cph.fit(df_train, duration_col='time', event_col='event')
            
#             df_test = pd.DataFrame(X_test, columns=[f'X{i}' for i in range(X_test.shape[1])])
            
#             # Predict Risk (for C-index)
#             cox_risk = cph.predict_partial_hazard(df_test).values
            
#             # Predict Median (for MAE)
#             cox_median = cph.predict_median(df_test).values.flatten()
            
#             # --- FIX FOR INFINITE MAE ---
#             # If predicted median is Inf (curve never drops below 0.5), 
#             # replace it with the max observed time in training data.
#             max_train_time = np.max(t_train)
#             cox_median[np.isinf(cox_median)] = max_train_time
            
#             # Calculate metrics
#             cox_cindex = concordance_index(t_test, cox_median, e_test)
#             cox_mae = np.mean(np.abs(true_T_test - cox_median))
            
#         except Exception as e:
#             print(f"CoxPH failed: {e}")
#             cox_cindex, cox_mae = 0.5, 999.0
#         # print("Training CoxPH...")
#         # try:
#         #     cox_risk, cox_median, _ = train_coxph(X_train, t_train, e_train, X_test)
#         #     cox_cindex = concordance_index(t_test, -cox_risk, e_test)
#         #     cox_mae = np.mean(np.abs(true_T_test - cox_median))
#         # except Exception as e:
#         #     print(f"CoxPH failed: {e}")
#         #     cox_cindex, cox_mae = 0.5, 999
            
#         # --- Model 2: DeepSurv ---
#         print("Training DeepSurv...")
#         ds_risk, ds_median = train_deepsurv(X_train, t_train, e_train, X_test, device=device)
#         ds_cindex = concordance_index(t_test, ds_median, e_test)
#         ds_mae = np.mean(np.abs(true_T_test - ds_median))
        
#         # --- Model 3: MTLR (Simplified) ---
#         print("Training MTLR...")
#         mtlr_risk, mtlr_median = train_mtlr(X_train, t_train, e_train, X_test, device=device, num_bins = 20)
#         # Note: MTLR outputs expected time, so risk is negative expected time
#         mtlr_cindex = concordance_index(t_test, mtlr_median, e_test) 
#         mtlr_mae = np.mean(np.abs(true_T_test - mtlr_median))

#         # --- Model 4: DVFM (Proposed) ---
#         print("Training DVFM...")
#         # Prepare Data Loaders
#         train_ds = SurvivalDataset(X_train, t_train, e_train)
#         val_ds = SurvivalDataset(X_test, t_test, e_test)
#         train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
#         val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
        
#         # Init Model
#         dvfm = DVFM(input_dim=X.shape[1], latent_dim=4).to(device)
        
#         # Train
#         train_dvfm(dvfm, train_loader, val_loader, n_epochs=200, 
#                    beta_max=1, free_bits=0, device=device)
        
#         # Predict (Monte Carlo integration)
#         # Use simple linspace for time points for curve generation
#         max_time = np.max(t_train) * 1.5
#         time_points = np.linspace(0, max_time, 1000)
        
#         surv_curves = predict_survival_curves(dvfm, X_test, time_points, n_samples=100, device=device)
#         dvfm_median = get_median_survival_time(surv_curves, time_points)
        
#         # For C-index, we use negative median time (higher median = lower risk)
#         dvfm_cindex = concordance_index(t_test, dvfm_median, e_test)
#         dvfm_mae = np.mean(np.abs(true_T_test - dvfm_median))
        

#         #######################################
#         learned_tau = measure_learned_dependence(dvfm, X_test, device=device)

#         print(f"Copula: {copula.upper()}")
#         print(f"Ground Truth Dependence (approx theta): {2.0}") # Or whatever theta you used
#         print(f"DVFM Learned Kendall's Tau: {learned_tau:.4f}")
#         ########################################################################################
#         # Store Results
#         row = {
#             'Copula': copula,
#             'CoxPH C-Idx': cox_cindex, 'CoxPH MAE': cox_mae,
#             'DeepSurv C-Idx': ds_cindex, 'DeepSurv MAE': ds_mae,
#             'MTLR C-Idx': mtlr_cindex, 'MTLR MAE': mtlr_mae,
#             'DVFM C-Idx': dvfm_cindex, 'DVFM MAE': dvfm_mae
#         }
#         results.append(row)

#     # # 9. Display Results
#     # res_df = pd.DataFrame(results)
#     # print("\n" + "="*50)
#     # print("FINAL RESULTS COMPARISON")
#     # print("="*50)
#     # print(res_df.round(4).to_string())
    
#     # # Optional: Plot comparison for last copula
#     # res_df.plot(x='Copula', y=['CoxPH C-Idx', 'DeepSurv C-Idx', 'MTLR C-Idx', 'DVFM C-Idx'], 
#     #             kind='bar', figsize=(10, 6), title='C-Index Comparison')
#     # plt.show()

#         # ... (Previous code inside run_experiment) ...
    
#     # 9. Display Results and Plot
#     res_df = pd.DataFrame(results)
#     print("\n" + "="*50)
#     print("FINAL RESULTS COMPARISON")
#     print("="*50)
#     print(res_df.round(4).to_string())
    
#     # ==========================================
#     # PLOT 1: C-INDEX (Higher is Better)
#     # ==========================================
#     plt.figure(figsize=(14, 6))
    
#     # Subplot 1: C-Index
#     plt.subplot(1, 2, 1)
#     c_cols = ['DeepSurv C-Idx', 'MTLR C-Idx', 'DVFM C-Idx'] #'CoxPH C-Idx', 
    
#     # Use Pandas plotting wrapper
#     ax1 = res_df.plot(x='Copula', y=c_cols, kind='bar', ax=plt.gca(), width=0.8,
#                       color=['#e74c3c', '#f39c12', '#2ecc71', '#3498db'])
    
#     plt.title('C-Index Comparison (Higher is Better)', fontsize=12, fontweight='bold')
#     plt.ylabel('Concordance Index')
#     plt.xlabel('Copula Type')
#     plt.xticks(rotation=0)
#     plt.ylim(0, 1.0) # Zoom in on the relevant range
#     plt.grid(axis='y', linestyle='--', alpha=0.6)
#     plt.legend(title='Models', loc='lower right', frameon=True)

#     # ==========================================
#     # PLOT 2: MAE (Lower is Better)
#     # ==========================================
#     plt.subplot(1, 2, 2)
#     mae_cols = [ 'DeepSurv MAE', 'MTLR MAE', 'DVFM MAE'] #'CoxPH MAE',
    
#     # Plot bars
#     ax2 = res_df.plot(x='Copula', y=mae_cols, kind='bar', ax=plt.gca(), width=0.8,
#                       color=['#e74c3c', '#f39c12', '#2ecc71', '#3498db'])
    
#     plt.title('MAE Comparison (Lower is Better)', fontsize=12, fontweight='bold')
#     plt.ylabel('Mean Absolute Error (Time Units)')
#     plt.xlabel('Copula Type')
#     plt.xticks(rotation=0)
#     plt.grid(axis='y', linestyle='--', alpha=0.6)
#     plt.legend(title='Models', loc='upper right', frameon=True)
    
#     # Optional: Log scale if CoxPH MAE is huge
#     # plt.yscale('log') 

#     # Add text labels on top of the bars for DVFM (to highlight your method)
#     # Find the index of DVFM column (3rd in the list of Y columns)
#     for i, p in enumerate(ax2.patches):
#         # Only label the DVFM bars (usually the last set of bars in the loop)
#         # Or label all if you prefer
#         if p.get_height() > 0 and p.get_height() != float('inf'):
#             ax2.annotate(f"{p.get_height():.2f}",
#                         (p.get_x() + p.get_width() / 2., p.get_height()),
#                         ha='center', va='bottom', xytext=(0, 5),
#                         textcoords='offset points', fontsize=8)

#     plt.tight_layout()
#     plt.show()


# if __name__ == "__main__":
#     run_experiment()
