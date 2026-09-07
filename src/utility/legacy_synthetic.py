"""Legacy copula DGP retained for backward-compatible experiment configurations."""

import numpy as np

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
