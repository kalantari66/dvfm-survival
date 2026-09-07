"""Paper baselines: CoxPH, DeepSurv, MTLR, and Clayton-Weibull AFT."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from lifelines import CoxPHFitter
from torch.utils.data import DataLoader

from utility.data import SurvivalDataset

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
    # Cox risk sets must span the complete cohort. Minibatch-local risk sets
    # optimize a different objective and make the baseline batch-dependent.
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    
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


def train_mtlr(X_train, time_train, event_train, X_test, num_bins=45,
               n_epochs=200, lr=0.005, device='cpu', eval_time_points=None):
    """Neural MTLR with an explicit tail category and censored likelihood."""
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
            # MTLR converts interval scores to density logits by reverse
            # cumulative summation and appends an explicit beyond-grid tail.
            scores = self.net(x)
            interval_logits = torch.flip(
                torch.cumsum(torch.flip(scores, dims=[1]), dim=1), dims=[1]
            )
            return torch.cat(
                [interval_logits, torch.zeros((len(x), 1), device=x.device)], dim=1
            )
    
    model = N_MTLR(X_train.shape[1], actual_num_bins).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = y_train_bins.to(device)
    event_t = torch.FloatTensor(event_train).to(device)
    
    # 4. Observed-data MTLR likelihood.
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        logits = model(X_train_t)
        log_probs = torch.log_softmax(logits, dim=1)
        event_log_likelihood = log_probs.gather(1, y_train_t[:, None]).squeeze(1)
        # A censored subject contributes log P(T > c). With discretized
        # intervals this is the log-sum of all later intervals and the tail.
        category = torch.arange(actual_num_bins + 1, device=device)[None, :]
        later = category > y_train_t[:, None]
        censored_log_likelihood = torch.logsumexp(
            log_probs.masked_fill(~later, float('-inf')), dim=1
        )
        log_likelihood = (
            event_t * event_log_likelihood
            + (1 - event_t) * censored_log_likelihood
        )
        loss = -log_likelihood.mean()
        
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
        cdf = np.cumsum(probs[:, :-1], axis=1)
        survival_probs = 1.0 - cdf
        
        # Calculate Risk Score (Expected Time)
        # Midpoints of bins
        bin_mids = (bins[:-1] + bins[1:]) / 2
        tail_time = bins[-1] + max(bins[-1] - bins[-2], 1e-5)
        category_times = np.r_[bin_mids, tail_time]
        predicted_means = np.sum(probs * category_times, axis=1)
        
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


def _fit_cox_and_predict_survival(X_train, t_train, e_train, X_test, time_points):
    """CoxPH evaluation matching the active supplied reference run."""
    df_train = pd.DataFrame(X_train, columns=[f"X{i}" for i in range(X_train.shape[1])])
    df_train["time"] = t_train
    df_train["event"] = e_train

    cph = CoxPHFitter()
    cph.fit(df_train, duration_col="time", event_col="event")

    df_test = pd.DataFrame(X_test, columns=[f"X{i}" for i in range(X_test.shape[1])])
    partial_hazard = cph.predict_partial_hazard(df_test).values.flatten()

    baseline_survival = cph.baseline_survival_
    baseline_times = baseline_survival.index.values
    baseline_s = baseline_survival.values.flatten()
    baseline_interp = np.interp(
        time_points, baseline_times, baseline_s, left=1.0, right=baseline_s[-1]
    )
    surv_curves = np.power(baseline_interp[None, :], partial_hazard[:, None])

    max_train_time = float(np.max(t_train))
    medians = []
    for risk_score in partial_hazard:
        patient_survival = baseline_s ** risk_score
        crossing_idx = np.where(patient_survival <= 0.5)[0]
        medians.append(
            baseline_times[crossing_idx[0]] if len(crossing_idx) > 0 else max_train_time
        )
    return np.asarray(medians, dtype=float), surv_curves


class ClaytonWeibullAFT(nn.Module):
    """
    Parametric copula model for dependent censoring:
    - Weibull margins for T and C with log-scale linear in X
    - Clayton copula links margins
    """
    def __init__(self, n_features):
        super().__init__()
        self.beta_t = nn.Parameter(torch.zeros(n_features + 1))
        self.beta_c = nn.Parameter(torch.zeros(n_features + 1))
        self.log_shape_t = nn.Parameter(torch.tensor(0.0))
        self.log_shape_c = nn.Parameter(torch.tensor(0.0))
        self.raw_theta = nn.Parameter(torch.tensor(0.2))

    def _margin_stats(self, x_aug, t):
        eps = 1e-8
        t = torch.clamp(t, min=eps)

        shape_t = torch.exp(self.log_shape_t) + eps
        shape_c = torch.exp(self.log_shape_c) + eps
        scale_t = torch.exp(x_aug @ self.beta_t) + eps
        scale_c = torch.exp(x_aug @ self.beta_c) + eps

        z_t = torch.clamp(t / scale_t, min=eps)
        z_c = torch.clamp(t / scale_c, min=eps)
        h_t = z_t ** shape_t
        h_c = z_c ** shape_c

        s_t = torch.exp(-h_t)
        s_c = torch.exp(-h_c)
        f_t = (shape_t / scale_t) * (z_t ** (shape_t - 1.0)) * s_t
        f_c = (shape_c / scale_c) * (z_c ** (shape_c - 1.0)) * s_c
        u = torch.clamp(1.0 - s_t, min=eps, max=1.0 - eps)
        v = torch.clamp(1.0 - s_c, min=eps, max=1.0 - eps)
        return f_t, f_c, u, v, shape_t, scale_t

    def neg_log_lik(self, x_aug, t_obs, event):
        eps = 1e-8
        f_t, f_c, u, v, _, _ = self._margin_stats(x_aug, t_obs)
        theta = torch.nn.functional.softplus(self.raw_theta) + 1e-4

        a = torch.clamp(u ** (-theta) + v ** (-theta) - 1.0, min=eps)
        c_pow = a ** (-1.0 / theta - 1.0)
        dC_du = c_pow * (u ** (-theta - 1.0))
        dC_dv = c_pow * (v ** (-theta - 1.0))

        l_event = f_t * torch.clamp(1.0 - dC_du, min=eps)
        l_cens = f_c * torch.clamp(1.0 - dC_dv, min=eps)
        l = event * l_event + (1.0 - event) * l_cens
        return -torch.mean(torch.log(torch.clamp(l, min=eps)))

    def predict_survival(self, x, time_points):
        eps = 1e-8
        x = np.asarray(x, dtype=np.float32)
        x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)
        # Prediction must follow the fitted module. In particular, a CUDA-fitted
        # model cannot multiply CPU covariates by CUDA parameters.
        x_t = torch.from_numpy(x_aug).to(self.beta_t.device)

        with torch.no_grad():
            scale_t = torch.exp(x_t @ self.beta_t).cpu().numpy() + eps
            shape_t = float((torch.exp(self.log_shape_t) + eps).cpu().item())

        t = np.asarray(time_points, dtype=float)[None, :]
        z = np.maximum(t / scale_t[:, None], eps)
        surv = np.exp(-(z ** shape_t))
        return surv


def fit_clayton_weibull_aft(X_train, t_train, e_train, X_test, time_points, epochs=400, lr=5e-3, device="cpu", return_model=False):
    x_train = np.asarray(X_train, dtype=np.float32)
    x_aug = np.concatenate([x_train, np.ones((x_train.shape[0], 1), dtype=np.float32)], axis=1)
    t_obs = np.asarray(t_train, dtype=np.float32)
    e_obs = np.asarray(e_train, dtype=np.float32)

    x_t = torch.from_numpy(x_aug).to(device)
    t_t = torch.from_numpy(t_obs).to(device)
    e_t = torch.from_numpy(e_obs).to(device)

    model = ClaytonWeibullAFT(n_features=X_train.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        optimizer.zero_grad()
        loss = model.neg_log_lik(x_t, t_t, e_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

    surv = model.predict_survival(X_test, time_points)
    medians = np.zeros(len(X_test), dtype=float)
    max_train_time = float(np.max(t_train))
    for i in range(len(X_test)):
        idx = np.where(surv[i] <= 0.5)[0]
        medians[i] = time_points[idx[0]] if len(idx) > 0 else max_train_time

    if return_model:
        return medians, surv, model
    return medians, surv


__all__ = ["train_deepsurv", "train_mtlr", "_fit_cox_and_predict_survival", "ClaytonWeibullAFT", "fit_clayton_weibull_aft"]
