"""Paper baselines: CoxPH, DeepSurv, MTLR, and Clayton-Weibull AFT."""

from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .coxph import _fit_cox_and_predict_survival
from .mtlr import train_mtlr
from utility.data import SurvivalDataset

def train_deepsurv(X_train, time_train, event_train, X_test,
                   n_epochs=200, batch_size=None, lr=1e-3, device='cpu',
                   eval_time_points=None, hidden_dims=None, dropout=0.3,
                   weight_decay=0.0, X_val=None, time_val=None,
                   event_val=None, early_stopping_patience=None):
    """
    Simple DeepSurv implementation (Cox proportional hazards with neural network)
    """
    class DeepSurv(nn.Module):
        def __init__(self, input_dim, hidden_dims, dropout):
            super(DeepSurv, self).__init__()
            layers = []
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(float(dropout)))
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
    # Katzman et al. (2017) list no batch size among DeepSurv's hyper-parameters
    # (Appendix A.1, Table 3) because the partial likelihood of their Eq. (3) is
    # defined over the full risk set, so the batch is the cohort and one epoch
    # is one gradient step. ``batch_size=None`` requests exactly that.
    cohort_size = len(train_dataset)
    batch_size = cohort_size if batch_size is None else int(batch_size)
    if batch_size < cohort_size:
        raise ValueError(
            f"DeepSurv batch_size={batch_size} is smaller than the training "
            f"cohort ({cohort_size}). Minibatch-local risk sets optimize a "
            f"different objective than the Cox partial likelihood; pass "
            f"batch_size=None (YAML null) to train on the full cohort."
        )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    input_dim = X_train.shape[1]
    model = DeepSurv(input_dim, [64, 32] if hidden_dims is None else hidden_dims, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=float(weight_decay))
    
    use_early_stopping = (
        early_stopping_patience is not None
        and int(early_stopping_patience) > 0
    )
    if use_early_stopping:
        if X_val is None or time_val is None or event_val is None:
            raise ValueError("Early stopping requires X_val, time_val, and event_val")
        X_val_tensor = torch.as_tensor(
            np.asarray(X_val, dtype=np.float32), device=device
        )
        time_val_tensor = torch.as_tensor(
            np.asarray(time_val, dtype=np.float32), device=device
        )
        event_val_tensor = torch.as_tensor(
            np.asarray(event_val, dtype=np.float32), device=device
        )
        if float(event_val_tensor.sum()) <= 0.0:
            raise ValueError(
                "Early stopping requires at least one validation event; the "
                "Cox partial likelihood is undefined on a fully censored split"
            )
        best_validation_nll = float("inf")
        best_state = None
        stale_epochs = 0

    # Training Loop
    for epoch in range(n_epochs):
        model.train()
        for x_b, t_b, e_b in train_loader:
            x_b, t_b, e_b = x_b.to(device), t_b.to(device), e_b.to(device)
            optimizer.zero_grad()
            scores = model(x_b)
            loss = cox_loss(scores, t_b, e_b)
            loss.backward()
            optimizer.step()

        if use_early_stopping:
            model.eval()
            with torch.no_grad():
                # Weight decay belongs to optimization only. Checkpoint
                # selection uses the unregularized partial-likelihood NLL.
                validation_nll = float(
                    cox_loss(model(X_val_tensor), time_val_tensor, event_val_tensor)
                )
            if validation_nll < best_validation_nll:
                best_validation_nll = validation_nll
                best_state = deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= int(early_stopping_patience):
                    break

    if use_early_stopping and best_state is not None:
        model.load_state_dict(best_state)

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
