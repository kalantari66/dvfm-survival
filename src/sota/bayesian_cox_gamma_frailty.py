"""Bayesian individual Cox model with Gamma frailty.

This is the scalable analogue of the individual-frailty model in the PyMC
frailty tutorial.  The event hazard is piecewise exponential,

    h_i(t | u_i) = u_i exp(x_i' beta) h_0(t),

with ``u_i ~ Gamma(alpha, rate=alpha)``.  Integrating ``u_i`` gives an exact
marginal likelihood for the population parameters, and conditioning on an
observed ``(time, event)`` gives an exact Gamma posterior for every subject.
Population parameters are MAP estimates; individual frailties retain their
full conjugate posterior conditional on those estimates (empirical Bayes).
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from torch import nn


class BayesianIndividualCoxGammaFrailty(nn.Module):
    """Piecewise-exponential Cox model with subject-specific Gamma frailty."""

    def __init__(self, n_features: int, interval_starts, *, dtype=torch.float64):
        super().__init__()
        starts = torch.as_tensor(interval_starts, dtype=dtype)
        if starts.ndim != 1 or len(starts) < 1 or starts[0] != 0:
            raise ValueError("interval_starts must be one-dimensional and start at zero")
        if len(starts) > 1 and not bool(torch.all(starts[1:] > starts[:-1])):
            raise ValueError("interval_starts must be strictly increasing")
        self.register_buffer("interval_starts", starts)
        self.beta = nn.Parameter(torch.zeros(int(n_features), dtype=dtype))
        self.log_baseline_hazards = nn.Parameter(torch.zeros(len(starts), dtype=dtype))
        self.log_alpha = nn.Parameter(torch.tensor(np.log(2.0), dtype=dtype))

    @property
    def dtype(self):
        return self.beta.dtype

    def alpha(self):
        return torch.exp(self.log_alpha.clamp(np.log(1e-3), np.log(1e4)))

    def baseline_hazards(self):
        return torch.exp(self.log_baseline_hazards.clamp(-30.0, 30.0))

    def _cumulative_baseline_hazard(self, time):
        time = torch.as_tensor(time, dtype=self.dtype, device=self.beta.device)
        starts = self.interval_starts
        if len(starts) == 1:
            durations = time[:, None].clamp_min(0.0)
        else:
            widths = starts[1:] - starts[:-1]
            finite = torch.minimum(
                (time[:, None] - starts[:-1]).clamp_min(0.0), widths[None, :]
            )
            final = (time - starts[-1]).clamp_min(0.0)[:, None]
            durations = torch.cat((finite, final), dim=1)
        return durations @ self.baseline_hazards()

    def _event_baseline_hazard(self, time):
        time = torch.as_tensor(time, dtype=self.dtype, device=self.beta.device)
        index = torch.bucketize(time, self.interval_starts[1:], right=True)
        return self.baseline_hazards()[index]

    def marginal_log_likelihood(self, X, time, event):
        """Per-subject likelihood after analytically integrating frailty."""
        X = torch.as_tensor(X, dtype=self.dtype, device=self.beta.device)
        time = torch.as_tensor(time, dtype=self.dtype, device=self.beta.device)
        event = torch.as_tensor(event, dtype=self.dtype, device=self.beta.device)
        alpha = self.alpha()
        relative_risk = torch.exp((X @ self.beta).clamp(-30.0, 30.0))
        cumulative_risk = relative_risk * self._cumulative_baseline_hazard(time)
        log_event_hazard = (
            torch.log(self._event_baseline_hazard(time).clamp_min(1e-30))
            + X @ self.beta
        )
        return (
            event * log_event_hazard
            + torch.lgamma(alpha + event)
            - torch.lgamma(alpha)
            + alpha * torch.log(alpha)
            - (alpha + event) * torch.log(alpha + cumulative_risk)
        )

    def map_objective(
        self, X, time, event, *, beta_prior_sd=2.5,
        log_hazard_prior_sd=5.0, log_alpha_prior_sd=2.0,
        baseline_smoothness=1.0,
    ):
        """Negative log posterior per observation for stable optimization."""
        n = max(int(len(time)), 1)
        log_likelihood = self.marginal_log_likelihood(X, time, event).sum()
        penalty = 0.5 * torch.sum((self.beta / float(beta_prior_sd)) ** 2)
        penalty = penalty + 0.5 * torch.sum(
            (self.log_baseline_hazards / float(log_hazard_prior_sd)) ** 2
        )
        penalty = penalty + 0.5 * (
            self.log_alpha / float(log_alpha_prior_sd)
        ) ** 2
        if len(self.log_baseline_hazards) > 1:
            penalty = penalty + float(baseline_smoothness) * torch.sum(
                torch.diff(self.log_baseline_hazards) ** 2
            )
        return (-log_likelihood + penalty) / n

    @torch.no_grad()
    def posterior_parameters(self, X, time, event):
        """Return exact posterior Gamma shape and rate for each subject."""
        X = torch.as_tensor(X, dtype=self.dtype, device=self.beta.device)
        time = torch.as_tensor(time, dtype=self.dtype, device=self.beta.device)
        event = torch.as_tensor(event, dtype=self.dtype, device=self.beta.device)
        alpha = self.alpha()
        risk = torch.exp((X @ self.beta).clamp(-30.0, 30.0))
        shape = alpha + event
        rate = alpha + risk * self._cumulative_baseline_hazard(time)
        return shape, rate

    @torch.no_grad()
    def posterior_log_frailty_mean(self, X, time, event):
        """Posterior expectation E[log(u_i) | observed time and status]."""
        shape, rate = self.posterior_parameters(X, time, event)
        return (torch.digamma(shape) - torch.log(rate)).cpu().numpy()

    @torch.no_grad()
    def posterior_frailty_mean(self, X, time, event):
        shape, rate = self.posterior_parameters(X, time, event)
        return (shape / rate).cpu().numpy()

    @torch.no_grad()
    def predict_marginal_survival(self, X, time_points):
        """Population-averaged survival for a new subject of unknown frailty."""
        X = torch.as_tensor(X, dtype=self.dtype, device=self.beta.device)
        grid = torch.as_tensor(
            time_points, dtype=self.dtype, device=self.beta.device
        ).clamp_min(0.0)
        alpha = self.alpha()
        risk = torch.exp((X @ self.beta).clamp(-30.0, 30.0))
        cumulative = self._cumulative_baseline_hazard(grid)
        log_survival = -alpha * torch.log1p(
            risk[:, None] * cumulative[None, :] / alpha
        )
        return torch.exp(log_survival).cpu().numpy()


def _interval_starts(time, event, n_intervals: int):
    event_times = np.asarray(time, dtype=float)[np.asarray(event, dtype=int) == 1]
    reference = event_times if len(event_times) >= 2 else np.asarray(time, dtype=float)
    quantiles = np.linspace(0.0, 1.0, int(n_intervals) + 1)[1:-1]
    interior = np.unique(np.quantile(reference, quantiles)) if len(quantiles) else []
    interior = np.asarray(interior, dtype=float)
    interior = interior[np.isfinite(interior) & (interior > 0.0)]
    return np.r_[0.0, interior]


class _PiecewiseExponentialHazard(nn.Module):
    """One cause's log-linear predictor and piecewise-constant baseline hazard,
    parameterised exactly as in ``BayesianIndividualCoxGammaFrailty``."""

    def __init__(self, n_features: int, interval_starts, *, dtype=torch.float64):
        super().__init__()
        starts = torch.as_tensor(interval_starts, dtype=dtype)
        if starts.ndim != 1 or len(starts) < 1 or starts[0] != 0:
            raise ValueError("interval_starts must be one-dimensional and start at zero")
        if len(starts) > 1 and not bool(torch.all(starts[1:] > starts[:-1])):
            raise ValueError("interval_starts must be strictly increasing")
        self.register_buffer("interval_starts", starts)
        self.beta = nn.Parameter(torch.zeros(int(n_features), dtype=dtype))
        self.log_baseline_hazards = nn.Parameter(torch.zeros(len(starts), dtype=dtype))

    dtype = BayesianIndividualCoxGammaFrailty.dtype
    baseline_hazards = BayesianIndividualCoxGammaFrailty.baseline_hazards
    _cumulative_baseline_hazard = BayesianIndividualCoxGammaFrailty._cumulative_baseline_hazard
    _event_baseline_hazard = BayesianIndividualCoxGammaFrailty._event_baseline_hazard

    def cumulative_hazard(self, X, time):
        """exp(x' beta) H_0(t): the cumulative hazard at unit frailty."""
        return torch.exp((X @ self.beta).clamp(-30.0, 30.0)) * self._cumulative_baseline_hazard(time)

    def log_hazard(self, X, time):
        return torch.log(self._event_baseline_hazard(time).clamp_min(1e-30)) + X @ self.beta

    def penalty(self, beta_prior_sd, log_hazard_prior_sd, baseline_smoothness):
        penalty = 0.5 * torch.sum((self.beta / float(beta_prior_sd)) ** 2)
        penalty = penalty + 0.5 * torch.sum(
            (self.log_baseline_hazards / float(log_hazard_prior_sd)) ** 2
        )
        if len(self.log_baseline_hazards) > 1:
            penalty = penalty + float(baseline_smoothness) * torch.sum(
                torch.diff(self.log_baseline_hazards) ** 2
            )
        return penalty


class BayesianJointCoxGammaFrailty(nn.Module):
    """Event and censoring hazards that share one Gamma frailty.

        h_E(t | x, u) = u exp(x' beta_E) h_0E(t),
        h_C(t | x, u) = u exp(x' beta_C) h_0C(t),    u ~ Gamma(alpha, rate=alpha),

    each cause with its own piecewise-exponential baseline. Given u the two
    times are independent, so for an observed (t, delta) with cause
    j = E if delta = 1 else C, and Lambda = Lambda_E(t | x) + Lambda_C(t | x),

        p(t, delta | x) = h_j(t | x) alpha^alpha Gamma(alpha + 1) / Gamma(alpha)
                          (alpha + Lambda)^-(alpha + 1),

    and the frailty posterior is Gamma(alpha + 1, alpha + Lambda) for every
    subject, event or censored.
    """

    def __init__(self, n_features: int, event_interval_starts, censor_interval_starts,
                 *, dtype=torch.float64):
        super().__init__()
        self.event = _PiecewiseExponentialHazard(n_features, event_interval_starts, dtype=dtype)
        self.censor = _PiecewiseExponentialHazard(n_features, censor_interval_starts, dtype=dtype)
        self.log_alpha = nn.Parameter(torch.tensor(np.log(2.0), dtype=dtype))

    dtype = BayesianIndividualCoxGammaFrailty.dtype
    alpha = BayesianIndividualCoxGammaFrailty.alpha

    @property
    def beta(self):
        # dtype/device lookups shared with the event-only model read ``beta``.
        return self.event.beta

    def _tensors(self, X, time, event):
        return (torch.as_tensor(value, dtype=self.dtype, device=self.beta.device)
                for value in (X, time, event))

    def total_cumulative_hazard(self, X, time):
        return self.event.cumulative_hazard(X, time) + self.censor.cumulative_hazard(X, time)

    def marginal_log_likelihood(self, X, time, event):
        """Per-subject likelihood after analytically integrating frailty."""
        X, time, event = self._tensors(X, time, event)
        alpha = self.alpha()
        log_hazard = (
            event * self.event.log_hazard(X, time)
            + (1.0 - event) * self.censor.log_hazard(X, time)
        )
        return (
            log_hazard
            + alpha * torch.log(alpha)
            + torch.lgamma(alpha + 1.0)
            - torch.lgamma(alpha)
            - (alpha + 1.0) * torch.log(alpha + self.total_cumulative_hazard(X, time))
        )

    def map_objective(
        self, X, time, event, *, beta_prior_sd=2.5,
        log_hazard_prior_sd=5.0, log_alpha_prior_sd=2.0,
        baseline_smoothness=1.0,
    ):
        """Negative log posterior per observation; both causes get the
        event-only model's priors."""
        n = max(int(len(time)), 1)
        log_likelihood = self.marginal_log_likelihood(X, time, event).sum()
        penalty = sum(
            cause.penalty(beta_prior_sd, log_hazard_prior_sd, baseline_smoothness)
            for cause in (self.event, self.censor)
        )
        penalty = penalty + 0.5 * (self.log_alpha / float(log_alpha_prior_sd)) ** 2
        return (-log_likelihood + penalty) / n

    @torch.no_grad()
    def posterior_parameters(self, X, time, event):
        """Return exact posterior Gamma shape and rate for each subject."""
        X, time, event = self._tensors(X, time, event)
        alpha = self.alpha()
        rate = alpha + self.total_cumulative_hazard(X, time)
        return torch.full_like(rate, 1.0) * (alpha + 1.0), rate

    posterior_log_frailty_mean = BayesianIndividualCoxGammaFrailty.posterior_log_frailty_mean
    posterior_frailty_mean = BayesianIndividualCoxGammaFrailty.posterior_frailty_mean

    @torch.no_grad()
    def predict_marginal_survival(self, X, time_points):
        """Population-averaged event-time survival, (1 + Lambda_E / alpha)^-alpha."""
        X = torch.as_tensor(X, dtype=self.dtype, device=self.beta.device)
        grid = torch.as_tensor(time_points, dtype=self.dtype, device=self.beta.device).clamp_min(0.0)
        alpha = self.alpha()
        risk = torch.exp((X @ self.event.beta).clamp(-30.0, 30.0))
        cumulative = self.event._cumulative_baseline_hazard(grid)
        return torch.exp(-alpha * torch.log1p(risk[:, None] * cumulative[None, :] / alpha)).cpu().numpy()


def fit_bayesian_cox_gamma_frailty(
    X_train, time_train, event_train,
    X_validation, time_validation, event_validation,
    X_test, time_points, config: dict, device="cpu",
):
    """Fit the scalable Cox--Gamma model and return benchmark artifacts."""
    dtype = torch.float64 if str(config.get("dtype", "float64")) == "float64" else torch.float32
    device = torch.device(device)
    starts = _interval_starts(time_train, event_train, int(config.get("n_intervals", 10)))
    model = BayesianIndividualCoxGammaFrailty(
        np.asarray(X_train).shape[1], starts, dtype=dtype
    ).to(device)

    event_rate = max(float(np.sum(event_train)) / max(float(np.sum(time_train)), 1e-12), 1e-8)
    with torch.no_grad():
        model.log_baseline_hazards.fill_(np.log(event_rate))
    return _fit_map(
        model, X_train, time_train, event_train,
        X_validation, time_validation, event_validation,
        X_test, time_points, config, device, dtype,
    )


def fit_joint_cox_gamma_frailty(
    X_train, time_train, event_train,
    X_validation, time_validation, event_validation,
    X_test, time_points, config: dict, device="cpu",
):
    """Fit the shared-frailty event + censoring model with the event-only
    model's procedure and hyperparameters; the censoring baseline gets the same
    number of intervals, placed at censoring-time quantiles."""
    dtype = torch.float64 if str(config.get("dtype", "float64")) == "float64" else torch.float32
    device = torch.device(device)
    n_intervals = int(config.get("n_intervals", 10))
    censored = 1 - np.asarray(event_train, dtype=int)
    model = BayesianJointCoxGammaFrailty(
        np.asarray(X_train).shape[1],
        _interval_starts(time_train, event_train, n_intervals),
        _interval_starts(time_train, censored, n_intervals),
        dtype=dtype,
    ).to(device)

    exposure = max(float(np.sum(time_train)), 1e-12)
    with torch.no_grad():
        for cause, count in ((model.event, np.sum(event_train)), (model.censor, np.sum(censored))):
            cause.log_baseline_hazards.fill_(np.log(max(float(count) / exposure, 1e-8)))
    survival, fit_info, history, model = _fit_map(
        model, X_train, time_train, event_train,
        X_validation, time_validation, event_validation,
        X_test, time_points, config, device, dtype,
    )
    fit_info["frailty_shared_with_censoring"] = True
    return survival, fit_info, history, model


def _fit_map(
    model, X_train, time_train, event_train,
    X_validation, time_validation, event_validation,
    X_test, time_points, config, device, dtype,
):
    X_train_t = torch.as_tensor(X_train, dtype=dtype, device=device)
    time_train_t = torch.as_tensor(time_train, dtype=dtype, device=device)
    event_train_t = torch.as_tensor(event_train, dtype=dtype, device=device)
    X_val_t = torch.as_tensor(X_validation, dtype=dtype, device=device)
    time_val_t = torch.as_tensor(time_validation, dtype=dtype, device=device)
    event_val_t = torch.as_tensor(event_validation, dtype=dtype, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", 0.03)))
    maximum_epochs = int(config.get("epochs", 500))
    minimum_epochs = int(config.get("minimum_epochs", 100))
    patience = int(config.get("early_stopping_patience", 50))
    best_loss, best_epoch, stale, best_state = float("inf"), 0, 0, None
    history = []
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        optimizer.zero_grad()
        objective = model.map_objective(
            X_train_t, time_train_t, event_train_t,
            beta_prior_sd=float(config.get("beta_prior_sd", 2.5)),
            log_hazard_prior_sd=float(config.get("log_hazard_prior_sd", 5.0)),
            log_alpha_prior_sd=float(config.get("log_alpha_prior_sd", 2.0)),
            baseline_smoothness=float(config.get("baseline_smoothness", 1.0)),
        )
        if not bool(torch.isfinite(objective)):
            raise FloatingPointError("Cox--Gamma frailty objective became non-finite")
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("gradient_clip", 10.0)))
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_nll = float(-model.marginal_log_likelihood(
                X_val_t, time_val_t, event_val_t
            ).mean().cpu())
        history.append({
            "epoch": epoch,
            "train_negative_log_posterior": float(objective.detach().cpu()),
            "validation_marginal_nll": validation_nll,
            "frailty_alpha": float(model.alpha().detach().cpu()),
            "frailty_variance": float((1.0 / model.alpha()).detach().cpu()),
        })
        if np.isfinite(validation_nll) and validation_nll < best_loss - float(config.get("minimum_delta", 1e-6)):
            best_loss, best_epoch, stale = validation_nll, epoch, 0
            best_state = deepcopy(model.state_dict())
        else:
            stale += 1
        if epoch >= minimum_epochs and stale >= patience:
            break

    if best_state is None:
        raise FloatingPointError("Cox--Gamma frailty produced no finite checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    survival = model.predict_marginal_survival(X_test, time_points)
    fit_info = {
        "checkpoint": "best_validation_marginal_nll",
        "checkpoint_epoch": int(best_epoch),
        "checkpoint_validation_marginal_nll": float(best_loss),
        "frailty_distribution": "gamma",
        "frailty_alpha": float(model.alpha().detach().cpu()),
        "frailty_variance": float((1.0 / model.alpha()).detach().cpu()),
        "frailty_estimation": "exact_conditional_posterior_empirical_bayes",
        "learned_conditional_kendall_tau": np.nan,
        "conditional_kendall_tau_error": np.nan,
        "absolute_conditional_kendall_tau_error": np.nan,
        "oracle_joint_survival_ise": np.nan,
    }
    return survival, fit_info, history, model


__all__ = [
    "BayesianIndividualCoxGammaFrailty",
    "BayesianJointCoxGammaFrailty",
    "fit_bayesian_cox_gamma_frailty",
    "fit_joint_cox_gamma_frailty",
]
