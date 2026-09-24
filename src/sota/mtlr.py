"""Multi-task logistic regression for right-censored survival data."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MTLR(nn.Module):
    """Neural feature extractor followed by an MTLR output layer."""

    def __init__(self, in_features, num_intervals, hidden_dims=None, dropout=0.0):
        super().__init__()
        if in_features < 1 or num_intervals < 2:
            raise ValueError("MTLR needs at least one feature and two intervals")

        layers = []
        previous = int(in_features)
        for width in ([] if hidden_dims is None else hidden_dims):
            layers.extend([nn.Linear(previous, int(width)), nn.ReLU()])
            if dropout:
                layers.append(nn.Dropout(float(dropout)))
            previous = int(width)
        self.features = nn.Sequential(*layers) if layers else nn.Identity()

        # There is one less logistic task than event-time intervals. G maps
        # their scores to interval logits and fixes the tail logit at zero.
        self.mtlr_weight = nn.Parameter(torch.empty(previous, num_intervals - 1))
        self.mtlr_bias = nn.Parameter(torch.zeros(num_intervals - 1))
        self.register_buffer(
            "G", torch.tril(torch.ones(num_intervals - 1, num_intervals))
        )
        nn.init.xavier_normal_(self.mtlr_weight)

    def forward(self, x):
        scores = self.features(x) @ self.mtlr_weight + self.mtlr_bias
        return scores @ self.G


def make_time_bins(times, events, num_bins):
    """Make finite interval boundaries using training observations only."""
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=bool)
    if times.ndim != 1 or events.shape != times.shape or not len(times):
        raise ValueError("times and events must be non-empty one-dimensional arrays")
    if int(num_bins) < 1:
        raise ValueError("num_bins must be positive")
    if not np.isfinite(times).all() or (times < 0).any():
        raise ValueError("training times must be finite and non-negative")

    # Event quantiles avoid letting a few long censoring times determine the
    # resolution. In an all-censored split, observed durations are the only
    # available training-only fallback.
    support = times[events]
    if not len(support):
        support = times
    quantiles = np.linspace(0.0, 1.0, int(num_bins) + 1)[1:]
    boundaries = np.unique(np.quantile(support, quantiles))
    boundaries = boundaries[np.isfinite(boundaries) & (boundaries > 0.0)]
    if not len(boundaries):
        boundaries = np.array([max(float(np.max(times)), 1e-8)])
    return boundaries.astype(np.float32, copy=False)


def encode_mtlr_targets(times, events, boundaries):
    """Encode events as one-hot and censoring as compatible interval masks."""
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=bool)
    boundaries = np.asarray(boundaries, dtype=float)
    if times.ndim != 1 or events.shape != times.shape:
        raise ValueError("times and events must have matching one-dimensional shapes")
    if boundaries.ndim != 1 or not len(boundaries):
        raise ValueError("boundaries must be a non-empty one-dimensional array")

    # side='left' defines intervals (-inf, b0], (b0, b1], ..., (b_last, inf).
    interval = np.searchsorted(boundaries, times, side="left")
    target = np.zeros((len(times), len(boundaries) + 1), dtype=np.float32)
    for row, (index, observed) in enumerate(zip(interval, events)):
        if observed:
            target[row, index] = 1.0
        else:
            # The censoring interval and all later intervals contain possible
            # event times strictly beyond the censoring time.
            target[row, index:] = 1.0
    return torch.from_numpy(target)


def mtlr_neg_log_likelihood(logits, target, reduction="mean"):
    """Observed-data negative log likelihood for events and right censoring."""
    if logits.shape != target.shape:
        raise ValueError("logits and target must have identical shapes")
    mask = target.to(dtype=torch.bool)
    log_numerator = torch.logsumexp(
        logits.masked_fill(~mask, -torch.inf), dim=1
    )
    losses = torch.logsumexp(logits, dim=1) - log_numerator
    if reduction == "mean":
        return losses.mean()
    if reduction == "sum":
        return losses.sum()
    if reduction == "none":
        return losses
    raise ValueError(f"Unsupported reduction: {reduction}")


def interval_probabilities(logits):
    """Return the normalized event-time interval probabilities."""
    return torch.softmax(logits, dim=1)


def mtlr_survival(logits):
    """Return survival at zero and after each finite interval boundary."""
    probabilities = interval_probabilities(logits)
    later_probability = torch.flip(
        torch.cumsum(torch.flip(probabilities[:, 1:], dims=[1]), dim=1), dims=[1]
    )
    # S(0) is exact rather than a floating-point sum of softmax probabilities.
    survival = torch.cat([torch.ones_like(probabilities[:, :1]), later_probability], dim=1)
    return survival.clamp(0.0, 1.0)


def train_mtlr(
    X_train,
    time_train,
    event_train,
    X_test,
    num_bins=45,
    n_epochs=200,
    lr=0.005,
    device="cpu",
    eval_time_points=None,
    hidden_dims=None,
    dropout=0.0,
    weight_decay=0.0,
    batch_size=64,
    X_val=None,
    time_val=None,
    event_val=None,
    early_stopping_patience=None,
):
    """Fit MTLR on CPU and return risk, median, and requested survival curves."""
    del device  # MTLR intentionally remains on CPU.
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    boundaries = make_time_bins(time_train, event_train, num_bins)
    targets = encode_mtlr_targets(time_train, event_train, boundaries)

    model = MTLR(
        X_train.shape[1], len(boundaries) + 1,
        [64, 32] if hidden_dims is None else hidden_dims, dropout,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(lr), weight_decay=float(weight_decay)
    )
    dataset = TensorDataset(torch.from_numpy(X_train), targets)
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True)

    use_early_stopping = (
        early_stopping_patience is not None
        and int(early_stopping_patience) > 0
    )
    if use_early_stopping:
        if X_val is None or time_val is None or event_val is None:
            raise ValueError("Early stopping requires X_val, time_val, and event_val")
        X_val_tensor = torch.from_numpy(np.asarray(X_val, dtype=np.float32))
        val_targets = encode_mtlr_targets(time_val, event_val, boundaries)
        best_validation_nll = float("inf")
        best_state = None
        stale_epochs = 0

    for _ in range(int(n_epochs)):
        model.train()
        for features, target in loader:
            optimizer.zero_grad()
            loss = mtlr_neg_log_likelihood(model(features), target)
            loss.backward()
            optimizer.step()

        if use_early_stopping:
            model.eval()
            with torch.no_grad():
                # Weight decay belongs to optimization only. Checkpoint
                # selection uses the unregularized observed-data NLL.
                validation_nll = float(
                    mtlr_neg_log_likelihood(model(X_val_tensor), val_targets)
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

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test))
        probabilities = interval_probabilities(logits)
        survival = mtlr_survival(logits)

        # Match the reference risk definition: sum over cumulative hazards.
        hazards = probabilities[:, :-1] / survival[:, 1:].clamp_min(1e-12)
        risk_scores = hazards.cumsum(dim=1).sum(dim=1).cpu().numpy()
        survival_at_bins = survival.cpu().numpy()

    knot_times = np.r_[0.0, boundaries.astype(float)]
    medians = np.empty(len(X_test), dtype=float)
    for row, curve in enumerate(survival_at_bins):
        crossing = np.flatnonzero(curve <= 0.5)
        medians[row] = knot_times[crossing[0]] if len(crossing) else boundaries[-1]

    curves = None
    if eval_time_points is not None:
        eval_time_points = np.asarray(eval_time_points, dtype=float)
        curves = np.vstack([
            np.interp(eval_time_points, knot_times, curve, left=1.0, right=curve[-1])
            for curve in survival_at_bins
        ])
        curves = np.minimum.accumulate(np.clip(curves, 0.0, 1.0), axis=1)

    return risk_scores, medians, curves


__all__ = [
    "MTLR", "encode_mtlr_targets", "interval_probabilities", "make_time_bins",
    "mtlr_neg_log_likelihood", "mtlr_survival", "train_mtlr",
]
