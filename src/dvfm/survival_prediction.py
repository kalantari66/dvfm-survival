"""Oracle conditional survival evaluation for a trained DVFM.

The DVFM encoder consumes (x, observed_time, event), while its decoder maps
(x, z) to Weibull event/censoring parameters. This module asks whether the
subject-specific posterior latent improves recovery of the known event-time
distribution relative to the explicit ablation z = 0.

This is conditional event-time recovery after observing follow-up, not a
baseline/time-zero survival prediction task. The censored-only subset is the
primary proof-of-concept evaluation because its true event time is not supplied
to the encoder.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.integrate import trapezoid
from scipy.special import gamma
from torch.utils.data import DataLoader

EPS = 1e-10


def _weibull_survival(
    times: np.ndarray,
    shape: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    times = np.asarray(times, dtype=float).reshape(1, -1)
    shape = np.asarray(shape, dtype=float).reshape(-1, 1)
    scale = np.asarray(scale, dtype=float).reshape(-1, 1)
    ratio = np.maximum(times, 0.0) / np.maximum(scale, EPS)
    return np.exp(-np.power(ratio, shape))


def _oracle_ci(true_time: np.ndarray, predicted_time: np.ndarray) -> float:
    """Concordance with every oracle event observed.

    lifelines.concordance_index expects larger predicted scores to correspond
    to longer survival, so a predicted event-time summary can be passed
    directly.
    """
    true_time = np.asarray(true_time, dtype=float)
    predicted_time = np.asarray(predicted_time, dtype=float)
    if len(true_time) < 2 or np.unique(true_time).size < 2:
        return float("nan")
    concordant = 0.0
    comparable = 0
    for i in range(len(true_time) - 1):
        time_diff = true_time[i] - true_time[i + 1 :]
        valid = time_diff != 0
        if not np.any(valid):
            continue
        score_diff = predicted_time[i] - predicted_time[i + 1 :]
        # Earlier true event should have a smaller predicted event time.
        product = time_diff[valid] * score_diff[valid]
        concordant += float(np.sum(product > 0))
        concordant += 0.5 * float(np.sum(product == 0))
        comparable += int(valid.sum())
    return float(concordant / comparable) if comparable else float("nan")


def _oracle_ibs(
    true_time: np.ndarray,
    survival: np.ndarray,
    grid: np.ndarray,
) -> float:
    true_time = np.asarray(true_time, dtype=float)
    grid = np.asarray(grid, dtype=float)
    truth = (true_time[:, None] > grid[None, :]).astype(float)
    brier = np.mean((truth - survival) ** 2, axis=0)
    width = float(grid[-1] - grid[0])
    if width <= 0:
        return float("nan")
    return float(trapezoid(brier, grid) / width)


def _posterior_statistics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    mus: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x, observed_time, event in loader:
            x = x.to(device)
            observed_time = observed_time.to(device)
            event = event.to(device)
            mu, logvar = model.encoder(x, observed_time, event)
            mus.append(mu.cpu().numpy())
            stds.append(torch.exp(0.5 * logvar).cpu().numpy())
    return np.concatenate(mus), np.concatenate(stds)


def _predict_decoder_distribution(
    model: torch.nn.Module,
    loader: DataLoader,
    grid_normalized: np.ndarray,
    device: str,
    mode: str,
    mc_samples: int,
) -> dict[str, np.ndarray]:
    """Predict event survival through the actual DVFM Weibull decoder.

    For posterior_mc, survival curves and event-time means are averaged across
    posterior draws. Weibull parameters themselves are retained only as their
    draw-wise averages for diagnostics; those averages do not define the
    mixture distribution used for scoring.
    """
    all_survival: list[np.ndarray] = []
    all_mean_time: list[np.ndarray] = []
    all_shape: list[np.ndarray] = []
    all_scale: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for x, observed_time, event in loader:
            x = x.to(device)
            observed_time = observed_time.to(device)
            event = event.to(device)
            mu, logvar = model.encoder(x, observed_time, event)
            std = torch.exp(0.5 * logvar)

            if mode == "zero":
                latent_draws = [torch.zeros_like(mu)]
            elif mode == "posterior_mean":
                latent_draws = [mu]
            elif mode == "posterior_mc":
                if mc_samples < 1:
                    raise ValueError("survival_prediction.mc_samples must be >= 1.")
                latent_draws = [
                    mu + torch.randn_like(std) * std
                    for _ in range(mc_samples)
                ]
            else:
                raise ValueError(
                    "survival_prediction.posterior_mode must be "
                    "'posterior_mean' or 'posterior_mc'."
                )

            survival_draws: list[np.ndarray] = []
            mean_draws: list[np.ndarray] = []
            shape_draws: list[np.ndarray] = []
            scale_draws: list[np.ndarray] = []
            for z in latent_draws:
                shape_t, scale_t, _, _ = model.decoder(x, z)
                shape_np = shape_t.cpu().numpy()
                scale_np = scale_t.cpu().numpy()
                survival_draws.append(
                    _weibull_survival(grid_normalized, shape_np, scale_np)
                )
                mean_draws.append(scale_np * gamma(1.0 + 1.0 / shape_np))
                shape_draws.append(shape_np)
                scale_draws.append(scale_np)

            all_survival.append(np.mean(survival_draws, axis=0))
            all_mean_time.append(np.mean(mean_draws, axis=0))
            all_shape.append(np.mean(shape_draws, axis=0))
            all_scale.append(np.mean(scale_draws, axis=0))

    return {
        "survival": np.concatenate(all_survival, axis=0),
        "mean_time": np.concatenate(all_mean_time),
        "mean_shape": np.concatenate(all_shape),
        "mean_scale": np.concatenate(all_scale),
    }


def evaluate_survival_prediction(
    model: torch.nn.Module,
    loader: DataLoader,
    test_split: dict[str, np.ndarray],
    train_split: dict[str, np.ndarray],
    time_scale: float,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    cfg = config.get("survival_prediction", {})
    device = str(config["dvfm"].get("device", "cpu"))
    posterior_mode = str(cfg.get("posterior_mode", "posterior_mean"))
    mc_samples = int(cfg.get("mc_samples", 50))
    n_time_points = int(cfg.get("n_time_points", 200))
    max_quantile = float(cfg.get("grid_max_quantile", 0.90))
    grid_min_original = max(float(cfg.get("grid_min", 0.0)), 0.0)

    if not 0.0 < max_quantile <= 1.0:
        raise ValueError("survival_prediction.grid_max_quantile must be in (0,1].")
    if n_time_points < 2:
        raise ValueError("survival_prediction.n_time_points must be >= 2.")

    # Fix the evaluation interval before inspecting test oracle outcomes.
    # The semi-synthetic training event times are known to the experimenter,
    # but never enter model fitting as oracle labels.
    train_true_original = (
        np.asarray(train_split["true_event_time"], dtype=float) * time_scale
    )
    grid_max_original = float(np.quantile(train_true_original, max_quantile))
    if grid_max_original <= grid_min_original:
        raise ValueError("Invalid survival prediction evaluation interval.")
    grid_original = np.linspace(
        grid_min_original,
        grid_max_original,
        n_time_points,
    )
    grid_normalized = grid_original / time_scale

    posterior = _predict_decoder_distribution(
        model=model,
        loader=loader,
        grid_normalized=grid_normalized,
        device=device,
        mode=posterior_mode,
        mc_samples=mc_samples,
    )
    zero = _predict_decoder_distribution(
        model=model,
        loader=loader,
        grid_normalized=grid_normalized,
        device=device,
        mode="zero",
        mc_samples=1,
    )
    mu, std = _posterior_statistics(model, loader, device)

    true_event = np.asarray(test_split["true_event_time"], dtype=float) * time_scale
    observed_time = np.asarray(test_split["time"], dtype=float) * time_scale
    event = np.asarray(test_split["event"], dtype=int)
    posterior_mean_original = posterior["mean_time"] * time_scale
    zero_mean_original = zero["mean_time"] * time_scale

    def score(mask: np.ndarray) -> dict[str, float | int]:
        p_ci = _oracle_ci(true_event[mask], posterior_mean_original[mask])
        z_ci = _oracle_ci(true_event[mask], zero_mean_original[mask])
        p_ibs = _oracle_ibs(
            true_event[mask], posterior["survival"][mask], grid_original
        )
        z_ibs = _oracle_ibs(
            true_event[mask], zero["survival"][mask], grid_original
        )
        return {
            "n": int(mask.sum()),
            "posterior_oracle_ci": p_ci,
            "zero_oracle_ci": z_ci,
            "delta_oracle_ci": p_ci - z_ci,
            "posterior_oracle_ibs": p_ibs,
            "zero_oracle_ibs": z_ibs,
            "delta_oracle_ibs": p_ibs - z_ibs,
            "posterior_oracle_mae": float(
                np.mean(np.abs(true_event[mask] - posterior_mean_original[mask]))
            ),
            "zero_oracle_mae": float(
                np.mean(np.abs(true_event[mask] - zero_mean_original[mask]))
            ),
            "mean_absolute_survival_change": float(
                np.mean(
                    np.abs(
                        posterior["survival"][mask] - zero["survival"][mask]
                    )
                )
            ),
        }

    all_mask = np.ones(len(event), dtype=bool)
    censored_mask = event == 0
    uncensored_mask = event == 1
    metrics: dict[str, Any] = {
        "estimand": "conditional oracle event-time recovery after observed follow-up",
        "primary_subset": "censored_test_primary",
        "posterior_mode": posterior_mode,
        "mc_samples": mc_samples if posterior_mode == "posterior_mc" else 1,
        "ablation": "same trained decoder with z set to zero",
        "grid_source": "training true event-time quantile",
        "grid_max_quantile": max_quantile,
        "grid_min": float(grid_original[0]),
        "grid_max": float(grid_original[-1]),
        "n_time_points": n_time_points,
        "all_test": score(all_mask),
        "censored_test_primary": (
            score(censored_mask) if censored_mask.sum() >= 2 else {}
        ),
        "uncensored_test_diagnostic": (
            score(uncensored_mask) if uncensored_mask.sum() >= 2 else {}
        ),
    }

    if mu.shape[1] != 1:
        raise ValueError("This proof of concept expects dvfm.latent_dim=1.")

    predictions = pd.DataFrame(
        {
            "row_index": test_split["row_index"],
            "observed_time": observed_time,
            "event": event,
            "true_event_time": true_event,
            "posterior_mu": mu[:, 0],
            "posterior_std": std[:, 0],
            "posterior_mean_weibull_shape": posterior["mean_shape"],
            "posterior_mean_weibull_scale": posterior["mean_scale"] * time_scale,
            "posterior_predicted_mean_event_time": posterior_mean_original,
            "zero_weibull_shape": zero["mean_shape"],
            "zero_weibull_scale": zero["mean_scale"] * time_scale,
            "zero_predicted_mean_event_time": zero_mean_original,
            "absolute_predicted_time_change": np.abs(
                posterior_mean_original - zero_mean_original
            ),
        }
    )
    return metrics, predictions
