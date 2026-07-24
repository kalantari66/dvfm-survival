"""Matched latent-vs-no-latent DVFM experiment.

Run from the repository root:

    python -m dvfm.latent_vs_no_latent \
        --config configs/latent_vs_no_latent.yaml

The experiment generates one SUPPORT semi-synthetic Clayton-frailty dataset,
uses one fixed train/validation/test split, and fits two models:

1. Latent DVFM
   q_phi(z | x, t, delta), decoder g_theta(x, z).

2. No-latent model
   decoder g_theta(x), trained from scratch with the same four Weibull outputs,
   observed-data likelihood, optimizer, validation criterion, and stopping rule.

The primary estimand is oracle conditional event-time recovery among censored
test subjects. For the latent model, posterior inference uses (x, C, delta=0).
The no-latent model uses x only. Oracle event times are used only for evaluation.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from scipy.integrate import trapezoid
from scipy.special import gamma
from torch.utils.data import DataLoader

from .reference_core import Decoder, SurvivalDataset
from .semi_synthetic_frailty_prediction import (
    generate_semi_synthetic,
    load_config,
    load_source_data,
    prepare_model_data,
    set_seed,
    split_indices,
    train_and_measure_latent,
)

EPS = 1e-10


class NoLatentSurvivalModel(nn.Module):
    """Joint Weibull event/censoring model conditioned only on covariates.

    The network reuses the reference DVFM Decoder with latent_dim=0. This gives
    the same hidden-layer structure and four Weibull outputs as DVFM, while
    removing the encoder and every subject-specific latent input.
    """

    def __init__(
        self,
        input_dim: int,
        decoder_hidden: list[int],
    ) -> None:
        super().__init__()
        self.decoder = Decoder(
            input_dim=input_dim,
            latent_dim=0,
            hidden_dims=decoder_hidden,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        empty_z = x.new_empty((x.shape[0], 0))
        return self.decoder(x, empty_z)

    @staticmethod
    def weibull_log_pdf(
        t: torch.Tensor,
        shape: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        t = torch.clamp(t, min=EPS)
        shape = torch.clamp(shape, min=EPS)
        scale = torch.clamp(scale, min=EPS)
        return (
            torch.log(shape)
            - torch.log(scale)
            + (shape - 1.0) * (torch.log(t) - torch.log(scale))
            - (t / scale).pow(shape)
        )

    @staticmethod
    def weibull_log_survival(
        t: torch.Tensor,
        shape: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        t = torch.clamp(t, min=EPS)
        shape = torch.clamp(shape, min=EPS)
        scale = torch.clamp(scale, min=EPS)
        return -(t / scale).pow(shape)

    def reconstruction_nll(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        event: torch.Tensor,
    ) -> torch.Tensor:
        shape_t, scale_t, shape_c, scale_c = self(x)
        log_f_t = self.weibull_log_pdf(time, shape_t, scale_t)
        log_s_t = self.weibull_log_survival(time, shape_t, scale_t)
        log_f_c = self.weibull_log_pdf(time, shape_c, scale_c)
        log_s_c = self.weibull_log_survival(time, shape_c, scale_c)
        log_likelihood = event * (log_f_t + log_s_c) + (1.0 - event) * (
            log_s_t + log_f_c
        )
        return -log_likelihood.mean()


def _make_loaders(
    train: dict[str, np.ndarray],
    valid: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = SurvivalDataset(train["X"], train["time"], train["event"])
    valid_ds = SurvivalDataset(valid["X"], valid["time"], valid["event"])
    test_ds = SurvivalDataset(test["X"], test["time"], test["event"])
    return (
        DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=(len(train_ds) % batch_size == 1),
        ),
        DataLoader(valid_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


def train_no_latent_model(
    train: dict[str, np.ndarray],
    valid: dict[str, np.ndarray],
    config: dict[str, Any],
) -> tuple[NoLatentSurvivalModel, pd.DataFrame, dict[str, Any]]:
    dvfm_cfg = config["dvfm"]
    training_cfg = config.get("training", {})
    no_latent_cfg = config.get("no_latent", {})
    device = str(dvfm_cfg.get("device", "cpu"))
    batch_size = int(dvfm_cfg["batch_size"])

    train_loader, valid_loader, _ = _make_loaders(
        train,
        valid,
        valid,
        batch_size,
    )

    model = NoLatentSurvivalModel(
        input_dim=train["X"].shape[1],
        decoder_hidden=list(dvfm_cfg["decoder_hidden"]),
    ).to(device)

    learning_rate = float(
        no_latent_cfg.get("learning_rate", dvfm_cfg["learning_rate"])
    )
    epochs = int(no_latent_cfg.get("epochs", dvfm_cfg["epochs"]))
    minimum_epochs = int(
        no_latent_cfg.get(
            "minimum_epochs",
            training_cfg.get("minimum_epochs", 0),
        )
    )
    patience = int(
        no_latent_cfg.get(
            "early_stopping_patience",
            training_cfg.get("early_stopping_patience", epochs),
        )
    )
    lr_patience = int(
        no_latent_cfg.get("lr_patience", training_cfg.get("lr_patience", 12))
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=lr_patience,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_val_nll = math.inf
    stale = 0
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        train_total = 0.0
        for x, time, event in train_loader:
            x = x.to(device)
            time = time.to(device)
            event = event.to(device)

            optimizer.zero_grad()
            loss = model.reconstruction_nll(x, time, event)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite no-latent training loss at epoch {epoch + 1}."
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_total += float(loss.item())

        train_nll = train_total / len(train_loader)

        model.eval()
        valid_total = 0.0
        with torch.no_grad():
            for x, time, event in valid_loader:
                x = x.to(device)
                time = time.to(device)
                event = event.to(device)
                valid_total += float(
                    model.reconstruction_nll(x, time, event).item()
                )
        valid_nll = valid_total / len(valid_loader)
        scheduler.step(valid_nll)

        history.append(
            {
                "epoch": float(epoch + 1),
                "train_reconstruction_nll": train_nll,
                "validation_reconstruction_nll": valid_nll,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )

        if valid_nll < best_val_nll - 1e-6:
            best_val_nll = valid_nll
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(
                f"[No latent] Epoch {epoch + 1}/{epochs}, "
                f"Train NLL: {train_nll:.4f}, "
                f"Val NLL: {valid_nll:.4f}"
            )

        if epoch + 1 >= minimum_epochs and stale >= patience:
            print(
                f"[No latent] Early stopping at epoch {epoch + 1}; "
                f"best epoch was {best_epoch}."
            )
            break

    if best_state is None:
        raise RuntimeError("No-latent training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    model.to(device)

    metrics = {
        "best_epoch": int(best_epoch),
        "best_validation_reconstruction_nll": float(best_val_nll),
        "epochs_completed": int(len(history)),
        "checkpoint_selection_metric": "validation_reconstruction_nll",
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
    }
    return model, pd.DataFrame(history), metrics


def _weibull_survival(
    grid_normalized: np.ndarray,
    shape: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    grid = np.asarray(grid_normalized, dtype=float).reshape(1, -1)
    shape = np.asarray(shape, dtype=float).reshape(-1, 1)
    scale = np.asarray(scale, dtype=float).reshape(-1, 1)
    return np.exp(-np.power(grid / np.maximum(scale, EPS), shape))


def _predict_latent(
    model: nn.Module,
    loader: DataLoader,
    grid_normalized: np.ndarray,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    cfg = config.get("survival_prediction", {})
    mode = str(cfg.get("posterior_mode", "posterior_mean"))
    mc_samples = int(cfg.get("mc_samples", 50))
    device = str(config["dvfm"].get("device", "cpu"))

    survival_batches: list[np.ndarray] = []
    mean_batches: list[np.ndarray] = []
    mu_batches: list[np.ndarray] = []
    std_batches: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for x, observed_time, event in loader:
            x = x.to(device)
            observed_time = observed_time.to(device)
            event = event.to(device)
            mu, logvar = model.encoder(x, observed_time, event)
            std = torch.exp(0.5 * logvar)
            mu_batches.append(mu.cpu().numpy())
            std_batches.append(std.cpu().numpy())

            if mode == "posterior_mean":
                draws = [mu]
            elif mode == "posterior_mc":
                if mc_samples < 1:
                    raise ValueError("survival_prediction.mc_samples must be >= 1.")
                draws = [mu + std * torch.randn_like(std) for _ in range(mc_samples)]
            else:
                raise ValueError(
                    "survival_prediction.posterior_mode must be "
                    "'posterior_mean' or 'posterior_mc'."
                )

            survival_draws: list[np.ndarray] = []
            mean_draws: list[np.ndarray] = []
            for z in draws:
                shape_t, scale_t, _, _ = model.decoder(x, z)
                shape_np = shape_t.cpu().numpy()
                scale_np = scale_t.cpu().numpy()
                survival_draws.append(
                    _weibull_survival(grid_normalized, shape_np, scale_np)
                )
                mean_draws.append(scale_np * gamma(1.0 + 1.0 / shape_np))

            survival_batches.append(np.mean(survival_draws, axis=0))
            mean_batches.append(np.mean(mean_draws, axis=0))

    return {
        "survival": np.concatenate(survival_batches, axis=0),
        "mean_time": np.concatenate(mean_batches),
        "posterior_mu": np.concatenate(mu_batches, axis=0).reshape(-1),
        "posterior_std": np.concatenate(std_batches, axis=0).reshape(-1),
    }


def _predict_no_latent(
    model: NoLatentSurvivalModel,
    loader: DataLoader,
    grid_normalized: np.ndarray,
    device: str,
) -> dict[str, np.ndarray]:
    survival_batches: list[np.ndarray] = []
    mean_batches: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for x, _, _ in loader:
            x = x.to(device)
            shape_t, scale_t, _, _ = model(x)
            shape_np = shape_t.cpu().numpy()
            scale_np = scale_t.cpu().numpy()
            survival_batches.append(
                _weibull_survival(grid_normalized, shape_np, scale_np)
            )
            mean_batches.append(scale_np * gamma(1.0 + 1.0 / shape_np))

    return {
        "survival": np.concatenate(survival_batches, axis=0),
        "mean_time": np.concatenate(mean_batches),
    }


def _oracle_ci(true_time: np.ndarray, predicted_time: np.ndarray) -> float:
    true_time = np.asarray(true_time, dtype=float)
    predicted_time = np.asarray(predicted_time, dtype=float)
    if len(true_time) < 2 or np.unique(true_time).size < 2:
        return float("nan")
    concordant = 0.0
    comparable = 0
    for i in range(len(true_time) - 1):
        dt = true_time[i] - true_time[i + 1 :]
        valid = dt != 0
        if not np.any(valid):
            continue
        dp = predicted_time[i] - predicted_time[i + 1 :]
        product = dt[valid] * dp[valid]
        concordant += float(np.sum(product > 0))
        concordant += 0.5 * float(np.sum(product == 0))
        comparable += int(valid.sum())
    return float(concordant / comparable) if comparable else float("nan")


def _oracle_ibs(
    true_time: np.ndarray,
    survival: np.ndarray,
    grid: np.ndarray,
) -> float:
    truth = (
        np.asarray(true_time, dtype=float)[:, None]
        > np.asarray(grid, dtype=float)[None, :]
    ).astype(float)
    brier = np.mean((truth - survival) ** 2, axis=0)
    width = float(grid[-1] - grid[0])
    return float(trapezoid(brier, grid) / width) if width > 0 else float("nan")


def evaluate_models(
    latent_model: nn.Module,
    no_latent_model: NoLatentSurvivalModel,
    train: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    time_scale: float,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    cfg = config.get("survival_prediction", {})
    batch_size = int(config["dvfm"]["batch_size"])
    device = str(config["dvfm"].get("device", "cpu"))
    n_time_points = int(cfg.get("n_time_points", 200))
    max_quantile = float(cfg.get("grid_max_quantile", 0.95))
    grid_min = max(float(cfg.get("grid_min", 0.0)), 0.0)

    if n_time_points < 2:
        raise ValueError("survival_prediction.n_time_points must be >= 2.")
    if not 0.0 < max_quantile <= 1.0:
        raise ValueError("survival_prediction.grid_max_quantile must be in (0,1].")

    grid_max = float(
        np.quantile(train["true_event_time"] * time_scale, max_quantile)
    )
    if grid_max <= grid_min:
        raise ValueError("Invalid oracle IBS evaluation interval.")
    grid_original = np.linspace(grid_min, grid_max, n_time_points)
    grid_normalized = grid_original / time_scale

    test_loader = DataLoader(
        SurvivalDataset(test["X"], test["time"], test["event"]),
        batch_size=batch_size,
        shuffle=False,
    )
    latent = _predict_latent(
        latent_model,
        test_loader,
        grid_normalized,
        config,
    )
    no_latent = _predict_no_latent(
        no_latent_model,
        test_loader,
        grid_normalized,
        device,
    )

    true_event = test["true_event_time"] * time_scale
    observed_time = test["time"] * time_scale
    event = test["event"].astype(int)
    latent_mean = latent["mean_time"] * time_scale
    no_latent_mean = no_latent["mean_time"] * time_scale

    def score(mask: np.ndarray) -> dict[str, float | int]:
        latent_ci = _oracle_ci(true_event[mask], latent_mean[mask])
        no_latent_ci = _oracle_ci(true_event[mask], no_latent_mean[mask])
        latent_ibs = _oracle_ibs(
            true_event[mask], latent["survival"][mask], grid_original
        )
        no_latent_ibs = _oracle_ibs(
            true_event[mask], no_latent["survival"][mask], grid_original
        )
        latent_mae = float(np.mean(np.abs(true_event[mask] - latent_mean[mask])))
        no_latent_mae = float(
            np.mean(np.abs(true_event[mask] - no_latent_mean[mask]))
        )
        return {
            "n": int(mask.sum()),
            "latent_oracle_ci": latent_ci,
            "no_latent_oracle_ci": no_latent_ci,
            "delta_oracle_ci": latent_ci - no_latent_ci,
            "latent_oracle_ibs": latent_ibs,
            "no_latent_oracle_ibs": no_latent_ibs,
            "delta_oracle_ibs": latent_ibs - no_latent_ibs,
            "latent_oracle_mae": latent_mae,
            "no_latent_oracle_mae": no_latent_mae,
            "delta_oracle_mae": latent_mae - no_latent_mae,
            "mean_absolute_survival_difference": float(
                np.mean(
                    np.abs(
                        latent["survival"][mask]
                        - no_latent["survival"][mask]
                    )
                )
            ),
        }

    all_mask = np.ones(len(event), dtype=bool)
    censored_mask = event == 0
    uncensored_mask = event == 1
    metrics = {
        "estimand": "conditional oracle event-time recovery after observed follow-up",
        "primary_subset": "censored_test_primary",
        "comparison": (
            "separately trained latent DVFM versus separately trained "
            "covariate-only joint Weibull event/censoring model"
        ),
        "fairness_controls": [
            "identical generated dataset and split",
            "identical preprocessing and time normalization",
            "same decoder hidden widths and four Weibull outputs",
            "same observed-data joint event/censoring likelihood",
            "same optimizer family and validation NLL checkpoint selection",
            "same oracle evaluation grid fixed from training data",
        ],
        "posterior_mode": str(cfg.get("posterior_mode", "posterior_mean")),
        "grid_source": "training true event-time quantile",
        "grid_max_quantile": max_quantile,
        "grid_min": float(grid_original[0]),
        "grid_max": float(grid_original[-1]),
        "n_time_points": n_time_points,
        "all_test": score(all_mask),
        "censored_test_primary": score(censored_mask),
        "uncensored_test_diagnostic": score(uncensored_mask),
    }

    predictions = pd.DataFrame(
        {
            "row_index": test["row_index"],
            "observed_time": observed_time,
            "event": event,
            "true_event_time": true_event,
            "latent_posterior_mu": latent["posterior_mu"],
            "latent_posterior_std": latent["posterior_std"],
            "latent_predicted_mean_event_time": latent_mean,
            "no_latent_predicted_mean_event_time": no_latent_mean,
            "latent_absolute_error": np.abs(true_event - latent_mean),
            "no_latent_absolute_error": np.abs(true_event - no_latent_mean),
            "absolute_prediction_difference": np.abs(
                latent_mean - no_latent_mean
            ),
        }
    )
    return metrics, predictions


def _parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def run(config_path: Path) -> None:
    config = load_config(config_path)
    seed = int(config["experiment"]["seed"])
    set_seed(seed)

    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/7] Loading real SUPPORT source dataset")
    source = load_source_data(config)

    print("[2/7] Generating one semi-synthetic Clayton-frailty dataset")
    generated, generation_metrics = generate_semi_synthetic(source, config)
    generated_path = Path(config["data"]["generated_csv"])
    generated_path.parent.mkdir(parents=True, exist_ok=True)
    generated.to_csv(generated_path, index=False)

    print("[3/7] Creating one shared split and preprocessing transform")
    indices = split_indices(generated, config)
    train, valid, test, _, time_scale = prepare_model_data(
        generated,
        indices,
        config,
    )

    print("[4/7] Training latent DVFM")
    # Reset model RNG so model initialization is reproducible independently of
    # generation and preprocessing operations.
    set_seed(seed + 1000)
    (
        latent_model,
        validation_latent_predictions,
        test_latent_predictions,
        latent_results,
    ) = train_and_measure_latent(train, valid, test, config)
    latent_history = pd.DataFrame(latent_results.pop("training_history", []))

    print("[5/7] Training matched no-latent model from scratch")
    set_seed(seed + 2000)
    no_latent_model, no_latent_history, no_latent_results = (
        train_no_latent_model(train, valid, config)
    )

    print("[6/7] Comparing oracle survival performance")
    comparison_results, predictions = evaluate_models(
        latent_model=latent_model,
        no_latent_model=no_latent_model,
        train=train,
        test=test,
        time_scale=time_scale,
        config=config,
    )

    latent_results["n_parameters"] = _parameter_count(latent_model)
    no_latent_results["n_parameters"] = _parameter_count(no_latent_model)

    print("[7/7] Saving results")
    torch.save(
        latent_model.state_dict(),
        output_dir / "latent_dvfm_state_dict.pt",
    )
    torch.save(
        no_latent_model.state_dict(),
        output_dir / "no_latent_state_dict.pt",
    )
    latent_history.to_csv(
        output_dir / "latent_training_history.csv",
        index=False,
    )
    no_latent_history.to_csv(
        output_dir / "no_latent_training_history.csv",
        index=False,
    )
    validation_latent_predictions.to_csv(
        output_dir / "latent_recovery_validation.csv",
        index=False,
    )
    test_latent_predictions.to_csv(
        output_dir / "latent_recovery_test.csv",
        index=False,
    )
    predictions.to_csv(
        output_dir / "latent_vs_no_latent_test_predictions.csv",
        index=False,
    )

    results = {
        "generation": generation_metrics,
        "latent_model": latent_results,
        "no_latent_model": no_latent_results,
        "survival_comparison": comparison_results,
        "time_scale": time_scale,
        "n_source": len(source),
        "n_train": len(train["time"]),
        "n_validation": len(valid["time"]),
        "n_test": len(test["time"]),
    }
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    primary = comparison_results["censored_test_primary"]
    print()
    print("Matched latent versus no-latent comparison")
    print("-------------------------------------------")
    print(f"N censored:                 {primary['n']}")
    print(f"Oracle CI, latent:          {primary['latent_oracle_ci']:.4f}")
    print(f"Oracle CI, no latent:       {primary['no_latent_oracle_ci']:.4f}")
    print(f"Delta CI:                   {primary['delta_oracle_ci']:+.4f}")
    print(f"Oracle IBS, latent:         {primary['latent_oracle_ibs']:.4f}")
    print(f"Oracle IBS, no latent:      {primary['no_latent_oracle_ibs']:.4f}")
    print(f"Delta IBS:                  {primary['delta_oracle_ibs']:+.4f}")
    print(f"Oracle MAE, latent:         {primary['latent_oracle_mae']:.4f}")
    print(f"Oracle MAE, no latent:      {primary['no_latent_oracle_mae']:.4f}")
    print(f"Delta MAE:                  {primary['delta_oracle_mae']:+.4f}")
    print()
    print(f"Generated dataset: {generated_path}")
    print(f"Experiment outputs: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train matched latent and no-latent joint Weibull survival "
            "models on one SUPPORT semi-synthetic frailty dataset."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to latent-vs-no-latent YAML configuration.",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
