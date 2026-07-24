"""Baseline survival prediction: marginalized latent DVFM vs no-latent model.

Run from the repository root:

    python -m dvfm.baseline_latent_vs_no_latent \
        --config configs/baseline_latent_vs_no_latent.yaml

This experiment trains two models on the same SUPPORT semi-synthetic
Clayton-frailty dataset and evaluates baseline survival prediction when only
covariates x are available for test subjects.

1. Latent DVFM:
   Training uses q_phi(z | x, t, delta), as required by the variational model.
   At test time the encoder is NOT called on test subjects. Predictions are
   marginalized over a latent reference distribution estimated exclusively
   from training subjects:

       S(t | x) ~= (1/M) sum_m S(t | x, z_m),
       z_m ~ q_train(z).

2. No-latent model:
   A separately trained joint Weibull event/censoring model g_theta(x).

The models share one generated dataset, split, preprocessing transform,
decoder widths, observed-data likelihood, optimizer settings, validation
criterion, and oracle evaluation grid.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from scipy.integrate import trapezoid
from scipy.special import gamma
from torch.utils.data import DataLoader

from .reference_core import SurvivalDataset
from .semi_synthetic_frailty_prediction import (
    generate_semi_synthetic,
    load_config,
    load_source_data,
    prepare_model_data,
    set_seed,
    split_indices,
    train_and_measure_latent,
)
from .latent_vs_no_latent import (
    NoLatentSurvivalModel,
    train_no_latent_model,
)

EPS = 1e-10


def _weibull_survival(
    grid_normalized: np.ndarray,
    shape: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    grid = np.asarray(grid_normalized, dtype=float).reshape(1, -1)
    shape = np.asarray(shape, dtype=float).reshape(-1, 1)
    scale = np.asarray(scale, dtype=float).reshape(-1, 1)
    return np.exp(-np.power(grid / np.maximum(scale, EPS), shape))


def _collect_training_posterior(
    model: nn.Module,
    train: dict[str, np.ndarray],
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect q_phi(z_i|x_i,t_i,delta_i) parameters from training only."""
    loader = DataLoader(
        SurvivalDataset(train["X"], train["time"], train["event"]),
        batch_size=batch_size,
        shuffle=False,
    )
    mus: list[np.ndarray] = []
    stds: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for x, time, event in loader:
            x = x.to(device)
            time = time.to(device)
            event = event.to(device)
            mu, logvar = model.encoder(x, time, event)
            mus.append(mu.cpu().numpy())
            stds.append(torch.exp(0.5 * logvar).cpu().numpy())

    return np.concatenate(mus, axis=0), np.concatenate(stds, axis=0)


def _draw_latents(
    train_mu: np.ndarray,
    train_std: np.ndarray,
    n_draws: int,
    seed: int,
    distribution: str,
) -> np.ndarray:
    """Draw one population latent per MC draw from training information only."""
    rng = np.random.default_rng(seed)
    latent_dim = train_mu.shape[1]

    if distribution == "aggregate_posterior":
        indices = rng.integers(0, len(train_mu), size=n_draws)
        eps = rng.normal(size=(n_draws, latent_dim))
        return train_mu[indices] + train_std[indices] * eps

    if distribution == "aggregate_posterior_means":
        indices = rng.integers(0, len(train_mu), size=n_draws)
        return train_mu[indices]

    if distribution == "standard_normal_prior":
        return rng.normal(size=(n_draws, latent_dim))

    raise ValueError(
        "baseline_prediction.latent_distribution must be one of "
        "'aggregate_posterior', 'aggregate_posterior_means', or "
        "'standard_normal_prior'."
    )


def _predict_latent_baseline(
    model: nn.Module,
    X: np.ndarray,
    grid_normalized: np.ndarray,
    train_mu: np.ndarray,
    train_std: np.ndarray,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Predict S(t|x) by integrating over a training-derived latent distribution.

    No test time or event indicator is passed to the encoder.
    """
    cfg = config["baseline_prediction"]
    device = str(config["dvfm"].get("device", "cpu"))
    batch_size = int(config["dvfm"]["batch_size"])
    n_draws = int(cfg.get("mc_samples", 200))
    if n_draws < 1:
        raise ValueError("baseline_prediction.mc_samples must be >= 1.")

    latent_draws = _draw_latents(
        train_mu=train_mu,
        train_std=train_std,
        n_draws=n_draws,
        seed=int(config["experiment"]["seed"]) + 3000,
        distribution=str(cfg.get("latent_distribution", "aggregate_posterior")),
    )

    survival_parts: list[np.ndarray] = []
    mean_parts: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            x = torch.as_tensor(
                X[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            batch_survival = np.zeros(
                (len(x), len(grid_normalized)),
                dtype=np.float64,
            )
            batch_mean = np.zeros(len(x), dtype=np.float64)

            for latent in latent_draws:
                z = torch.as_tensor(
                    latent,
                    dtype=torch.float32,
                    device=device,
                ).reshape(1, -1).expand(len(x), -1)
                shape_t, scale_t, _, _ = model.decoder(x, z)
                shape_np = shape_t.cpu().numpy()
                scale_np = scale_t.cpu().numpy()
                batch_survival += _weibull_survival(
                    grid_normalized,
                    shape_np,
                    scale_np,
                )
                batch_mean += scale_np * gamma(1.0 + 1.0 / shape_np)

            survival_parts.append(batch_survival / n_draws)
            mean_parts.append(batch_mean / n_draws)

    return {
        "survival": np.concatenate(survival_parts, axis=0),
        "mean_time": np.concatenate(mean_parts, axis=0),
        "latent_draw_mean": np.mean(latent_draws, axis=0),
        "latent_draw_std": np.std(latent_draws, axis=0),
    }


def _predict_no_latent_baseline(
    model: NoLatentSurvivalModel,
    X: np.ndarray,
    grid_normalized: np.ndarray,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    device = str(config["dvfm"].get("device", "cpu"))
    batch_size = int(config["dvfm"]["batch_size"])
    survival_parts: list[np.ndarray] = []
    mean_parts: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            x = torch.as_tensor(
                X[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            shape_t, scale_t, _, _ = model(x)
            shape_np = shape_t.cpu().numpy()
            scale_np = scale_t.cpu().numpy()
            survival_parts.append(
                _weibull_survival(grid_normalized, shape_np, scale_np)
            )
            mean_parts.append(scale_np * gamma(1.0 + 1.0 / shape_np))

    return {
        "survival": np.concatenate(survival_parts, axis=0),
        "mean_time": np.concatenate(mean_parts, axis=0),
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


def evaluate_baseline_models(
    latent_model: nn.Module,
    no_latent_model: NoLatentSurvivalModel,
    train: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    time_scale: float,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    cfg = config["baseline_prediction"]
    batch_size = int(config["dvfm"]["batch_size"])
    device = str(config["dvfm"].get("device", "cpu"))
    n_time_points = int(cfg.get("n_time_points", 200))
    max_quantile = float(cfg.get("grid_max_quantile", 0.95))
    grid_min = max(float(cfg.get("grid_min", 0.0)), 0.0)

    if n_time_points < 2:
        raise ValueError("baseline_prediction.n_time_points must be >= 2.")
    if not 0.0 < max_quantile <= 1.0:
        raise ValueError(
            "baseline_prediction.grid_max_quantile must be in (0,1]."
        )

    grid_max = float(
        np.quantile(train["true_event_time"] * time_scale, max_quantile)
    )
    if grid_max <= grid_min:
        raise ValueError("Invalid oracle IBS evaluation interval.")

    grid_original = np.linspace(grid_min, grid_max, n_time_points)
    grid_normalized = grid_original / time_scale

    train_mu, train_std = _collect_training_posterior(
        model=latent_model,
        train=train,
        batch_size=batch_size,
        device=device,
    )
    latent = _predict_latent_baseline(
        model=latent_model,
        X=test["X"],
        grid_normalized=grid_normalized,
        train_mu=train_mu,
        train_std=train_std,
        config=config,
    )
    no_latent = _predict_no_latent_baseline(
        model=no_latent_model,
        X=test["X"],
        grid_normalized=grid_normalized,
        config=config,
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
            true_event[mask],
            latent["survival"][mask],
            grid_original,
        )
        no_latent_ibs = _oracle_ibs(
            true_event[mask],
            no_latent["survival"][mask],
            grid_original,
        )
        latent_mae = float(
            np.mean(np.abs(true_event[mask] - latent_mean[mask]))
        )
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
        "estimand": "baseline oracle survival prediction from covariates x only",
        "primary_subset": "all_test_primary",
        "test_information_latent_model": "x only",
        "test_information_no_latent_model": "x only",
        "latent_inference": (
            "Monte Carlo marginalization over a latent distribution estimated "
            "exclusively from training posterior distributions"
        ),
        "latent_distribution": str(
            cfg.get("latent_distribution", "aggregate_posterior")
        ),
        "mc_samples": int(cfg.get("mc_samples", 200)),
        "fairness_controls": [
            "neither model receives test observed time or event indicator",
            "identical generated dataset and split",
            "identical preprocessing and time normalization",
            "same decoder hidden widths and four Weibull outputs",
            "same observed-data joint event/censoring likelihood",
            "same optimizer family and validation NLL checkpoint selection",
            "same oracle evaluation grid fixed from training data",
        ],
        "grid_source": "training true event-time quantile",
        "grid_max_quantile": max_quantile,
        "grid_min": float(grid_original[0]),
        "grid_max": float(grid_original[-1]),
        "n_time_points": n_time_points,
        "all_test_primary": score(all_mask),
        "censored_test_diagnostic": score(censored_mask),
        "uncensored_test_diagnostic": score(uncensored_mask),
        "training_aggregate_posterior": {
            "mu_mean": train_mu.mean(axis=0).tolist(),
            "mu_std": train_mu.std(axis=0).tolist(),
            "posterior_std_mean": train_std.mean(axis=0).tolist(),
            "mc_draw_mean": np.asarray(latent["latent_draw_mean"]).tolist(),
            "mc_draw_std": np.asarray(latent["latent_draw_std"]).tolist(),
        },
    }

    predictions = pd.DataFrame(
        {
            "row_index": test["row_index"],
            "observed_time_diagnostic_only": observed_time,
            "event_diagnostic_only": event,
            "true_event_time": true_event,
            "latent_predicted_mean_event_time": latent_mean,
            "no_latent_predicted_mean_event_time": no_latent_mean,
            "latent_absolute_error": np.abs(true_event - latent_mean),
            "no_latent_absolute_error": np.abs(true_event - no_latent_mean),
            "absolute_prediction_difference": np.abs(
                latent_mean - no_latent_mean
            ),
        }
    )

    brier_rows = []
    truth = (true_event[:, None] > grid_original[None, :]).astype(float)
    latent_brier = np.mean((truth - latent["survival"]) ** 2, axis=0)
    no_latent_brier = np.mean((truth - no_latent["survival"]) ** 2, axis=0)
    for t, latent_bs, no_latent_bs in zip(
        grid_original,
        latent_brier,
        no_latent_brier,
    ):
        brier_rows.append(
            {
                "time": t,
                "latent_oracle_brier": latent_bs,
                "no_latent_oracle_brier": no_latent_bs,
                "delta_oracle_brier": latent_bs - no_latent_bs,
            }
        )

    return metrics, predictions, pd.DataFrame(brier_rows)


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
    set_seed(seed + 1000)
    (
        latent_model,
        validation_latent_predictions,
        test_latent_predictions,
        latent_results,
    ) = train_and_measure_latent(train, valid, test, config)
    latent_history = pd.DataFrame(
        latent_results.pop("training_history", [])
    )

    print("[5/7] Training matched no-latent model")
    set_seed(seed + 2000)
    no_latent_model, no_latent_history, no_latent_results = (
        train_no_latent_model(train, valid, config)
    )

    print("[6/7] Evaluating baseline survival prediction from x only")
    comparison_results, predictions, brier_curves = (
        evaluate_baseline_models(
            latent_model=latent_model,
            no_latent_model=no_latent_model,
            train=train,
            test=test,
            time_scale=time_scale,
            config=config,
        )
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
        output_dir / "latent_recovery_test_diagnostic.csv",
        index=False,
    )
    predictions.to_csv(
        output_dir / "baseline_latent_vs_no_latent_test_predictions.csv",
        index=False,
    )
    brier_curves.to_csv(
        output_dir / "baseline_oracle_brier_curves.csv",
        index=False,
    )

    results = {
        "generation": generation_metrics,
        "latent_model": latent_results,
        "no_latent_model": no_latent_results,
        "baseline_survival_comparison": comparison_results,
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

    primary = comparison_results["all_test_primary"]
    print()
    print("Baseline survival prediction from x only")
    print("----------------------------------------")
    print(f"N test:                     {primary['n']}")
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
    print("Neither model received test t or delta.")
    print(f"Generated dataset: {generated_path}")
    print(f"Experiment outputs: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare marginalized latent DVFM and no-latent model for "
            "baseline survival prediction from x only."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to baseline latent-vs-no-latent YAML configuration.",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
