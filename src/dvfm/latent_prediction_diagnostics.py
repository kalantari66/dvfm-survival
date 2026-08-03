"""Diagnose how a trained DVFM decoder behaves under four latent inputs.

Run from the repository root:

    python -m dvfm.latent_prediction_diagnostics \
        --config configs/latent_prediction_diagnostics.yaml

The original DVFM architecture and C-ELBO training are unchanged. One trained
DVFM decoder is evaluated using:

1. posterior_mean:
   z_i = E[q_phi(z_i | x_i, t_i, delta_i)].
   This uses test follow-up and is diagnostic, not baseline prediction.

2. true_latent_aligned:
   The simulated true scalar frailty is linearly mapped into the learned raw
   latent coordinate using validation subjects only, then passed to the decoder.
   This is an oracle decoder diagnostic, not baseline prediction.

3. aggregate_posterior:
   Shared Monte Carlo draws are sampled from the aggregate training posterior
   and used for every test subject. This is an x-only baseline prediction.

4. zero:
   z_i = 0 for every test subject. This tests whether the decoder retains a
   useful x-only pathway.

A separately trained no-latent joint Weibull model is included as a reference.
Oracle event times are used only for evaluation.
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
MODES = (
    "posterior_mean",
    "true_latent_aligned",
    "aggregate_posterior",
    "zero",
)


def _weibull_survival(
    grid_normalized: np.ndarray,
    shape: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    grid = np.asarray(grid_normalized, dtype=float).reshape(1, -1)
    shape = np.asarray(shape, dtype=float).reshape(-1, 1)
    scale = np.asarray(scale, dtype=float).reshape(-1, 1)
    return np.exp(-np.power(grid / np.maximum(scale, EPS), shape))


def _median_from_survival(
    survival: np.ndarray,
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """First linearly interpolated S(t)=0.5 crossing for each curve."""
    survival = np.asarray(survival, dtype=float)
    grid = np.asarray(grid, dtype=float)
    medians = np.full(survival.shape[0], np.nan, dtype=float)
    crossed = np.zeros(survival.shape[0], dtype=bool)

    for i, curve in enumerate(survival):
        indices = np.flatnonzero(curve <= 0.5)
        if len(indices) == 0:
            continue
        j = int(indices[0])
        crossed[i] = True
        if j == 0:
            medians[i] = grid[0]
            continue

        t0, t1 = grid[j - 1], grid[j]
        s0, s1 = curve[j - 1], curve[j]
        if abs(s1 - s0) < EPS:
            medians[i] = t1
        else:
            weight = (0.5 - s0) / (s1 - s0)
            medians[i] = t0 + weight * (t1 - t0)

    return medians, crossed


def _oracle_ci(
    true_time: np.ndarray,
    predicted_time: np.ndarray,
) -> float:
    true_time = np.asarray(true_time, dtype=float)
    predicted_time = np.asarray(predicted_time, dtype=float)
    finite = np.isfinite(true_time) & np.isfinite(predicted_time)
    true_time = true_time[finite]
    predicted_time = predicted_time[finite]
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


def _collect_posterior(
    model: nn.Module,
    data: dict[str, np.ndarray],
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        SurvivalDataset(data["X"], data["time"], data["event"]),
        batch_size=batch_size,
        shuffle=False,
    )
    mus: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x, time, event in loader:
            mu, logvar = model.encoder(
                x.to(device),
                time.to(device),
                event.to(device),
            )
            mus.append(mu.cpu().numpy())
            stds.append(torch.exp(0.5 * logvar).cpu().numpy())
    return np.concatenate(mus), np.concatenate(stds)


def _fit_true_to_raw_latent_map(
    validation_latent_predictions: pd.DataFrame,
) -> dict[str, float]:
    """Fit learned_mu_raw = intercept + slope * true_z on validation only."""
    required = {"true_z", "learned_mu_raw"}
    missing = required.difference(validation_latent_predictions.columns)
    if missing:
        raise KeyError(
            "Validation latent predictions are missing: "
            + ", ".join(sorted(missing))
        )

    true_z = validation_latent_predictions["true_z"].to_numpy(dtype=float)
    learned_raw = validation_latent_predictions[
        "learned_mu_raw"
    ].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(true_z)), true_z])
    intercept, slope = np.linalg.lstsq(
        design,
        learned_raw,
        rcond=None,
    )[0]
    predicted = intercept + slope * true_z
    residual = learned_raw - predicted
    total = learned_raw - learned_raw.mean()
    r2 = 1.0 - (
        np.sum(residual**2) / max(np.sum(total**2), EPS)
    )
    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "validation_r2": float(r2),
    }


def _predict_from_subject_latents(
    model: nn.Module,
    X: np.ndarray,
    z_values: np.ndarray,
    grid_normalized: np.ndarray,
    batch_size: int,
    device: str,
) -> np.ndarray:
    z_values = np.asarray(z_values, dtype=np.float32)
    if z_values.ndim == 1:
        z_values = z_values[:, None]
    if len(z_values) != len(X):
        raise ValueError("One latent vector is required per test subject.")

    pieces: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            x = torch.as_tensor(
                X[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            z = torch.as_tensor(
                z_values[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            shape_t, scale_t, _, _ = model.decoder(x, z)
            pieces.append(
                _weibull_survival(
                    grid_normalized,
                    shape_t.cpu().numpy(),
                    scale_t.cpu().numpy(),
                )
            )
    return np.concatenate(pieces, axis=0)


def _predict_aggregate_posterior(
    model: nn.Module,
    X: np.ndarray,
    train_mu: np.ndarray,
    train_std: np.ndarray,
    grid_normalized: np.ndarray,
    n_draws: int,
    batch_size: int,
    device: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Marginalize with common latent draws shared by all test subjects."""
    if n_draws < 1:
        raise ValueError("diagnostics.mc_samples must be >= 1.")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(train_mu), size=n_draws)
    eps = rng.normal(size=(n_draws, train_mu.shape[1]))
    draws = train_mu[indices] + train_std[indices] * eps

    pieces: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            x = torch.as_tensor(
                X[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            accumulator = np.zeros(
                (len(x), len(grid_normalized)),
                dtype=np.float64,
            )
            for draw in draws:
                z = torch.as_tensor(
                    draw,
                    dtype=torch.float32,
                    device=device,
                ).reshape(1, -1).expand(len(x), -1)
                shape_t, scale_t, _, _ = model.decoder(x, z)
                accumulator += _weibull_survival(
                    grid_normalized,
                    shape_t.cpu().numpy(),
                    scale_t.cpu().numpy(),
                )
            pieces.append(accumulator / n_draws)
    return np.concatenate(pieces, axis=0), draws


def _predict_no_latent(
    model: NoLatentSurvivalModel,
    X: np.ndarray,
    grid_normalized: np.ndarray,
    batch_size: int,
    device: str,
) -> np.ndarray:
    pieces: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            x = torch.as_tensor(
                X[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            shape_t, scale_t, _, _ = model(x)
            pieces.append(
                _weibull_survival(
                    grid_normalized,
                    shape_t.cpu().numpy(),
                    scale_t.cpu().numpy(),
                )
            )
    return np.concatenate(pieces, axis=0)


def _score_mode(
    true_event: np.ndarray,
    survival: np.ndarray,
    grid_original: np.ndarray,
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray]:
    median, crossed = _median_from_survival(survival, grid_original)
    valid = crossed & np.isfinite(median)
    mae = (
        float(np.mean(np.abs(true_event[valid] - median[valid])))
        if np.any(valid)
        else float("nan")
    )
    metrics: dict[str, float | int] = {
        "n": int(len(true_event)),
        "oracle_ci": _oracle_ci(true_event[valid], median[valid]),
        "oracle_ibs": _oracle_ibs(true_event, survival, grid_original),
        "median_time_mae": mae,
        "median_crossing_fraction": float(np.mean(crossed)),
        "n_median_crossings": int(crossed.sum()),
    }
    return metrics, median, crossed


def evaluate_diagnostics(
    latent_model: nn.Module,
    no_latent_model: NoLatentSurvivalModel,
    train: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    validation_latent_predictions: pd.DataFrame,
    test_latent_predictions: pd.DataFrame,
    time_scale: float,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    cfg = config.get("diagnostics", {})
    device = str(config["dvfm"].get("device", "cpu"))
    batch_size = int(config["dvfm"]["batch_size"])
    n_points = int(cfg.get("n_time_points", 200))
    max_quantile = float(cfg.get("grid_max_quantile", 0.95))
    n_draws = int(cfg.get("mc_samples", 500))

    grid_max = float(
        np.quantile(train["true_event_time"] * time_scale, max_quantile)
    )
    grid_original = np.linspace(0.0, grid_max, n_points)
    grid_normalized = grid_original / time_scale

    train_mu, train_std = _collect_posterior(
        latent_model,
        train,
        batch_size,
        device,
    )
    test_mu, _ = _collect_posterior(
        latent_model,
        test,
        batch_size,
        device,
    )

    if test_mu.shape[1] != 1:
        raise ValueError(
            "The true-latent oracle diagnostic currently requires latent_dim=1."
        )

    alignment = _fit_true_to_raw_latent_map(
        validation_latent_predictions
    )
    true_z = test_latent_predictions["true_z"].to_numpy(dtype=float)
    true_z_in_learned_coordinate = (
        alignment["intercept"] + alignment["slope"] * true_z
    )[:, None]

    posterior_survival = _predict_from_subject_latents(
        latent_model,
        test["X"],
        test_mu,
        grid_normalized,
        batch_size,
        device,
    )
    true_survival = _predict_from_subject_latents(
        latent_model,
        test["X"],
        true_z_in_learned_coordinate,
        grid_normalized,
        batch_size,
        device,
    )
    aggregate_survival, aggregate_draws = (
        _predict_aggregate_posterior(
            latent_model,
            test["X"],
            train_mu,
            train_std,
            grid_normalized,
            n_draws,
            batch_size,
            device,
            int(config["experiment"]["seed"]) + 4000,
        )
    )
    zero_survival = _predict_from_subject_latents(
        latent_model,
        test["X"],
        np.zeros_like(test_mu),
        grid_normalized,
        batch_size,
        device,
    )
    no_latent_survival = _predict_no_latent(
        no_latent_model,
        test["X"],
        grid_normalized,
        batch_size,
        device,
    )

    true_event = test["true_event_time"] * time_scale
    event = test["event"].astype(int)
    observed_time = test["time"] * time_scale

    curves = {
        "posterior_mean": posterior_survival,
        "true_latent_aligned": true_survival,
        "aggregate_posterior": aggregate_survival,
        "zero": zero_survival,
        "no_latent": no_latent_survival,
    }
    metric_rows: dict[str, dict[str, float | int]] = {}
    medians: dict[str, np.ndarray] = {}
    crossings: dict[str, np.ndarray] = {}

    for name, survival in curves.items():
        scored, median, crossed = _score_mode(
            true_event,
            survival,
            grid_original,
        )
        metric_rows[name] = scored
        medians[name] = median
        crossings[name] = crossed

    metrics = {
        "purpose": (
            "decompose decoder quality, posterior usefulness, and "
            "test-time marginalization failure without changing DVFM"
        ),
        "mode_information": {
            "posterior_mean": "x, observed test time, and test event indicator",
            "true_latent_aligned": "x and oracle simulated frailty",
            "aggregate_posterior": "x only; common training-posterior draws",
            "zero": "x only; fixed z=0",
            "no_latent": "x only; separately trained reference model",
        },
        "valid_baseline_modes": [
            "aggregate_posterior",
            "zero",
            "no_latent",
        ],
        "diagnostic_only_modes": [
            "posterior_mean",
            "true_latent_aligned",
        ],
        "grid_max_quantile": max_quantile,
        "grid_max": grid_max,
        "n_time_points": n_points,
        "mc_samples": n_draws,
        "true_to_raw_latent_alignment": alignment,
        "training_posterior": {
            "mu_mean": train_mu.mean(axis=0).tolist(),
            "mu_std": train_mu.std(axis=0).tolist(),
            "posterior_std_mean": train_std.mean(axis=0).tolist(),
            "aggregate_draw_mean": aggregate_draws.mean(axis=0).tolist(),
            "aggregate_draw_std": aggregate_draws.std(axis=0).tolist(),
        },
        "all_test": metric_rows,
        "interpretation_flags": {
            "marginalization_failure_supported": bool(
                metric_rows["posterior_mean"]["oracle_ibs"]
                < metric_rows["aggregate_posterior"]["oracle_ibs"]
                and metric_rows["true_latent_aligned"]["oracle_ibs"]
                < metric_rows["aggregate_posterior"]["oracle_ibs"]
            ),
            "x_only_decoder_path_weaker_than_no_latent": bool(
                metric_rows["zero"]["oracle_ibs"]
                > metric_rows["no_latent"]["oracle_ibs"]
            ),
            "decoder_fails_even_with_true_latent": bool(
                metric_rows["true_latent_aligned"]["oracle_ibs"]
                >= metric_rows["no_latent"]["oracle_ibs"]
            ),
            "aggregate_and_zero_similar": bool(
                abs(
                    metric_rows["aggregate_posterior"]["oracle_ibs"]
                    - metric_rows["zero"]["oracle_ibs"]
                )
                < float(cfg.get("similarity_tolerance_ibs", 0.005))
            ),
        },
    }

    prediction_columns: dict[str, Any] = {
        "row_index": test["row_index"],
        "observed_time_diagnostic_only": observed_time,
        "event_diagnostic_only": event,
        "true_event_time": true_event,
        "true_z": true_z,
        "true_z_mapped_to_raw_decoder_coordinate": (
            true_z_in_learned_coordinate.reshape(-1)
        ),
        "posterior_mu_raw": test_mu.reshape(-1),
    }
    for name in curves:
        prediction_columns[f"{name}_predicted_median"] = medians[name]
        prediction_columns[f"{name}_median_crossed"] = crossings[name]
        prediction_columns[f"{name}_median_absolute_error"] = np.where(
            crossings[name],
            np.abs(true_event - medians[name]),
            np.nan,
        )
    predictions = pd.DataFrame(prediction_columns)

    truth = (true_event[:, None] > grid_original[None, :]).astype(float)
    brier = {"time": grid_original}
    for name, survival in curves.items():
        brier[f"{name}_oracle_brier"] = np.mean(
            (truth - survival) ** 2,
            axis=0,
        )
    return metrics, predictions, pd.DataFrame(brier)


def run(config_path: Path) -> None:
    config = load_config(config_path)
    seed = int(config["experiment"]["seed"])
    set_seed(seed)

    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/7] Loading SUPPORT source data")
    source = load_source_data(config)

    print("[2/7] Generating one semi-synthetic frailty dataset")
    generated, generation_metrics = generate_semi_synthetic(source, config)
    generated_path = Path(config["data"]["generated_csv"])
    generated_path.parent.mkdir(parents=True, exist_ok=True)
    generated.to_csv(generated_path, index=False)

    print("[3/7] Creating shared split and preprocessing")
    indices = split_indices(generated, config)
    train, valid, test, _, time_scale = prepare_model_data(
        generated,
        indices,
        config,
    )

    print("[4/7] Training the original DVFM")
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

    print("[5/7] Training the matched no-latent reference")
    set_seed(seed + 2000)
    no_latent_model, no_latent_history, no_latent_results = (
        train_no_latent_model(train, valid, config)
    )

    print("[6/7] Running four latent-input diagnostics")
    diagnostics, predictions, brier = evaluate_diagnostics(
        latent_model=latent_model,
        no_latent_model=no_latent_model,
        train=train,
        test=test,
        validation_latent_predictions=validation_latent_predictions,
        test_latent_predictions=test_latent_predictions,
        time_scale=time_scale,
        config=config,
    )

    print("[7/7] Saving outputs")
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
        output_dir / "latent_prediction_diagnostic_predictions.csv",
        index=False,
    )
    brier.to_csv(
        output_dir / "latent_prediction_diagnostic_brier.csv",
        index=False,
    )

    results = {
        "generation": generation_metrics,
        "latent_model": latent_results,
        "no_latent_model": no_latent_results,
        "latent_prediction_diagnostics": diagnostics,
        "time_scale": time_scale,
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

    print()
    print("DVFM latent-input diagnostic")
    print("----------------------------")
    print(
        f"{'Mode':<24} {'Oracle CI':>10} "
        f"{'Oracle IBS':>11} {'Median MAE':>12} {'Crossing':>10}"
    )
    for mode in (*MODES, "no_latent"):
        row = diagnostics["all_test"][mode]
        print(
            f"{mode:<24} "
            f"{row['oracle_ci']:>10.4f} "
            f"{row['oracle_ibs']:>11.4f} "
            f"{row['median_time_mae']:>12.2f} "
            f"{row['median_crossing_fraction']:>10.3f}"
        )

    print()
    print("Interpretation flags:")
    for name, value in diagnostics["interpretation_flags"].items():
        print(f"  {name}: {value}")
    print()
    print(f"Outputs: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one original DVFM decoder under posterior, true, "
            "aggregate-posterior, and zero latent inputs."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
