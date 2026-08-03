"""Train the explicit shared-frailty DVFM in the existing baseline experiment.

Place this file and shared_frailty_core.py under src/dvfm/, then run:

    python -m dvfm.shared_frailty_baseline_experiment \
        --config configs/shared_frailty_baseline.yaml

The no-latent comparator and baseline evaluation are reused from the existing
matched experiment.  Only the latent architecture is changed.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader

from .baseline_latent_vs_no_latent import evaluate_baseline_models
from .latent_vs_no_latent import train_no_latent_model
from .reference_core import SurvivalDataset
from .semi_synthetic_frailty_prediction import (
    generate_semi_synthetic,
    load_config,
    load_source_data,
    prepare_model_data,
    split_indices,
)
from .shared_frailty_core import SharedFrailtyDVFM, WeibullBounds

EPS = 1e-10


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_corr(function: Any, x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) < EPS or np.std(y) < EPS:
        return float("nan")
    return float(function(x, y).statistic)


def _encode(
    model: SharedFrailtyDVFM,
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
                x.to(device), time.to(device), event.to(device)
            )
            mus.append(mu.cpu().numpy().reshape(-1))
            stds.append(torch.exp(0.5 * logvar).cpu().numpy().reshape(-1))
    return np.concatenate(mus), np.concatenate(stds)


def _latent_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    linear = LinearRegression().fit(prediction.reshape(-1, 1), target)
    calibrated = linear.predict(prediction.reshape(-1, 1))
    return {
        "pearson": _safe_corr(pearsonr, prediction, target),
        "spearman": _safe_corr(spearmanr, prediction, target),
        "linear_alignment_r2": float(r2_score(target, calibrated)),
        "linear_alignment_rmse": float(
            mean_squared_error(target, calibrated) ** 0.5
        ),
    }


def train_shared_frailty_dvfm(
    train: dict[str, np.ndarray],
    valid: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    config: dict[str, Any],
) -> tuple[SharedFrailtyDVFM, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    dvfm_cfg = config["dvfm"]
    training_cfg = config.get("training", {})
    frailty_cfg = config["shared_frailty"]
    bounds_cfg = config["weibull_constraints"]
    device = str(dvfm_cfg.get("device", "cpu"))
    batch_size = int(dvfm_cfg["batch_size"])

    model = SharedFrailtyDVFM(
        input_dim=train["X"].shape[1],
        latent_dim=int(dvfm_cfg["latent_dim"]),
        encoder_hidden=list(dvfm_cfg["encoder_hidden"]),
        decoder_hidden=list(dvfm_cfg["decoder_hidden"]),
        bounds=WeibullBounds(
            min_shape=float(bounds_cfg["min_shape"]),
            max_shape=float(bounds_cfg["max_shape"]),
            min_scale=float(bounds_cfg["min_scale"]),
            max_scale=float(bounds_cfg["max_scale"]),
        ),
        min_loading=float(frailty_cfg.get("min_loading", 0.0)),
        max_loading=float(frailty_cfg.get("max_loading", 3.0)),
        loading_init=float(frailty_cfg.get("loading_init", 0.5)),
    ).to(device)

    train_loader = DataLoader(
        SurvivalDataset(train["X"], train["time"], train["event"]),
        batch_size=batch_size,
        shuffle=True,
        drop_last=(len(train["X"]) % batch_size == 1),
    )
    valid_loader = DataLoader(
        SurvivalDataset(valid["X"], valid["time"], valid["event"]),
        batch_size=batch_size,
        shuffle=False,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(dvfm_cfg["learning_rate"])
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=int(training_cfg.get("lr_patience", 12)),
    )

    epochs = int(dvfm_cfg["epochs"])
    beta_max = float(dvfm_cfg["beta_max"])
    warmup_epochs = int(dvfm_cfg["warmup_epochs"])
    free_bits = float(dvfm_cfg.get("free_bits", 0.0))
    minimum_epochs = int(training_cfg.get("minimum_epochs", 0))
    patience = int(training_cfg.get("early_stopping_patience", epochs))

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_nll = np.inf
    stale = 0
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        beta = (
            beta_max * min(1.0, (epoch + 1) / warmup_epochs)
            if warmup_epochs > 0
            else beta_max
        )
        model.train()
        train_loss = train_nll = train_kl = 0.0

        for x, time, event in train_loader:
            x, time, event = x.to(device), time.to(device), event.to(device)
            optimizer.zero_grad()
            outputs = model(x, time, event)
            loss, nll, kl = model.loss_function(
                *outputs, time, event, beta, free_bits
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += float(loss.item())
            train_nll += float(nll.item())
            train_kl += float(kl.item())

        train_loss /= len(train_loader)
        train_nll /= len(train_loader)
        train_kl /= len(train_loader)

        model.eval()
        valid_loss = valid_nll = valid_kl = 0.0
        with torch.no_grad():
            for x, time, event in valid_loader:
                x, time, event = x.to(device), time.to(device), event.to(device)
                outputs = model(x, time, event)
                loss, nll, kl = model.loss_function(
                    *outputs, time, event, beta, free_bits
                )
                valid_loss += float(loss.item())
                valid_nll += float(nll.item())
                valid_kl += float(kl.item())
        valid_loss /= len(valid_loader)
        valid_nll /= len(valid_loader)
        valid_kl /= len(valid_loader)
        scheduler.step(valid_nll)

        valid_mu, _ = _encode(model, valid, batch_size, device)
        valid_target = valid["target_z"].reshape(-1)
        valid_abs_r = abs(_safe_corr(pearsonr, valid_mu, valid_target))
        loadings = model.decoder.diagnostics()

        history.append(
            {
                "epoch": epoch + 1,
                "beta": beta,
                "train_loss": train_loss,
                "train_reconstruction": train_nll,
                "train_kl": train_kl,
                "validation_loss": valid_loss,
                "validation_reconstruction_nll": valid_nll,
                "validation_kl": valid_kl,
                "validation_abs_pearson": valid_abs_r,
                "event_frailty_loading": loadings["event_frailty_loading"],
                "censor_frailty_loading": loadings["censor_frailty_loading"],
                "loading_product": loadings["loading_product"],
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )

        if valid_nll < best_nll - 1e-6:
            best_nll = valid_nll
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1

        if epoch == 0 or (epoch + 1) % 25 == 0:
            print(
                f"[Shared frailty] Epoch {epoch + 1}/{epochs}, "
                f"Train NLL: {train_nll:.4f}, Val NLL: {valid_nll:.4f}, "
                f"KL: {valid_kl:.4f}, |r|: {valid_abs_r:.4f}, "
                f"a_T: {loadings['event_frailty_loading']:.3f}, "
                f"a_C: {loadings['censor_frailty_loading']:.3f}"
            )

        if epoch + 1 >= minimum_epochs and stale >= patience:
            print(
                f"[Shared frailty] Early stopping at epoch {epoch + 1}; "
                f"best epoch was {best_epoch}."
            )
            break

    if best_state is None:
        raise RuntimeError("Shared-frailty training produced no checkpoint.")
    model.load_state_dict(best_state)
    model.to(device)

    valid_mu, valid_std = _encode(model, valid, batch_size, device)
    valid_target = valid["target_z"].reshape(-1)
    raw_validation_r = _safe_corr(pearsonr, valid_mu, valid_target)
    sign = 1.0 if raw_validation_r >= 0 else -1.0

    test_mu, test_std = _encode(model, test, batch_size, device)
    test_target = test["target_z"].reshape(-1)
    aligned_test_mu = sign * test_mu

    valid_frame = pd.DataFrame(
        {
            "row_index": valid["row_index"],
            "event": valid["event"].astype(int),
            "true_z": valid_target,
            "learned_mu_raw": valid_mu,
            "learned_mu_aligned": sign * valid_mu,
            "learned_std": valid_std,
        }
    )
    test_frame = pd.DataFrame(
        {
            "row_index": test["row_index"],
            "event": test["event"].astype(int),
            "true_z": test_target,
            "learned_mu_raw": test_mu,
            "learned_mu_aligned": aligned_test_mu,
            "learned_std": test_std,
        }
    )

    metrics = _latent_metrics(aligned_test_mu, test_target)
    metrics.update(
        {
            "raw_test_pearson": _safe_corr(pearsonr, test_mu, test_target),
            "validation_pearson_raw": raw_validation_r,
            "sign_alignment_from_validation": sign,
            "posterior_std_mean": float(test_std.mean()),
            "posterior_std_median": float(np.median(test_std)),
            "best_epoch": best_epoch,
            "best_validation_reconstruction_nll": float(best_nll),
            "epochs_completed": len(history),
            **model.decoder.diagnostics(),
            "architecture": (
                "x-only Weibull shapes/base scales with a scalar z entering "
                "event and censoring hazards through explicit PH loadings"
            ),
            "training_history": history,
        }
    )
    return model, valid_frame, test_frame, metrics


def _parameter_count(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def run(config_path: Path) -> None:
    config = load_config(config_path)
    seed = int(config["experiment"]["seed"])
    set_seed(seed)

    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/7] Loading SUPPORT")
    source = load_source_data(config)
    print("[2/7] Generating shared-frailty semi-synthetic data")
    generated, generation_metrics = generate_semi_synthetic(source, config)
    generated_path = Path(config["data"]["generated_csv"])
    generated_path.parent.mkdir(parents=True, exist_ok=True)
    generated.to_csv(generated_path, index=False)

    print("[3/7] Shared split and preprocessing")
    indices = split_indices(generated, config)
    train, valid, test, _, time_scale = prepare_model_data(
        generated, indices, config
    )

    print("[4/7] Training explicit shared-frailty DVFM")
    set_seed(seed + 1000)
    model, latent_valid, latent_test, latent_results = train_shared_frailty_dvfm(
        train, valid, test, config
    )
    latent_history = pd.DataFrame(latent_results.pop("training_history"))

    print("[5/7] Training matched no-latent model")
    set_seed(seed + 2000)
    no_latent_model, no_latent_history, no_latent_results = train_no_latent_model(
        train, valid, config
    )

    print("[6/7] Baseline prediction from x only")
    comparison, predictions, brier = evaluate_baseline_models(
        latent_model=model,
        no_latent_model=no_latent_model,
        train=train,
        test=test,
        time_scale=time_scale,
        config=config,
    )

    print("[7/7] Saving")
    latent_results["n_parameters"] = _parameter_count(model)
    no_latent_results["n_parameters"] = _parameter_count(no_latent_model)
    torch.save(model.state_dict(), output_dir / "shared_frailty_dvfm.pt")
    torch.save(no_latent_model.state_dict(), output_dir / "no_latent.pt")
    latent_history.to_csv(output_dir / "shared_frailty_training_history.csv", index=False)
    no_latent_history.to_csv(output_dir / "no_latent_training_history.csv", index=False)
    latent_valid.to_csv(output_dir / "latent_recovery_validation.csv", index=False)
    latent_test.to_csv(output_dir / "latent_recovery_test.csv", index=False)
    predictions.to_csv(output_dir / "baseline_predictions.csv", index=False)
    brier.to_csv(output_dir / "oracle_brier_curves.csv", index=False)

    results = {
        "generation": generation_metrics,
        "shared_frailty_model": latent_results,
        "no_latent_model": no_latent_results,
        "baseline_survival_comparison": comparison,
        "time_scale": time_scale,
        "n_train": len(train["time"]),
        "n_validation": len(valid["time"]),
        "n_test": len(test["time"]),
    }
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    primary = comparison["all_test_primary"]
    print()
    print("Explicit shared-frailty DVFM vs no latent")
    print("------------------------------------------")
    print(f"Learned event loading:      {latent_results['event_frailty_loading']:.4f}")
    print(f"Learned censor loading:     {latent_results['censor_frailty_loading']:.4f}")
    print(f"Latent recovery Pearson:    {latent_results['pearson']:.4f}")
    print(f"Oracle CI, shared frailty:  {primary['latent_oracle_ci']:.4f}")
    print(f"Oracle CI, no latent:       {primary['no_latent_oracle_ci']:.4f}")
    print(f"Delta CI:                   {primary['delta_oracle_ci']:+.4f}")
    print(f"Oracle IBS, shared frailty: {primary['latent_oracle_ibs']:.4f}")
    print(f"Oracle IBS, no latent:      {primary['no_latent_oracle_ibs']:.4f}")
    print(f"Delta IBS:                  {primary['delta_oracle_ibs']:+.4f}")
    print(f"Outputs: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
