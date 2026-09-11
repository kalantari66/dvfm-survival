"""Focused training diagnostic for population and subject-level frailty recovery."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utility.data import SurvivalData
from utility.metrics import (
    compute_oracle_metrics, frailty_regression_metrics,
    learned_conditional_kendall_tau, pearson_correlation,
    spearman_correlation,
)
from dvfm.model import DVFM
from utility.data import SurvivalDataset
from dvfm.prediction import (
    predict_survival_curves,
    predict_survival_from_prior,
)
from dvfm.model import Decoder
from utility.synthetic import generate_clayton_gamma_frailty, generate_gaussian_shared_frailty
from utility.runtime import clone_state, seed_everything
from utility.splitting import three_way_split_indices
from .config import expand_seed_streams


def _seed(seed: int) -> None:
    seed_everything(seed)


def _clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return clone_state(model)


def _split(data: SurvivalData, cfg: dict, seed: int):
    return three_way_split_indices(data, cfg, seed)


def _part(data: SurvivalData, index: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "X": np.asarray(data.X[index], dtype=np.float32),
        "time": np.asarray(data.time[index], dtype=np.float32),
        "event": np.asarray(data.event[index], dtype=np.float32),
        "true_event_time": np.asarray(data.true_event_time[index], dtype=float),
        "true_censor_time": np.asarray(data.true_censor_time[index], dtype=float),
        "true_z": np.asarray(data.true_z[index], dtype=np.float32),
        "row_index": np.asarray(index, dtype=int),
    }


def _loader(part: dict, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    dataset = TensorDataset(
        torch.as_tensor(part["X"]), torch.as_tensor(part["time"]),
        torch.as_tensor(part["event"]), torch.as_tensor(part["true_z"]),
        torch.as_tensor(part["row_index"]),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, generator=generator,
        drop_last=bool(shuffle and len(dataset) % batch_size == 1),
    )


def _evaluate_dvfm(model, loader, beta, free_bits, device):
    totals = {"loss": 0.0, "reconstruction_nll": 0.0, "kl": 0.0, "n": 0}
    mus, logvars, targets = [], [], []
    model.eval()
    with torch.no_grad():
        for x, observed_time, event, true_z, _ in loader:
            x, observed_time, event = x.to(device), observed_time.to(device), event.to(device)
            outputs = model(x, observed_time, event)
            loss, reconstruction_nll, kl = model.loss_function(
                *outputs[:4], outputs[4], outputs[5], observed_time, event, beta, free_bits
            )
            n = len(x)
            totals["loss"] += float(loss.item()) * n
            totals["reconstruction_nll"] += float(reconstruction_nll.item()) * n
            totals["kl"] += float(kl.item()) * n
            totals["n"] += n
            mus.append(outputs[4].cpu().numpy())
            logvars.append(outputs[5].cpu().numpy())
            targets.append(true_z.numpy())
    n = max(totals.pop("n"), 1)
    result = {key: value / n for key, value in totals.items()}
    mu, logvar, target = np.concatenate(mus), np.concatenate(logvars), np.concatenate(targets)
    per_dimension_kl = np.mean(-0.5 * (1 + logvar - mu ** 2 - np.exp(logvar)), axis=0)
    result.update({
        "frailty_pearson": pearson_correlation(mu[:, 0], target),
        "frailty_spearman": spearman_correlation(mu[:, 0], target),
        "active_latent_dimensions": int(np.sum(per_dimension_kl > 0.01)),
    })
    return result


def _checkpoint_diagnostics(
    model, validation, x_reference, variant, common, checkpoint_epoch, cfg, device, seed
):
    """Evaluate a stored state without using test outcomes or test frailties."""
    batch_size = int(variant.get("batch_size", common["batch_size"]))
    loader = _loader(validation, batch_size, False, seed)
    warmup = int(variant["warmup_epochs"])
    beta = float(variant["beta_max"]) * min(1.0, checkpoint_epoch / warmup) if warmup > 0 else float(variant["beta_max"])
    metrics = _evaluate_dvfm(model, loader, beta, float(common.get("free_bits", 0.0)), device)
    learned_tau = learned_conditional_kendall_tau(
        model,
        x_reference,
        int(cfg["evaluation"]["dependence_samples_checkpoint"]),
        device,
        seed + 20_000,
    )
    raw_pearson = metrics["frailty_pearson"]
    alignment_sign = 1.0 if not np.isfinite(raw_pearson) or raw_pearson >= 0 else -1.0
    return {
        "checkpoint_epoch": int(checkpoint_epoch),
        "checkpoint_validation_elbo": metrics["loss"],
        "checkpoint_validation_reconstruction_nll": metrics["reconstruction_nll"],
        "checkpoint_validation_kl": metrics["kl"],
        "checkpoint_validation_frailty_pearson_raw": metrics["frailty_pearson"],
        "checkpoint_validation_frailty_spearman_raw": metrics["frailty_spearman"],
        "checkpoint_validation_alignment_sign": alignment_sign,
        "checkpoint_validation_frailty_pearson": alignment_sign * metrics["frailty_pearson"],
        "checkpoint_validation_frailty_spearman": alignment_sign * metrics["frailty_spearman"],
        "checkpoint_active_latent_dimensions": metrics["active_latent_dimensions"],
        "learned_conditional_kendall_tau": learned_tau,
        "conditional_kendall_tau_error": learned_tau - float(cfg["data"]["kendall_tau"]),
        "absolute_conditional_kendall_tau_error": abs(learned_tau - float(cfg["data"]["kendall_tau"])),
    }


def _train_dvfm_variant(train, validation, variant, common, device, model_seed, tau_samples):
    _seed(model_seed)
    batch_size = int(variant.get("batch_size", common["batch_size"]))
    train_loader = _loader(train, batch_size, True, model_seed)
    validation_loader = _loader(validation, batch_size, False, model_seed)
    model = DVFM(
        input_dim=train["X"].shape[1], latent_dim=1,
        encoder_hidden=list(common.get("encoder_hidden", [64, 32])),
        decoder_hidden=list(common.get("decoder_hidden", [32, 64])),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(variant["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=int(variant["lr_patience"]),
        min_lr=float(variant["minimum_learning_rate"]),
    )
    maximum_epochs = int(variant["maximum_epochs"])
    minimum_epochs = int(variant.get("minimum_epochs", 0))
    patience = variant.get("early_stopping_patience")
    patience = None if patience is None else int(patience)
    beta_max, warmup = float(variant["beta_max"]), int(variant["warmup_epochs"])
    free_bits = float(common.get("free_bits", 0.0))
    scheduler_metric = str(variant["scheduler_metric"])
    best_state, best_epoch, best_reconstruction = None, -1, float("inf")
    no_improvement, history = 0, []
    x_reference = np.mean(train["X"], axis=0)

    for epoch in range(1, maximum_epochs + 1):
        beta = beta_max * min(1.0, epoch / warmup) if warmup > 0 else beta_max
        model.train()
        totals = {"loss": 0.0, "reconstruction_nll": 0.0, "kl": 0.0, "n": 0}
        for x, observed_time, event, _, _ in train_loader:
            x, observed_time, event = x.to(device), observed_time.to(device), event.to(device)
            optimizer.zero_grad()
            outputs = model(x, observed_time, event)
            loss, reconstruction_nll, kl = model.loss_function(
                *outputs[:4], outputs[4], outputs[5], observed_time, event, beta, free_bits
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            n = len(x)
            totals["loss"] += float(loss.item()) * n
            totals["reconstruction_nll"] += float(reconstruction_nll.item()) * n
            totals["kl"] += float(kl.item()) * n
            totals["n"] += n
        n = max(totals.pop("n"), 1)
        training = {key: value / n for key, value in totals.items()}
        validation_metrics = _evaluate_dvfm(model, validation_loader, beta, free_bits, device)
        scheduler_value = (
            validation_metrics["loss"] if scheduler_metric == "validation_elbo"
            else validation_metrics["reconstruction_nll"]
        )
        scheduler.step(scheduler_value)
        learned_tau = learned_conditional_kendall_tau(
            model, x_reference, int(tau_samples), device, model_seed + 10_000
        )
        raw_pearson = validation_metrics["frailty_pearson"]
        alignment_sign = 1.0 if not np.isfinite(raw_pearson) or raw_pearson >= 0 else -1.0
        history.append({
            "epoch": epoch, "beta": beta,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_elbo": training["loss"],
            "train_reconstruction_nll": training["reconstruction_nll"],
            "train_kl": training["kl"],
            "validation_elbo": validation_metrics["loss"],
            "validation_reconstruction_nll": validation_metrics["reconstruction_nll"],
            "validation_kl": validation_metrics["kl"],
            "validation_alignment_sign": alignment_sign,
            "validation_frailty_pearson_raw": raw_pearson,
            "validation_frailty_spearman_raw": validation_metrics["frailty_spearman"],
            "validation_frailty_pearson": alignment_sign * raw_pearson,
            "validation_frailty_spearman": alignment_sign * validation_metrics["frailty_spearman"],
            "active_latent_dimensions": validation_metrics["active_latent_dimensions"],
            "learned_conditional_kendall_tau": learned_tau,
        })
        if validation_metrics["reconstruction_nll"] < best_reconstruction - 1e-6:
            best_reconstruction = validation_metrics["reconstruction_nll"]
            best_epoch, best_state, no_improvement = epoch, _clone_state(model), 0
        else:
            no_improvement += 1
        if patience is not None and epoch >= minimum_epochs and no_improvement >= patience:
            break
    if best_state is None:
        raise RuntimeError("Training did not produce a finite checkpoint")
    return model, _clone_state(model), best_state, best_epoch, history


def _encode(model, part, batch_size, device):
    loader = _loader(part, batch_size, False, 0)
    mus, stds = [], []
    model.eval()
    with torch.no_grad():
        for x, observed_time, event, _, _ in loader:
            mu, logvar = model.encoder(x.to(device), observed_time.to(device), event.to(device))
            mus.append(mu.cpu().numpy()[:, 0])
            stds.append(torch.exp(0.5 * logvar).cpu().numpy()[:, 0])
    return np.concatenate(mus), np.concatenate(stds)


def _recovery_for_checkpoint(model, validation, test, batch_size, device, context):
    validation_mu, validation_std = _encode(model, validation, batch_size, device)
    test_mu, test_std = _encode(model, test, batch_size, device)
    raw_validation_pearson = pearson_correlation(validation_mu, validation["true_z"])
    sign = 1.0 if not np.isfinite(raw_validation_pearson) or raw_validation_pearson >= 0 else -1.0
    validation_aligned, test_aligned = sign * validation_mu, sign * test_mu
    calibrator = LinearRegression().fit(validation_aligned.reshape(-1, 1), validation["true_z"])
    validation_calibrated = calibrator.predict(validation_aligned.reshape(-1, 1))
    test_calibrated = calibrator.predict(test_aligned.reshape(-1, 1))
    subject_rows, metric_rows = [], []
    for split_name, part, raw_mu, std, aligned, calibrated in (
        ("validation", validation, validation_mu, validation_std, validation_aligned, validation_calibrated),
        ("test", test, test_mu, test_std, test_aligned, test_calibrated),
    ):
        for index in range(len(part["time"])):
            subject_rows.append({
                **context, "split": split_name, "row_index": int(part["row_index"][index]),
                "event": int(part["event"][index]), "true_z": float(part["true_z"][index]),
                "posterior_mu_raw": float(raw_mu[index]), "posterior_std": float(std[index]),
                "posterior_mu_aligned": float(aligned[index]),
                "frailty_z_calibrated": float(calibrated[index]),
            })
        masks = {
            "all": np.ones(len(part["time"]), dtype=bool),
            "event_observed": part["event"] == 1,
            "censored": part["event"] == 0,
        }
        for subgroup, mask in masks.items():
            truth, estimate = part["true_z"][mask], calibrated[mask]
            metric_rows.append({
                **context, "split": split_name, "subgroup": subgroup, "n": int(mask.sum()),
                "frailty_pearson": pearson_correlation(aligned[mask], truth),
                "frailty_spearman": spearman_correlation(aligned[mask], truth),
                **frailty_regression_metrics(truth, estimate),
                "validation_alignment_sign": sign,
                "validation_calibration_intercept": float(calibrator.intercept_),
                "validation_calibration_slope": float(calibrator.coef_[0]),
            })
    return subject_rows, metric_rows


def _prediction_rows(model, train, test, cfg, device, context):
    evaluation = cfg["evaluation"]
    grid = np.linspace(
        0.0, float(np.quantile(train["true_event_time"], evaluation["grid_max_quantile"])),
        int(evaluation["n_time_points"]),
    )
    batch_size, mc_samples = int(cfg["models"]["dvfm"]["batch_size"]), int(evaluation["mc_samples"])
    prediction_loader = DataLoader(
        SurvivalDataset(train["X"], train["time"], train["event"]),
        batch_size=batch_size, shuffle=False,
    )
    predictions = {
        "prior": predict_survival_from_prior(model, test["X"], grid, mc_samples, device),
        "aggregate_posterior": predict_survival_curves(
            model, test["X"], grid, prediction_loader, mc_samples, device
        ),
    }
    rows, calibration = [], []
    for mode, survival in predictions.items():
        metrics = compute_oracle_metrics(
            survival, grid, test["true_event_time"], test["event"]
        )
        rows.append({
            **context, "prediction_mode": mode, **metrics,
        })
        for grid_index in np.linspace(0, len(grid) - 1, 10, dtype=int):
            calibration.append({
                **context, "prediction_mode": mode, "time": float(grid[grid_index]),
                "mean_predicted_survival": float(survival[:, grid_index].mean()),
                "empirical_oracle_survival": float(np.mean(test["true_event_time"] > grid[grid_index])),
            })
    return rows, calibration


class OracleZDecoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.decoder = Decoder(input_dim, 1, hidden_dims)

    def reconstruction_nll(self, x, z, observed_time, event):
        shape_t, scale_t, shape_c, scale_c = self.decoder(x, z.reshape(-1, 1))
        eps = 1e-8
        observed_time = torch.clamp(observed_time, min=eps)
        log_f_t = torch.log(shape_t) - torch.log(scale_t) + (shape_t - 1) * (torch.log(observed_time) - torch.log(scale_t)) - (observed_time / scale_t) ** shape_t
        log_s_t = -(observed_time / scale_t) ** shape_t
        log_f_c = torch.log(shape_c) - torch.log(scale_c) + (shape_c - 1) * (torch.log(observed_time) - torch.log(scale_c)) - (observed_time / scale_c) ** shape_c
        log_s_c = -(observed_time / scale_c) ** shape_c
        return -(event * (log_f_t + log_s_c) + (1 - event) * (log_s_t + log_f_c)).mean()


def _train_oracle_decoder(train, validation, cfg, device, seed):
    settings = cfg["models"]["oracle_z_decoder"]
    _seed(seed)
    batch_size = int(settings["batch_size"])
    train_loader = _loader(train, batch_size, True, seed)
    validation_loader = _loader(validation, batch_size, False, seed)
    model = OracleZDecoder(train["X"].shape[1], list(settings.get("decoder_hidden", [32, 64]))).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(settings["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=int(settings["lr_patience"]),
        min_lr=float(settings["minimum_learning_rate"]),
    )
    best_state, best_epoch, best_value, no_improvement, history = None, -1, float("inf"), 0, []
    for epoch in range(1, int(settings["maximum_epochs"]) + 1):
        model.train(); train_total = 0.0; train_n = 0
        for x, observed_time, event, true_z, _ in train_loader:
            x, observed_time, event, true_z = x.to(device), observed_time.to(device), event.to(device), true_z.to(device)
            optimizer.zero_grad(); loss = model.reconstruction_nll(x, true_z, observed_time, event)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite oracle-Z loss at epoch {epoch}")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            train_total += float(loss.item()) * len(x); train_n += len(x)
        model.eval(); validation_total = 0.0; validation_n = 0
        with torch.no_grad():
            for x, observed_time, event, true_z, _ in validation_loader:
                loss = model.reconstruction_nll(x.to(device), true_z.to(device), observed_time.to(device), event.to(device))
                validation_total += float(loss.item()) * len(x); validation_n += len(x)
        validation_nll = validation_total / validation_n
        scheduler.step(validation_nll)
        history.append({
            "epoch": epoch, "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_reconstruction_nll": train_total / train_n,
            "validation_reconstruction_nll": validation_nll,
        })
        if validation_nll < best_value - 1e-6:
            best_value, best_epoch, best_state, no_improvement = validation_nll, epoch, _clone_state(model), 0
        else:
            no_improvement += 1
        if epoch >= int(settings["minimum_epochs"]) and no_improvement >= int(settings["early_stopping_patience"]):
            break
    if best_state is None:
        raise RuntimeError("Oracle-Z training did not produce a checkpoint")
    return model, _clone_state(model), best_state, best_epoch, history


def _oracle_prediction_rows(model, train, test, cfg, device, context):
    evaluation = cfg["evaluation"]
    grid = np.linspace(0.0, float(np.quantile(train["true_event_time"], evaluation["grid_max_quantile"])), int(evaluation["n_time_points"]))
    x = torch.as_tensor(test["X"], dtype=torch.float32, device=device)
    grid_tensor = torch.as_tensor(grid, dtype=torch.float32, device=device)
    mc_samples = int(evaluation["mc_samples"])
    rng = np.random.default_rng(int(context["model_seed"]) + 40_000)
    curves = {}
    with torch.no_grad():
        z = torch.as_tensor(test["true_z"], dtype=torch.float32, device=device)
        shape, scale, _, _ = model.decoder(x, z[:, None])
        curves["oracle_subject_z"] = torch.exp(-((grid_tensor[None, :] / scale[:, None]) ** shape[:, None])).cpu().numpy()
        population = torch.zeros((len(x), len(grid)), device=device)
        for sampled_z in rng.choice(train["true_z"], size=mc_samples, replace=True):
            z_draw = torch.full((len(x), 1), float(sampled_z), device=device)
            shape, scale, _, _ = model.decoder(x, z_draw)
            population += torch.exp(-((grid_tensor[None, :] / scale[:, None]) ** shape[:, None]))
        curves["empirical_train_true_z"] = (population / mc_samples).cpu().numpy()
    rows, calibration = [], []
    for mode, survival in curves.items():
        metrics = compute_oracle_metrics(
            survival, grid, test["true_event_time"], test["event"]
        )
        rows.append({
            **context, "prediction_mode": mode, **metrics,
        })
        for grid_index in np.linspace(0, len(grid) - 1, 10, dtype=int):
            calibration.append({
                **context, "prediction_mode": mode, "time": float(grid[grid_index]),
                "mean_predicted_survival": float(survival[:, grid_index].mean()),
                "empirical_oracle_survival": float(np.mean(test["true_event_time"] > grid[grid_index])),
            })
    return rows, calibration


def _generate(mechanism, cfg, seeds):
    common = dict(
        n_samples=int(cfg["data"]["n_samples"]), n_features=int(cfg["data"]["n_features"]),
        kendall_tau=float(cfg["data"]["kendall_tau"]),
        censoring_rate=float(cfg["data"]["censoring_rate"]),
        dgp_seed=int(seeds["dgp"]), sampling_seed=int(seeds["sampling"]),
    )
    if mechanism == "gaussian_shared_frailty":
        return generate_gaussian_shared_frailty(
            **common, calibration_samples=int(cfg["data"].get("calibration_samples", 50_000))
        )
    if mechanism == "clayton_gamma_frailty":
        return generate_clayton_gamma_frailty(**common)
    raise ValueError(f"Unknown frailty mechanism: {mechanism}")


def run_frailty_recovery_diagnostic(cfg: dict, out_dir: Path, device: torch.device) -> pd.DataFrame:
    """Run the four training variants plus the oracle-Z decoder control."""
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    seed_streams = expand_seed_streams(cfg["seeds"])
    rows, recovery, subjects, histories, calibration, manifest = [], [], [], [], [], []
    for repeat, seeds in enumerate(seed_streams):
        dgp_seed = seeds["dgp"]
        sampling_seed = seeds["sampling"]
        split_seed = seeds["split"]
        model_seed = seeds["model"]
        seed_values = {"dgp": int(dgp_seed), "sampling": int(sampling_seed)}
        for mechanism in cfg["data"]["mechanisms"]:
            generated = _generate(mechanism, cfg, seed_values)
            full = SurvivalData(
                generated.X, generated.observed_time, generated.event,
                [f"X{i}" for i in range(generated.X.shape[1])], generated.event_time,
                generated.censor_time, generated.true_z,
            )
            train_index, validation_index, test_index = _split(full, cfg["split"], int(split_seed))
            train, validation, test = _part(full, train_index), _part(full, validation_index), _part(full, test_index)
            base = {
                "study": cfg["study"]["name"], "mechanism": mechanism,
                "target_kendall_tau": float(cfg["data"]["kendall_tau"]),
                "empirical_conditional_kendall_tau": generated.empirical_conditional_kendall_tau,
                "empirical_marginal_kendall_tau": generated.empirical_marginal_kendall_tau,
                "target_censoring_rate": float(cfg["data"]["censoring_rate"]),
                "achieved_censoring_rate": generated.achieved_censoring_rate,
                "repeat": repeat, "dgp_seed": int(dgp_seed),
                "sampling_seed": int(sampling_seed), "split_seed": int(split_seed),
                "model_seed": int(model_seed),
            }
            # Arms that differ only in checkpoint selection share one fitted
            # trajectory, making the checkpoint comparison exactly paired.
            training_cache: dict[str, tuple] = {}
            for variant in cfg["training_variants"]:
                started = time.perf_counter(); name = str(variant["name"])
                context = {**base, "architecture": "dvfm", "training_variant": name}
                training_key = json.dumps(
                    {key: value for key, value in variant.items()
                     if key not in {"name", "checkpoint_selection"}},
                    sort_keys=True,
                )
                reused_from = ""
                try:
                    if training_key in training_cache:
                        final_state, best_state, best_epoch, history, reused_from = training_cache[training_key]
                        model = DVFM(
                            input_dim=train["X"].shape[1], latent_dim=1,
                            encoder_hidden=list(cfg["models"]["dvfm"].get("encoder_hidden", [64, 32])),
                            decoder_hidden=list(cfg["models"]["dvfm"].get("decoder_hidden", [32, 64])),
                        ).to(device)
                    else:
                        model, final_state, best_state, best_epoch, history = _train_dvfm_variant(
                            train, validation, variant, cfg["models"]["dvfm"], device,
                            int(model_seed), cfg["evaluation"]["dependence_samples_per_epoch"],
                        )
                        training_cache[training_key] = (
                            final_state, best_state, best_epoch, history, name,
                        )
                    primary = str(variant["checkpoint_selection"])
                    for checkpoint, state in (("final", final_state), ("best_validation_reconstruction_nll", best_state)):
                        model.load_state_dict(state); model.to(device)
                        checkpoint_epoch = len(history) if checkpoint == "final" else best_epoch
                        diagnostics = _checkpoint_diagnostics(
                            model, validation, np.mean(train["X"], axis=0),
                            variant, cfg["models"]["dvfm"],
                            checkpoint_epoch, cfg, device,
                            int(model_seed) + (0 if checkpoint == "final" else 50_000),
                        )
                        checkpoint_context = {
                            **context, "checkpoint": checkpoint,
                            "is_primary_checkpoint": checkpoint == primary,
                            "best_epoch": best_epoch, "epochs_completed": len(history),
                            **diagnostics,
                        }
                        tag = f"{mechanism}_{name}_repeat{repeat}_{checkpoint}"
                        torch.save(state, checkpoint_dir / f"{tag}.pt")
                        subject_part, recovery_part = _recovery_for_checkpoint(
                            model, validation, test,
                            int(variant.get("batch_size", cfg["models"]["dvfm"]["batch_size"])),
                            device, checkpoint_context,
                        )
                        subjects.extend(subject_part); recovery.extend(recovery_part)
                        _seed(int(model_seed) + (0 if checkpoint == "final" else 50_000))
                        prediction_part, calibration_part = _prediction_rows(
                            model, train, test, cfg, device, checkpoint_context
                        )
                        rows.extend(prediction_part); calibration.extend(calibration_part)
                    histories.extend([{**context, **item} for item in history])
                    manifest.append({
                        **context, "status": "success",
                        "training_reused_from": reused_from,
                        "runtime_seconds": time.perf_counter() - started, "error": "",
                    })
                except Exception as error:
                    manifest.append({
                        **context, "status": "failed",
                        "training_reused_from": reused_from,
                        "runtime_seconds": time.perf_counter() - started, "error": repr(error),
                    })
            oracle_context = {**base, "architecture": "oracle_z_decoder", "training_variant": "oracle_z"}
            started = time.perf_counter()
            try:
                model, final_state, best_state, best_epoch, history = _train_oracle_decoder(
                    train, validation, cfg, device, int(model_seed) + 70_000
                )
                for checkpoint, state in (("final", final_state), ("best_validation_reconstruction_nll", best_state)):
                    model.load_state_dict(state); model.to(device)
                    checkpoint_context = {
                        **oracle_context, "checkpoint": checkpoint,
                        "is_primary_checkpoint": checkpoint == "best_validation_reconstruction_nll",
                        "best_epoch": best_epoch, "epochs_completed": len(history),
                    }
                    tag = f"{mechanism}_oracle_z_repeat{repeat}_{checkpoint}"
                    torch.save(state, checkpoint_dir / f"{tag}.pt")
                    prediction_part, calibration_part = _oracle_prediction_rows(
                        model, train, test, cfg, device, checkpoint_context
                    )
                    rows.extend(prediction_part); calibration.extend(calibration_part)
                histories.extend([{**oracle_context, **item} for item in history])
                manifest.append({**oracle_context, "status": "success", "runtime_seconds": time.perf_counter() - started, "error": ""})
            except Exception as error:
                manifest.append({**oracle_context, "status": "failed", "runtime_seconds": time.perf_counter() - started, "error": repr(error)})

    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("All frailty diagnostic fits failed")
    pd.DataFrame(recovery).to_csv(out_dir / "frailty_recovery.csv", index=False)
    pd.DataFrame(subjects).to_csv(out_dir / "subject_latent_diagnostics.csv.gz", index=False, compression="gzip")
    pd.DataFrame(histories).to_csv(out_dir / "training_history.csv.gz", index=False, compression="gzip")
    pd.DataFrame(calibration).to_csv(out_dir / "calibration_curves.csv.gz", index=False, compression="gzip")
    pd.DataFrame(manifest).to_csv(out_dir / "run_manifest.csv", index=False)
    results.to_csv(out_dir / "results_raw.csv", index=False)
    groups = ["mechanism", "architecture", "training_variant", "checkpoint", "is_primary_checkpoint", "prediction_mode"]
    metrics = ["oracle_ibs", "oracle_ci", "oracle_mae", "oracle_mae_censored", "oracle_mae_uncensored"]
    results.groupby(groups)[metrics].mean().reset_index().to_csv(out_dir / "results_mean.csv", index=False)
    results.groupby(groups)[metrics].std().reset_index().to_csv(out_dir / "results_std.csv", index=False)
    with (out_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=2, default=str)
    failures = [item for item in manifest if item["status"] == "failed"]
    if failures:
        raise RuntimeError(
            f"{len(failures)} frailty diagnostic fits failed; see {out_dir / 'run_manifest.csv'}"
        )
    return results


__all__ = ["run_frailty_recovery_diagnostic"]
