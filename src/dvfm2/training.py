"""Training utilities for DVFM2.

By default this trainer runs the full configured number of epochs. It does not
restore an early checkpoint. That matches the current diagnostic strategy used
for DVFM while keeping best validation reconstruction NLL as a diagnostic only.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import torch

from .model import SharedPrivateDVFM


def _evaluate(
    model: SharedPrivateDVFM,
    loader,
    beta_shared: float,
    beta_private: float,
    free_bits_shared: float,
    free_bits_private: float,
    device: torch.device,
    mc_samples: int,
):
    totals = {
        "loss": 0.0,
        "reconstruction_nll": 0.0,
        "kl_shared": 0.0,
        "kl_event_private": 0.0,
        "kl_censor_private": 0.0,
        "kl_private_total": 0.0,
        "kl_total": 0.0,
    }
    n_batches = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x, time, event = [v.to(device) for v in batch]
            batch_acc = {k: 0.0 for k in totals}
            for _ in range(max(1, int(mc_samples))):
                output = model(x, time, event)
                loss, metrics = model.loss_function(
                    output,
                    time,
                    event,
                    beta_shared=beta_shared,
                    beta_private=beta_private,
                    free_bits_shared=free_bits_shared,
                    free_bits_private=free_bits_private,
                )
                batch_acc["loss"] += float(loss.item())
                for key in metrics:
                    batch_acc[key] += float(metrics[key].item())
            for key in totals:
                totals[key] += batch_acc[key] / max(1, int(mc_samples))
            n_batches += 1

    if n_batches == 0:
        raise RuntimeError("Validation loader is empty.")
    return {k: v / n_batches for k, v in totals.items()}


def train_dvfm2(
    model: SharedPrivateDVFM,
    train_loader,
    val_loader,
    epochs: int = 200,
    lr: float = 1e-3,
    warmup_epochs: int = 50,
    beta_shared_max: float = 1.0,
    private_kl_multiplier: float = 2.0,
    free_bits_shared: float = 0.0,
    free_bits_private: float = 0.0,
    validation_mc_samples: int = 5,
    grad_clip: float = 1.0,
    lr_factor: float = 0.5,
    lr_patience: int = 10,
    device: str | torch.device = "cpu",
    log_every: int = 25,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Train for all epochs and evaluate the final model.

    Private latent KL receives

        beta_private = beta_shared * private_kl_multiplier

    so that event/censor-specific explanations are more expensive than a single
    shared explanation.
    """
    device = torch.device(device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(lr_factor),
        patience=int(lr_patience),
    )

    best_val_recon = float("inf")
    best_epoch = -1
    history: list[dict[str, float]] = []

    for epoch in range(int(epochs)):
        progress = min(1.0, (epoch + 1) / max(1, int(warmup_epochs)))
        beta_s = float(beta_shared_max) * progress
        beta_p = beta_s * float(private_kl_multiplier)

        model.train()
        train_sum = {
            "loss": 0.0,
            "reconstruction_nll": 0.0,
            "kl_shared": 0.0,
            "kl_event_private": 0.0,
            "kl_censor_private": 0.0,
            "kl_private_total": 0.0,
            "kl_total": 0.0,
        }
        n_batches = 0

        for batch in train_loader:
            x, time, event = [v.to(device) for v in batch]
            optimizer.zero_grad()

            output = model(x, time, event)
            loss, metrics = model.loss_function(
                output,
                time,
                event,
                beta_shared=beta_s,
                beta_private=beta_p,
                free_bits_shared=free_bits_shared,
                free_bits_private=free_bits_private,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite DVFM2 loss at epoch {epoch + 1}."
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()

            train_sum["loss"] += float(loss.item())
            for key in metrics:
                train_sum[key] += float(metrics[key].item())
            n_batches += 1

        if n_batches == 0:
            raise RuntimeError("Training loader is empty.")
        train_mean = {k: v / n_batches for k, v in train_sum.items()}

        val = _evaluate(
            model,
            val_loader,
            beta_shared=beta_s,
            beta_private=beta_p,
            free_bits_shared=free_bits_shared,
            free_bits_private=free_bits_private,
            device=device,
            mc_samples=validation_mc_samples,
        )

        # Stationary observed-data reconstruction objective for LR scheduling.
        scheduler.step(val["reconstruction_nll"])

        if val["reconstruction_nll"] < best_val_recon:
            best_val_recon = val["reconstruction_nll"]
            best_epoch = epoch + 1

        history.append(
            {
                "epoch": epoch + 1,
                "beta_shared": beta_s,
                "beta_private": beta_p,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **{f"train_{k}": v for k, v in train_mean.items()},
                **{f"validation_{k}": v for k, v in val.items()},
                "shared_loading_event": float(
                    model.decoder.shared_loading_event.detach().cpu().item()
                ),
                "shared_loading_censor": float(
                    model.decoder.shared_loading_censor.detach().cpu().item()
                ),
                "event_private_loading": float(
                    model.decoder.event_private_loading.detach().cpu().item()
                ),
                "censor_private_loading": float(
                    model.decoder.censor_private_loading.detach().cpu().item()
                ),
            }
        )

        if (
            epoch == 0
            or (epoch + 1) % max(1, int(log_every)) == 0
            or epoch + 1 == int(epochs)
        ):
            print(
                f"DVFM2 epoch={epoch+1:03d} "
                f"recon={val['reconstruction_nll']:.4f} "
                f"KL_s={val['kl_shared']:.4f} "
                f"KL_e={val['kl_event_private']:.4f} "
                f"KL_c={val['kl_censor_private']:.4f} "
                f"load_s=({history[-1]['shared_loading_event']:.3f},"
                f"{history[-1]['shared_loading_censor']:.3f})"
            )

    info = {
        "epochs_completed": int(epochs),
        "evaluated_epoch": int(epochs),
        "best_validation_reconstruction_epoch": int(best_epoch),
        "best_validation_reconstruction_nll": float(best_val_recon),
        "checkpoint_selection_metric": "none_final_epoch_evaluated",
        "private_kl_multiplier": float(private_kl_multiplier),
    }
    return pd.DataFrame(history), info
