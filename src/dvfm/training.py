"""Training loop and checkpoint selection for DVFM."""

from __future__ import annotations

import numpy as np
import torch


def train_dvfm(
    model,
    train_loader,
    val_loader,
    n_epochs=200,
    lr=1e-3,
    beta_max=1.0,
    warmup_epochs=50,
    free_bits=0.0,
    device="cpu",
    return_history=False,
    checkpoint_min_epoch=None,
    return_artifacts=False,
    numerical_failure_threshold=None,
    weight_decay=0.0,
):
    """Train DVFM for fixed epochs and retain the best eligible validation ELBO."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=float(weight_decay))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    train_losses, val_losses, history = [], [], []
    best_validation_elbo = float("inf")
    best_validation_elbo_epoch = None
    best_validation_elbo_state = None
    model.to(device)

    for epoch in range(n_epochs):
        beta = min(beta_max, (epoch + 1) / warmup_epochs) if warmup_epochs > 0 else beta_max
        model.train()
        train_loss = train_recon = train_kl = 0.0
        for x, time, event in train_loader:
            x, time, event = x.to(device), time.to(device), event.to(device)
            optimizer.zero_grad()
            outputs = model(x, time, event)
            loss, recon, kl = model.loss_function(*outputs, time, event, beta, free_bits)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite DVFM training loss at epoch {epoch + 1}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"Non-finite DVFM gradient norm at epoch {epoch + 1}")
            optimizer.step()
            train_loss += loss.item()
            train_recon += recon.item()
            train_kl += kl.item()

        train_loss /= len(train_loader)
        train_recon /= len(train_loader)
        train_kl /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = val_recon = val_kl = 0.0
        with torch.no_grad():
            for x, time, event in val_loader:
                x, time, event = x.to(device), time.to(device), event.to(device)
                outputs = model(x, time, event)
                loss, recon, kl = model.loss_function(*outputs, time, event, beta, free_bits)
                val_loss += loss.item()
                val_recon += recon.item()
                val_kl += kl.item()
        val_loss /= len(val_loader)
        val_recon /= len(val_loader)
        val_kl /= len(val_loader)
        val_losses.append(val_loss)

        monitored = np.asarray(
            [train_loss, train_recon, train_kl, val_loss, val_recon, val_kl], dtype=float
        )
        numerical_valid = bool(np.all(np.isfinite(monitored)))
        if numerical_valid and numerical_failure_threshold is not None:
            numerical_valid = bool(
                np.max(np.abs(monitored)) <= float(numerical_failure_threshold)
            )
        if numerical_valid:
            scheduler.step(val_loss)

        history.append(
            {
                "epoch": epoch + 1,
                "beta": beta,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "train_reconstruction": train_recon,
                "train_kl": train_kl,
                "validation_loss": val_loss,
                "validation_reconstruction": val_recon,
                "validation_kl": val_kl,
                "numerical_valid": numerical_valid,
            }
        )
        eligible = checkpoint_min_epoch is None or epoch + 1 >= int(checkpoint_min_epoch)
        if eligible and numerical_valid and val_loss < best_validation_elbo:
            best_validation_elbo = val_loss
            best_validation_elbo_epoch = epoch + 1
            best_validation_elbo_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }

        if (epoch + 1) % 100 == 0:
            print(
                f"Epoch {epoch + 1}/{n_epochs}, Beta: {beta:.3f}, "
                f"Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, "
                f"KL: {train_kl:.4f}), Val Loss: {val_loss:.4f}"
            )

    if return_artifacts:
        if best_validation_elbo_state is None:
            raise RuntimeError("No finite validation ELBO was available for checkpointing")
        return {
            "history": history,
            "final_state": {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            },
            "best_validation_elbo_state": best_validation_elbo_state,
            "best_validation_elbo_epoch": best_validation_elbo_epoch,
            "best_validation_elbo": best_validation_elbo,
            "numerically_invalid_epochs": sum(not row["numerical_valid"] for row in history),
            "final_checkpoint_valid": bool(history[-1]["numerical_valid"]),
        }
    if return_history:
        return history
    return train_losses, val_losses

__all__ = ["train_dvfm"]
