"""Configuration loading with reference-code defaults."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

# These defaults reproduce the active run_experiment() settings in
# reference/VAE_montcarlo.py. They are intentionally not shortened.
REFERENCE_DEFAULTS: dict[str, Any] = {
    "seed": 42,
    "output_dir": "outputs/reference_run",
    "device": "auto",
    "torch_num_threads": 1,
    "repeats": 1,
    "split": {"strategy": "holdout", "test_size": 0.30, "folds": 5},
    "preprocessing": {"standardize": False, "time_normalize": "none"},
    "evaluation": {"n_time_points": 1000, "max_time_factor": 1.5, "save_predictions": False},
    "models": {
        "enabled": ["coxph", "deepsurv", "mtlr", "clayton_aft", "dvfm"],
        "dvfm": {
            "latent_dim": 20,
            "epochs": 200,
            "lr": 1e-3,
            "batch_size": 64,
            "beta_max": 1.0,
            "warmup_epochs": 50,
            "free_bits": 0.0,
            "mc_samples": 100,
        },
        "deepsurv": {"epochs": 200, "lr": 1e-3, "batch_size": 64},
        "mtlr": {"epochs": 200, "lr": 5e-3, "bins": 200},
        # This baseline exists in the modified reference experiment file.
        "clayton_aft": {"epochs": 100, "lr": 5e-3},
    },
}


def _deep_update(base: dict, update: dict) -> dict:
    out = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    cfg = _deep_update(REFERENCE_DEFAULTS, user_cfg)
    cfg["_config_path"] = str(path.resolve())
    validate_reference_parameters(cfg)
    return cfg


def validate_reference_parameters(cfg: dict) -> None:
    """Reject accidental shortened DVFM settings in benchmark configurations."""
    dvfm = cfg["models"]["dvfm"]
    if int(dvfm["epochs"]) != 200:
        raise ValueError(
            f"DVFM epochs must be 200 for the reference benchmark; received {dvfm['epochs']}. "
            "Create a separately named exploratory config if you intentionally change it."
        )
    required = {
        "latent_dim": 20,
        "batch_size": 64,
        "mc_samples": 100,
        "warmup_epochs": 50,
    }
    for key, expected in required.items():
        if int(dvfm[key]) != expected:
            raise ValueError(f"Reference DVFM parameter {key} must be {expected}; received {dvfm[key]}.")
    if abs(float(dvfm["lr"]) - 1e-3) > 1e-12:
        raise ValueError(f"Reference DVFM learning rate must be 0.001; received {dvfm['lr']}.")
    if abs(float(dvfm["beta_max"]) - 1.0) > 1e-12:
        raise ValueError(f"Reference beta_max must be 1.0; received {dvfm['beta_max']}.")
    if abs(float(dvfm["free_bits"])) > 1e-12:
        raise ValueError(f"Reference free_bits must be 0.0; received {dvfm['free_bits']}.")
