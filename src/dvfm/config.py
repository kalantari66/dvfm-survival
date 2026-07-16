"""YAML configuration helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULTS: dict[str, Any] = {
    "seed": 42,
    "output_dir": "outputs/run",
    "device": "auto",
        "torch_num_threads": 1,
    "repeats": 1,
    "split": {"strategy": "holdout", "test_size": 0.2, "folds": 5},
    "preprocessing": {"standardize": True, "time_normalize": "none"},
    "evaluation": {"n_time_points": 600, "max_time_factor": 1.5, "save_predictions": False},
    "models": {
        "enabled": ["coxph", "deepsurv", "mtlr", "clayton_aft", "dvfm"],
        "dvfm": {"latent_dim": 20, "epochs": 300, "lr": 0.005, "batch_size": 64, "beta_max": 1.0, "warmup_epochs": 50, "free_bits": 0.0, "mc_samples": 100},
        "deepsurv": {"epochs": 120, "lr": 0.001, "batch_size": 64},
        "mtlr": {"epochs": 120, "lr": 0.005, "bins": 60},
        "clayton_aft": {"epochs": 100, "lr": 0.005},
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
    cfg = _deep_update(DEFAULTS, user_cfg)
    cfg["_config_path"] = str(path.resolve())
    return cfg


def apply_quick_mode(cfg: dict) -> dict:
    cfg = deepcopy(cfg)
    cfg["repeats"] = 1
    cfg["split"] = {"strategy": "holdout", "test_size": 0.2, "folds": 5}
    cfg["evaluation"]["n_time_points"] = min(150, cfg["evaluation"]["n_time_points"])
    cfg["models"]["dvfm"].update({"epochs": 5, "mc_samples": 5, "latent_dim": 5})
    cfg["models"]["deepsurv"]["epochs"] = 5
    cfg["models"]["mtlr"].update({"epochs": 5, "bins": min(20, int(cfg["models"]["mtlr"]["bins"]))})
    cfg["models"]["clayton_aft"]["epochs"] = 5
    if "synthetic" in cfg:
        cfg["synthetic"]["n_samples"] = min(500, int(cfg["synthetic"].get("n_samples", 500)))
        scenarios = cfg["synthetic"].get("scenarios", [])
        if scenarios:
            cfg["synthetic"]["scenarios"] = scenarios[:1]
    return cfg
