"""Load and validate the canonical DVFM experiment specification."""

from __future__ import annotations

from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

import yaml


DEFAULTS: dict[str, Any] = {
    "schema_version": 1,
    "study": {"stage": "exploratory", "output_dir": "results/experiment"},
    "compute": {"device": "auto", "torch_num_threads": 1},
    "split": {"strategy": "holdout", "validation_fraction": 0.15, "test_fraction": 0.30, "folds": 5},
    "preprocessing": {"standardize_x": False, "time_normalization": "none"},
    "models": {
        "enabled": ["dvfm"],
        "dvfm": {"latent_dim": 20, "epochs": 200, "learning_rate": 1e-3, "batch_size": 64, "beta_max": 1.0, "warmup_epochs": 50, "free_bits": 0.0, "mc_samples": 100},
        "deepsurv": {"epochs": 200, "learning_rate": 1e-3, "batch_size": 64},
        "mtlr": {"epochs": 200, "learning_rate": 5e-3, "bins": 200},
        "clayton_aft": {"epochs": 100, "learning_rate": 5e-3},
    },
    "evaluation": {"n_time_points": 200, "max_time_factor": 1.2, "save_predictions": False, "primary_metrics": ["ibs_oracle", "mae_oracle"]},
}


def _deep_update(base: dict, update: dict) -> dict:
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _require(mapping: dict, key: str, location: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required key '{location}.{key}'")
    return mapping[key]


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def expand_scenarios(data_cfg: dict) -> list[dict]:
    """Expand explicit scenarios or a Cartesian ``data.grid`` into atomic scenarios."""
    if "scenarios" in data_cfg:
        return [deepcopy(item) for item in data_cfg["scenarios"]]
    grid = data_cfg.get("grid")
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, values)) for values in product(*[_as_list(grid[k]) for k in keys])]


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    cfg = _deep_update(DEFAULTS, raw)
    cfg["_config_path"] = str(path.resolve())
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    if int(cfg.get("schema_version", 0)) != 1:
        raise ValueError("schema_version must be 1")

    if "workflow" in cfg:
        _require(cfg["workflow"], "target_name", "workflow")
    if "resources" in cfg:
        resources = cfg["resources"]
        for key in ("cores", "memory", "walltime", "partition", "account"):
            _require(resources, key, "resources")
        if int(resources["cores"]) < 1:
            raise ValueError("resources.cores must be at least 1")
    study = _require(cfg, "study", "config")
    _require(study, "name", "study")

    data = _require(cfg, "data", "config")
    source = str(_require(data, "source", "data")).lower()
    if source == "synthetic_copula":
        study_seeds = _require(study, "seeds", "study")
        if not study_seeds or not all(isinstance(seed, int) for seed in study_seeds):
            raise ValueError("study.seeds must be a non-empty list of integers")
        _require(data, "n_samples", "data")
        _require(data, "n_features", "data")
        scenarios = expand_scenarios(data)
        if not scenarios:
            raise ValueError("Synthetic data requires at least one scenario")
        for scenario in scenarios:
            _require(scenario, "copula", "data scenario")
            if "theta" not in scenario:
                raise ValueError("Each current synthetic_copula scenario requires theta; do not substitute Kendall's tau without explicit calibration")
    elif source == "gaussian_shared_frailty":
        _require(data, "n_samples", "data"); _require(data, "n_features", "data")
        for scenario in expand_scenarios(data):
            kendall_tau = float(_require(scenario, "kendall_tau", "data scenario"))
            censoring_rate = float(_require(scenario, "censoring_rate", "data scenario"))
            if not 0.0 <= kendall_tau < 1.0:
                raise ValueError("data scenario kendall_tau must be in [0, 1)")
            if not 0.0 < censoring_rate < 1.0:
                raise ValueError("data scenario censoring_rate must be between 0 and 1")
        seeds = _require(cfg, "seeds", "config")
        for key in ("dgp", "sampling", "split", "model"):
            _require(seeds, key, "seeds")
        if not isinstance(seeds["dgp"], int):
            raise ValueError("seeds.dgp must be one integer")
        for key in ("sampling", "split", "model"):
            if not seeds[key] or not all(isinstance(seed, int) for seed in seeds[key]):
                raise ValueError(f"seeds.{key} must be a non-empty list of integers")
        if len({len(seeds[key]) for key in ("sampling", "split", "model")}) != 1:
            raise ValueError("seeds.sampling, seeds.split, and seeds.model must have equal length")
        _require(cfg["split"], "validation_fraction", "split")
        latent_dims = _require(cfg["models"]["dvfm"], "latent_dims", "models.dvfm")
        if not latent_dims or any(int(value) < 0 for value in latent_dims):
            raise ValueError("models.dvfm.latent_dims must contain nonnegative integers")
    elif source in {"real_file", "semi_synthetic_file"}:
        study_seeds = _require(study, "seeds", "study")
        if not study_seeds or not all(isinstance(seed, int) for seed in study_seeds):
            raise ValueError("study.seeds must be a non-empty list of integers")
        _require(data, "path", "data")
        if source == "real_file":
            _require(data, "time_column", "data")
            _require(data, "event_column", "data")
        else:
            _require(data, "true_event_time_column", "data")
            _require(data, "true_censor_time_column", "data")
    else:
        raise ValueError(f"Unsupported data.source: {source}")

    split = cfg["split"]
    validation_fraction = float(split["validation_fraction"])
    if not 0 < validation_fraction < 1:
        raise ValueError("split.validation_fraction must be between 0 and 1")
    if split["strategy"] == "holdout":
        if not 0 < float(split["test_fraction"]) < 1:
            raise ValueError("split.test_fraction must be between 0 and 1")
        if validation_fraction + float(split["test_fraction"]) >= 1:
            raise ValueError("validation and test fractions must sum to less than 1")
    elif split["strategy"] != "kfold":
        raise ValueError("split.strategy must be holdout or kfold")

    supported = {"coxph", "deepsurv", "mtlr", "clayton_aft", "dvfm"}
    unknown = set(cfg["models"]["enabled"]) - supported
    if unknown:
        raise ValueError(f"Unsupported models: {sorted(unknown)}")
