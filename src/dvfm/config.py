"""Load and validate the canonical DVFM experiment specification."""

from __future__ import annotations

from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

import yaml


DEFAULTS: dict[str, Any] = {
    "schema_version": 1,
    "study": {"stage": "exploratory", "output_dir": "results/experiment", "seeds": [42]},
    "compute": {"device": "auto", "torch_num_threads": 1},
    "split": {"strategy": "holdout", "test_fraction": 0.30, "folds": 5},
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
    seeds = _require(study, "seeds", "study")
    if not seeds or not all(isinstance(seed, int) for seed in seeds):
        raise ValueError("study.seeds must be a non-empty list of integers")

    data = _require(cfg, "data", "config")
    source = str(_require(data, "source", "data")).lower()
    if source == "synthetic_copula":
        _require(data, "n_samples", "data")
        _require(data, "n_features", "data")
        scenarios = expand_scenarios(data)
        if not scenarios:
            raise ValueError("Synthetic data requires at least one scenario")
        for scenario in scenarios:
            _require(scenario, "copula", "data scenario")
            if "theta" not in scenario:
                raise ValueError("Each current synthetic_copula scenario requires theta; do not substitute Kendall's tau without explicit calibration")
    elif source in {"real_file", "semi_synthetic_file"}:
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
    if split["strategy"] == "holdout":
        if not 0 < float(split["test_fraction"]) < 1:
            raise ValueError("split.test_fraction must be between 0 and 1")
    elif split["strategy"] != "kfold":
        raise ValueError("split.strategy must be holdout or kfold")

    supported = {"coxph", "deepsurv", "mtlr", "clayton_aft", "dvfm"}
    unknown = set(cfg["models"]["enabled"]) - supported
    if unknown:
        raise ValueError(f"Unsupported models: {sorted(unknown)}")
