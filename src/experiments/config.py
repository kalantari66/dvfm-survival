"""Load and validate the canonical experiment specification."""

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
    "preprocessing": {"zscore_x": False},
    "models": {
        "enabled": ["dvfm"],
        "dvfm": {"latent_dim": 20, "epochs": 200, "learning_rate": 1e-3, "batch_size": 64, "beta_max": 1.0, "warmup_epochs": 50, "free_bits": 0.0, "mc_samples": 100},
        "deepsurv": {"epochs": 200, "learning_rate": 1e-3, "batch_size": 64},
        "mtlr": {"epochs": 200, "learning_rate": 5e-3, "bins": 200},
        "clayton_aft": {"epochs": 100, "learning_rate": 5e-3},
        "deephit": {
            "epochs": 200, "batch_size": 256, "learning_rate": 1e-3,
            "time_bins": 100, "num_nodes_shared": [64, 32],
            "batch_norm": True, "dropout": 0.1, "alpha": 0.2,
            "sigma": 0.1, "early_stop": True, "patience": 10,
            "verbose": False,
        },
        "gbsa": {
            "n_estimators": 100, "learning_rate": 0.1, "max_depth": 3,
            "loss": "coxph", "min_samples_split": 2, "min_samples_leaf": 1,
            "max_features": "sqrt", "subsample": 0.8, "random_state": 0,
        },
        "rsf": {
            "n_estimators": 100, "max_depth": 3, "min_samples_split": 2,
            "min_samples_leaf": 1, "max_features": "sqrt",
            "random_state": 0, "n_jobs": 1,
        },
        "weibull_aft": {"penalizer": 0.0, "l1_ratio": 0.0},
        "bayesian_cox_gamma_frailty": {
            "n_intervals": 10, "epochs": 500, "minimum_epochs": 100,
            "early_stopping_patience": 50, "learning_rate": 0.03,
            "minimum_delta": 1e-6, "gradient_clip": 10.0,
            "beta_prior_sd": 2.5, "log_hazard_prior_sd": 5.0,
            "log_alpha_prior_sd": 2.0, "baseline_smoothness": 1.0,
            "dtype": "float64",
        },
        "hacsurv_2d": {
            "epochs": 1000, "batch_size": 512, "learning_rate": 1e-4,
            "copula_learning_rate": 1e-4, "copula_start_epoch": 200,
            "minimum_epochs": 400, "checkpoint_min_epoch": 201,
            "early_stopping_patience": 200,
            "generator_samples": 200, "validation_generator_samples": 500,
            "hidden_size": 32, "hidden_survival": 32,
            "inverse_iterations": 200, "inverse_tolerance": 1e-8,
            "scale_regularization": 1.0, "numerical_failure_threshold": 100.0,
            "dtype": "float64",
        },
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
    if "standardize_x" in cfg["preprocessing"]:
        raise ValueError("preprocessing.standardize_x was renamed to preprocessing.zscore_x")

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
    elif source in {"gaussian_shared_frailty", "frailty_recovery_diagnostic"}:
        _require(data, "n_features", "data")
        scenarios = expand_scenarios(data) if source == "gaussian_shared_frailty" else [data]
        for scenario in scenarios:
            n_samples = scenario.get("n_samples", data.get("n_samples"))
            if n_samples is None or int(n_samples) < 10:
                raise ValueError("Each Gaussian frailty scenario requires n_samples >= 10")
            kendall_tau = float(_require(scenario, "kendall_tau", "data scenario"))
            censoring_rate = float(_require(scenario, "censoring_rate", "data scenario"))
            if not 0.0 <= kendall_tau < 1.0:
                raise ValueError("data scenario kendall_tau must be in [0, 1)")
            if not 0.0 < censoring_rate < 1.0:
                raise ValueError("data scenario censoring_rate must be between 0 and 1")
            mechanism = str(
                scenario.get("mechanism", "gaussian_shared_frailty")
            ).lower()
            if mechanism not in {
                "gaussian_shared_frailty", "clayton_gamma_frailty"
            }:
                raise ValueError(f"Unsupported data scenario mechanism: {mechanism}")
            if mechanism == "clayton_gamma_frailty" and kendall_tau <= 0.0:
                raise ValueError(
                    "Clayton Gamma-frailty scenarios require kendall_tau > 0"
                )
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
        if source == "gaussian_shared_frailty":
            dvfm = cfg["models"]["dvfm"]
            variants = []
            if "dvfm" in {str(name).lower() for name in cfg["models"]["enabled"]}:
                latent_dims = _require(dvfm, "latent_dims", "models.dvfm")
                if not latent_dims or any(int(value) < 0 for value in latent_dims):
                    raise ValueError("models.dvfm.latent_dims must contain nonnegative integers")
                checkpoint_min_epoch = int(_require(
                    dvfm, "checkpoint_min_epoch", "models.dvfm"
                ))
                if checkpoint_min_epoch < int(dvfm["warmup_epochs"]):
                    raise ValueError(
                        "models.dvfm.checkpoint_min_epoch must be at or after warmup_epochs"
                    )
                if checkpoint_min_epoch > int(dvfm["epochs"]):
                    raise ValueError(
                        "models.dvfm.checkpoint_min_epoch cannot exceed epochs"
                    )
                if _require(dvfm, "primary_checkpoint", "models.dvfm") not in {
                    "final", "best_validation_elbo_post_warmup"
                }:
                    raise ValueError("Invalid models.dvfm.primary_checkpoint")
                if float(_require(
                    dvfm, "numerical_failure_threshold", "models.dvfm"
                )) <= 0:
                    raise ValueError(
                        "models.dvfm.numerical_failure_threshold must be positive"
                    )
                variants = dvfm.get("variants", [])
            if variants:
                names = [item.get("name") for item in variants]
                if any(not name for name in names) or len(set(names)) != len(names):
                    raise ValueError("models.dvfm.variants must have unique names")
                for variant in variants:
                    resolved = _deep_update(dvfm, variant)
                    epochs = int(resolved["epochs"])
                    minimum = int(resolved["checkpoint_min_epoch"])
                    warmup = int(resolved["warmup_epochs"])
                    if epochs < 1 or not warmup <= minimum <= epochs:
                        raise ValueError(
                            f"Invalid epoch/checkpoint settings for DVFM variant {variant['name']}"
                        )
                    dropout = float(resolved.get("dropout", 0.0))
                    if not 0.0 <= dropout < 1.0:
                        raise ValueError("DVFM variant dropout must be in [0, 1)")
                    if float(resolved.get("weight_decay", 0.0)) < 0.0:
                        raise ValueError("DVFM variant weight_decay cannot be negative")
                    if resolved.get("scale_link", "softplus") not in {"softplus", "exp"}:
                        raise ValueError("DVFM variant scale_link must be softplus or exp")
                    if resolved.get("latent_path", "nonlinear") not in {
                        "nonlinear", "additive_scale"
                    }:
                        raise ValueError(
                            "DVFM variant latent_path must be nonlinear or additive_scale"
                        )
                    if resolved.get("shape_mode", "conditional") not in {
                        "conditional", "global"
                    }:
                        raise ValueError(
                            "DVFM variant shape_mode must be conditional or global"
                        )
                    if float(resolved.get("latent_loading_l1", 0.0)) < 0.0:
                        raise ValueError("DVFM variant latent_loading_l1 cannot be negative")
                    if float(resolved.get("latent_group_lasso", 0.0)) < 0.0:
                        raise ValueError("DVFM variant latent_group_lasso cannot be negative")
                    if float(resolved.get("gate_l1", 0.0)) < 0.0:
                        raise ValueError("DVFM variant gate_l1 cannot be negative")
                    latent_gate = resolved.get("latent_gate", "none")
                    if latent_gate not in {"none", "hard_concrete", "sigmoid"}:
                        raise ValueError(
                            "DVFM variant latent_gate must be none, hard_concrete, or sigmoid"
                        )
                    if float(resolved.get("gate_l1", 0.0)) > 0.0 and latent_gate == "none":
                        raise ValueError("DVFM gate_l1 requires an enabled latent_gate")
                    if not 0.0 < float(resolved.get("gate_initial_value", 0.9)) < 1.0:
                        raise ValueError("DVFM variant gate_initial_value must be in (0, 1)")
                    if float(resolved.get("gate_temperature", 0.67)) <= 0.0:
                        raise ValueError("DVFM variant gate_temperature must be positive")
                    for key in ("encoder_hidden", "decoder_hidden"):
                        widths = resolved.get(key, [])
                        if not widths or any(int(width) < 1 for width in widths):
                            raise ValueError(
                                f"DVFM variant {key} must contain positive widths"
                            )
                reference = str(
                    cfg.get("hyperparameter_sweep", {}).get(
                        "reference_variant", "reference"
                    )
                )
                if reference not in names:
                    raise ValueError(
                        "hyperparameter_sweep.reference_variant must name a DVFM variant"
                    )
                sweep_cfg = cfg.get("hyperparameter_sweep", {})
                if sweep_cfg.get("selection_partition", "validation") != "validation":
                    raise ValueError("Hyperparameters must be selected on validation")
                if sweep_cfg.get("selection_prediction_mode", "prior") not in cfg["evaluation"].get("prediction_modes", []):
                    raise ValueError(
                        "hyperparameter_sweep.selection_prediction_mode must be evaluated"
                    )
                if float(sweep_cfg.get("frailty_spearman_tolerance", 0.03)) < 0:
                    raise ValueError("frailty_spearman_tolerance cannot be negative")
                partitions = cfg["evaluation"].get("evaluate_partitions", [])
                if "validation" not in partitions or "test" not in partitions:
                    raise ValueError(
                        "DVFM sweeps must evaluate both validation and test partitions"
                    )
            if "hacsurv_2d" in {
                str(name).lower() for name in cfg["models"]["enabled"]
            }:
                hac = cfg["models"]["hacsurv_2d"]
                required = {
                    "epochs", "batch_size", "learning_rate",
                    "copula_learning_rate", "copula_start_epoch",
                    "minimum_epochs", "checkpoint_min_epoch",
                    "early_stopping_patience",
                    "generator_samples", "validation_generator_samples",
                    "hidden_size", "hidden_survival", "inverse_iterations",
                    "inverse_tolerance", "scale_regularization",
                    "numerical_failure_threshold", "dtype",
                }
                missing = required - set(hac)
                if missing:
                    raise ValueError(
                        f"models.hacsurv_2d is missing keys: {sorted(missing)}"
                    )
                if int(hac["epochs"]) < 1 or int(hac["batch_size"]) < 2:
                    raise ValueError("HACSurv epochs and batch_size must be positive")
                if not 0 <= int(hac["copula_start_epoch"]) < int(hac["epochs"]):
                    raise ValueError("HACSurv copula_start_epoch must precede epochs")
                if not 0 <= int(hac["minimum_epochs"]) <= int(hac["epochs"]):
                    raise ValueError("HACSurv minimum_epochs must be within training")
                if not int(hac["copula_start_epoch"]) < int(hac["checkpoint_min_epoch"]) <= int(hac["epochs"]):
                    raise ValueError(
                        "HACSurv checkpoint_min_epoch must follow copula_start_epoch"
                    )
                if str(hac["dtype"]) not in {"float32", "float64"}:
                    raise ValueError("HACSurv dtype must be float32 or float64")
            if "bayesian_cox_gamma_frailty" in {
                str(name).lower() for name in cfg["models"]["enabled"]
            }:
                frailty = cfg["models"]["bayesian_cox_gamma_frailty"]
                for key in (
                    "n_intervals", "epochs", "minimum_epochs",
                    "early_stopping_patience", "learning_rate",
                ):
                    _require(frailty, key, "models.bayesian_cox_gamma_frailty")
                if int(frailty["n_intervals"]) < 1:
                    raise ValueError("Cox--Gamma frailty n_intervals must be positive")
                if not 1 <= int(frailty["minimum_epochs"]) <= int(frailty["epochs"]):
                    raise ValueError(
                        "Cox--Gamma frailty minimum_epochs must be within training"
                    )
                if int(frailty["early_stopping_patience"]) < 1:
                    raise ValueError(
                        "Cox--Gamma frailty early_stopping_patience must be positive"
                    )
                if float(frailty["learning_rate"]) <= 0:
                    raise ValueError("Cox--Gamma frailty learning_rate must be positive")
                if str(frailty.get("dtype", "float64")) not in {"float32", "float64"}:
                    raise ValueError("Cox--Gamma frailty dtype must be float32 or float64")
        else:
            mechanisms = _require(data, "mechanisms", "data")
            allowed_mechanisms = {"gaussian_shared_frailty", "clayton_gamma_frailty"}
            if not mechanisms or set(mechanisms) - allowed_mechanisms:
                raise ValueError(f"data.mechanisms must contain only {sorted(allowed_mechanisms)}")
            if int(_require(cfg["models"]["dvfm"], "latent_dim", "models.dvfm")) != 1:
                raise ValueError("Frailty recovery diagnostic requires models.dvfm.latent_dim = 1")
            variants = _require(cfg, "training_variants", "config")
            required_variant_keys = {
                "name", "maximum_epochs", "learning_rate", "beta_max", "warmup_epochs",
                "lr_patience", "minimum_learning_rate", "scheduler_metric", "checkpoint_selection",
            }
            if not variants or len({item.get("name") for item in variants}) != len(variants):
                raise ValueError("training_variants must have unique names")
            for variant in variants:
                missing = required_variant_keys - set(variant)
                if missing:
                    raise ValueError(f"Training variant is missing keys: {sorted(missing)}")
                if variant["scheduler_metric"] not in {"validation_elbo", "validation_reconstruction_nll"}:
                    raise ValueError("Invalid training variant scheduler_metric")
                if variant["checkpoint_selection"] not in {"final", "best_validation_reconstruction_nll"}:
                    raise ValueError("Invalid training variant checkpoint_selection")
            _require(cfg["models"], "oracle_z_decoder", "models")
            for key in ("dependence_samples_per_epoch", "dependence_samples_checkpoint"):
                if int(_require(cfg["evaluation"], key, "evaluation")) < 2:
                    raise ValueError(f"evaluation.{key} must be at least 2")
    elif source == "support_cox_clayton_semisynthetic":
        _require(data, "path", "data")
        if str(_require(data, "copula", "data")).lower() != "clayton":
            raise ValueError("SUPPORT semi-synthetic generation currently requires copula: clayton")
        kendall_tau = float(_require(data, "kendall_tau", "data"))
        if not 0.0 < kendall_tau < 1.0:
            raise ValueError("data.kendall_tau must be in (0, 1) for Clayton")
        rates = _require(data, "censoring_rates", "data")
        if not rates or any(not 0.0 < float(rate) < 1.0 for rate in rates):
            raise ValueError("data.censoring_rates must contain values in (0, 1)")
        seeds = _require(cfg, "seeds", "config")
        for key in ("sampling", "split", "model"):
            values = _require(seeds, key, "seeds")
            if not values or not all(isinstance(seed, int) for seed in values):
                raise ValueError(f"seeds.{key} must be a non-empty list of integers")
        if len({len(seeds[key]) for key in ("sampling", "split", "model")}) != 1:
            raise ValueError("seeds.sampling, seeds.split, and seeds.model must have equal length")
        if str(cfg["split"].get("stratify", "")).lower() != "time_event":
            raise ValueError("SUPPORT semi-synthetic splits require split.stratify: time_event")
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

    supported = {
        "coxph", "deepsurv", "mtlr", "clayton_aft", "hacsurv_2d", "dvfm",
        "deephit", "gbsa", "rsf", "weibull_aft",
        "bayesian_cox_gamma_frailty",
    }
    unknown = set(cfg["models"]["enabled"]) - supported
    if unknown:
        raise ValueError(f"Unsupported models: {sorted(unknown)}")
    if "dvfm" in {str(name).lower() for name in cfg["models"]["enabled"]}:
        dvfm = cfg["models"]["dvfm"]
        scale_link = str(dvfm.get("scale_link", "softplus"))
        if scale_link not in {"softplus", "exp"}:
            raise ValueError("models.dvfm.scale_link must be softplus or exp")
        if float(dvfm.get("latent_loading_l1", 0.0)) < 0.0:
            raise ValueError("models.dvfm.latent_loading_l1 cannot be negative")
        if float(dvfm.get("latent_group_lasso", 0.0)) < 0.0:
            raise ValueError("models.dvfm.latent_group_lasso cannot be negative")
        if float(dvfm.get("gate_l1", 0.0)) < 0.0:
            raise ValueError("models.dvfm.gate_l1 cannot be negative")
        if str(dvfm.get("latent_gate", "none")) not in {
            "none", "hard_concrete", "sigmoid"
        }:
            raise ValueError(
                "models.dvfm.latent_gate must be none, hard_concrete, or sigmoid"
            )
    if cfg["evaluation"].get("compute_oracle_joint_survival_ise", False):
        for key in (
            "joint_n_time_points", "joint_n_subjects", "joint_dgp_samples",
            "joint_model_samples", "joint_model_batch_size",
        ):
            if int(_require(cfg["evaluation"], key, "evaluation")) < 2:
                raise ValueError(f"evaluation.{key} must be at least 2")
        quantile = float(_require(
            cfg["evaluation"], "joint_grid_max_quantile", "evaluation"
        ))
        if not 0.0 < quantile <= 1.0:
            raise ValueError("evaluation.joint_grid_max_quantile must be in (0, 1]")
