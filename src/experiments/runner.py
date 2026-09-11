"""Dataset-independent runner using the supplied algorithms and reference parameters."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sota.adapters import fit_deephit, fit_sksurv_ensemble, fit_weibull_aft
from sota.bayesian_cox_gamma_frailty import fit_bayesian_cox_gamma_frailty

from sota.baselines import (
    _fit_cox_and_predict_survival,
    fit_clayton_weibull_aft,
    train_deepsurv,
    train_mtlr,
)
from .config import expand_scenarios, expand_seed_streams
from utility.data import SurvivalData, load_real_data, load_semi_synthetic_data
from utility.metrics import censoring_rate, collect_metrics
from dvfm.model import DVFM
from utility.data import SurvivalDataset
from dvfm.prediction import get_median_survival_time, predict_survival_curves
from utility.synthetic import generate_copula_data
from utility.runtime import resolve_device, seed_everything
from utility.splitting import (
    iter_split_indices, preprocess_covariates, subset_survival_data,
    time_event_stratified_split_indices,
)
from dvfm.training import train_dvfm


def _splits(data: SurvivalData, split_cfg: dict, seed: int):
    yield from iter_split_indices(data, split_cfg, seed)


def _preprocess_three(train, validation, test, cfg):
    return preprocess_covariates(train, validation, test, cfg)


def _subset(data: SurvivalData, idx: np.ndarray) -> SurvivalData:
    return subset_survival_data(data, idx)


def _fit_one_split(
    train: SurvivalData,
    validation: SurvivalData,
    test: SurvivalData,
    cfg: dict,
    device: torch.device,
    context: dict,
) -> tuple[list[dict], dict]:
    eval_cfg = cfg["evaluation"]
    model_cfg = cfg["models"]
    enabled = {str(x).lower() for x in model_cfg["enabled"]}

    max_time = max(float(np.max(train.time) * float(eval_cfg["max_time_factor"])), 1e-8)
    time_points = np.linspace(0.0, max_time, int(eval_cfg["n_time_points"]))
    predictions: dict[str, dict] = {}

    if "coxph" in enabled:
        try:
            median, survival = _fit_cox_and_predict_survival(
                train.X, train.time, train.event, test.X, time_points
            )
        except Exception as exc:
            print(f"CoxPH failed on this split: {exc}")
            median = np.full(len(test.time), float(np.max(train.time)), dtype=float)
            survival = np.ones((len(test.time), len(time_points)), dtype=float)
        predictions["CoxPH"] = {"median": median, "survival": survival}

    if "deepsurv" in enabled:
        c = model_cfg["deepsurv"]
        _, median, survival = train_deepsurv(
            train.X,
            train.time,
            train.event,
            test.X,
            n_epochs=int(c["epochs"]),
            batch_size=int(c["batch_size"]),
            lr=float(c["learning_rate"]),
            device=device,
            eval_time_points=time_points,
        )
        predictions["DeepSurv"] = {"median": median, "survival": survival}

    if "mtlr" in enabled:
        c = model_cfg["mtlr"]
        _, median, survival = train_mtlr(
            train.X,
            train.time,
            train.event,
            test.X,
            num_bins=int(c["bins"]),
            n_epochs=int(c["epochs"]),
            lr=float(c["learning_rate"]),
            device=device,
            eval_time_points=time_points,
        )
        predictions["MTLR"] = {"median": median, "survival": survival}

    if "clayton_aft" in enabled:
        c = model_cfg["clayton_aft"]
        median, survival = fit_clayton_weibull_aft(
            train.X,
            train.time,
            train.event,
            test.X,
            time_points,
            epochs=int(c["epochs"]),
            lr=float(c["learning_rate"]),
            device=device,
        )
        predictions["ClaytonAFT"] = {"median": median, "survival": survival}

    if "deephit" in enabled:
        median, survival = fit_deephit(
            train.X, train.time, train.event,
            validation.X, validation.time, validation.event,
            test.X, time_points, model_cfg["deephit"], device,
        )
        predictions["DeepHit"] = {"median": median, "survival": survival}

    for ensemble_name, display_name in (("gbsa", "GBSA"), ("rsf", "RSF")):
        if ensemble_name in enabled:
            median, survival = fit_sksurv_ensemble(
                ensemble_name, train.X, train.time, train.event,
                test.X, time_points, model_cfg[ensemble_name],
            )
            predictions[display_name] = {"median": median, "survival": survival}

    if "weibull_aft" in enabled:
        median, survival = fit_weibull_aft(
            train.X, train.time, train.event, test.X, time_points,
            model_cfg["weibull_aft"],
        )
        predictions["WeibullAFT"] = {"median": median, "survival": survival}

    if "bayesian_cox_gamma_frailty" in enabled:
        survival, _, _, _ = fit_bayesian_cox_gamma_frailty(
            train.X, train.time, train.event,
            validation.X, validation.time, validation.event,
            test.X, time_points,
            model_cfg["bayesian_cox_gamma_frailty"], device,
        )
        median = get_median_survival_time(survival, time_points)
        predictions["BayesianCoxGammaFrailty"] = {
            "median": median, "survival": survival,
        }

    if "dvfm" in enabled:
        c = model_cfg["dvfm"]
        train_loader = DataLoader(
            SurvivalDataset(train.X, train.time, train.event),
            batch_size=int(c["batch_size"]),
            shuffle=True,
        )
        val_loader = DataLoader(
            SurvivalDataset(validation.X, validation.time, validation.event),
            batch_size=int(c["batch_size"]),
            shuffle=False,
        )
        model = DVFM(
            input_dim=train.X.shape[1],
            latent_dim=int(c["latent_dim"]),
            encoder_hidden=list(c.get("encoder_hidden", [64, 32])),
            decoder_hidden=list(c.get("decoder_hidden", [32, 64])),
            dropout=float(c.get("dropout", 0.0)),
            scale_link=str(c.get("scale_link", "softplus")),
            latent_path=str(c.get("latent_path", "nonlinear")),
            shape_mode=str(c.get("shape_mode", "conditional")),
            latent_gate=str(c.get("latent_gate", "none")),
            gate_initial_value=float(c.get("gate_initial_value", 0.9)),
            gate_temperature=float(c.get("gate_temperature", 0.67)),
            latent_loading_l1=float(c.get("latent_loading_l1", 0.1)),
        ).to(device)
        artifacts = train_dvfm(
            model,
            train_loader,
            val_loader,
            n_epochs=int(c["epochs"]),
            lr=float(c["learning_rate"]),
            beta_max=float(c["beta_max"]),
            warmup_epochs=int(c["warmup_epochs"]),
            free_bits=float(c["free_bits"]),
            device=device,
            checkpoint_min_epoch=int(c.get("checkpoint_min_epoch", c["warmup_epochs"])),
            numerical_failure_threshold=float(c.get("numerical_failure_threshold", 100.0)),
            weight_decay=float(c.get("weight_decay", 0.0)),
            latent_group_lasso=float(c.get("latent_group_lasso", 0.0)),
            gate_l1=float(c.get("gate_l1", 0.0)),
            return_artifacts=True,
        )
        checkpoint = str(c.get("primary_checkpoint", "best_validation_elbo_post_warmup"))
        if checkpoint == "best_validation_elbo_post_warmup":
            model.load_state_dict(artifacts["best_validation_elbo_state"])
        elif checkpoint != "final":
            raise ValueError(f"Unsupported DVFM primary checkpoint: {checkpoint}")
        survival = predict_survival_curves(
            model=model,
            X=test.X,
            time_points=time_points,
            train_loader=train_loader,
            n_samples=int(c["mc_samples"]),
            device=device,
        )
        median = get_median_survival_time(survival, time_points)
        predictions["DVFM"] = {
            "median": median,
            "survival": survival,
            "learned_gate": float(
                model.decoder.gate_value(stochastic=False).detach().cpu()
            ),
            "gate_is_open": bool(
                model.decoder.gate_value(stochastic=False).detach().cpu().item() >= 0.5
            ),
            "latent_loading_l1_magnitude": float(
                model.decoder.latent_loading_l1().detach().cpu()
            ),
            "latent_loading_group_norm": float(
                model.decoder.latent_loading_group_norm().detach().cpu()
            ),
        }

    metadata = dict(context)
    metadata.update(
        {
            "Num Samples": int(len(train.time) + len(validation.time) + len(test.time)),
            "Num Features": int(train.X.shape[1]),
            "Train Size": int(len(train.time)),
            "Validation Size": int(len(validation.time)),
            "Test Size": int(len(test.time)),
            "Event Rate Train": float(np.mean(train.event)),
            "Event Rate Validation": float(np.mean(validation.event)),
            "Event Rate Test": float(np.mean(test.event)),
            "Censoring Rate Train": censoring_rate(train.event),
            "Censoring Rate Validation": censoring_rate(validation.event),
            "Censoring Rate Test": censoring_rate(test.event),
        }
    )

    rows: list[dict] = []
    tau = None
    for name, pred in predictions.items():
        is_synthetic = "synthetic" in str(context.get("Dataset Type", "")).lower()
        metrics = collect_metrics(
            name,
            pred["median"],
            pred["survival"],
            test.time,
            test.event,
            test.true_event_time,
            time_points,
            tau=tau,
            t_train=train.time,
            e_train=train.event,
            oracle_only=is_synthetic,
        )
        row = dict(metadata)
        row["Model"] = name
        prefix = f"{name} "
        clean_metrics = {key.removeprefix(prefix): value for key, value in metrics.items()}
        row.update(clean_metrics)
        row.update({
            key: value for key, value in pred.items()
            if key not in {"median", "survival"}
        })
        rows.append(row)
        tau = clean_metrics.get("evaluation_time_horizon", tau)

    return rows, {
        "time_points": time_points,
        "test_time": test.time,
        "test_event": test.event,
        **predictions,
    }


def _load_dataset(spec: dict, source: str) -> SurvivalData:
    if source == "real_file":
        return load_real_data(
            spec["path"], spec["time_column"], spec["event_column"], spec.get("feature_columns")
        )
    if source == "semi_synthetic_file":
        return load_semi_synthetic_data(
            spec["path"],
            spec["true_event_time_column"],
            spec["true_censor_time_column"],
            spec.get("feature_columns"),
        )
    raise ValueError(source)


def validate_inputs(cfg: dict) -> None:
    data_cfg = cfg["data"]
    source = str(data_cfg["source"]).lower()
    if source in {"synthetic_copula", "gaussian_shared_frailty", "frailty_recovery_diagnostic"}:
        return
    path = Path(data_cfg["path"])
    if not path.exists():
        raise FileNotFoundError(path)
    if source == "support_cox_clayton_semisynthetic":
        if path.suffix.lower() not in {".feather", ".ftr"}:
            raise ValueError("SUPPORT semi-synthetic input must be a Feather file")
        required = {"duration", "event", *(f"x{i}" for i in range(14))}
        missing = sorted(required - set(pd.read_feather(path).columns))
        if missing:
            raise ValueError(f"SUPPORT input is missing columns: {missing}")
        return
    _load_dataset(data_cfg, source)


def run(cfg: dict) -> pd.DataFrame:
    validate_inputs(cfg)
    study_cfg = cfg["study"]
    data_cfg = cfg["data"]
    source = str(data_cfg["source"]).lower()
    out_dir = Path(study_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(str(cfg["compute"].get("device", "auto")))
    if device.type == "cpu":
        torch.set_num_threads(max(1, int(cfg["compute"].get("torch_num_threads", 1))))
    print(f"Using device: {device}")
    rows: list[dict] = []

    if source == "gaussian_shared_frailty":
        from .synthetic import run_synthetic_pilot

        rows = run_synthetic_pilot(cfg, out_dir, device).to_dict("records")
    elif source == "frailty_recovery_diagnostic":
        from .frailty_recovery import run_frailty_recovery_diagnostic

        return run_frailty_recovery_diagnostic(cfg, out_dir, device)
    elif source == "support_cox_clayton_semisynthetic":
        from utility.semisynthetic import (
            fit_support_semisynthetic_dgp, generate_support_semisynthetic,
        )

        dgp = fit_support_semisynthetic_dgp(
            data_cfg["path"], cox_penalizer=float(data_cfg.get("cox_penalizer", 0.01))
        )
        diagnostics = []
        seed_streams = expand_seed_streams(cfg["seeds"])
        for repeat, seeds in enumerate(seed_streams):
            dgp_seed = seeds["dgp"]
            sampling_seed = seeds["sampling"]
            split_seed = seeds["split"]
            model_seed = seeds["model"]
            for target_rate in data_cfg["censoring_rates"]:
                generated = generate_support_semisynthetic(
                    dgp, kendall_tau=float(data_cfg["kendall_tau"]),
                    censoring_rate=float(target_rate), sampling_seed=int(sampling_seed),
                )
                train_idx, validation_idx, test_idx = time_event_stratified_split_indices(
                    generated.data, cfg["split"], int(split_seed)
                )
                train, validation, test = _preprocess_three(
                    _subset(generated.data, train_idx),
                    _subset(generated.data, validation_idx),
                    _subset(generated.data, test_idx),
                    cfg["preprocessing"],
                )
                seed_everything(int(model_seed))
                scenario = f"support_clayton_tau_0.50_censor_{float(target_rate):.2f}"
                context = {
                    "Study": study_cfg["name"], "Stage": study_cfg["stage"],
                    "Dataset Type": source, "Dataset": data_cfg.get("name", "support"),
                    "Source Path": str(data_cfg["path"]), "Scenario": scenario,
                    "Copula": "clayton", "Dependence": "dependent_censoring",
                    "Theta": generated.clayton_theta,
                    "Target Kendall Tau": generated.target_kendall_tau,
                    "Empirical Copula Kendall Tau": generated.empirical_copula_kendall_tau,
                    "Empirical Marginal Kendall Tau": generated.empirical_marginal_kendall_tau,
                    "Target Censoring Rate": generated.target_censoring_rate,
                    "Achieved Censoring Rate": generated.achieved_censoring_rate,
                    "Repeat": repeat, "Fold": 0, "DGP Seed": int(dgp_seed),
                    "Sampling Seed": int(sampling_seed), "Split Seed": int(split_seed),
                    "Model Seed": int(model_seed),
                }
                split_rows, preds = _fit_one_split(
                    train, validation, test, cfg, device, context
                )
                rows.extend(split_rows)
                diagnostics.append({
                    **context, "Censor Time Scale": generated.censor_time_scale,
                    "Source Samples": dgp.source_n_samples,
                    "Source Event Rate": dgp.source_event_rate,
                    "Cox Penalizer": dgp.cox_penalizer,
                })
                if cfg["evaluation"].get("save_predictions", False):
                    _save_predictions(out_dir, context, preds)
        pd.DataFrame(diagnostics).to_csv(out_dir / "dgp_diagnostics.csv", index=False)
    elif source == "synthetic_copula":
        scenarios = expand_scenarios(data_cfg)
        for scenario in scenarios:
            for repeat, seed in enumerate(study_cfg["seeds"]):
                seed_everything(seed)
                X, time, event, true_t, true_c = generate_copula_data(
                    n_samples=int(scenario.get("n_samples", data_cfg["n_samples"])),
                    n_features=int(scenario.get("n_features", data_cfg["n_features"])),
                    copula_type=str(scenario["copula"]),
                    theta=float(scenario["theta"]),
                    seed=seed,
                )
                data = SurvivalData(
                    X,
                    time,
                    event,
                    [f"X{i}" for i in range(X.shape[1])],
                    true_t,
                    true_c,
                )
                for fold, (train_idx, validation_idx, test_idx) in enumerate(_splits(data, cfg["split"], seed)):
                    train, validation, test = _preprocess_three(
                        _subset(data, train_idx), _subset(data, validation_idx),
                        _subset(data, test_idx), cfg["preprocessing"]
                    )
                    context = {
                        "Study": study_cfg["name"],
                        "Stage": study_cfg["stage"],
                        "Dataset Type": source,
                        "Dataset": scenario.get("name", scenario["copula"]),
                        "Scenario": scenario.get("id", scenario.get("name", scenario["copula"])),
                        "Copula": scenario["copula"],
                        "Dependence": scenario.get("dependence", "custom"),
                        "Theta": float(scenario["theta"]),
                        "Repeat": repeat,
                        "Fold": fold,
                        "Seed": seed,
                    }
                    split_rows, preds = _fit_one_split(train, validation, test, cfg, device, context)
                    rows.extend(split_rows)
                    if cfg["evaluation"].get("save_predictions", False):
                        _save_predictions(out_dir, context, preds)
    else:
        data = _load_dataset(data_cfg, source)
        for repeat, seed in enumerate(study_cfg["seeds"]):
            seed_everything(seed)
            for fold, (train_idx, validation_idx, test_idx) in enumerate(_splits(data, cfg["split"], seed)):
                train, validation, test = _preprocess_three(
                    _subset(data, train_idx), _subset(data, validation_idx),
                    _subset(data, test_idx), cfg["preprocessing"]
                )
                context = {
                    "Study": study_cfg["name"],
                    "Stage": study_cfg["stage"],
                    "Dataset Type": source,
                    "Dataset": data_cfg.get("name", Path(data_cfg["path"]).stem),
                    "Source Path": str(data_cfg["path"]),
                    "Copula": None,
                    "Dependence": source,
                    "Theta": None,
                    "Repeat": repeat,
                    "Fold": fold,
                    "Seed": seed,
                }
                split_rows, preds = _fit_one_split(train, validation, test, cfg, device, context)
                rows.extend(split_rows)
                if cfg["evaluation"].get("save_predictions", False):
                    _save_predictions(out_dir, context, preds)

    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("No experiments were completed")
    results.to_csv(out_dir / "results_raw.csv", index=False)

    group_cols = [
        c
        for c in ("Study", "Stage", "Dataset", "Scenario", "Copula", "Dependence", "Theta", "Model",
                  "study", "scenario", "n_samples", "target_kendall_tau", "target_censoring_rate",
                  "mechanism", "model", "latent_dim", "hyperparameter_variant",
                  "comparison_parent",
                  "epochs", "batch_size", "dropout", "weight_decay", "encoder_hidden",
                  "decoder_hidden", "scale_link", "latent_path", "shape_mode",
                  "latent_loading_l1", "latent_group_lasso", "latent_gate", "gate_l1",
                  "gate_initial_value", "gate_temperature",
                  "checkpoint", "is_primary_checkpoint", "prediction_mode",
                  "partition")
        if c in results and not results[c].isna().all()
    ]
    numeric = results.select_dtypes(include=[np.number]).columns.tolist()
    excluded = {
        "Repeat", "Fold", "Seed", "Sampling Seed", "Split Seed", "Model Seed",
        "repeat", "dgp_seed", "sampling_seed", "split_seed", "model_seed",
    }
    metric_cols = [c for c in numeric if c not in excluded and c not in group_cols]
    results.groupby(group_cols, dropna=False)[metric_cols].mean().reset_index().to_csv(
        out_dir / "results_mean.csv", index=False
    )
    results.groupby(group_cols, dropna=False)[metric_cols].std().reset_index().to_csv(
        out_dir / "results_std.csv", index=False
    )
    with (out_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, default=str)
    print(f"Saved results to {out_dir.resolve()}")
    return results


def _save_predictions(out_dir: Path, context: dict, preds: dict) -> None:
    name = "_".join(str(context.get(k, "")) for k in ("Dataset", "Scenario", "Repeat", "Fold"))
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
    payload = {
        "time_points": preds["time_points"],
        "test_time": preds["test_time"],
        "test_event": preds["test_event"],
    }
    for model, values in preds.items():
        if isinstance(values, dict):
            payload[f"{model}_median"] = values["median"]
            payload[f"{model}_survival"] = values["survival"]
    np.savez_compressed(out_dir / f"predictions_{safe}.npz", **payload)
