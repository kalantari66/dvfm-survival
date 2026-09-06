"""Dataset-independent runner using the supplied algorithms and reference parameters."""

from __future__ import annotations

import json
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from lifelines.utils import concordance_index
from torch.utils.data import DataLoader

from .baselines import (
    _fit_cox_and_predict_survival,
    fit_clayton_weibull_aft,
    train_deepsurv,
    train_mtlr,
)
from .config import expand_scenarios
from .data import SurvivalData, load_real_data, load_semi_synthetic_data
from .metrics import _censoring_rate, collect_metrics, compute_oracle_brier_ibs
from .model import DVFM, SurvivalDataset
from .prediction import (get_median_survival_time, learned_conditional_kendall_tau,
                         posterior_diagnostics, predict_survival_curves,
                         predict_survival_from_prior)
from .synthetic import generate_copula_data, generate_gaussian_shared_frailty
from .training import train_dvfm


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _splits(data: SurvivalData, split_cfg: dict, seed: int):
    indices = np.arange(len(data.time))
    strategy = str(split_cfg.get("strategy", "holdout")).lower()
    if strategy == "kfold":
        folds = int(split_cfg.get("folds", 5))
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for fold, (train_val_idx, test_idx) in enumerate(splitter.split(indices, data.event)):
            train_idx, validation_idx = train_test_split(
                train_val_idx,
                test_size=float(split_cfg["validation_fraction"]),
                random_state=seed + fold + 1,
                stratify=data.event[train_val_idx],
            )
            yield train_idx, validation_idx, test_idx
        return
    if strategy != "holdout":
        raise ValueError(f"Unknown split strategy: {strategy}")

    yield _three_way_split(data, split_cfg, seed)


def _three_way_split(data: SurvivalData, split_cfg: dict, seed: int):
    indices = np.arange(len(data.time))
    test_fraction = float(split_cfg["test_fraction"])
    validation_fraction = float(split_cfg["validation_fraction"])
    train_val, test = train_test_split(indices, test_size=test_fraction, random_state=seed,
                                       stratify=data.event)
    relative_validation = validation_fraction / (1.0 - test_fraction)
    train, validation = train_test_split(train_val, test_size=relative_validation,
                                         random_state=seed + 1, stratify=data.event[train_val])
    return train, validation, test


def _preprocess_three(train, validation, test, cfg):
    if bool(cfg.get("standardize_x", False)):
        scaler = StandardScaler().fit(train.X)
        train.X = scaler.transform(train.X); validation.X = scaler.transform(validation.X); test.X = scaler.transform(test.X)
    scale = 1.0
    if cfg.get("time_normalization", "none") == "train_max":
        scale = max(float(np.max(train.time)), 1e-8)
    for part in (train, validation, test):
        part.time /= scale
        if part.true_event_time is not None:
            part.true_event_time /= scale
        if part.true_censor_time is not None:
            part.true_censor_time /= scale
    return train, validation, test, scale


def _subset(data: SurvivalData, idx: np.ndarray) -> SurvivalData:
    return SurvivalData(
        X=data.X[idx].copy(),
        time=data.time[idx].copy(),
        event=data.event[idx].copy(),
        feature_names=list(data.feature_names),
        true_event_time=None if data.true_event_time is None else data.true_event_time[idx].copy(),
        true_censor_time=None if data.true_censor_time is None else data.true_censor_time[idx].copy(),
        true_z=None if data.true_z is None else data.true_z[idx].copy(),
    )


def _preprocess(
    train: SurvivalData, test: SurvivalData, cfg: dict
) -> tuple[SurvivalData, SurvivalData, float]:
    if bool(cfg.get("standardize_x", False)):
        scaler = StandardScaler()
        train.X = scaler.fit_transform(train.X)
        test.X = scaler.transform(test.X)

    method = str(cfg.get("time_normalization", "none")).lower()
    scale = 1.0
    if method == "train_max":
        scale = max(float(np.max(train.time)), 1e-8)
    elif method != "none":
        raise ValueError(f"Unsupported time_normalization method: {method}")

    if scale != 1.0:
        for part in (train, test):
            part.time = part.time / scale
            if part.true_event_time is not None:
                part.true_event_time = part.true_event_time / scale
            if part.true_censor_time is not None:
                part.true_censor_time = part.true_censor_time / scale
    return train, test, scale


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
        model = DVFM(input_dim=train.X.shape[1], latent_dim=int(c["latent_dim"])).to(device)
        train_dvfm(
            model,
            train_loader,
            val_loader,
            n_epochs=int(c["epochs"]),
            lr=float(c["learning_rate"]),
            beta_max=float(c["beta_max"]),
            warmup_epochs=int(c["warmup_epochs"]),
            free_bits=float(c["free_bits"]),
            device=device,
        )
        survival = predict_survival_curves(
            model=model,
            X=test.X,
            time_points=time_points,
            train_loader=train_loader,
            n_samples=int(c["mc_samples"]),
            device=device,
        )
        median = get_median_survival_time(survival, time_points)
        predictions["DVFM"] = {"median": median, "survival": survival}

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
            "Censoring Rate Train": _censoring_rate(train.event),
            "Censoring Rate Validation": _censoring_rate(validation.event),
            "Censoring Rate Test": _censoring_rate(test.event),
        }
    )

    rows: list[dict] = []
    tau = None
    dep_copula = context.get("Copula")
    dep_theta = context.get("Theta")
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
            true_t_train=train.true_event_time,
            true_c_train=train.true_censor_time,
            dep_copula_name=dep_copula,
            dep_alpha=dep_theta,
            oracle_only=is_synthetic,
        )
        row = dict(metadata)
        row["Model"] = name
        prefix = f"{name} "
        clean_metrics = {key.removeprefix(prefix): value for key, value in metrics.items()}
        row.update(clean_metrics)
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


def _calibration_rows(survival, grid, true_event_time, context, bins=10):
    rows = []
    for index in np.linspace(0, len(grid) - 1, bins, dtype=int):
        rows.append({**context, "time": float(grid[index]),
                     "mean_predicted_survival": float(survival[:, index].mean()),
                     "empirical_oracle_survival": float(np.mean(true_event_time > grid[index]))})
    return rows


def _run_gaussian_frailty(cfg: dict, out_dir: Path, device: torch.device) -> pd.DataFrame:
    data_cfg, split_cfg, model_cfg, eval_cfg = cfg["data"], cfg["split"], cfg["models"], cfg["evaluation"]
    seed_cfg = cfg["seeds"]
    lengths = {len(seed_cfg[name]) for name in ("sampling", "split", "model")}
    if len(lengths) != 1:
        raise ValueError("sampling, split, and model seed lists must have equal length")
    rows, histories, diagnostics, calibration = [], [], [], []
    for scenario in expand_scenarios(data_cfg):
        for repeat, (sampling_seed, split_seed, model_seed) in enumerate(zip(seed_cfg["sampling"], seed_cfg["split"], seed_cfg["model"])):
            generated = generate_gaussian_shared_frailty(
                n_samples=int(data_cfg["n_samples"]), n_features=int(data_cfg["n_features"]),
                kendall_tau=float(scenario["kendall_tau"]), censoring_rate=float(scenario["censoring_rate"]),
                dgp_seed=int(seed_cfg["dgp"]), sampling_seed=int(sampling_seed),
                calibration_samples=int(data_cfg.get("calibration_samples", 50_000)))
            full = SurvivalData(generated.X, generated.observed_time, generated.event,
                                [f"X{i}" for i in range(generated.X.shape[1])], generated.event_time,
                                generated.censor_time, generated.true_z)
            train_i, val_i, test_i = _three_way_split(full, split_cfg, int(split_seed))
            train, validation, test, scale = _preprocess_three(_subset(full, train_i), _subset(full, val_i), _subset(full, test_i), cfg["preprocessing"])
            grid = np.linspace(0.0, float(np.quantile(train.true_event_time, eval_cfg.get("grid_max_quantile", .95))), int(eval_cfg["n_time_points"]))
            for latent_dim in model_cfg["dvfm"]["latent_dims"]:
                seed_everything(int(model_seed))
                c = model_cfg["dvfm"]
                train_loader = DataLoader(SurvivalDataset(train.X, train.time, train.event), batch_size=int(c["batch_size"]), shuffle=True)
                validation_loader = DataLoader(SurvivalDataset(validation.X, validation.time, validation.event), batch_size=int(c["batch_size"]), shuffle=False)
                test_loader = DataLoader(SurvivalDataset(test.X, test.time, test.event), batch_size=int(c["batch_size"]), shuffle=False)
                model = DVFM(train.X.shape[1], int(latent_dim)).to(device)
                history = train_dvfm(model, train_loader, validation_loader, n_epochs=int(c["epochs"]),
                                     lr=float(c["learning_rate"]), beta_max=float(c["beta_max"]),
                                     warmup_epochs=int(c["warmup_epochs"]), free_bits=float(c["free_bits"]),
                                     device=device, return_history=True)
                base = {"study": cfg["study"]["name"], "scenario": f"kendall_tau-{scenario['kendall_tau']}-censoring_rate-{scenario['censoring_rate']}",
                        "target_kendall_tau": float(scenario["kendall_tau"]),
                        "empirical_conditional_kendall_tau": generated.empirical_conditional_kendall_tau,
                        "empirical_marginal_kendall_tau": generated.empirical_marginal_kendall_tau,
                        "target_censoring_rate": float(scenario["censoring_rate"]), "achieved_censoring_rate": generated.achieved_censoring_rate,
                        "latent_dim": int(latent_dim), "repeat": repeat, "dgp_seed": int(seed_cfg["dgp"]),
                        "sampling_seed": int(sampling_seed), "split_seed": int(split_seed), "model_seed": int(model_seed)}
                histories.extend([{**base, **item} for item in history])
                diag = posterior_diagnostics(model, test_loader, test.true_z, device=device,
                                             active_kl_threshold=float(eval_cfg.get("active_kl_threshold", .01)))
                learned_tau = learned_conditional_kendall_tau(model, np.mean(train.X, axis=0),
                                                              int(eval_cfg.get("dependence_samples", 2000)), device, int(model_seed))
                diagnostics.append({**base, "learned_conditional_kendall_tau": learned_tau,
                                    "active_latent_dimensions": diag["active_latent_dimensions"], "mean_kl": diag["mean_kl"],
                                    "best_abs_z_pearson": diag["best_abs_z_pearson"]})
                predictions = {
                    "aggregate_posterior": predict_survival_curves(model, test.X, grid, train_loader, int(c["mc_samples"]), device),
                    "prior": predict_survival_from_prior(model, test.X, grid, int(c["mc_samples"]), device),
                }
                for mode, survival in predictions.items():
                    median = get_median_survival_time(survival, grid)
                    _, ibs = compute_oracle_brier_ibs(survival, grid, test.true_event_time)
                    result_context = {**base, "prediction_mode": mode}
                    rows.append({**result_context, "oracle_ibs": ibs,
                                 "oracle_ci": float(concordance_index(test.true_event_time, median)),
                                 "oracle_mae": float(np.mean(np.abs(test.true_event_time - median))),
                                 "oracle_mae_censored": float(np.mean(np.abs(test.true_event_time[test.event == 0] - median[test.event == 0]))),
                                 "oracle_mae_uncensored": float(np.mean(np.abs(test.true_event_time[test.event == 1] - median[test.event == 1])))})
                    calibration.extend(_calibration_rows(survival, grid, test.true_event_time, result_context))
                tag = f"kendall_tau{scenario['kendall_tau']}_censoring_rate{scenario['censoring_rate']}_repeat{repeat}_latent_dim{latent_dim}"
                np.savez_compressed(out_dir / f"predictions_{tag}.npz", time_points=grid, test_indices=test_i,
                                    test_event=test.event, true_event_time=test.true_event_time,
                                    true_censor_time=test.true_censor_time, true_z=test.true_z,
                                    survival_aggregate_posterior=predictions["aggregate_posterior"], survival_prior=predictions["prior"],
                                    posterior_mu=diag.get("posterior_mu", np.empty((len(test.time), 0))),
                                    posterior_logvar=diag.get("posterior_logvar", np.empty((len(test.time), 0))))
    pd.DataFrame(histories).to_csv(out_dir / "training_history.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(out_dir / "dvfm_diagnostics.csv", index=False)
    pd.DataFrame(calibration).to_csv(out_dir / "calibration_curves.csv", index=False)
    return pd.DataFrame(rows)


def validate_inputs(cfg: dict) -> None:
    data_cfg = cfg["data"]
    source = str(data_cfg["source"]).lower()
    if source in {"synthetic_copula", "gaussian_shared_frailty"}:
        return
    path = Path(data_cfg["path"])
    if not path.exists():
        raise FileNotFoundError(path)
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
        rows = _run_gaussian_frailty(cfg, out_dir, device).to_dict("records")
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
                    train, validation, test, _ = _preprocess_three(
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
                train, validation, test, scale = _preprocess_three(
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
                    "Time Scale": scale,
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
                  "study", "scenario", "target_kendall_tau", "target_censoring_rate", "latent_dim", "prediction_mode")
        if c in results and not results[c].isna().all()
    ]
    numeric = results.select_dtypes(include=[np.number]).columns.tolist()
    excluded = {"Repeat", "Fold", "Seed", "repeat", "dgp_seed", "sampling_seed", "split_seed", "model_seed"}
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
