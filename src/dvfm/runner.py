"""Dataset-independent runner using the supplied algorithms and reference parameters."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from .baselines import (
    _fit_cox_and_predict_survival,
    fit_clayton_weibull_aft,
    train_deepsurv,
    train_mtlr,
)
from .data import SurvivalData, load_real_data, load_semi_synthetic_data
from .metrics import _censoring_rate, collect_metrics
from .model import DVFM, SurvivalDataset
from .prediction import get_median_survival_time, predict_survival_curves
from .synthetic import generate_copula_data
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
        yield from splitter.split(indices, data.event)
        return
    if strategy != "holdout":
        raise ValueError(f"Unknown split strategy: {strategy}")

    # The supplied synthetic reference code uses train_test_split without stratification.
    train_idx, test_idx = train_test_split(
        indices,
        test_size=float(split_cfg.get("test_size", 0.30)),
        random_state=seed,
    )
    yield train_idx, test_idx


def _subset(data: SurvivalData, idx: np.ndarray) -> SurvivalData:
    return SurvivalData(
        X=data.X[idx].copy(),
        time=data.time[idx].copy(),
        event=data.event[idx].copy(),
        feature_names=list(data.feature_names),
        true_event_time=None if data.true_event_time is None else data.true_event_time[idx].copy(),
        true_censor_time=None if data.true_censor_time is None else data.true_censor_time[idx].copy(),
    )


def _preprocess(
    train: SurvivalData, test: SurvivalData, cfg: dict
) -> tuple[SurvivalData, SurvivalData, float]:
    if bool(cfg.get("standardize", False)):
        scaler = StandardScaler()
        train.X = scaler.fit_transform(train.X)
        test.X = scaler.transform(test.X)

    method = str(cfg.get("time_normalize", "none")).lower()
    scale = 1.0
    if method == "train_max":
        scale = max(float(np.max(train.time)), 1e-8)
    elif method != "none":
        raise ValueError(f"Unsupported time_normalize method: {method}")

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
    test: SurvivalData,
    cfg: dict,
    device: torch.device,
    context: dict,
) -> tuple[dict, dict]:
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
            lr=float(c["lr"]),
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
            lr=float(c["lr"]),
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
            lr=float(c["lr"]),
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
            SurvivalDataset(test.X, test.time, test.event),
            batch_size=int(c["batch_size"]),
            shuffle=False,
        )
        model = DVFM(input_dim=train.X.shape[1], latent_dim=int(c["latent_dim"])).to(device)
        train_dvfm(
            model,
            train_loader,
            val_loader,
            n_epochs=int(c["epochs"]),
            lr=float(c["lr"]),
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

    row = dict(context)
    row.update(
        {
            "Num Samples": int(len(train.time) + len(test.time)),
            "Num Features": int(train.X.shape[1]),
            "Train Size": int(len(train.time)),
            "Test Size": int(len(test.time)),
            "Event Rate Train": float(np.mean(train.event)),
            "Event Rate Test": float(np.mean(test.event)),
            "Censoring Rate Train": _censoring_rate(train.event),
            "Censoring Rate Test": _censoring_rate(test.event),
        }
    )

    tau = None
    dep_copula = context.get("Copula")
    dep_theta = context.get("Theta")
    for name, pred in predictions.items():
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
        )
        row.update(metrics)
        tau = metrics.get("Eval Tau", tau)

    return row, {
        "time_points": time_points,
        "test_time": test.time,
        "test_event": test.event,
        **predictions,
    }


def _load_dataset(spec: dict, mode: str) -> SurvivalData:
    if mode == "real":
        return load_real_data(
            spec["path"], spec["time_col"], spec["event_col"], spec.get("feature_cols")
        )
    if mode == "semi_synthetic":
        return load_semi_synthetic_data(
            spec["path"],
            spec["true_event_time_col"],
            spec["true_censor_time_col"],
            spec.get("feature_cols"),
        )
    raise ValueError(mode)


def validate_inputs(cfg: dict) -> None:
    mode = str(cfg.get("mode", "synthetic")).lower()
    if mode == "synthetic":
        scenarios = cfg.get("synthetic", {}).get("scenarios", [])
        if not scenarios:
            raise ValueError("Synthetic mode requires synthetic.scenarios")
        for scenario in scenarios:
            if "copula" not in scenario or "theta" not in scenario:
                raise ValueError(f"Invalid synthetic scenario: {scenario}")
        return
    if mode not in {"real", "semi_synthetic"}:
        raise ValueError(f"Unknown mode: {mode}")
    specs = cfg.get("datasets") or [cfg.get("data", {})]
    for spec in specs:
        path = Path(spec.get("path", ""))
        if not path.exists():
            raise FileNotFoundError(path)
        _load_dataset(spec, mode)


def run(cfg: dict) -> pd.DataFrame:
    validate_inputs(cfg)
    mode = str(cfg.get("mode", "synthetic")).lower()
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(str(cfg.get("device", "auto")))
    if device.type == "cpu":
        torch.set_num_threads(max(1, int(cfg.get("torch_num_threads", 1))))
    print(f"Using device: {device}")
    rows: list[dict] = []

    if mode == "synthetic":
        scenarios = cfg["synthetic"]["scenarios"]
        for scenario in scenarios:
            for repeat in range(int(cfg.get("repeats", 1))):
                seed = int(cfg.get("seed", 42)) + repeat
                seed_everything(seed)
                X, time, event, true_t, true_c = generate_copula_data(
                    n_samples=int(
                        scenario.get("n_samples", cfg["synthetic"].get("n_samples", 10000))
                    ),
                    n_features=int(
                        scenario.get("n_features", cfg["synthetic"].get("n_features", 10))
                    ),
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
                for fold, (train_idx, test_idx) in enumerate(_splits(data, cfg["split"], seed)):
                    train, test, _ = _preprocess(
                        _subset(data, train_idx), _subset(data, test_idx), cfg["preprocessing"]
                    )
                    context = {
                        "Dataset Type": "synthetic",
                        "Dataset": scenario.get("name", scenario["copula"]),
                        "Scenario": scenario.get("id", scenario.get("name", scenario["copula"])),
                        "Copula": scenario["copula"],
                        "Dependence": scenario.get("dependence", "custom"),
                        "Theta": float(scenario["theta"]),
                        "Repeat": repeat,
                        "Fold": fold,
                        "Seed": seed,
                    }
                    row, preds = _fit_one_split(train, test, cfg, device, context)
                    rows.append(row)
                    if cfg["evaluation"].get("save_predictions", False):
                        _save_predictions(out_dir, context, preds)
    else:
        specs = cfg.get("datasets") or [cfg.get("data", {})]
        for spec in specs:
            data = _load_dataset(spec, mode)
            for repeat in range(int(cfg.get("repeats", 1))):
                seed = int(cfg.get("seed", 42)) + repeat
                seed_everything(seed)
                for fold, (train_idx, test_idx) in enumerate(_splits(data, cfg["split"], seed)):
                    train, test, scale = _preprocess(
                        _subset(data, train_idx), _subset(data, test_idx), cfg["preprocessing"]
                    )
                    context = {
                        "Dataset Type": mode,
                        "Dataset": spec.get("name", Path(spec["path"]).stem),
                        "Source Path": str(spec["path"]),
                        "Copula": None,
                        "Dependence": mode,
                        "Theta": None,
                        "Repeat": repeat,
                        "Fold": fold,
                        "Seed": seed,
                        "Time Scale": scale,
                    }
                    row, preds = _fit_one_split(train, test, cfg, device, context)
                    rows.append(row)
                    if cfg["evaluation"].get("save_predictions", False):
                        _save_predictions(out_dir, context, preds)

    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("No experiments were completed")
    results.to_csv(out_dir / "results_raw.csv", index=False)

    group_cols = [
        c
        for c in ("Dataset", "Scenario", "Copula", "Dependence", "Theta")
        if c in results and not results[c].isna().all()
    ]
    numeric = results.select_dtypes(include=[np.number]).columns.tolist()
    excluded = {"Repeat", "Fold", "Seed"}
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
