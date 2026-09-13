"""Tune semi-synthetic model configurations on one 70/30 development split."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from experiments.config import load_config, semisynthetic_datasets
from experiments.runner import _fit_one_split
from utility.runtime import resolve_device, seed_everything
from utility.semisynthetic import fit_semisynthetic_dgp, generate_semisynthetic
from utility.splitting import (
    preprocess_train_validation, subset_survival_data,
    time_event_stratified_train_validation_indices,
)


MODEL_LABELS = {
    "deepsurv": "DeepSurv",
    "mtlr": "MTLR",
    "clayton_aft": "ClaytonAFT",
    "hacsurv_2d": "HACSurv",
    "bayesian_cox_gamma_frailty": "BayesianCoxGammaFrailty",
    "dvfm": "DVFM",
}


def _read_tuning_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    required = {"experiment_config", "study", "development", "tuning"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Tuning config is missing keys: {sorted(missing)}")
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("schema_version must be 1")
    if int(config["tuning"].get("trials", 0)) < 1:
        raise ValueError("tuning.trials must be positive")
    if str(config["tuning"].get("selection_metric", "")) != "IBS Oracle":
        raise ValueError("tuning.selection_metric must be 'IBS Oracle'")
    if not 0 < float(config["development"].get("validation_fraction", 0)) < 1:
        raise ValueError("development.validation_fraction must be between 0 and 1")
    return config


def _sample(space: dict, rng: np.random.Generator) -> dict:
    sampled = {}
    for key, choices in space.items():
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"Search-space entry {key!r} must be a non-empty list")
        sampled[key] = deepcopy(choices[int(rng.integers(len(choices)))])
    return sampled


def _apply_trial(base: dict, model: str, parameters: dict) -> dict:
    config = deepcopy(base)
    parameters = deepcopy(parameters)
    config["models"]["enabled"] = [model]
    target = config["models"][model]
    if model == "dvfm":
        target["latent_dim"] = 1
        if "architecture" in parameters:
            encoder, decoder = parameters.pop("architecture")
            target["encoder_hidden"] = encoder
            target["decoder_hidden"] = decoder
    if model == "hacsurv_2d" and "copula_lr_multiplier" in parameters:
        multiplier = parameters.pop("copula_lr_multiplier")
        target["copula_learning_rate"] = float(parameters["learning_rate"]) * float(multiplier)
    target.update(parameters)
    return config


def _tune_dataset(base: dict, tuning: dict, dataset: dict, out_dir: Path, device) -> list[dict]:
    development = tuning["development"]
    dgp = fit_semisynthetic_dgp(dataset, cox_penalizer=float(base["data"].get("cox_penalizer", 0.01)))
    generated = generate_semisynthetic(
        dgp, kendall_tau=float(development["kendall_tau"]),
        censoring_rate=1.0 - float(dgp.source_event_rate),
        sampling_seed=int(development["seed"]), copula=str(development["copula"]),
    )
    train_idx, validation_idx = time_event_stratified_train_validation_indices(
        generated.data, development, int(development["seed"])
    )
    train, validation = preprocess_train_validation(
        subset_survival_data(generated.data, train_idx),
        subset_survival_data(generated.data, validation_idx),
        {**base["preprocessing"], "numeric_features": dataset["numeric_features"]},
    )
    trial_rows = []
    for model, space in tuning["tuning"]["search_spaces"].items():
        if model not in MODEL_LABELS:
            raise ValueError(f"Unsupported tuning model: {model}")
        rng = np.random.default_rng(int(development["seed"]) + sum(map(ord, dataset["name"] + model)))
        for trial in range(int(tuning["tuning"]["trials"])):
            parameters = _sample(space, rng)
            trial_config = _apply_trial(base, model, parameters)
            seed_everything(int(development["seed"]) + trial)
            context = {
                "Study": tuning["study"]["name"], "Stage": "tuning",
                "Dataset Type": "development", "Dataset": dataset["name"],
                "Copula": str(development["copula"]).lower(),
                "Target Kendall Tau": float(development["kendall_tau"]),
                "Model Seed": int(development["seed"]) + trial,
            }
            row = {"dataset": dataset["name"], "model": model, "trial": trial,
                   "parameters": json.dumps(parameters, sort_keys=True), "status": "ok"}
            try:
                metrics, _ = _fit_one_split(
                    train, validation, validation, trial_config, device, context,
                )
                selected = next(item for item in metrics if item["Model"] == MODEL_LABELS[model])
                row["validation_oracle_ibs"] = float(selected["IBS Oracle"])
                # Retain the observed-data estimate as a sensitivity result;
                # it is explicitly not the hyperparameter selection score.
                row["validation_ibs_ipcw"] = float(selected["IBS IPCW"])
                row["validation_ci_ipcw"] = float(selected["CI IPCW"])
                row["validation_mae_margin"] = float(selected["MAE Margin"])
                row["selection_score"] = row["validation_oracle_ibs"]
            except Exception as exc:  # preserve failed trials for auditability
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}", selection_score=np.inf)
            trial_rows.append(row)
    frame = pd.DataFrame(trial_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "trials.csv", index=False)
    return trial_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset", action="append", help="Optional dataset name; may be repeated")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    tuning = _read_tuning_config(args.config)
    base_path = Path(tuning["experiment_config"])
    if not base_path.is_absolute():
        base_path = args.config.parent.parent / base_path
    base = load_config(base_path)
    datasets = semisynthetic_datasets(base["data"])
    if args.dataset:
        wanted = {name.lower() for name in args.dataset}
        datasets = [item for item in datasets if item["name"].lower() in wanted]
        if not datasets or len(datasets) != len(wanted):
            raise ValueError("--dataset must name configured semi-synthetic datasets")
    if args.validate_only:
        print(f"Configuration is valid for {len(datasets)} datasets. No model was trained.")
        return
    out_root = args.output_dir or Path(tuning["study"]["output_dir"])
    device = resolve_device(str(base["compute"].get("device", "auto")))
    if device.type == "cpu":
        torch.set_num_threads(max(1, int(base["compute"].get("torch_num_threads", 1))))
    rows = []
    for dataset in datasets:
        print(f"[tuning] {dataset['name']}", flush=True)
        # A GWF per-dataset target passes an already dataset-specific output
        # directory. Direct all-dataset CLI use retains the nested layout.
        dataset_out = out_root if args.dataset else out_root / dataset["name"]
        rows.extend(_tune_dataset(base, tuning, dataset, dataset_out, device))
    trials = pd.DataFrame(rows)
    trials.to_csv(out_root / "tuning_trials.csv", index=False)
    winners = []
    for (dataset, model), group in trials.query("status == 'ok'").groupby(["dataset", "model"]):
        best = group.loc[group["selection_score"].idxmin()]
        winners.append({"dataset": dataset, "model": model, "trial": int(best["trial"]),
                        "parameters": json.loads(best["parameters"]),
                        "validation_oracle_ibs": float(best["validation_oracle_ibs"]),
                        "validation_ibs_ipcw_secondary": float(best["validation_ibs_ipcw"])})
    (out_root / "selected_hyperparameters.json").write_text(
        json.dumps(winners, indent=2) + "\n", encoding="utf-8"
    )
    (out_root / "resolved_tuning_config.json").write_text(
        json.dumps(tuning, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
