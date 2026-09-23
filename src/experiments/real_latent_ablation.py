"""Paired latent-dimension ablation of DVFM on a real cohort.

Real cohorts have no generating truth, so every metric is observed-data only:

* held-out log-likelihood of (t, delta). For ``latent_dim == 0`` it is exact;
  for a latent model the ELBO and IWAE are lower bounds, so a latent win
  against the exact latent-free likelihood is conservative;
* event-margin and censor-margin log-likelihoods, which show whether a gain
  comes from the death margin or only from fitting censoring;
* C-index and IBS-IPCW on the predicted marginal event survival. IPCW assumes
  censoring independent of death given covariates and is descriptive here.

Every latent dimension sees the same split and model seed within a repeat, so
differences are paired per repeat. A latent win shows the latent carries
signal the covariates do not; it is not a test of event-censoring dependence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import MissingIndicator

from dvfm.likelihood import (
    encode_posterior, event_martingale_residual, heldout_log_likelihood,
)
from dvfm.prediction import get_median_survival_time, predict_survival_curves
from utility.data import SurvivalData
from utility.metrics import censoring_rate, collect_metrics, spearman_correlation
from utility.runtime import seed_everything
from utility.semisynthetic import _semisynthetic_preprocessor, load_semisynthetic_source
from utility.splitting import time_event_stratified_split_indices

from .config import expand_seed_streams, semisynthetic_datasets


# Every latent dimension is compared with the latent-free, exact-likelihood fit.
REFERENCE_LATENT_DIM = 0
# (metric column, higher is better) for the paired latent-versus-reference table.
PAIRED_METRICS = {
    "test_joint_elbo": True,
    "test_joint_iwae": True,
    "test_event_margin_loglik": True,
    "test_censor_margin_loglik": True,
    "C-Idx": True,
    "CI IPCW": True,
    "IBS IPCW": False,
}


def load_real_cohort(spec: dict) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Read one configured cohort and return (frame, time, event)."""
    frame = load_semisynthetic_source(spec)
    time = pd.to_numeric(frame[spec["time_column"]], errors="coerce")
    keep = np.isfinite(time) & (time > 0)
    frame = frame.loc[keep].reset_index(drop=True)
    time = time.loc[keep].to_numpy(float)
    event = (pd.to_numeric(frame[spec["event_column"]], errors="coerce").fillna(0) > 0)
    return frame, time, event.astype(int).to_numpy()


def apply_missingness_filter(frame: pd.DataFrame, spec: dict) -> tuple[dict, pd.DataFrame]:
    """Drop configured features whose cohort-wide missing fraction exceeds the limit.

    Only covariate missingness is inspected, never outcomes, and it is computed
    once on the whole cohort so every repeat uses the same feature set.
    """
    features = spec["numeric_features"] + spec["categorical_features"]
    limit = spec.get("max_missing_fraction")
    missing = frame[features].isna().mean()
    report = pd.DataFrame({
        "feature": features, "missing_fraction": missing.to_numpy(),
        "kept": True if limit is None else (missing <= float(limit)).to_numpy(),
    })
    kept = set(report.loc[report.kept, "feature"])
    if not kept:
        raise ValueError(f"{spec['name']}: max_missing_fraction removes every feature")
    filtered = {
        **spec,
        "numeric_features": [f for f in spec["numeric_features"] if f in kept],
        "categorical_features": [f for f in spec["categorical_features"] if f in kept],
    }
    return filtered, report.assign(Dataset=spec["name"], max_missing_fraction=limit)


def _encode_split(frame, spec, train_idx, validation_idx, test_idx):
    """Impute, z-score and one-hot encode using the training partition only.

    With ``missing_indicators`` every covariate missing somewhere in the
    training partition also gets an unscaled 0/1 column, so imputed values
    remain distinguishable from observed ones.
    """
    preprocessor = _semisynthetic_preprocessor(
        spec["numeric_features"], spec["categorical_features"],
        numeric_imputation=spec.get("numeric_imputation", "mean"),
    )
    if spec.get("missing_indicators", False):
        preprocessor.transformers.append((
            "missing", MissingIndicator(features="missing-only", error_on_new=False),
            spec["numeric_features"] + spec["categorical_features"],
        ))
    preprocessor.fit(frame.iloc[train_idx])
    names = list(preprocessor.get_feature_names_out())
    return [
        np.asarray(preprocessor.transform(frame.iloc[idx]), dtype=np.float32)
        for idx in (train_idx, validation_idx, test_idx)
    ], names


def _partial_spearman(x, y, control) -> float:
    """Spearman correlation of x and y after removing a rank-linear control."""
    frame = pd.DataFrame({"x": x, "y": y, "c": control}).dropna()
    if len(frame) < 4:
        return np.nan
    ranks = frame.rank()
    residuals = {}
    for column in ("x", "y"):
        slope, intercept = np.polyfit(ranks["c"], ranks[column], 1)
        residuals[column] = ranks[column] - (slope * ranks["c"] + intercept)
    return float(np.corrcoef(residuals["x"], residuals["y"])[0, 1])


def _latent_external_rows(subjects: pd.DataFrame, external: list[str], context: dict) -> list[dict]:
    rows = []
    test = subjects.loc[subjects["split"].eq("test")]
    for subgroup, subset in (("All", test), ("Event observed", test[test.event == 1]),
                             ("Censored", test[test.event == 0])):
        for column in external:
            usable = subset.dropna(subset=[column, "z_mu_oriented", "martingale_residual"])
            rows.append({
                **context, "split": "test", "subgroup": subgroup, "external": column,
                "n": len(usable),
                "spearman_z_external": spearman_correlation(usable.z_mu_oriented, usable[column]),
                "spearman_martingale_external": spearman_correlation(
                    usable.martingale_residual, usable[column]
                ),
                "spearman_z_martingale": spearman_correlation(
                    usable.z_mu_oriented, usable.martingale_residual
                ),
                "partial_spearman_z_external_given_martingale": _partial_spearman(
                    usable.z_mu_oriented, usable[column], usable.martingale_residual
                ),
            })
    return rows


def _paired_differences(results: pd.DataFrame, reference_dim: int) -> pd.DataFrame:
    fitted = results.loc[~results["numerical_failure"].astype(bool)]
    keys = ["Dataset", "Repeat"]
    reference = fitted.loc[fitted.latent_dim.eq(reference_dim)].set_index(keys)
    rows = []
    for latent_dim, group in fitted.loc[fitted.latent_dim.ne(reference_dim)].groupby("latent_dim"):
        group = group.set_index(keys)
        common = group.index.intersection(reference.index)
        for metric, higher_is_better in PAIRED_METRICS.items():
            for key in common:
                value, base = group.at[key, metric], reference.at[key, metric]
                difference = value - base
                rows.append({
                    "Dataset": key[0], "Repeat": key[1], "latent_dim": int(latent_dim),
                    "reference_latent_dim": reference_dim, "metric": metric,
                    "value": value, "reference_value": base, "difference": difference,
                    "higher_is_better": higher_is_better,
                    "latent_better": bool(difference > 0 if higher_is_better else difference < 0),
                })
    return pd.DataFrame(rows)


def _paired_summary(paired: pd.DataFrame) -> pd.DataFrame:
    if paired.empty:
        return paired
    grouped = paired.groupby(
        ["Dataset", "latent_dim", "reference_latent_dim", "metric", "higher_is_better"]
    )
    summary = grouped["difference"].agg(["count", "mean", "std"]).reset_index()
    summary["se"] = summary["std"] / np.sqrt(summary["count"])
    summary["n_latent_better"] = grouped["latent_better"].sum().to_numpy()
    return summary.rename(columns={
        "count": "n_repeats", "mean": "mean_difference", "std": "sd_difference",
    })


def run_real_latent_ablation(cfg: dict, out_dir: Path, device: torch.device) -> pd.DataFrame:
    from .runner import (
        _models_for_dataset, evaluation_time_grid, fit_dvfm,
        scale_observed_time, training_time_scale,
    )

    study_cfg, eval_cfg = cfg["study"], cfg["evaluation"]
    likelihood_samples = int(eval_cfg["likelihood_samples"])
    # Natural time units. Held-out likelihoods treat anyone still event- and
    # censoring-free at the horizon as administratively censored there.
    horizon = eval_cfg.get("likelihood_horizon")
    latent_dims = [int(dim) for dim in cfg["models"]["dvfm"]["latent_dims"]]
    # The reference fit supplies the martingale residuals used by later dims.
    latent_dims = [REFERENCE_LATENT_DIM] + [
        dim for dim in latent_dims if dim != REFERENCE_LATENT_DIM
    ]
    rows, manifest, latent_rows, missingness = [], [], [], []
    for spec in semisynthetic_datasets(cfg["data"]):
        frame, time, event = load_real_cohort(spec)
        spec, report = apply_missingness_filter(frame, spec)
        missingness.append(report)
        dropped = report.loc[~report.kept, "feature"].tolist()
        if dropped:
            print(f"{spec['name']}: dropping features above max_missing_fraction: {dropped}")
        id_column = spec.get("id_column")
        subject_ids = frame[id_column].to_numpy() if id_column else np.arange(len(frame))
        external = list(spec.get("external_columns", []))
        cohort = SurvivalData(np.zeros((len(time), 0)), time, event, [])
        model_cfg = _models_for_dataset(cfg, spec["name"])
        for repeat, seeds in enumerate(expand_seed_streams(cfg["seeds"])):
            split_idx = time_event_stratified_split_indices(cohort, cfg["split"], int(seeds["split"]))
            (x_train, x_validation, x_test), names = _encode_split(frame, spec, *split_idx)
            train, validation, test = (
                SurvivalData(x, time[idx], event[idx], names)
                for x, idx in zip((x_train, x_validation, x_test), split_idx)
            )
            time_points = evaluation_time_grid(train, eval_cfg)
            time_scale = training_time_scale(train)
            model_train, model_validation, model_test = (
                scale_observed_time(part, time_scale) for part in (train, validation, test)
            )
            parts = dict(zip(("train", "validation", "test"), split_idx))
            subjects = pd.concat([
                pd.DataFrame({
                    "split": name, "row_index": idx, id_column or "row_id": subject_ids[idx],
                    "time": time[idx], "event": event[idx],
                    **{column: frame[column].to_numpy()[idx] for column in external},
                })
                for name, idx in parts.items()
            ], ignore_index=True)
            model_parts = {"train": model_train, "validation": model_validation, "test": model_test}
            base_context = {
                "Study": study_cfg["name"], "Stage": study_cfg.get("stage"),
                "Dataset Type": "real_latent_ablation", "Dataset": spec["name"],
                "Source Path": str(spec.get("path")), "Repeat": repeat, "Fold": 0,
                "Split Seed": int(seeds["split"]), "Model Seed": int(seeds["model"]),
                "Num Samples": len(time), "Num Features": len(names),
                "Train Size": len(train.time), "Validation Size": len(validation.time),
                "Test Size": len(test.time),
                "Censoring Rate Train": censoring_rate(train.event),
                "Censoring Rate Test": censoring_rate(test.event),
                "Training Time Scale": time_scale,
                "evaluation_time_horizon": float(time_points[-1]),
            }
            for latent_dim in latent_dims:
                context = {**base_context, "Model": "DVFM", "latent_dim": latent_dim}
                c = {**model_cfg["dvfm"], "latent_dim": latent_dim}
                seed_everything(int(seeds["model"]))
                try:
                    model, artifacts, train_loader, checkpoint = fit_dvfm(
                        model_train, model_validation, c, device
                    )
                    survival = predict_survival_curves(
                        model, model_test.X, time_points / time_scale, train_loader,
                        n_samples=int(c["mc_samples"]), device=device,
                    )
                    median = get_median_survival_time(survival, time_points / time_scale) * time_scale
                    metrics = collect_metrics(
                        "DVFM", median, survival, test.time, test.event, None, time_points,
                        t_train=train.time, e_train=train.event,
                    )
                    mixing_mu, mixing_std = encode_posterior(
                        model, model_train.X, model_train.time, model_train.event,
                        batch_size=int(c["batch_size"]), device=device,
                    )
                    likelihood_args = dict(
                        mixing_mu=mixing_mu, mixing_std=mixing_std,
                        n_samples=likelihood_samples, time_scale=time_scale,
                        seed=int(seeds["model"]), device=device,
                    )
                    likelihood = heldout_log_likelihood(
                        model, model_test.X, model_test.time, model_test.event,
                        horizon=None if horizon is None else float(horizon) / time_scale,
                        **likelihood_args,
                    )
                    untruncated = likelihood if horizon is None else heldout_log_likelihood(
                        model, model_test.X, model_test.time, model_test.event,
                        **likelihood_args,
                    )
                except Exception as exc:  # recorded so the pairing stays auditable
                    print(f"DVFM latent_dim={latent_dim} repeat={repeat} failed: {exc!r}")
                    rows.append({**context, "numerical_failure": True, "error": repr(exc)})
                    manifest.append({**context, "status": "failed", "error": repr(exc)})
                    continue
                subjects[f"test_joint_elbo_dz{latent_dim}"] = np.nan
                subjects.loc[subjects.split.eq("test"), f"test_joint_elbo_dz{latent_dim}"] = likelihood["joint_elbo"]
                if latent_dim == REFERENCE_LATENT_DIM:
                    for name, part in model_parts.items():
                        subjects.loc[subjects.split.eq(name), "martingale_residual"] = (
                            event_martingale_residual(model, part.X, part.time, part.event, device=device)
                        )
                if latent_dim == 1:
                    for name, part in model_parts.items():
                        mu, std = encode_posterior(
                            model, part.X, part.time, part.event,
                            batch_size=int(c["batch_size"]), device=device,
                        )
                        subjects.loc[subjects.split.eq(name), "z_mu"] = mu[:, 0]
                        subjects.loc[subjects.split.eq(name), "z_std"] = std[:, 0]
                clean = {key.removeprefix("DVFM "): value for key, value in metrics.items()}
                rows.append({
                    **context, **clean,
                    "checkpoint": checkpoint, "is_primary_checkpoint": True,
                    "prediction_mode": "aggregate_posterior",
                    "best_validation_elbo_epoch": artifacts["best_validation_elbo_epoch"],
                    "best_validation_loss_model_units": artifacts["best_validation_elbo"],
                    "numerically_invalid_epochs": artifacts["numerically_invalid_epochs"],
                    "likelihood_samples": 1 if latent_dim == 0 else likelihood_samples,
                    "likelihood_horizon": horizon,
                    "test_n_beyond_likelihood_horizon": (
                        0 if horizon is None else int(np.sum(test.time > float(horizon)))
                    ),
                    "test_joint_elbo_untruncated": float(np.mean(untruncated["joint_elbo"])),
                    "test_joint_elbo": float(np.mean(likelihood["joint_elbo"])),
                    "test_joint_iwae": float(np.mean(likelihood["joint_iwae"])),
                    "test_event_margin_loglik": float(np.mean(likelihood["event_margin"])),
                    "test_censor_margin_loglik": float(np.mean(likelihood["censor_margin"])),
                    "latent_loading_l1_magnitude": float(
                        model.decoder.latent_loading_l1().detach().cpu()
                    ),
                    "numerical_failure": False,
                })
                manifest.append({**context, "status": "complete", "error": None})
            if {"z_mu", "martingale_residual"} <= set(subjects.columns):
                # The latent's sign is arbitrary. Orient it on the training
                # split only, so that larger z means more deaths than the
                # covariate-only model expects; the external signal is unused.
                train_subjects = subjects.loc[subjects.split.eq("train")]
                sign = 1.0 if spearman_correlation(
                    train_subjects.z_mu, train_subjects.martingale_residual
                ) >= 0 else -1.0
                subjects["z_orientation_sign"] = sign
                subjects["z_mu_oriented"] = sign * subjects["z_mu"]
                latent_rows.extend(_latent_external_rows(
                    subjects, external, {**base_context, "z_orientation_sign": sign},
                ))
            subject_dir = out_dir / "subjects"
            subject_dir.mkdir(parents=True, exist_ok=True)
            subjects.to_csv(subject_dir / f"{spec['name']}_repeat_{repeat}.csv", index=False)

    results = pd.DataFrame(rows)
    pd.DataFrame(manifest).to_csv(out_dir / "run_manifest.csv", index=False)
    pd.concat(missingness).to_csv(out_dir / "feature_missingness.csv", index=False)
    paired = _paired_differences(results, REFERENCE_LATENT_DIM)
    paired.to_csv(out_dir / "paired_differences.csv", index=False)
    _paired_summary(paired).to_csv(out_dir / "paired_summary.csv", index=False)
    pd.DataFrame(latent_rows).to_csv(out_dir / "latent_external_validation.csv", index=False)
    if all(item["status"] == "complete" for item in manifest):
        (out_dir / "_SUCCESS").write_text("", encoding="utf-8")
    return results


__all__ = ["PAIRED_METRICS", "REFERENCE_LATENT_DIM", "apply_missingness_filter", "load_real_cohort", "run_real_latent_ablation"]
