import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from SurvivalEVAL.Evaluations.util import predict_multi_probs_from_curve

from VAE_montcarlo import (
    DVFM,
    SurvivalDataset,
    analyze_conditional_dependence,
    compute_ipcw_brier_ibs,
    compute_oracle_brier_ibs,
    generate_copula_data,
    get_median_survival_time,
    predict_survival_curves,
    train_deepsurv,
    train_dvfm,
    train_mtlr,
)


@dataclass
class ExperimentConfig:
    copulas: tuple = ("clayton_covdep",)  # includes non-standard latent frailty dependence
    theta_low: float = 1
    theta_high: float = 10.0
    n_samples: int = 5000
    n_features: int = 10
    test_size: float = 0.2
    repeats: int = 5

    latent_dim: int = 20
    latent_dim_grid: tuple = (20,)
    latent_sensitivity_runs: int = 10
    dvfm_beta_max: float = 1.0
    real_beta_grid: tuple = (0.2, 0.5, 1.0)
    real_beta_sensitivity_runs: int = 5
    dvfm_epochs: int = 300
    dvfm_lr: float = 5e-3
    dvfm_batch_size: int = 64
    mc_samples: int = 100

    deepsurv_epochs: int = 120
    deepsurv_lr: float = 1e-3
    mtlr_epochs: int = 120
    mtlr_lr: float = 5e-3
    mtlr_bins: int = 60
    copula_aft_epochs: int = 100
    copula_aft_lr: float = 5e-3

    n_time_points: int = 600
    max_time_factor: float = 1.5
    analyze_dependence: bool = True
    make_plots: bool = True
    output_prefix: str = "survival_results"
    real_output_prefix: str = "survival_realdata_results"
    real_time_normalize: bool = True
    real_time_norm_method: str = "train_max"
    real_dataset_names: tuple = ("ALS_PROACT", "Cancer_METABRIC", "MI_DEPENDENT")
    real_mtlr_bins_mode: str = "fixed"  # "fixed" or "sqrt_train"
    synthetic_sensitivity_output: str = "dvfm_latent_sensitivity_synthetic.xlsx"
    real_sensitivity_output: str = "dvfm_latent_sensitivity_real.xlsx"
    real_beta_sensitivity_output: str = "dvfm_real_beta_sensitivity.xlsx"


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_subset_mae(y_true, y_pred, mask):
    if np.sum(mask) == 0:
        return np.nan
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def _censoring_rate(event):
    event = np.asarray(event, dtype=float)
    if event.size == 0:
        return np.nan
    return float(1.0 - np.mean(event))


def _fit_cox_and_predict_survival(X_train, t_train, e_train, X_test, time_points):
    df_train = pd.DataFrame(X_train, columns=[f"X{i}" for i in range(X_train.shape[1])])
    df_train["time"] = t_train
    df_train["event"] = e_train

    cph = None
    for pen in [0.0, 0.01, 0.1, 1.0]:
        try:
            cph_try = CoxPHFitter(penalizer=pen)
            cph_try.fit(df_train, duration_col="time", event_col="event")
            cph = cph_try
            break
        except Exception:
            cph = None

    if cph is None:
        max_train_time = float(np.max(t_train))
        medians = np.full(len(X_test), max_train_time, dtype=float)
        surv_curves = np.ones((len(X_test), len(time_points)), dtype=float)
        return medians, surv_curves

    df_test = pd.DataFrame(X_test, columns=[f"X{i}" for i in range(X_test.shape[1])])
    partial_hazard = cph.predict_partial_hazard(df_test).values.flatten()

    baseline_survival = cph.baseline_survival_
    baseline_times = baseline_survival.index.values
    baseline_s = baseline_survival.values.flatten()

    baseline_interp = np.interp(time_points, baseline_times, baseline_s, left=1.0, right=baseline_s[-1])
    surv_curves = np.power(baseline_interp[None, :], partial_hazard[:, None])

    medians = np.zeros(len(X_test), dtype=float)
    max_train_time = float(np.max(t_train))
    for i in range(len(X_test)):
        idx = np.where(surv_curves[i] <= 0.5)[0]
        medians[i] = time_points[idx[0]] if len(idx) > 0 else max_train_time

    return medians, surv_curves


def _collect_metrics(
    prefix,
    medians,
    surv_curves,
    t_test,
    e_test,
    true_t_test,
    time_points,
    tau=None,
    t_train=None,
    e_train=None,
    true_t_train=None,
    true_c_train=None,
    dep_copula_name=None,
    dep_alpha=None,
):
    out = {}
    out[f"{prefix} C-Idx"] = float(concordance_index(t_test, medians, e_test))
    unc_mask = e_test == 1
    cen_mask = e_test == 0
    mae_target = t_test if true_t_test is None else true_t_test
    out[f"{prefix} MAE"] = _safe_subset_mae(mae_target, medians, unc_mask)  # kept for backward compatibility
    out[f"{prefix} MAE Uncensored"] = _safe_subset_mae(mae_target, medians, unc_mask)
    out[f"{prefix} MAE Censored"] = _safe_subset_mae(mae_target, medians, cen_mask)
    out[f"{prefix} MAE All"] = float(np.mean(np.abs(mae_target - medians)))
    out[f"{prefix} MAE Oracle"] = out[f"{prefix} MAE All"] if true_t_test is not None else np.nan

    if true_t_test is not None:
        _, ibs_oracle = compute_oracle_brier_ibs(surv_curves, time_points, true_t_test)
        out[f"{prefix} IBS Oracle"] = float(ibs_oracle)
    else:
        out[f"{prefix} IBS Oracle"] = np.nan
    _, ibs_ipcw, tau_used = compute_ipcw_brier_ibs(surv_curves, time_points, t_test, e_test, tau=tau)
    out[f"{prefix} IBS IPCW"] = float(ibs_ipcw)
    out["Eval Tau"] = float(tau_used)
    dep_metrics = _collect_dep_metrics(
        prefix=prefix,
        medians=medians,
        surv_curves=surv_curves,
        time_points=time_points,
        t_test=t_test,
        e_test=e_test,
        t_train=t_train,
        e_train=e_train,
        true_t_train=true_t_train,
        true_c_train=true_c_train,
        dep_copula_name=dep_copula_name,
        dep_alpha=dep_alpha,
    )
    out.update(dep_metrics)
    return out


def _median_from_survival(surv_curves, time_points, fallback_times):
    if surv_curves is None:
        return np.asarray(fallback_times, dtype=float)
    return get_median_survival_time(np.asarray(surv_curves, dtype=float), np.asarray(time_points, dtype=float))


def _build_km_area(times, indicators):
    try:
        from SurvivalEVAL.Evaluations.util import KaplanMeierArea
    except Exception:
        return None
    times = np.asarray(times, dtype=float)
    indicators = np.asarray(indicators, dtype=int)
    mask = np.isfinite(times) & np.isfinite(indicators) & (times > 0)
    if np.sum(mask) < 3:
        return None
    return KaplanMeierArea(times[mask], indicators[mask])


def _weighted_ci_dep(medians, t_test, e_test, g_hat):
    medians = np.asarray(medians, dtype=float)
    t_test = np.asarray(t_test, dtype=float)
    e_test = np.asarray(e_test, dtype=int)
    g_hat = np.asarray(g_hat, dtype=float)

    num = 0.0
    den = 0.0
    n = len(t_test)
    for i in range(n):
        if e_test[i] != 1:
            continue
        w_i = 1.0 / max(g_hat[i], 1e-8) ** 2
        for j in range(n):
            if t_test[i] < t_test[j]:
                den += w_i
                if medians[i] < medians[j]:
                    num += w_i
                elif medians[i] == medians[j]:
                    num += 0.5 * w_i
    return float(num / den) if den > 0 else np.nan


class _CopulaGraphic:
    """
    Copula-Graphic estimator (DependentEVAL-style) for dependent censoring.
    Supports: clayton, gumbel, frank.
    """

    def __init__(self, event_times, event_indicators, alpha=1.0, copula_type="clayton"):
        event_times = np.asarray(event_times, dtype=float)
        event_indicators = np.asarray(event_indicators, dtype=int)
        alpha = float(max(alpha, 1e-9))
        n_samples = len(event_times)

        order = np.lexsort((event_indicators, event_times))
        uniq_t, uniq_counts = np.unique(event_times[order], return_counts=True)
        self.survival_times = uniq_t
        self.population_count = np.flip(np.flip(uniq_counts).cumsum())

        event_counter = np.append(0, uniq_counts.cumsum()[:-1])
        idx = []
        for i in range(np.size(event_counter[:-1])):
            idx.append(event_counter[i])
            idx.append(event_counter[i + 1])
        idx.append(event_counter[-1])
        idx.append(len(event_indicators))
        events = np.add.reduceat(np.append(event_indicators[order], 0), idx)[::2]
        self.events = events

        event_diff = self.population_count - self.events
        with np.errstate(divide="ignore", invalid="ignore"):
            if copula_type == "clayton":
                diff_ = (event_diff / n_samples) ** (-alpha) - (self.population_count / n_samples) ** (-alpha)
                if len(diff_) > 0:
                    diff_[-1] = 0
                self.survival_probabilities = (1.0 + np.cumsum(diff_)) ** (-1.0 / alpha)
            elif copula_type == "gumbel":
                diff_ = ((-np.log(event_diff / n_samples)) ** (alpha + 1) -
                         (-np.log(self.population_count / n_samples)) ** (alpha + 1))
                if len(diff_) > 0:
                    diff_[-1] = 0
                self.survival_probabilities = np.exp(-np.cumsum(diff_) ** (1.0 / (1.0 + alpha)))
            elif copula_type == "frank":
                log_diff_ = np.log(
                    (np.exp(-alpha * event_diff / n_samples) - 1.0) /
                    (np.exp(-alpha * self.population_count / n_samples) - 1.0)
                )
                if len(log_diff_) > 0:
                    log_diff_[-1] = 0
                self.survival_probabilities = -1.0 / alpha * np.log(
                    1.0 + (np.exp(-alpha) - 1.0) * np.exp(np.cumsum(log_diff_))
                )
            else:
                raise ValueError(f"Unknown copula type: {copula_type}")

        self.survival_probabilities = np.asarray(self.survival_probabilities, dtype=float)
        if self.survival_probabilities.size > 0:
            self.survival_probabilities[0] = 1.0

    def predict(self, prediction_times):
        prediction_times = np.asarray(prediction_times, dtype=float)
        p_idx = np.digitize(prediction_times, self.survival_times)
        p_idx = np.where(p_idx == self.survival_times.size + 1, p_idx - 1, p_idx)
        return np.append(1.0, self.survival_probabilities)[p_idx]


class _CopulaGraphicWrapper:
    def __init__(self, event_times, event_indicators, copula_name="clayton", alpha=1.0):
        self.cg = _CopulaGraphic(event_times, event_indicators, alpha=alpha, copula_type=copula_name)
        event_times = np.asarray(event_times, dtype=float)
        event_indicators = np.asarray(event_indicators, dtype=int)

        order = np.lexsort((event_indicators, event_times))
        uniq_t = np.unique(event_times[order], return_counts=True)[0]
        self.survival_times = uniq_t
        self.survival_probabilities = self.cg.predict(self.survival_times)
        if len(self.survival_probabilities) > 0:
            self.survival_probabilities[-1] = 0.0

        area_prob = np.append(1.0, self.survival_probabilities)
        area_times = np.append(0.0, self.survival_times)
        if len(area_times) > 1:
            denom = (area_prob[-1] - 1.0) / max(area_times[-1], 1e-12)
            self.cg_linear_zero = -1.0 / denom if denom != 0 else np.inf
        else:
            self.cg_linear_zero = np.inf

        if len(self.survival_probabilities) > 0 and self.survival_probabilities[-1] != 0:
            area_times = np.append(area_times, self.cg_linear_zero)
            area_prob = np.append(area_prob, 0.0)

        area_diff = np.diff(area_times)
        avg_prob = (area_prob[:-1] + area_prob[1:]) / 2.0
        area = np.flip(np.flip(area_diff * avg_prob).cumsum())
        self.area_times = np.append(area_times, np.inf).astype(float)
        self.area_probabilities = np.asarray(area_prob, dtype=float)
        self.area = np.append(area, 0.0).astype(float)

    def predict(self, prediction_times):
        return self.cg.predict(prediction_times)

    def best_guess(self, censor_times):
        censor_times = np.asarray(censor_times, dtype=float)
        if censor_times.size == 0:
            return censor_times
        slope = (1.0 - float(np.min(self.survival_probabilities))) / (0.0 - float(np.max(self.survival_times)))

        before_last = censor_times <= np.max(self.survival_times)
        after_last = ~before_last
        surv_prob = np.empty_like(censor_times, dtype=float)
        surv_prob[after_last] = 1.0 + censor_times[after_last] * slope
        surv_prob[before_last] = self.predict(censor_times[before_last])
        surv_prob = np.clip(surv_prob, 1e-10, None)

        c_idx = np.digitize(censor_times, self.area_times)
        c_idx = np.where(c_idx == self.area_times.size + 1, c_idx - 1, c_idx)
        beyond = c_idx > len(self.area_times) - 2
        censor_area = np.zeros_like(censor_times, dtype=float)
        keep = ~beyond
        censor_area[keep] = (
            (self.area_times[c_idx[keep]] - censor_times[keep]) *
            (self.area_probabilities[c_idx[keep]] + surv_prob[keep]) * 0.5
        )
        censor_area[keep] += self.area[c_idx[keep]]
        return censor_times + censor_area / surv_prob


def _predict_event_probs_from_curve(surv_curve, curve_times, eval_times):
    """Predict event probabilities F(t)=1-S(t) at eval_times using SurvivalEVAL."""
    surv_curve = np.asarray(surv_curve, dtype=float)
    curve_times = np.asarray(curve_times, dtype=float)
    eval_times = np.asarray(eval_times, dtype=float)
    surv_probs = predict_multi_probs_from_curve(surv_curve, curve_times, eval_times, interpolation="Linear")
    return 1.0 - np.asarray(surv_probs, dtype=float)


def _ibs_dep_copula_graphic(
    surv_curves,
    time_points,
    t_test,
    e_test,
    t_train,
    e_train,
    copula_name="clayton",
    alpha=1.0,
    num_points=10,
    interpolation="Linear",
    uncertainty_weighting=True,
):
    t_test = np.asarray(t_test, dtype=float)
    e_test = np.asarray(e_test, dtype=int)
    t_train = np.asarray(t_train, dtype=float)
    e_train = np.asarray(e_train, dtype=int)
    surv_curves = np.asarray(surv_curves, dtype=float)
    time_points = np.asarray(time_points, dtype=float)

    max_target_time = float(np.max(np.concatenate((t_test, t_train))))
    if max_target_time <= 0:
        return np.nan
    eval_times = np.linspace(0.0, max_target_time, int(max(2, num_points)))

    predict_probs_mat = np.vstack([
        _predict_event_probs_from_curve(surv_curves[i], time_points, eval_times)
        for i in range(surv_curves.shape[0])
    ])

    censored_mask = ~e_test.astype(bool)
    censored_times = t_test[censored_mask]

    cg_wrapper = _CopulaGraphicWrapper(
        t_train,
        e_train,
        copula_name=copula_name,
        alpha=alpha,
    )
    event_times_bg = t_test.copy()
    if censored_times.size > 0:
        event_times_bg[censored_mask] = cg_wrapper.best_guess(censored_times)

    target_times_mat = np.repeat(eval_times.reshape(1, -1), repeats=len(t_test), axis=0)
    event_times_mat = np.repeat(event_times_bg.reshape(-1, 1), repeats=len(eval_times), axis=1)

    weight_cat1 = (event_times_mat <= target_times_mat).astype(float)
    weight_cat2 = (event_times_mat > target_times_mat).astype(float)

    if uncertainty_weighting:
        if censored_times.size > 0:
            cg_uncert = _CopulaGraphic(t_train, e_train, alpha=alpha, copula_type=copula_name)
            s_e = cg_uncert.predict(censored_times)
            w_c = np.clip(1.0 - s_e, 0.0, 1.0)
        else:
            w_c = np.array([])
        w_row = np.ones(len(t_test), dtype=float)
        if w_c.size > 0:
            w_row[censored_mask] = w_c
        w_row /= (w_row.mean() + 1e-12)
        w_mat = np.repeat(w_row.reshape(-1, 1), repeats=len(eval_times), axis=1)
        weight_cat1 = weight_cat1 * w_mat
        weight_cat2 = weight_cat2 * w_mat

    se_mat = (
        np.square(predict_probs_mat) * weight_cat1
        + np.square(1.0 - predict_probs_mat) * weight_cat2
    )
    brier_t = np.mean(se_mat, axis=0)
    return float(np.trapz(brier_t, eval_times) / max_target_time)


def _collect_dep_metrics(
    prefix,
    medians,
    surv_curves,
    time_points,
    t_test,
    e_test,
    t_train=None,
    e_train=None,
    true_t_train=None,
    true_c_train=None,
    dep_copula_name=None,
    dep_alpha=None,
    dep_num_points=10,
):
    out = {
        f"{prefix} CI-DEP": np.nan,
        f"{prefix} IBS-DEP": np.nan,
        f"{prefix} MAE-DEP": np.nan,
    }
    if t_train is None or e_train is None:
        return out

    # Practical estimator choice:
    # - Synthetic: if true marginals are available, use them.
    # - Real: fallback to KM-based marginals from observed data.
    event_model = _build_km_area(true_t_train, np.ones_like(true_t_train)) if true_t_train is not None else None
    if event_model is None:
        event_model = _build_km_area(t_train, e_train)
    if event_model is None:
        return out

    censor_model = _build_km_area(true_c_train, np.ones_like(true_c_train)) if true_c_train is not None else None
    if censor_model is None:
        censor_model = _build_km_area(t_train, 1 - np.asarray(e_train, dtype=int))
    if censor_model is None:
        return out

    t_test = np.asarray(t_test, dtype=float)
    e_test = np.asarray(e_test, dtype=int)
    medians = np.asarray(medians, dtype=float)
    surv_curves = np.asarray(surv_curves, dtype=float)
    time_points = np.asarray(time_points, dtype=float)

    # DependentEVAL-style IBS-DEP with Copula-Graphic best-guess estimator.
    copula_name = dep_copula_name if dep_copula_name is not None else "clayton"
    alpha = 1.0 if dep_alpha is None else float(dep_alpha)
    supported_copulas = {"clayton", "gumbel", "frank"}
    compute_dep_ibs = copula_name in supported_copulas

    if compute_dep_ibs:
        try:
            out[f"{prefix} IBS-DEP"] = _ibs_dep_copula_graphic(
                surv_curves=surv_curves,
                time_points=time_points,
                t_test=t_test,
                e_test=e_test,
                t_train=np.asarray(t_train, dtype=float),
                e_train=np.asarray(e_train, dtype=int),
                copula_name=copula_name,
                alpha=alpha,
                num_points=dep_num_points,
                interpolation="Linear",
                uncertainty_weighting=True,
            )
        except Exception:
            out[f"{prefix} IBS-DEP"] = np.nan
    elif dep_copula_name is None:
        # Real-data fallback when the dependency copula is unknown.
        try:
            out[f"{prefix} IBS-DEP"] = _ibs_dep_copula_graphic(
                surv_curves=surv_curves,
                time_points=time_points,
                t_test=t_test,
                e_test=e_test,
                t_train=np.asarray(t_train, dtype=float),
                e_train=np.asarray(e_train, dtype=int),
                copula_name="clayton",
                alpha=1.0,
                num_points=dep_num_points,
                interpolation="Linear",
                uncertainty_weighting=True,
            )
        except Exception:
            out[f"{prefix} IBS-DEP"] = np.nan

    # 1) CI-DEP
    g_hat = np.clip(np.asarray(censor_model.predict(t_test), dtype=float), 1e-8, 1.0)
    out[f"{prefix} CI-DEP"] = _weighted_ci_dep(medians, t_test, e_test, g_hat)

    # 2) MAE-DEP
    t_tilde = t_test.copy()
    cen_mask = e_test == 0
    if np.any(cen_mask):
        t_tilde[cen_mask] = event_model.best_guess(t_test[cen_mask])
    w = np.ones_like(t_test, dtype=float)
    if np.any(cen_mask):
        s_event_at_censor = np.clip(np.asarray(event_model.predict(t_test[cen_mask]), dtype=float), 0.0, 1.0)
        w[cen_mask] = np.clip(1.0 - s_event_at_censor, 1e-8, 1.0)
    mae_dep = np.average(np.abs(t_tilde - medians), weights=w)
    out[f"{prefix} MAE-DEP"] = float(mae_dep)
    return out


def _to_binary_event(values):
    s = pd.Series(values)
    if pd.api.types.is_bool_dtype(s):
        return s.astype(int).values
    if pd.api.types.is_numeric_dtype(s):
        return (pd.to_numeric(s, errors="coerce").fillna(0) > 0).astype(int).values

    mapped = (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "1": 1,
                "true": 1,
                "yes": 1,
                "y": 1,
                "dead": 1,
                "event": 1,
                "0": 0,
                "false": 0,
                "no": 0,
                "n": 0,
                "alive": 0,
                "censored": 0,
            }
        )
        .fillna(0)
        .astype(int)
    )
    return mapped.values


def _prepare_real_dataset(file_path, time_col, event_col):
    file_path = Path(file_path)
    if file_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)

    needed = [time_col, event_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {file_path}")

    df = df.copy()
    t = pd.to_numeric(df[time_col], errors="coerce")
    e = pd.Series(_to_binary_event(df[event_col]))
    valid = t.notna() & (t > 0) & e.notna()
    df = df.loc[valid].copy()
    t = t.loc[valid].values.astype(float)
    e = e.loc[valid].values.astype(int)

    x_df = df.drop(columns=[time_col, event_col])
    x_df = pd.get_dummies(x_df, drop_first=True)
    x_df = x_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x = x_df.values.astype(float)
    return x, t, e


class ClaytonWeibullAFT(nn.Module):
    """
    Parametric copula model for dependent censoring:
    - Weibull margins for T and C with log-scale linear in X
    - Clayton copula links margins
    """
    def __init__(self, n_features):
        super().__init__()
        self.beta_t = nn.Parameter(torch.zeros(n_features + 1))
        self.beta_c = nn.Parameter(torch.zeros(n_features + 1))
        self.log_shape_t = nn.Parameter(torch.tensor(0.0))
        self.log_shape_c = nn.Parameter(torch.tensor(0.0))
        self.raw_theta = nn.Parameter(torch.tensor(0.2))

    def _margin_stats(self, x_aug, t):
        eps = 1e-8
        t = torch.clamp(t, min=eps)

        shape_t = torch.exp(self.log_shape_t) + eps
        shape_c = torch.exp(self.log_shape_c) + eps
        scale_t = torch.exp(x_aug @ self.beta_t) + eps
        scale_c = torch.exp(x_aug @ self.beta_c) + eps

        z_t = torch.clamp(t / scale_t, min=eps)
        z_c = torch.clamp(t / scale_c, min=eps)
        h_t = z_t ** shape_t
        h_c = z_c ** shape_c

        s_t = torch.exp(-h_t)
        s_c = torch.exp(-h_c)
        f_t = (shape_t / scale_t) * (z_t ** (shape_t - 1.0)) * s_t
        f_c = (shape_c / scale_c) * (z_c ** (shape_c - 1.0)) * s_c
        u = torch.clamp(1.0 - s_t, min=eps, max=1.0 - eps)
        v = torch.clamp(1.0 - s_c, min=eps, max=1.0 - eps)
        return f_t, f_c, u, v, shape_t, scale_t

    def neg_log_lik(self, x_aug, t_obs, event):
        eps = 1e-8
        f_t, f_c, u, v, _, _ = self._margin_stats(x_aug, t_obs)
        theta = torch.nn.functional.softplus(self.raw_theta) + 1e-4

        a = torch.clamp(u ** (-theta) + v ** (-theta) - 1.0, min=eps)
        c_pow = a ** (-1.0 / theta - 1.0)
        dC_du = c_pow * (u ** (-theta - 1.0))
        dC_dv = c_pow * (v ** (-theta - 1.0))

        l_event = f_t * torch.clamp(1.0 - dC_du, min=eps)
        l_cens = f_c * torch.clamp(1.0 - dC_dv, min=eps)
        l = event * l_event + (1.0 - event) * l_cens
        return -torch.mean(torch.log(torch.clamp(l, min=eps)))

    def predict_survival(self, x, time_points):
        eps = 1e-8
        x = np.asarray(x, dtype=np.float32)
        x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)
        x_t = torch.from_numpy(x_aug)

        with torch.no_grad():
            scale_t = torch.exp(x_t @ self.beta_t).cpu().numpy() + eps
            shape_t = float((torch.exp(self.log_shape_t) + eps).cpu().item())

        t = np.asarray(time_points, dtype=float)[None, :]
        z = np.maximum(t / scale_t[:, None], eps)
        surv = np.exp(-(z ** shape_t))
        return surv


def fit_clayton_weibull_aft(X_train, t_train, e_train, X_test, time_points, epochs=400, lr=5e-3, device="cpu"):
    x_train = np.asarray(X_train, dtype=np.float32)
    x_aug = np.concatenate([x_train, np.ones((x_train.shape[0], 1), dtype=np.float32)], axis=1)
    t_obs = np.asarray(t_train, dtype=np.float32)
    e_obs = np.asarray(e_train, dtype=np.float32)

    x_t = torch.from_numpy(x_aug).to(device)
    t_t = torch.from_numpy(t_obs).to(device)
    e_t = torch.from_numpy(e_obs).to(device)

    model = ClaytonWeibullAFT(n_features=X_train.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        optimizer.zero_grad()
        loss = model.neg_log_lik(x_t, t_t, e_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

    surv = model.predict_survival(X_test, time_points)
    medians = np.zeros(len(X_test), dtype=float)
    max_train_time = float(np.max(t_train))
    for i in range(len(X_test)):
        idx = np.where(surv[i] <= 0.5)[0]
        medians[i] = time_points[idx[0]] if len(idx) > 0 else max_train_time

    return medians, surv


def run_experiment_v2(cfg: ExperimentConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    rows = []
    # Restricted to low dependence for this experiment run.
    dep_settings = [("low", cfg.theta_low)]

    for copula in cfg.copulas:
        for dep_label, theta_val in dep_settings:
            for rep in range(cfg.repeats):
                seed = 42 + rep
                _seed_everything(seed)

                X, obs_time, event, true_t, true_c = generate_copula_data(
                    n_samples=cfg.n_samples,
                    n_features=cfg.n_features,
                    copula_type=copula,
                    theta=theta_val,
                    seed=seed,
                )

                (
                    X_train,
                    X_test,
                    t_train,
                    t_test,
                    e_train,
                    e_test,
                    true_t_train,
                    true_t_test,
                    true_c_train,
                    true_c_test,
                ) = train_test_split(X, obs_time, event, true_t, true_c, test_size=cfg.test_size, random_state=seed)

                max_time = float(np.max(t_train) * cfg.max_time_factor)
                time_points = np.linspace(0.0, max_time, cfg.n_time_points)

                # CoxPH
                cox_median_raw, cox_surv = _fit_cox_and_predict_survival(
                    X_train, t_train, e_train, X_test, time_points
                )
                cox_median = _median_from_survival(cox_surv, time_points, cox_median_raw)

                # DeepSurv
                _, ds_median_raw, ds_surv = train_deepsurv(
                    X_train,
                    t_train,
                    e_train,
                    X_test,
                    n_epochs=cfg.deepsurv_epochs,
                    lr=cfg.deepsurv_lr,
                    device=device,
                    eval_time_points=time_points,
                )
                ds_median = _median_from_survival(ds_surv, time_points, ds_median_raw)

                # MTLR
                _, mtlr_median_raw, mtlr_surv = train_mtlr(
                    X_train,
                    t_train,
                    e_train,
                    X_test,
                    num_bins=cfg.mtlr_bins,
                    n_epochs=cfg.mtlr_epochs,
                    lr=cfg.mtlr_lr,
                    device=device,
                    eval_time_points=time_points,
                )
                mtlr_median = _median_from_survival(mtlr_surv, time_points, mtlr_median_raw)

                # Copula baseline (Clayton-Weibull AFT)
                copula_median_raw, copula_surv = fit_clayton_weibull_aft(
                    X_train,
                    t_train,
                    e_train,
                    X_test,
                    time_points,
                    epochs=cfg.copula_aft_epochs,
                    lr=cfg.copula_aft_lr,
                    device=device,
                )
                copula_median = _median_from_survival(copula_surv, time_points, copula_median_raw)

                # DVFM
                train_ds = SurvivalDataset(X_train, t_train, e_train)
                val_ds = SurvivalDataset(X_test, t_test, e_test)
                train_loader = DataLoader(train_ds, batch_size=cfg.dvfm_batch_size, shuffle=True)
                val_loader = DataLoader(val_ds, batch_size=cfg.dvfm_batch_size, shuffle=False)

                dvfm = DVFM(input_dim=cfg.n_features, latent_dim=cfg.latent_dim).to(device)
                train_dvfm(
                    dvfm,
                    train_loader,
                    val_loader,
                    n_epochs=cfg.dvfm_epochs,
                    lr=cfg.dvfm_lr,
                    beta_max=cfg.dvfm_beta_max,
                    free_bits=0.0,
                    device=device,
                )

                dvfm_surv = predict_survival_curves(
                    model=dvfm,
                    X=X_test,
                    time_points=time_points,
                    train_loader=train_loader,
                    n_samples=cfg.mc_samples,
                    device=device,
                )
                dvfm_median = _median_from_survival(dvfm_surv, time_points, t_test)

                row = {
                    "Copula": copula,
                    "Dependence": dep_label,
                    "Theta": float(theta_val),
                    "Repeat": rep,
                    "Seed": seed,
                    "Dataset Type": "synthetic",
                    "Dataset": f"{copula}_{dep_label}",
                    "Censoring Rate All": _censoring_rate(event),
                    "Censoring Rate Train": _censoring_rate(e_train),
                    "Censoring Rate Test": _censoring_rate(e_test),
                }

                row.update(
                    _collect_metrics(
                        "CoxPH",
                        cox_median,
                        cox_surv,
                        t_test,
                        e_test,
                        true_t_test,
                        time_points,
                        tau=None,
                        t_train=t_train,
                        e_train=e_train,
                        true_t_train=true_t_train,
                        true_c_train=true_c_train,
                        dep_copula_name=copula,
                        dep_alpha=theta_val,
                    )
                )
                tau_eval = row["Eval Tau"]
                row.update(
                    _collect_metrics(
                        "DeepSurv",
                        ds_median,
                        ds_surv,
                        t_test,
                        e_test,
                        true_t_test,
                        time_points,
                        tau=tau_eval,
                        t_train=t_train,
                        e_train=e_train,
                        true_t_train=true_t_train,
                        true_c_train=true_c_train,
                        dep_copula_name=copula,
                        dep_alpha=theta_val,
                    )
                )
                row.update(
                    _collect_metrics(
                        "MTLR",
                        mtlr_median,
                        mtlr_surv,
                        t_test,
                        e_test,
                        true_t_test,
                        time_points,
                        tau=tau_eval,
                        t_train=t_train,
                        e_train=e_train,
                        true_t_train=true_t_train,
                        true_c_train=true_c_train,
                        dep_copula_name=copula,
                        dep_alpha=theta_val,
                    )
                )
                row.update(
                    _collect_metrics(
                        "ClaytonAFT",
                        copula_median,
                        copula_surv,
                        t_test,
                        e_test,
                        true_t_test,
                        time_points,
                        tau=tau_eval,
                        t_train=t_train,
                        e_train=e_train,
                        true_t_train=true_t_train,
                        true_c_train=true_c_train,
                        dep_copula_name=copula,
                        dep_alpha=theta_val,
                    )
                )
                row.update(
                    _collect_metrics(
                        "DVFM",
                        dvfm_median,
                        dvfm_surv,
                        t_test,
                        e_test,
                        true_t_test,
                        time_points,
                        tau=tau_eval,
                        t_train=t_train,
                        e_train=e_train,
                        true_t_train=true_t_train,
                        true_c_train=true_c_train,
                        dep_copula_name=copula,
                        dep_alpha=theta_val,
                    )
                )

                if cfg.analyze_dependence:
                    tau_true, tau_x, tau_xz = analyze_conditional_dependence(
                        dvfm, X_test, copula_type=copula, theta=theta_val, device=device
                    )
                    row["True Tau"] = float(tau_true)
                    row["Tau(T,C|X)"] = float(tau_x)
                    row["Tau(T,C|X,Z)"] = float(tau_xz)
                    row["Tau Error |X|"] = float(abs(tau_x - tau_true))

                rows.append(row)
                print(
                    f"[{copula} | dep={dep_label} | theta={theta_val:.3f} | rep={rep}] "
                    f"DVFM IBS_IPCW={row['DVFM IBS IPCW']:.4f}, "
                    f"DVFM IBS_Oracle={row['DVFM IBS Oracle']:.4f}, "
                    f"DVFM C-Idx={row['DVFM C-Idx']:.4f}"
                )

    res = pd.DataFrame(rows)
    metric_cols = [c for c in res.columns if c not in ["Copula", "Dependence", "Theta", "Repeat", "Seed"]]

    summary_mean = res.groupby(["Copula", "Dependence", "Theta"])[metric_cols].mean().reset_index()
    summary_std = res.groupby(["Copula", "Dependence", "Theta"])[metric_cols].std().reset_index()

    print("\n" + "=" * 60)
    print("RAW RESULTS")
    print("=" * 60)
    print(res.round(4).to_string(index=False))

    print("\n" + "=" * 60)
    print("MEAN ACROSS REPEATS")
    print("=" * 60)
    print(summary_mean.round(4).to_string(index=False))

    print("\n" + "=" * 60)
    print("STD ACROSS REPEATS")
    print("=" * 60)
    print(summary_std.round(4).fillna(0.0).to_string(index=False))

    out_raw = Path(f"{cfg.output_prefix}_raw.csv")
    out_mean = Path(f"{cfg.output_prefix}_mean.csv")
    out_std = Path(f"{cfg.output_prefix}_std.csv")
    res.to_csv(out_raw, index=False)
    summary_mean.to_csv(out_mean, index=False)
    summary_std.to_csv(out_std, index=False)
    print("\nSaved CSV files:")
    print(f"  {out_raw.resolve()}")
    print(f"  {out_mean.resolve()}")
    print(f"  {out_std.resolve()}")

    if cfg.make_plots:
        _plot_summary(summary_mean)

    return res, summary_mean, summary_std


def run_real_data_experiments(cfg: ExperimentConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for real datasets: {device}")

    dataset_specs = [
        {
            "name": "ALS_PROACT",
            "path": Path("proact_processed_mod.xlsx"),
            "time_col": "merged_time",
            "event_col": "Status",
        },
        {
            "name": "Cancer_METABRIC",
            "path": Path("CensoredMAE-main/data/Metabric/Metabric.csv"),
            "time_col": "time",
            "event_col": "delta",
        },
        {
            "name": "MI_DEPENDENT",
            "path": Path("mi_dependent_survival.xlsx"),
            "time_col": "merged_time",
            "event_col": "Status",
        },
    ]

    rows = []
    for spec in dataset_specs:
        if spec["name"] not in set(cfg.real_dataset_names):
            continue
        if not spec["path"].exists():
            print(f"Skipping {spec['name']}: file not found at {spec['path']}")
            continue

        print(f"\nProcessing real dataset: {spec['name']} ({spec['path']})")
        X, obs_time, event = _prepare_real_dataset(spec["path"], spec["time_col"], spec["event_col"])
        if len(X) < 50 or len(np.unique(event)) < 2:
            print(f"Skipping {spec['name']}: not enough valid samples or event class imbalance")
            continue

        seed = 42
        _seed_everything(seed)
        stratify_arg = event if len(np.unique(event)) > 1 else None
        X_train, X_test, t_train, t_test, e_train, e_test = train_test_split(
            X,
            obs_time,
            event,
            test_size=cfg.test_size,
            random_state=seed,
            stratify=stratify_arg,
        )

        time_scale = 1.0
        if cfg.real_time_normalize:
            if cfg.real_time_norm_method == "train_max":
                time_scale = float(np.max(t_train))
            else:
                raise ValueError(f"Unsupported real_time_norm_method: {cfg.real_time_norm_method}")
            time_scale = max(time_scale, 1e-8)
            t_train = t_train / time_scale
            t_test = t_test / time_scale

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        max_time = float(np.max(t_train) * cfg.max_time_factor)
        time_points = np.linspace(0.0, max_time, cfg.n_time_points)

        cox_median_raw, cox_surv = _fit_cox_and_predict_survival(X_train, t_train, e_train, X_test, time_points)
        cox_median = _median_from_survival(cox_surv, time_points, cox_median_raw)

        _, ds_median_raw, ds_surv = train_deepsurv(
            X_train,
            t_train,
            e_train,
            X_test,
            n_epochs=cfg.deepsurv_epochs,
            lr=cfg.deepsurv_lr,
            device=device,
            eval_time_points=time_points,
        )
        ds_median = _median_from_survival(ds_surv, time_points, ds_median_raw)

        mtlr_bins_fold = cfg.mtlr_bins
        if cfg.real_mtlr_bins_mode == "sqrt_train":
            mtlr_bins_fold = max(2, int(np.floor(np.sqrt(len(t_train)))))

        _, mtlr_median_raw, mtlr_surv = train_mtlr(
            X_train,
            t_train,
            e_train,
            X_test,
            num_bins=mtlr_bins_fold,
            n_epochs=cfg.mtlr_epochs,
            lr=cfg.mtlr_lr,
            device=device,
            eval_time_points=time_points,
        )
        mtlr_median = _median_from_survival(mtlr_surv, time_points, mtlr_median_raw)

        copula_median_raw, copula_surv = fit_clayton_weibull_aft(
            X_train,
            t_train,
            e_train,
            X_test,
            time_points,
            epochs=cfg.copula_aft_epochs,
            lr=cfg.copula_aft_lr,
            device=device,
        )
        copula_median = _median_from_survival(copula_surv, time_points, copula_median_raw)

        train_ds = SurvivalDataset(X_train, t_train, e_train)
        val_ds = SurvivalDataset(X_test, t_test, e_test)
        train_loader = DataLoader(train_ds, batch_size=cfg.dvfm_batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.dvfm_batch_size, shuffle=False)

        dvfm = DVFM(input_dim=X_train.shape[1], latent_dim=cfg.latent_dim).to(device)
        train_dvfm(
            dvfm,
            train_loader,
            val_loader,
            n_epochs=cfg.dvfm_epochs,
            lr=cfg.dvfm_lr,
            beta_max=cfg.dvfm_beta_max,
            free_bits=0.0,
            device=device,
        )

        dvfm_surv = predict_survival_curves(
            model=dvfm,
            X=X_test,
            time_points=time_points,
            train_loader=train_loader,
            n_samples=cfg.mc_samples,
            device=device,
        )
        dvfm_median = _median_from_survival(dvfm_surv, time_points, t_test)

        row = {
            "Dataset Type": "real",
            "Dataset": spec["name"],
            "Source Path": str(spec["path"]),
            "Split": "80/20",
            "Time Normalized": bool(cfg.real_time_normalize),
            "Time Norm Method": cfg.real_time_norm_method if cfg.real_time_normalize else "none",
            "Time Scale": float(time_scale),
            "Copula": np.nan,
            "Dependence": "real",
            "Theta": np.nan,
            "Repeat": 0,
            "Seed": seed,
            "Num Samples": int(len(X)),
            "Num Features": int(X.shape[1]),
            "MTLR Bins Used": int(mtlr_bins_fold),
            "Event Rate All": float(np.mean(event)),
            "Event Rate Train": float(np.mean(e_train)),
            "Event Rate Test": float(np.mean(e_test)),
            "Censoring Rate All": _censoring_rate(event),
            "Censoring Rate Train": _censoring_rate(e_train),
            "Censoring Rate Test": _censoring_rate(e_test),
        }

        row.update(
            _collect_metrics(
                "CoxPH",
                cox_median,
                cox_surv,
                t_test,
                e_test,
                None,
                time_points,
                tau=None,
                t_train=t_train,
                e_train=e_train,
            )
        )
        tau_eval = row["Eval Tau"]
        row.update(
            _collect_metrics(
                "DeepSurv",
                ds_median,
                ds_surv,
                t_test,
                e_test,
                None,
                time_points,
                tau=tau_eval,
                t_train=t_train,
                e_train=e_train,
            )
        )
        row.update(
            _collect_metrics(
                "MTLR",
                mtlr_median,
                mtlr_surv,
                t_test,
                e_test,
                None,
                time_points,
                tau=tau_eval,
                t_train=t_train,
                e_train=e_train,
            )
        )
        row.update(
            _collect_metrics(
                "ClaytonAFT",
                copula_median,
                copula_surv,
                t_test,
                e_test,
                None,
                time_points,
                tau=tau_eval,
                t_train=t_train,
                e_train=e_train,
            )
        )
        row.update(
            _collect_metrics(
                "DVFM",
                dvfm_median,
                dvfm_surv,
                t_test,
                e_test,
                None,
                time_points,
                tau=tau_eval,
                t_train=t_train,
                e_train=e_train,
            )
        )
        rows.append(row)

        print(
            f"[{spec['name']} | split=80/20] "
            f"Censoring(Test)={row['Censoring Rate Test']:.3f}, "
            f"DVFM IBS_IPCW={row['DVFM IBS IPCW']:.4f}, "
            f"DVFM C-Idx={row['DVFM C-Idx']:.4f}"
        )

    if not rows:
        print("No real datasets were processed. No real-data CSV files written.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    res = pd.DataFrame(rows)
    group_cols = ["Dataset"]
    metric_cols = [c for c in res.columns if c not in ["Dataset Type", "Dataset", "Source Path", "Copula", "Dependence", "Theta", "Repeat", "Seed"]]
    summary_mean = res.groupby(group_cols)[metric_cols].mean(numeric_only=True).reset_index()
    summary_std = res.groupby(group_cols)[metric_cols].std(numeric_only=True).reset_index()

    out_raw = Path(f"{cfg.real_output_prefix}_raw.csv")
    out_mean = Path(f"{cfg.real_output_prefix}_mean.csv")
    out_std = Path(f"{cfg.real_output_prefix}_std.csv")
    res.to_csv(out_raw, index=False)
    summary_mean.to_csv(out_mean, index=False)
    summary_std.to_csv(out_std, index=False)

    print("\nSaved real-data CSV files:")
    print(f"  {out_raw.resolve()}")
    print(f"  {out_mean.resolve()}")
    print(f"  {out_std.resolve()}")

    return res, summary_mean, summary_std


def _parse_latent_dim_grid(latent_dim_grid) -> List[int]:
    if isinstance(latent_dim_grid, str):
        parts = [p.strip() for p in latent_dim_grid.split(",") if p.strip()]
        return [int(p) for p in parts]
    return [int(v) for v in latent_dim_grid]


def _parse_float_grid(value_grid) -> List[float]:
    if isinstance(value_grid, str):
        parts = [p.strip() for p in value_grid.split(",") if p.strip()]
        return [float(p) for p in parts]
    return [float(v) for v in value_grid]


def _run_dvfm_single_split(
    X_train,
    t_train,
    e_train,
    X_test,
    t_test,
    e_test,
    latent_dim: int,
    cfg: ExperimentConfig,
    device,
    time_points,
    seed: int,
    true_t_train=None,
    true_c_train=None,
    true_t_test=None,
    copula=None,
    theta=None,
    dataset_name=None,
    data_type: str = "synthetic",
    include_dependence: bool = False,
):
    _seed_everything(seed)

    train_ds = SurvivalDataset(X_train, t_train, e_train)
    val_ds = SurvivalDataset(X_test, t_test, e_test)
    train_loader = DataLoader(train_ds, batch_size=cfg.dvfm_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.dvfm_batch_size, shuffle=False)

    dvfm = DVFM(input_dim=X_train.shape[1], latent_dim=latent_dim).to(device)
    train_dvfm(
        dvfm,
        train_loader,
        val_loader,
        n_epochs=cfg.dvfm_epochs,
        lr=cfg.dvfm_lr,
        beta_max=cfg.dvfm_beta_max,
        free_bits=0.0,
        device=device,
    )

    dvfm_surv = predict_survival_curves(
        model=dvfm,
        X=X_test,
        time_points=time_points,
        train_loader=train_loader,
        n_samples=cfg.mc_samples,
        device=device,
    )
    dvfm_median = _median_from_survival(dvfm_surv, time_points, t_test)

    row = {
        "Dataset Type": data_type,
        "Dataset": dataset_name if dataset_name is not None else (f"{copula}_latent_sensitivity" if copula is not None else "real"),
        "Latent Dim": int(latent_dim),
        "Seed": int(seed),
        "Num Samples": int(len(X_train) + len(X_test)),
        "Num Features": int(X_train.shape[1]),
        "Train Size": int(len(X_train)),
        "Test Size": int(len(X_test)),
        "Censoring Rate All": _censoring_rate(np.concatenate([e_train, e_test])),
        "Censoring Rate Train": _censoring_rate(e_train),
        "Censoring Rate Test": _censoring_rate(e_test),
    }
    if copula is not None:
        row["Copula"] = copula
    if theta is not None:
        row["Theta"] = float(theta)

    row.update(
        _collect_metrics(
            "DVFM",
            dvfm_median,
            dvfm_surv,
            t_test,
            e_test,
            true_t_test,
            time_points,
            tau=None,
            t_train=t_train,
            e_train=e_train,
            true_t_train=true_t_train,
            true_c_train=true_c_train,
            dep_copula_name=copula,
            dep_alpha=theta,
        )
    )

    if include_dependence and data_type == "synthetic" and cfg.analyze_dependence and copula is not None and theta is not None:
        tau_true, tau_x, tau_xz = analyze_conditional_dependence(
            dvfm, X_test, copula_type=copula, theta=theta, device=device
        )
        row["True Tau"] = float(tau_true)
        row["Tau(T,C|X)"] = float(tau_x)
        row["Tau(T,C|X,Z)"] = float(tau_xz)
        row["Tau Error |X|"] = float(abs(tau_x - tau_true))

    return row


def _run_dvfm_real_single_split(
    X,
    obs_time,
    event,
    spec,
    cfg: ExperimentConfig,
    device,
    seed: int,
    beta_max: float,
    repeat: int,
):
    _seed_everything(seed)
    stratify_arg = event if len(np.unique(event)) > 1 else None
    X_train, X_test, t_train, t_test, e_train, e_test = train_test_split(
        X,
        obs_time,
        event,
        test_size=cfg.test_size,
        random_state=seed,
        stratify=stratify_arg,
    )

    time_scale = 1.0
    if cfg.real_time_normalize:
        if cfg.real_time_norm_method == "train_max":
            time_scale = float(np.max(t_train))
        else:
            raise ValueError(f"Unsupported real_time_norm_method: {cfg.real_time_norm_method}")
        time_scale = max(time_scale, 1e-8)
        t_train = t_train / time_scale
        t_test = t_test / time_scale

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    max_time = float(np.max(t_train) * cfg.max_time_factor)
    time_points = np.linspace(0.0, max_time, cfg.n_time_points)

    train_ds = SurvivalDataset(X_train, t_train, e_train)
    val_ds = SurvivalDataset(X_test, t_test, e_test)
    train_loader = DataLoader(train_ds, batch_size=cfg.dvfm_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.dvfm_batch_size, shuffle=False)

    dvfm = DVFM(input_dim=X_train.shape[1], latent_dim=cfg.latent_dim).to(device)
    train_dvfm(
        dvfm,
        train_loader,
        val_loader,
        n_epochs=cfg.dvfm_epochs,
        lr=cfg.dvfm_lr,
        beta_max=float(beta_max),
        free_bits=0.0,
        device=device,
    )

    dvfm_surv = predict_survival_curves(
        model=dvfm,
        X=X_test,
        time_points=time_points,
        train_loader=train_loader,
        n_samples=cfg.mc_samples,
        device=device,
    )
    dvfm_median = _median_from_survival(dvfm_surv, time_points, t_test)

    row = {
        "Dataset Type": "real",
        "Dataset": spec["name"],
        "Source Path": str(spec["path"]),
        "Split": "80/20",
        "Time Normalized": bool(cfg.real_time_normalize),
        "Time Norm Method": cfg.real_time_norm_method if cfg.real_time_normalize else "none",
        "Time Scale": float(time_scale),
        "Beta Max": float(beta_max),
        "Repeat": int(repeat),
        "Seed": int(seed),
        "Num Samples": int(len(X)),
        "Num Features": int(X.shape[1]),
        "Event Rate All": float(np.mean(event)),
        "Event Rate Train": float(np.mean(e_train)),
        "Event Rate Test": float(np.mean(e_test)),
        "Censoring Rate All": _censoring_rate(event),
        "Censoring Rate Train": _censoring_rate(e_train),
        "Censoring Rate Test": _censoring_rate(e_test),
    }

    row.update(
        _collect_metrics(
            "DVFM",
            dvfm_median,
            dvfm_surv,
            t_test,
            e_test,
            None,
            time_points,
            tau=None,
            t_train=t_train,
            e_train=e_train,
        )
    )
    return row


def _write_excel_with_summary(raw_df: pd.DataFrame, group_cols: List[str], metric_cols: List[str], output_path: Path):
    summary_mean = raw_df.groupby(group_cols)[metric_cols].mean(numeric_only=True).reset_index()
    summary_std = raw_df.groupby(group_cols)[metric_cols].std(numeric_only=True).reset_index()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path) as writer:
        raw_df.to_excel(writer, sheet_name="raw", index=False)
        summary_mean.to_excel(writer, sheet_name="mean", index=False)
        summary_std.to_excel(writer, sheet_name="std", index=False)
    return summary_mean, summary_std


def run_dvfm_latent_sensitivity_synthetic(cfg: ExperimentConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dims = _parse_latent_dim_grid(cfg.latent_dim_grid)
    rows = []

    synthetic_copulas = ("gaussian", "clayton", "frank", "gumbel", "frailty_mixture")
    theta_grid = (cfg.theta_low, cfg.theta_high)
    scenario_idx = 0
    for copula in synthetic_copulas:
        for theta_val in theta_grid:
            seed = 1000 + scenario_idx
            _seed_everything(seed)
            X, obs_time, event, true_t, true_c = generate_copula_data(
                n_samples=cfg.n_samples,
                n_features=cfg.n_features,
                copula_type=copula,
                theta=theta_val,
                seed=seed,
            )
            (
                X_train,
                X_test,
                t_train,
                t_test,
                e_train,
                e_test,
                true_t_train,
                true_t_test,
                true_c_train,
                true_c_test,
            ) = train_test_split(
                X, obs_time, event, true_t, true_c, test_size=cfg.test_size, random_state=seed
            )

            max_time = float(np.max(t_train) * cfg.max_time_factor)
            time_points = np.linspace(0.0, max_time, cfg.n_time_points)

            for latent_dim in latent_dims:
                row = _run_dvfm_single_split(
                    X_train,
                    t_train,
                    e_train,
                    X_test,
                    t_test,
                    e_test,
                    latent_dim=latent_dim,
                    cfg=cfg,
                    device=device,
                    time_points=time_points,
                    seed=seed,
                    true_t_train=true_t_train,
                    true_c_train=true_c_train,
                    true_t_test=true_t_test,
                    copula=copula,
                    theta=theta_val,
                    dataset_name=f"{copula}_theta{theta_val}",
                    data_type="synthetic",
                    include_dependence=False,
                )
                row["Copula"] = copula
                row["Theta"] = float(theta_val)
                row["Scenario"] = scenario_idx
                rows.append(row)
                print(
                    f"[synthetic | {copula} | theta={theta_val:.3f} | latent_dim={latent_dim}] "
                    f"DVFM C-Idx={row['DVFM C-Idx']:.4f}, "
                    f"DVFM IBS_IPCW={row['DVFM IBS IPCW']:.4f}, "
                    f"DVFM IBS_DEP={row['DVFM IBS-DEP']:.4f}"
                )
            scenario_idx += 1

    raw_df = pd.DataFrame(rows)
    metric_cols = [
        "DVFM C-Idx",
        "DVFM IBS IPCW",
        "DVFM IBS-DEP",
        "DVFM MAE Uncensored",
        "DVFM MAE Censored",
    ]
    summary_mean, summary_std = _write_excel_with_summary(
        raw_df,
        group_cols=["Latent Dim"],
        metric_cols=metric_cols,
        output_path=Path(cfg.synthetic_sensitivity_output),
    )
    print(f"\nSaved synthetic latent-sensitivity results to {Path(cfg.synthetic_sensitivity_output).resolve()}")
    return raw_df, summary_mean, summary_std


def run_dvfm_latent_sensitivity_real(cfg: ExperimentConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dims = _parse_latent_dim_grid(cfg.latent_dim_grid)
    n_scenarios = int(cfg.latent_sensitivity_runs)

    dataset_specs = [
        {
            "name": "ALS_PROACT",
            "path": Path("proact_processed_mod_no_missing.xlsx"),
            "time_col": "merged_time",
            "event_col": "Status",
        },
        {
            "name": "Cancer_METABRIC",
            "path": Path("CensoredMAE-main/data/Metabric/Metabric.csv"),
            "time_col": "time",
            "event_col": "delta",
        },
        {
            "name": "MI_DEPENDENT",
            "path": Path("mi_dependent_survival.xlsx"),
            "time_col": "merged_time",
            "event_col": "Status",
        },
    ]

    rows = []
    for spec in dataset_specs:
        if spec["name"] not in set(cfg.real_dataset_names):
            continue
        if not spec["path"].exists():
            print(f"Skipping {spec['name']}: file not found at {spec['path']}")
            continue

        print(f"\nProcessing real dataset for latent sweep: {spec['name']} ({spec['path']})")
        X, obs_time, event = _prepare_real_dataset(spec["path"], spec["time_col"], spec["event_col"])
        if len(X) < 50 or len(np.unique(event)) < 2:
            print(f"Skipping {spec['name']}: not enough valid samples or event class imbalance")
            continue

        for scenario_idx in range(n_scenarios):
            seed = 1000 + scenario_idx
            _seed_everything(seed)
            stratify_arg = event if len(np.unique(event)) > 1 else None
            X_train, X_test, t_train, t_test, e_train, e_test = train_test_split(
                X,
                obs_time,
                event,
                test_size=cfg.test_size,
                random_state=seed,
                stratify=stratify_arg,
            )

            time_scale = 1.0
            if cfg.real_time_normalize:
                if cfg.real_time_norm_method == "train_max":
                    time_scale = float(np.max(t_train))
                else:
                    raise ValueError(f"Unsupported real_time_norm_method: {cfg.real_time_norm_method}")
                time_scale = max(time_scale, 1e-8)
                t_train = t_train / time_scale
                t_test = t_test / time_scale

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            max_time = float(np.max(t_train) * cfg.max_time_factor)
            time_points = np.linspace(0.0, max_time, cfg.n_time_points)

            for latent_dim in latent_dims:
                row = _run_dvfm_single_split(
                    X_train,
                    t_train,
                    e_train,
                    X_test,
                    t_test,
                    e_test,
                    latent_dim=latent_dim,
                    cfg=cfg,
                    device=device,
                    time_points=time_points,
                    seed=seed,
                    dataset_name=spec["name"],
                    data_type="real",
                )
                row["Source Path"] = str(spec["path"])
                row["Scenario"] = scenario_idx
                row["Repeat"] = scenario_idx
                row["Time Normalized"] = bool(cfg.real_time_normalize)
                row["Time Norm Method"] = cfg.real_time_norm_method if cfg.real_time_normalize else "none"
                row["Time Scale"] = float(time_scale)
                rows.append(row)
                print(
                    f"[real | {spec['name']} | latent_dim={latent_dim} | scenario={scenario_idx + 1}/{n_scenarios}] "
                    f"DVFM IBS_IPCW={row['DVFM IBS IPCW']:.4f}, "
                    f"DVFM C-Idx={row['DVFM C-Idx']:.4f}"
                )

    if not rows:
        print("No real datasets were processed for latent sensitivity. No Excel file written.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    raw_df = pd.DataFrame(rows)
    metric_cols = [c for c in raw_df.columns if c not in {
        "Dataset Type",
        "Dataset",
        "Source Path",
        "Copula",
        "Dependence",
        "Theta",
        "Scenario",
        "Repeat",
        "Seed",
        "Latent Dim",
        "Time Normalized",
        "Time Norm Method",
        "Time Scale",
    }]
    summary_mean, summary_std = _write_excel_with_summary(
        raw_df,
        group_cols=["Dataset", "Latent Dim"],
        metric_cols=metric_cols,
        output_path=Path(cfg.real_sensitivity_output),
    )
    print(f"\nSaved real latent-sensitivity results to {Path(cfg.real_sensitivity_output).resolve()}")
    return raw_df, summary_mean, summary_std


def run_dvfm_beta_sensitivity_real(cfg: ExperimentConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    beta_grid = _parse_float_grid(cfg.real_beta_grid)
    n_repeats = int(cfg.real_beta_sensitivity_runs)

    dataset_specs = [
        {
            "name": "ALS_PROACT",
            "path": Path("proact_processed_mod.xlsx"),
            "time_col": "merged_time",
            "event_col": "Status",
        },
        {
            "name": "Cancer_METABRIC",
            "path": Path("CensoredMAE-main/data/Metabric/Metabric.csv"),
            "time_col": "time",
            "event_col": "delta",
        },
        {
            "name": "MI_DEPENDENT",
            "path": Path("mi_dependent_survival.xlsx"),
            "time_col": "merged_time",
            "event_col": "Status",
        },
    ]

    rows = []
    for spec in dataset_specs:
        if spec["name"] not in set(cfg.real_dataset_names):
            continue
        if not spec["path"].exists():
            print(f"Skipping {spec['name']}: file not found at {spec['path']}")
            continue

        print(f"\nProcessing real dataset for beta sweep: {spec['name']} ({spec['path']})")
        X, obs_time, event = _prepare_real_dataset(spec["path"], spec["time_col"], spec["event_col"])
        if len(X) < 50 or len(np.unique(event)) < 2:
            print(f"Skipping {spec['name']}: not enough valid samples or event class imbalance")
            continue

        for beta_max in beta_grid:
            for repeat in range(n_repeats):
                seed = 1000 + repeat
                row = _run_dvfm_real_single_split(
                    X,
                    obs_time,
                    event,
                    spec,
                    cfg,
                    device,
                    seed=seed,
                    beta_max=beta_max,
                    repeat=repeat,
                )
                row["Beta Max"] = float(beta_max)
                row["Repeat"] = int(repeat)
                row["Scenario"] = f"{spec['name']}_beta{beta_max}"
                rows.append(row)
                print(
                    f"[real | {spec['name']} | beta_max={beta_max:.3f} | repeat={repeat + 1}/{n_repeats}] "
                    f"DVFM IBS_IPCW={row['DVFM IBS IPCW']:.4f}, "
                    f"DVFM IBS_DEP={row['DVFM IBS-DEP']:.4f}, "
                    f"DVFM C-Idx={row['DVFM C-Idx']:.4f}"
                )

    if not rows:
        print("No real datasets were processed for beta sensitivity. No Excel file written.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    raw_df = pd.DataFrame(rows)
    metric_cols = [
        "DVFM C-Idx",
        "DVFM IBS IPCW",
        "DVFM IBS-DEP",
        "DVFM MAE Uncensored",
        "DVFM MAE Censored",
    ]
    summary_mean, summary_std = _write_excel_with_summary(
        raw_df,
        group_cols=["Dataset", "Beta Max"],
        metric_cols=metric_cols,
        output_path=Path(cfg.real_beta_sensitivity_output),
    )
    print(f"\nSaved real beta-sensitivity results to {Path(cfg.real_beta_sensitivity_output).resolve()}")
    return raw_df, summary_mean, summary_std


def _plot_summary(summary_mean):
    plot_df = summary_mean.copy()
    plot_df["Scenario"] = plot_df["Copula"] + " | " + plot_df["Dependence"]
    plt.figure(figsize=(14, 12))

    plt.subplot(3, 1, 1)
    c_cols = ["DeepSurv C-Idx", "MTLR C-Idx", "ClaytonAFT C-Idx", "DVFM C-Idx"]
    plot_df.plot(x="Scenario", y=c_cols, kind="bar", ax=plt.gca(), width=0.8,
                 color=["#e74c3c", "#f39c12", "#2980b9", "#2ecc71"])
    plt.title("C-Index (Mean over Repeats)")
    plt.ylim(0, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.xticks(rotation=20, ha="right")

    plt.subplot(3, 1, 2)
    mae_cols = ["DeepSurv MAE", "MTLR MAE", "ClaytonAFT MAE", "DVFM MAE"]
    plot_df.plot(x="Scenario", y=mae_cols, kind="bar", ax=plt.gca(), width=0.8,
                 color=["#e74c3c", "#f39c12", "#2980b9", "#2ecc71"])
    plt.title("Uncensored MAE (Mean over Repeats)")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.xticks(rotation=20, ha="right")

    plt.subplot(3, 1, 3)
    ibs_cols = ["DeepSurv IBS IPCW", "MTLR IBS IPCW", "ClaytonAFT IBS IPCW", "DVFM IBS IPCW"]
    plot_df.plot(x="Scenario", y=ibs_cols, kind="bar", ax=plt.gca(), width=0.8,
                 color=["#e74c3c", "#f39c12", "#2980b9", "#2ecc71"])
    plt.title("IPCW IBS (Mean over Repeats)")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.xticks(rotation=20, ha="right")

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Modified DVFM survival experiment runner")
    parser.add_argument("--quick", action="store_true", help="Run a faster debug configuration")
    parser.add_argument("--no-plots", action="store_true", help="Disable plotting")
    parser.add_argument("--run-original", action="store_true", help="Run the original full experiment pipeline")
    parser.add_argument("--run-real-data", action="store_true", help="Also run real datasets and save separate CSV files")
    parser.add_argument("--real-only", action="store_true", help="Run only real datasets (skip synthetic)")
    parser.add_argument("--latent-sensitivity", action="store_true", help="Run DVFM-only latent-dimension sensitivity studies and save Excel files")
    parser.add_argument("--real-beta-sensitivity", action="store_true", help="Run DVFM-only beta sensitivity on real datasets and save Excel files")
    parser.add_argument("--latent-dims", type=str, default=None, help="Comma-separated latent dimensions for sensitivity runs, e.g. 2,5,10,20")
    parser.add_argument("--latent-sensitivity-runs", type=int, default=None, help="Number of scenarios/repeats for each latent dimension")
    parser.add_argument("--sensitivity-output-prefix", type=str, default=None, help="Prefix for latent-sensitivity Excel output files")
    parser.add_argument("--beta-values", type=str, default=None, help="Comma-separated beta_max values for real beta sensitivity, e.g. 0.2,0.5,1")
    parser.add_argument("--real-beta-sensitivity-runs", type=int, default=None, help="Number of repeats for each real beta value")
    parser.add_argument("--real-beta-sensitivity-output", type=str, default=None, help="Output file for real beta-sensitivity Excel results")
    parser.add_argument("--real-datasets", type=str, default=None, help="Comma-separated real dataset names to run")
    parser.add_argument("--real-mtlr-bins", type=str, default=None, help="Real-data MTLR bins mode: fixed or sqrt_train")
    parser.add_argument("--theta-low", type=float, default=None, help="Theta value for low dependence")
    parser.add_argument("--theta-high", type=float, default=None, help="Theta value for very high dependence")
    parser.add_argument("--output-prefix", type=str, default=None, help="Prefix for saved CSV files")
    parser.add_argument("--real-output-prefix", type=str, default=None, help="Prefix for saved real-data CSV files")
    parser.add_argument("--dvfm-lr", type=float, default=None, help="Learning rate for DVFM")
    parser.add_argument("--beta-max", type=float, default=None, help="Maximum beta value for DVFM training")
    parser.add_argument("--deepsurv-lr", type=float, default=None, help="Learning rate for DeepSurv")
    parser.add_argument("--mtlr-lr", type=float, default=None, help="Learning rate for MTLR")
    args = parser.parse_args()

    cfg = ExperimentConfig()
    if args.quick:
        cfg.n_samples = 2500
        cfg.repeats = 1
        cfg.dvfm_epochs = 60
        cfg.deepsurv_epochs = 60
        cfg.mtlr_epochs = 60
        cfg.copula_aft_epochs = 150
        cfg.n_time_points = 300
        cfg.mc_samples = 50
        cfg.copulas = ("gaussian", "gumbel", "frailty_mixture", "clayton_covdep")

    if args.no_plots:
        cfg.make_plots = False
    if args.theta_low is not None:
        cfg.theta_low = float(args.theta_low)
    if args.theta_high is not None:
        cfg.theta_high = float(args.theta_high)
    if args.output_prefix is not None:
        cfg.output_prefix = args.output_prefix
    if args.real_output_prefix is not None:
        cfg.real_output_prefix = args.real_output_prefix
    if args.sensitivity_output_prefix is not None:
        cfg.synthetic_sensitivity_output = f"{args.sensitivity_output_prefix}_synthetic.xlsx"
        cfg.real_sensitivity_output = f"{args.sensitivity_output_prefix}_real.xlsx"
    if args.real_datasets is not None:
        cfg.real_dataset_names = tuple([s.strip() for s in args.real_datasets.split(",") if s.strip()])
    if args.real_mtlr_bins is not None:
        cfg.real_mtlr_bins_mode = str(args.real_mtlr_bins).strip().lower()
    if args.latent_dims is not None:
        cfg.latent_dim_grid = tuple(_parse_latent_dim_grid(args.latent_dims))
    if args.latent_sensitivity_runs is not None:
        cfg.latent_sensitivity_runs = max(1, int(args.latent_sensitivity_runs))
    if args.beta_values is not None:
        cfg.real_beta_grid = tuple(_parse_float_grid(args.beta_values))
    if args.real_beta_sensitivity_runs is not None:
        cfg.real_beta_sensitivity_runs = max(1, int(args.real_beta_sensitivity_runs))
    if args.real_beta_sensitivity_output is not None:
        cfg.real_beta_sensitivity_output = args.real_beta_sensitivity_output
    if args.dvfm_lr is not None:
        cfg.dvfm_lr = float(args.dvfm_lr)
    if args.beta_max is not None:
        cfg.dvfm_beta_max = float(args.beta_max)
    if args.deepsurv_lr is not None:
        cfg.deepsurv_lr = float(args.deepsurv_lr)
    if args.mtlr_lr is not None:
        cfg.mtlr_lr = float(args.mtlr_lr)

    explicit_original_mode = args.run_original or args.run_real_data or args.real_only
    explicit_any_mode = (
        args.run_original
        or args.run_real_data
        or args.real_only
        or args.latent_sensitivity
        or args.real_beta_sensitivity
    )

    if args.real_beta_sensitivity or not explicit_any_mode:
        run_dvfm_beta_sensitivity_real(cfg)
        if not explicit_original_mode and not args.latent_sensitivity:
            return

    if args.latent_sensitivity:
        run_dvfm_latent_sensitivity_synthetic(cfg)
        run_dvfm_latent_sensitivity_real(cfg)
        if not explicit_original_mode:
            return

    if args.run_original and not args.real_only:
        run_experiment_v2(cfg)
    if args.run_real_data or args.real_only:
        run_real_data_experiments(cfg)


if __name__ == "__main__":
    main()
