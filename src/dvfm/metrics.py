"""Evaluation metrics used by the supplied experiments.

Includes C-index, MAE decomposition, oracle and IPCW IBS, and the
DependentEVAL-style dependence-aware metrics used in the modified script.
"""

import numpy as np
from lifelines.utils import concordance_index

from .prediction import get_median_survival_time

from scipy.integrate import trapezoid as trapz

try:
    from SurvivalEVAL.Evaluations.util import predict_multi_probs_from_curve
except ImportError:  # IBS-DEP will be returned as NaN when SurvivalEVAL is absent
    predict_multi_probs_from_curve = None

def _km_step_survival(times, event_observed):
    """
    Kaplan-Meier step curve for P(T > t).
    event_observed is 1 when a failure/event is observed.
    """
    times = np.asarray(times, dtype=float)
    event_observed = np.asarray(event_observed, dtype=int)

    uniq_event_times = np.unique(times[event_observed == 1])
    surv = 1.0
    step_times = [0.0]
    step_surv = [1.0]

    for t in uniq_event_times:
        at_risk = np.sum(times >= t)
        d_t = np.sum((times == t) & (event_observed == 1))
        if at_risk > 0:
            surv *= (1.0 - d_t / at_risk)
        step_times.append(float(t))
        step_surv.append(float(surv))

    return np.asarray(step_times), np.asarray(step_surv)


def _eval_step_curve(query_t, step_t, step_s, side='right'):
    """
    Evaluate right-continuous ('right') or left-limit ('left') step survival.
    """
    query_t = np.asarray(query_t, dtype=float)
    idx = np.searchsorted(step_t, query_t, side=side) - 1
    idx = np.clip(idx, 0, len(step_s) - 1)
    return step_s[idx]


def compute_oracle_brier_ibs(survival_curves, time_points, true_event_times):
    """
    Oracle Brier/IBS using fully observed true event times from simulation.
    """
    time_points = np.asarray(time_points, dtype=float)
    true_event_times = np.asarray(true_event_times, dtype=float)
    y_true = (true_event_times[:, None] > time_points[None, :]).astype(float)

    brier = np.mean((y_true - survival_curves) ** 2, axis=0)
    if len(time_points) > 1:
        ibs = trapz(brier, time_points) / (time_points[-1] - time_points[0] + 1e-12)
    else:
        ibs = float(brier[0])

    return brier, float(ibs)


def compute_ipcw_brier_ibs(survival_curves, time_points, observed_time, event, tau=None, eps=1e-6):
    """
    IPCW Brier/IBS on observed data.
    Note: under dependent censoring this estimator is biased; comparing it
    to Oracle-IBS is useful as a diagnostic gap.
    """
    time_points = np.asarray(time_points, dtype=float)
    observed_time = np.asarray(observed_time, dtype=float)
    event = np.asarray(event, dtype=int)

    if tau is None:
        tau = np.quantile(observed_time, 0.8)
    eval_mask = time_points <= tau
    eval_times = time_points[eval_mask]
    eval_surv = survival_curves[:, eval_mask]

    # Censoring distribution G(t) = P(C > t)
    censor_event = 1 - event
    g_t, g_s = _km_step_survival(observed_time, censor_event)
    g_obs_left = np.maximum(_eval_step_curve(observed_time, g_t, g_s, side='left'), eps)

    brier = np.zeros(len(eval_times), dtype=float)
    n = len(observed_time)
    for j, t in enumerate(eval_times):
        s_hat = eval_surv[:, j]
        g_at_t = max(float(_eval_step_curve(np.array([t]), g_t, g_s, side='right')[0]), eps)

        event_before_t = (observed_time <= t) & (event == 1)
        survive_past_t = observed_time > t

        term_event = np.zeros(n, dtype=float)
        term_event[event_before_t] = ((0.0 - s_hat[event_before_t]) ** 2) / g_obs_left[event_before_t]

        term_survive = np.zeros(n, dtype=float)
        term_survive[survive_past_t] = ((1.0 - s_hat[survive_past_t]) ** 2) / g_at_t

        brier[j] = np.mean(term_event + term_survive)

    if len(eval_times) > 1:
        ibs = trapz(brier, eval_times) / (eval_times[-1] - eval_times[0] + 1e-12)
    else:
        ibs = float(brier[0]) if len(eval_times) == 1 else np.nan

    return brier, float(ibs), float(tau)


def _safe_subset_mae(y_true, y_pred, mask):
    if np.sum(mask) == 0:
        return np.nan
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def _censoring_rate(event):
    event = np.asarray(event, dtype=float)
    if event.size == 0:
        return np.nan
    return float(1.0 - np.mean(event))


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
        try:
            from SurvivalEVAL.Evaluations.util import KaplanMeierArea
        except ImportError:
            from SurvivalEVAL.NonparametricEstimator.SingleEvent import KaplanMeierArea
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


collect_metrics = _collect_metrics
