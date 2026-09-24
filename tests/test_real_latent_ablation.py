"""PRO-ACT cohort construction and the real-data latent ablation."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from dvfm.likelihood import (
    encode_posterior, event_martingale_residual, heldout_log_likelihood,
)
from dvfm.model import DVFM
from experiments.config import load_config
from experiments.runner import run
from utility.proact import build_proact_death_cohort


ROOT = Path(__file__).resolve().parents[1]


def _toy_model(latent_dim, seed=0):
    torch.manual_seed(seed)
    return DVFM(input_dim=3, latent_dim=latent_dim, encoder_hidden=[8], decoder_hidden=[8]).eval()


def _toy_outcomes(n=40, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3)).astype(np.float32)
    time = rng.uniform(0.2, 2.0, size=n).astype(np.float32)
    event = rng.binomial(1, 0.5, size=n).astype(np.float32)
    return X, time, event


def test_latent_free_likelihood_is_exact_and_matches_training_loss():
    model = _toy_model(0)
    X, time, event = _toy_outcomes()
    scale = 7.0
    estimate = heldout_log_likelihood(
        model, X, time, event, mixing_mu=np.empty((1, 0)),
        mixing_std=np.empty((1, 0)), time_scale=scale,
    )
    np.testing.assert_allclose(estimate["joint_elbo"], estimate["joint_iwae"])
    np.testing.assert_allclose(
        estimate["joint_elbo"], estimate["event_margin"] + estimate["censor_margin"],
        rtol=1e-6, atol=1e-6,
    )
    with torch.no_grad():
        tensors = [torch.as_tensor(v) for v in (X, time, event)]
        _, reconstruction_nll, kl = model.loss_function(*model(*tensors), *tensors[1:])
    assert float(kl) == 0.0
    # Each subject has exactly one density term, hence one log(scale) Jacobian.
    assert np.mean(estimate["joint_elbo"]) == pytest.approx(
        -float(reconstruction_nll) - math.log(scale), rel=1e-5
    )


def test_latent_likelihood_bounds_and_isolated_rng():
    model = _toy_model(1)
    X, time, event = _toy_outcomes()
    mu, std = encode_posterior(model, X, time, event)
    torch.manual_seed(5)
    expected_next = torch.rand(3)
    torch.manual_seed(5)
    estimate = heldout_log_likelihood(
        model, X, time, event, mixing_mu=mu, mixing_std=std, n_samples=2000,
    )
    torch.testing.assert_close(torch.rand(3), expected_next)
    assert all(np.isfinite(values).all() for values in estimate.values())
    assert np.mean(estimate["joint_iwae"]) >= np.mean(estimate["joint_elbo"]) - 1e-3
    with pytest.raises(ValueError, match="latent_dim = 0"):
        event_martingale_residual(model, X, time, event)


def _write_form(raw: Path, form: str, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(raw / f"PROACT_{form}.csv", index=False)


@pytest.fixture
def raw_proact(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    # 1 dies at day 300; 2 is alive, last assessed on day 250, but has an
    # adverse event dated day 900 and an implausible 10-day onset; 3 died with no recorded day; 4 has no
    # follow-up after baseline; 5 is absent from the mortality form, so its
    # vital status is unknown.
    subjects = [1, 2, 3, 4, 5]
    alsfrs = []
    for subject, days in {1: [0, 100, 200], 2: [0, 250], 3: [0, 50], 4: [0], 5: [0, 120]}.items():
        for index, day in enumerate(days):
            alsfrs.append({"subject_id": subject, "ALSFRS_Delta": day,
                           "ALSFRS_R_Total": 40 - 3 * index, "Q1_Speech": 4})
    _write_form(raw, "ALSFRS", alsfrs)
    _write_form(raw, "ALSHISTORY", [
        {"subject_id": s, "Onset_Delta": {1: np.nan, 2: -10}.get(s, -300), "Site_of_Onset": "Onset: Limb"}
        for s in subjects
    ] + [{"subject_id": 1, "Onset_Delta": -600, "Site_of_Onset": "Onset: Bulbar"}])
    _write_form(raw, "DEMOGRAPHICS", [
        {"subject_id": s, "Age": 50 + s, "Sex": "M" if s % 2 else "Female"} for s in subjects
    ])
    _write_form(raw, "RILUZOLE", [
        {"subject_id": s, "Subject_used_Riluzole": "Yes"} for s in subjects + [2]
    ])
    fvc = [{"subject_id": s, "Subject_Liters_Trial_1": 3.0, "Subject_Liters_Trial_2": 3.0,
            "Subject_Liters_Trial_3": 3.0, "Forced_Vital_Capacity_Delta": 0} for s in subjects]
    fvc.insert(0, {"subject_id": 1, "Subject_Liters_Trial_1": 1.0, "Subject_Liters_Trial_2": 1.0,
                   "Subject_Liters_Trial_3": 1.0, "Forced_Vital_Capacity_Delta": 200})
    _write_form(raw, "FVC", fvc)
    _write_form(raw, "DEATHDATA", [
        {"subject_id": 1, "Subject_Died": "Yes", "Death_Days": 300},
        {"subject_id": 2, "Subject_Died": "No", "Death_Days": np.nan},
        {"subject_id": 3, "Subject_Died": "Yes", "Death_Days": np.nan},
        {"subject_id": 4, "Subject_Died": "No", "Death_Days": np.nan},
    ])
    for form, column in {"SVC": "Slow_vital_Capacity_Delta", "VITALSIGNS": "Vital_Signs_Delta",
                         "HANDGRIPSTRENGTH": "MS_Delta", "MUSCLESTRENGTH": "MS_Delta"}.items():
        _write_form(raw, form, [{"subject_id": 2, column: 10}])
    # A stray lab record years after the last assessment is not a contact.
    _write_form(raw, "LABS", [{"subject_id": 2, "Laboratory_Delta": 3000}])
    _write_form(raw, "ADVERSEEVENTS", [{"subject_id": 2, "Start_Date_Delta": 900}])
    return raw


def test_proact_cohort_outcomes_and_deduplication(raw_proact):
    cohort, flow = build_proact_death_cohort(raw_proact)
    cohort = cohort.set_index("subject_id")
    assert list(cohort.index) == [1, 2]
    assert cohort.loc[1, ["time", "event"]].tolist() == [300, 1]
    # Last seen is the latest assessment, not the adverse-event or lab date.
    assert cohort.loc[2, ["time", "event"]].tolist() == [250, 0]
    assert cohort.loc[1, "Onset_Delta"] == 600
    assert cohort.loc[1, "FVC_Mean"] == 3.0
    assert cohort.loc[1, "alsfrs_r_decline_per_month"] == pytest.approx(3 * 30.44 / 100)
    assert flow["dropped_died_without_death_day"] == 1
    assert flow["dropped_nonpositive_time"] == 1
    assert flow["dropped_no_death_record"] == 1
    # An onset under 30 days is implausible: set missing, subject kept.
    assert cohort.loc[2, ["Onset_Delta", "DiseaseProgressionRate"]].isna().all()
    assert flow["onset_set_missing_below_min"] == 1


def _ablation_config(tmp_path, **dvfm_overrides):
    cfg = yaml.safe_load((ROOT / "configs/proact_latent_ablation.yaml").read_text(encoding="utf-8"))
    rng = np.random.default_rng(3)
    n = 240
    frame = pd.DataFrame({
        "subject_id": np.arange(n), "a": rng.normal(size=n), "b": rng.choice(["u", "v"], size=n),
        "decline": rng.normal(size=n),
    })
    frame.loc[::7, "a"] = np.nan
    frame["time"] = rng.exponential(np.exp(0.5 * frame["a"].fillna(0)), size=n) + 0.01
    frame["event"] = rng.binomial(1, 0.6, size=n)
    frame.to_csv(tmp_path / "cohort.csv", index=False)
    cfg["study"]["output_dir"] = str(tmp_path / "out")
    cfg["seeds"] = [0, 1]
    cfg["compute"] = {"device": "cpu", "torch_num_threads": 1}
    cfg["data"]["datasets"] = [{
        "name": "toy", "path": str(tmp_path / "cohort.csv"), "time_column": "time",
        "event_column": "event", "id_column": "subject_id", "numeric_features": ["a"],
        "categorical_features": ["b"], "external_columns": ["decline"],
    }]
    cfg["models"]["dvfm"].update({"epochs": 6, "warmup_epochs": 2, "checkpoint_min_epoch": 2,
                                  "mc_samples": 5, **dvfm_overrides})
    cfg["evaluation"]["likelihood_samples"] = 20
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_ablation_config_rejects_unpaired_or_leaky_specs(tmp_path):
    assert load_config(ROOT / "configs/proact_latent_ablation.yaml")["models"]["dvfm"]["latent_dims"] == [0, 1]
    with pytest.raises(ValueError, match="latent-free reference"):
        load_config(_ablation_config(tmp_path, latent_dims=[1]))
    path = _ablation_config(tmp_path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["data"]["datasets"][0]["external_columns"] = ["a"]
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot be features"):
        load_config(path)
    cfg["data"]["datasets"][0]["external_columns"] = ["decline"]
    cfg["evaluation"]["time_grid"] = "uniform_train_true_event_quantile"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="no true event times"):
        load_config(path)


def test_ablation_run_writes_paired_artifacts(tmp_path):
    results = run(load_config(_ablation_config(tmp_path)))
    out = tmp_path / "out"
    assert (out / "_SUCCESS").exists()
    assert sorted(results.latent_dim.unique()) == [0, 1]
    assert not results.numerical_failure.any()
    exact = results.loc[results.latent_dim.eq(0)]
    np.testing.assert_allclose(exact.test_joint_elbo, exact.test_joint_iwae)
    paired = pd.read_csv(out / "paired_differences.csv")
    assert len(paired) == 2 * 7
    subjects = pd.read_csv(out / "subjects" / "toy_repeat_0.csv")
    assert {"martingale_residual", "z_mu_oriented", "decline"} <= set(subjects.columns)
    assert subjects.subject_id.is_unique and len(subjects) == 240
    external = pd.read_csv(out / "latent_external_validation.csv")
    assert set(external.subgroup) == {"All", "Event observed", "Censored"}


def test_missing_indicators_are_unscaled_and_fitted_on_train_only():
    from experiments.real_latent_ablation import _encode_split

    frame = pd.DataFrame({
        "a": [1.0, np.nan, 3.0, 4.0, 5.0, np.nan],
        "b": ["u", "v", np.nan, "u", "v", "u"],
        "c": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan],
    })
    spec = {"numeric_features": ["a", "c"], "categorical_features": ["b"],
            "missing_indicators": True}
    train, validation, test = np.array([0, 1, 2, 3]), np.array([4]), np.array([5])
    (x_train, _, x_test), names = _encode_split(frame, spec, train, validation, test)
    # c is never missing in training, so it gets no indicator even though the
    # test subject lacks it.
    assert [name for name in names if name.startswith("missingindicator")] == [
        "missingindicator_a", "missingindicator_b",
    ]
    flags = x_train[:, [names.index("missingindicator_a"), names.index("missingindicator_b")]]
    np.testing.assert_array_equal(flags, [[0, 0], [1, 0], [0, 1], [0, 0]])
    assert x_test.shape[1] == len(names)
    spec["missing_indicators"] = False
    _, plain = _encode_split(frame, spec, train, validation, test)
    assert not any(name.startswith("missingindicator") for name in plain)


def test_missingness_filter_drops_features_above_limit_only():
    from experiments.real_latent_ablation import apply_missingness_filter

    frame = pd.DataFrame({
        "a": [1.0, np.nan, 3.0, 4.0, 5.0],        # 20% missing: kept at the limit
        "b": [np.nan, np.nan, 3.0, 4.0, 5.0],     # 40% missing: dropped
        "c": ["u", np.nan, np.nan, "v", "u"],     # 40% missing: dropped
        "d": ["u", "v", "u", "v", "u"],
    })
    spec = {"name": "toy", "numeric_features": ["a", "b"], "categorical_features": ["c", "d"],
            "max_missing_fraction": 0.2}
    filtered, report = apply_missingness_filter(frame, spec)
    assert filtered["numeric_features"] == ["a"] and filtered["categorical_features"] == ["d"]
    assert report.set_index("feature").kept.to_dict() == {"a": True, "b": False, "c": False, "d": True}
    unfiltered, _ = apply_missingness_filter(frame, {**spec, "max_missing_fraction": None})
    assert unfiltered["numeric_features"] == ["a", "b"]
    with pytest.raises(ValueError, match="removes every feature"):
        apply_missingness_filter(frame, {**spec, "max_missing_fraction": 0.0,
                                         "numeric_features": ["b"], "categorical_features": ["c"]})


def test_likelihood_ignores_overflow_in_the_unused_margin_term():
    from dvfm.likelihood import _margin_log_terms

    class ExtremeDecoder(torch.nn.Module):
        # Event scale so small that the event density overflows for a subject
        # who was censored and therefore never uses it.
        def forward(self, x, z):
            n = len(x)
            return (torch.full((n,), 50.0), torch.full((n,), 1e-6),
                    torch.ones(n), torch.ones(n))

    model = _toy_model(0)
    model.decoder = ExtremeDecoder()
    x = torch.zeros((1, 3))
    event_term, censor_term = _margin_log_terms(
        model, x, torch.tensor([2.0]), torch.tensor([0.0]), x.new_empty((1, 0))
    )
    assert not torch.isnan(event_term).any() and not torch.isnan(censor_term).any()
    assert event_term.item() == float("-inf")
    # Unit-shape, unit-scale Weibull censoring density at t = 2: log f = -2.
    assert censor_term.item() == pytest.approx(-2.0)


def test_likelihood_horizon_scores_joint_survival_beyond_it():
    model = _toy_model(0)
    x = np.zeros((4, 3), dtype=np.float32)
    # Inside the horizon, then three subjects beyond it with any status: all
    # three are only known to be free of both outcomes at the horizon.
    time = np.array([0.5, 2.0, 5.0, 9.0], dtype=np.float32)
    event = np.array([1, 0, 1, 0], dtype=np.float32)
    kwargs = dict(mixing_mu=np.empty((1, 0)), mixing_std=np.empty((1, 0)), time_scale=3.0)
    capped = heldout_log_likelihood(model, x, time, event, horizon=1.5, **kwargs)
    plain = heldout_log_likelihood(model, x, time, event, **kwargs)
    assert capped["joint_elbo"][0] == pytest.approx(plain["joint_elbo"][0])
    np.testing.assert_allclose(capped["joint_elbo"][1:], capped["joint_elbo"][1])
    with torch.no_grad():
        shape_t, scale_t, shape_c, scale_c = model.decoder(torch.zeros((1, 3)), torch.zeros((1, 0)))
        expected = float(DVFM.weibull_log_survival(torch.tensor(1.5), shape_t, scale_t)
                         + DVFM.weibull_log_survival(torch.tensor(1.5), shape_c, scale_c))
    # Survival terms only, so no log(time_scale) Jacobian.
    assert capped["joint_elbo"][1] == pytest.approx(expected, rel=1e-5)

    latent = _toy_model(1)
    mu, std = encode_posterior(latent, x, time, event)
    bounded = heldout_log_likelihood(
        latent, x, time, event, horizon=1.5, n_samples=500, **{**kwargs, "mixing_mu": mu, "mixing_std": std}
    )
    assert all(np.isfinite(values).all() for values in bounded.values())
    assert np.mean(bounded["joint_iwae"]) >= np.mean(bounded["joint_elbo"]) - 1e-3
