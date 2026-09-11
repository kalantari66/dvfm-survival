import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from pathlib import Path

from experiments.latent_recovery import export_latent_recovery
from experiments.config import load_config
from experiments.runner import run
from utility.data import SurvivalData
from utility.semisynthetic import generate_support_semisynthetic, sample_clayton_uniforms


def test_generic_cohort_loader_fits_custom_columns(tmp_path):
    from utility.semisynthetic import fit_semisynthetic_dgp
    rng = np.random.default_rng(9)
    frame = pd.DataFrame({"age": rng.normal(size=150), "group": ["a", "b", "c"] * 50,
                          "followup": rng.exponential(size=150) + .01,
                          "observed": rng.integers(0, 2, size=150)})
    path = tmp_path / "cohort.csv"
    frame.to_csv(path, index=False)
    dgp = fit_semisynthetic_dgp(dict(path=str(path), time_column="followup",
                                   event_column="observed", numeric_features=["age"],
                                   categorical_features=["group"]))
    assert dgp.X.shape == (150, 3)
    assert np.isfinite(dgp.X).all()
    generated = generate_support_semisynthetic(dgp, kendall_tau=.5,
                                               censoring_rate=.5, sampling_seed=2)
    assert len(generated.data.true_z) == 150


class KnownEncoder(torch.nn.Module):
    def encoder(self, x, time, event):
        return x[:, :1], torch.zeros_like(x[:, :1])


def test_validation_calibration_is_frozen_and_exports_notebook_inputs(tmp_path):
    mu = np.arange(6, dtype=np.float32)
    def data(truth):
        return SurvivalData(mu[:, None], np.ones(6), np.array([0, 1] * 3), ["x"], true_z=truth)
    valid = data(3 - 2 * mu)
    test = data(100 + mu)  # Deliberately incompatible: must not influence calibration.
    export_latent_recovery(KnownEncoder(), valid, test, np.arange(6), np.arange(10, 16),
                           tmp_path, {"Dataset": "example"}, 2, "cpu")
    frame = pd.read_csv(tmp_path / "latent_recovery_test.csv")
    np.testing.assert_allclose(frame.learned_z_calibrated, 3 - 2 * mu, atol=1e-5)
    np.testing.assert_allclose(frame.learned_z_calibrated_std, 2, atol=1e-5)
    np.testing.assert_array_equal(frame.row_index, np.arange(10, 16))
    assert set(["row_index", "event", "true_z", "learned_mu_raw",
                "learned_mu_aligned", "learned_std"]).issubset(frame.columns)
    metrics = pd.read_csv(tmp_path / "latent_calibration_metrics.csv")
    assert len(metrics) == 12
    assert set(metrics.subgroup) == {"All", "Event observed", "Censored"}
    assert metrics.loc[(metrics.split == "test") &
                       (metrics.representation == "Validation-calibrated"), "coverage_95"].eq(0).all()


def test_generator_preserves_sampled_truth_and_independence_has_no_target(tmp_path):
    class Margin:
        def inverse_survival(self, u, x):
            return -np.log(u)
    dgp = SimpleNamespace(X=np.zeros((100, 1)), feature_names=["x"],
                          event_margin=Margin(), censor_margin=Margin())
    u, theta, frailty = sample_clayton_uniforms(100, .5, 4, return_frailty=True)
    old_u, old_theta = sample_clayton_uniforms(100, .5, 4)
    np.testing.assert_array_equal(u, old_u)
    assert theta == old_theta
    generated = generate_support_semisynthetic(dgp, kendall_tau=.5, censoring_rate=.5, sampling_seed=4)
    log_w = np.log(np.clip(frailty, 1e-12, None))
    np.testing.assert_allclose(generated.data.true_z, (log_w - log_w.mean()) / log_w.std())
    independent = generate_support_semisynthetic(dgp, kendall_tau=0, censoring_rate=.5, sampling_seed=4)
    assert independent.data.true_z is None
    export_latent_recovery(KnownEncoder(), independent.data, independent.data,
                           np.arange(100), np.arange(100), tmp_path, {}, 16, "cpu")
    metadata = json.loads((tmp_path / "latent_recovery_metadata.json").read_text())
    assert metadata["status"] == "unavailable_no_true_frailty"
    assert not (tmp_path / "latent_recovery_test.csv").exists()


def test_semisynthetic_runner_exports_each_tau_and_repeat(tmp_path, monkeypatch):
    import utility.semisynthetic as semi
    class Margin:
        def inverse_survival(self, u, x):
            return -np.log(u)
    dgp = SimpleNamespace(X=np.random.default_rng(7).normal(size=(120, 2)).astype(np.float32),
                          feature_names=["x", "y"], event_margin=Margin(), censor_margin=Margin(),
                          source_n_samples=120, source_event_rate=.5, cox_penalizer=.01)
    fitted_datasets = []
    def fit_dataset(spec, **kwargs):
        fitted_datasets.append(spec["name"])
        return dgp
    monkeypatch.setattr(semi, "fit_semisynthetic_dgp", fit_dataset)
    monkeypatch.setattr("experiments.runner.validate_inputs", lambda cfg: None)
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/semi_synthetic.yaml")
    cfg["study"]["output_dir"] = str(tmp_path)
    cfg["compute"]["device"] = "cpu"
    cfg["seeds"] = [0, 1]
    cfg["data"].update(kendall_tau=[0, .5], censoring_rates=[.5])
    cfg["data"]["datasets"].append({**cfg["data"]["datasets"][0], "name": "second"})
    for spec in cfg["data"]["datasets"]:
        spec.update(numeric_features=["x", "y"], categorical_features=[])
    cfg["models"]["enabled"] = ["dvfm"]
    cfg["models"]["dvfm"].update(epochs=1, warmup_epochs=0, checkpoint_min_epoch=1, mc_samples=2)
    cfg["evaluation"]["n_time_points"] = 10
    results = run(cfg)
    assert len(results) == 8
    assert results.Scenario.nunique() == 4
    assert set(results.Dataset) == {"support", "second"}
    assert fitted_datasets == ["support", "second"]
    exports = list((tmp_path / "latent_recovery").glob("*/latent_recovery_metadata.json"))
    assert len(exports) == 8
    assert len(list((tmp_path / "latent_recovery").glob("*/latent_recovery_test.csv"))) == 4
    for path in (tmp_path / "latent_recovery").glob("*/latent_recovery_test.csv"):
        frame = pd.read_csv(path)
        assert len(frame) == 24
        assert np.isfinite(frame.learned_z_calibrated).all()
