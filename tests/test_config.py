from pathlib import Path

import pytest

from experiments.config import expand_scenarios, load_config


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_config_loads():
    cfg = load_config(ROOT / "configs" / "reference_original.yaml")
    assert cfg["schema_version"] == 1
    assert cfg["study"]["name"] == "reference-original"
    assert cfg["compute"]["device"] == "cuda"
    assert cfg["models"]["dvfm"]["epochs"] == 200


def test_grid_expands_to_atomic_scenarios():
    scenarios = expand_scenarios({"grid": {"copula": ["clayton", "frank"], "theta": [1, 2]}})
    assert len(scenarios) == 4
    assert scenarios[0] == {"copula": "clayton", "theta": 1}


def test_missing_synthetic_dependence_parameter_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "schema_version: 1\nstudy: {name: bad, seeds: [0]}\n"
        "data: {source: synthetic_copula, n_samples: 10, n_features: 2, scenarios: [{copula: clayton, kendall_tau: 0.5}]}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="requires theta"):
        load_config(path)


def test_gaussian_pilot_has_requested_grid_and_separate_seeds():
    cfg = load_config(ROOT / "configs" / "synthetic_pilot.yaml")
    scenarios = expand_scenarios(cfg["data"])
    assert len(scenarios) == 12
    assert cfg["data"]["n_samples"] == 10000
    assert {item["kendall_tau"] for item in scenarios} == {0.0, 0.25, 0.5, 0.75}
    assert {item["censoring_rate"] for item in scenarios} == {0.25, 0.5, 0.75}
    assert cfg["models"]["dvfm"]["latent_dims"] == [0, 1, 5]
    assert cfg["models"]["dvfm"]["epochs"] == 200
    assert cfg["models"]["dvfm"]["beta_max"] == 1.0
    assert cfg["models"]["dvfm"]["warmup_epochs"] == 50
    assert cfg["models"]["dvfm"]["checkpoint_min_epoch"] == 50
    assert cfg["models"]["dvfm"]["primary_checkpoint"] == "best_validation_elbo_post_warmup"
    assert cfg["models"]["dvfm"]["numerical_failure_threshold"] == 100.0
    assert set(cfg["models"]["enabled"]) == {"coxph", "deepsurv", "mtlr", "clayton_aft", "dvfm"}
    assert set(cfg["seeds"]) == {"dgp", "sampling", "split", "model"}
    assert len(cfg["seeds"]["sampling"]) == 5
    assert "seeds" not in cfg["study"]


def test_primary_synthetic_config_has_full_paired_benchmark():
    cfg = load_config(ROOT / "configs" / "synthetic.yaml")
    scenarios = expand_scenarios(cfg["data"])
    assert len(scenarios) == 12
    assert cfg["study"] == {
        "name": "synthetic", "stage": "primary", "output_dir": "results/synthetic"
    }
    assert cfg["data"]["n_samples"] == 10000
    assert cfg["data"]["n_features"] == 10
    assert cfg["split"]["strategy"] == "holdout"
    assert cfg["split"]["validation_fraction"] == 0.10
    assert cfg["split"]["test_fraction"] == 0.20
    assert {item["kendall_tau"] for item in scenarios} == {0.0, 0.25, 0.5, 0.75}
    assert {item["censoring_rate"] for item in scenarios} == {0.25, 0.5, 0.75}
    assert len(cfg["seeds"]["sampling"]) == 10
    assert len(cfg["seeds"]["split"]) == 10
    assert len(cfg["seeds"]["model"]) == 10
    assert cfg["models"]["enabled"] == ["dvfm"]
    assert cfg["models"]["dvfm"]["latent_dims"] == [0, 1]
    assert cfg["models"]["dvfm"]["weight_decay"] == 0.0001
    assert cfg["models"]["dvfm"]["scale_link"] == "exp"
    assert cfg["models"]["dvfm"]["primary_checkpoint"] == (
        "best_validation_elbo_post_warmup"
    )
    assert cfg["models"]["dvfm"]["checkpoint_min_epoch"] == 50


def test_support_semisynthetic_config_has_paired_generator_and_split():
    cfg = load_config(ROOT / "configs" / "semi_synthetic.yaml")
    assert cfg["data"]["source"] == "support_cox_clayton_semisynthetic"
    assert cfg["data"]["copula"] == "clayton"
    assert cfg["data"]["kendall_tau"] == 0.5
    assert cfg["data"]["censoring_rates"] == [0.25, 0.5, 0.75]
    assert cfg["split"]["stratify"] == "time_event"
    assert cfg["split"]["validation_fraction"] == 0.10
    assert cfg["split"]["test_fraction"] == 0.20
    assert len(cfg["seeds"]["sampling"]) == len(cfg["seeds"]["split"])
    assert len(cfg["seeds"]["split"]) == len(cfg["seeds"]["model"])
    assert cfg["models"]["dvfm"]["primary_checkpoint"] == (
        "best_validation_elbo_post_warmup"
    )
    assert cfg["models"]["dvfm"]["scale_link"] == "exp"


def test_hyperparameter_sweep_is_paired_and_keeps_reference_fixed():
    cfg = load_config(ROOT / "configs" / "synthetic_hyperparameter_sweep.yaml")
    scenarios = expand_scenarios(cfg["data"])
    variants = {item["name"]: item for item in cfg["models"]["dvfm"]["variants"]}
    assert len(scenarios) == 7
    assert len(variants) == 8
    assert cfg["models"]["dvfm"]["latent_dims"] == [1]
    assert cfg["models"]["dvfm"]["epochs"] == 200
    assert cfg["models"]["dvfm"]["warmup_epochs"] == 50
    assert cfg["models"]["dvfm"]["learning_rate"] == 0.001
    assert variants["reference"] == {"name": "reference"}
    assert variants["epochs_400"]["epochs"] == 400
    assert variants["batch_32"]["batch_size"] == 32
    assert variants["batch_128"]["batch_size"] == 128
    assert variants["wider_decoder"]["decoder_hidden"] == [64, 128]
    assert variants["weight_decay_1e_4"]["weight_decay"] == 0.0001
    assert {item.get("mechanism") for item in scenarios} == {
        "gaussian_shared_frailty", "clayton_gamma_frailty"
    }
    assert cfg["evaluation"]["evaluate_partitions"] == ["validation", "test"]
    assert len(cfg["seeds"]["sampling"]) == 3


def test_dependence_calibration_is_a_paired_sequential_ablation():
    cfg = load_config(ROOT / "configs" / "synthetic_dependence_calibration.yaml")
    scenarios = expand_scenarios(cfg["data"])
    variants = cfg["models"]["dvfm"]["variants"]
    assert [item["name"] for item in variants] == [
        "reference", "exp_scale_link", "additive_log_scale",
        "global_weibull_shape",
    ]
    assert {item["censoring_rate"] for item in scenarios} == {0.25}
    assert [item["kendall_tau"] for item in scenarios] == [0.0, 0.25, 0.5, 0.75]
    assert cfg["models"]["dvfm"]["latent_dims"] == [1]
    assert len(cfg["seeds"]["sampling"]) == 5
    assert variants[1]["comparison_parent"] == "reference"
    assert variants[1]["scale_link"] == "exp"
    assert variants[2]["comparison_parent"] == "exp_scale_link"
    assert variants[2]["latent_path"] == "additive_scale"
    assert variants[3]["comparison_parent"] == "additive_log_scale"
    assert variants[3]["shape_mode"] == "global"


def test_hacsurv_feasibility_pilot_runs_only_hacsurv():
    cfg = load_config(ROOT / "configs" / "hacsurv_synthetic_pilot.yaml")
    assert cfg["models"]["enabled"] == ["hacsurv_2d"]
    assert cfg["data"]["n_samples"] == 10000
    assert len(expand_scenarios(cfg["data"])) == 1
    assert len(cfg["seeds"]["sampling"]) == 1
    assert cfg["models"]["hacsurv_2d"]["checkpoint_min_epoch"] > cfg["models"]["hacsurv_2d"]["copula_start_epoch"]
    assert cfg["evaluation"]["compute_oracle_joint_survival_ise"] is True


def test_frailty_diagnostic_encodes_prespecified_four_way_comparison():
    cfg = load_config(ROOT / "configs" / "frailty_recovery_diagnostic.yaml")
    variants = {item["name"]: item for item in cfg["training_variants"]}
    assert set(variants) == {"current", "checkpoint_fix", "old_training", "schedule_isolation"}
    assert cfg["data"]["kendall_tau"] == 0.5
    assert cfg["data"]["censoring_rate"] == 0.5
    assert cfg["models"]["dvfm"]["latent_dim"] == 1
    assert variants["current"]["checkpoint_selection"] == "final"
    assert variants["checkpoint_fix"]["checkpoint_selection"] == "best_validation_reconstruction_nll"
    assert variants["old_training"]["beta_max"] == 0.2
    assert variants["old_training"]["warmup_epochs"] == 150
    assert variants["old_training"]["learning_rate"] == 0.0005
    assert variants["schedule_isolation"]["beta_max"] == 1.0
    assert len(cfg["seeds"]["sampling"]) == 5
