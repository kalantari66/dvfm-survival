from pathlib import Path

import pytest

from dvfm.config import expand_scenarios, load_config


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
    assert {item["kendall_tau"] for item in scenarios} == {0.0, 0.25, 0.5, 0.75}
    assert {item["censoring_rate"] for item in scenarios} == {0.25, 0.5, 0.75}
    assert cfg["models"]["dvfm"]["latent_dims"] == [0, 1, 5]
    assert set(cfg["seeds"]) == {"dgp", "sampling", "split", "model"}
    assert len(cfg["seeds"]["sampling"]) == 5
    assert "seeds" not in cfg["study"]


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
