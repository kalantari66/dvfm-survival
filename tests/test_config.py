from pathlib import Path

import pytest
import yaml

from experiments.config import expand_scenarios, expand_seed_streams, load_config, semisynthetic_datasets
from experiments.runner import _models_for_dataset


ROOT = Path(__file__).resolve().parents[1]


def test_semisynthetic_dataset_names_and_features_are_validated():
    data = load_config(ROOT / "configs/semi_synthetic.yaml")["data"]
    data["datasets"].append({**data["datasets"][0], "name": "another_cohort"})
    assert len(semisynthetic_datasets(data)) == 13
    data["datasets"][-1]["name"] = "WHAS"
    with pytest.raises(ValueError, match="Duplicate dataset"):
        semisynthetic_datasets(data)
    data["datasets"][-1]["name"] = "../outside"
    with pytest.raises(ValueError, match="Dataset names"):
        semisynthetic_datasets(data)
    data["datasets"][-1]["name"] = "another_cohort"
    data["datasets"][-1]["numeric_features"] = ["time"]
    with pytest.raises(ValueError, match="outcome columns"):
        semisynthetic_datasets(data)


def test_grid_expands_to_atomic_scenarios():
    scenarios = expand_scenarios({"grid": {"copula": ["clayton", "frank"], "theta": [1, 2]}})
    assert len(scenarios) == 4
    assert scenarios[0] == {"copula": "clayton", "theta": 1}


def test_seed_list_assigns_each_repeat_seed_to_every_component():
    streams = expand_seed_streams([0, 1, 2])
    assert streams == [
        {"dgp": 0, "sampling": 0, "split": 0, "model": 0},
        {"dgp": 1, "sampling": 1, "split": 1, "model": 1},
        {"dgp": 2, "sampling": 2, "split": 2, "model": 2},
    ]


def test_missing_synthetic_dependence_parameter_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "schema_version: 1\nstudy: {name: bad, seeds: [0]}\n"
        "data: {source: synthetic_copula, n_samples: 10, n_features: 2, scenarios: [{copula: clayton, kendall_tau: 0.5}]}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="requires theta"):
        load_config(path)


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
    assert cfg["seeds"] == list(range(10))
    assert cfg["models"]["enabled"] == ["dvfm"]
    assert cfg["models"]["dvfm"]["latent_dims"] == [0, 1]
    assert cfg["models"]["dvfm"]["weight_decay"] == 0.0001
    assert cfg["models"]["dvfm"]["latent_loading_l1"] == 0.1
    assert cfg["models"]["dvfm"]["scale_link"] == "exp"
    assert cfg["models"]["dvfm"]["latent_loading_l1"] == 0.1
    assert cfg["models"]["dvfm"]["primary_checkpoint"] == (
        "best_validation_elbo_post_warmup"
    )
    assert cfg["models"]["dvfm"]["checkpoint_min_epoch"] == 50


def test_support_semisynthetic_config_has_paired_generator_and_split():
    cfg = load_config(ROOT / "configs" / "semi_synthetic.yaml")
    assert cfg["data"]["source"] == "cox_clayton_semisynthetic"
    assert [item["name"] for item in cfg["data"]["datasets"]] == [
        "whas", "gbsg", "metabric", "churn", "nacd", "flchain", "support",
        "employee", "mimic_iv", "seer_brain", "seer_liver", "seer_stomach",
    ]
    assert cfg["data"]["copula"] == "clayton"
    assert cfg["data"]["copulas"] == ["gaussian", "clayton", "frank", "gumbel"]
    flchain = next(item for item in cfg["data"]["datasets"] if item["name"] == "flchain")
    assert "chapter" not in flchain["categorical_features"]
    assert "drop_encoded_features" not in flchain
    assert flchain["numeric_imputation"] == "median"
    subsampled = {
        item["name"]: item["subsample"] for item in cfg["data"]["datasets"]
        if "subsample" in item
    }
    assert set(subsampled) == {
        "employee", "mimic_iv", "seer_brain", "seer_liver", "seer_stomach",
    }
    assert all(item == {"target_size": 10000, "time_bins": 10, "random_seed": 42}
               for item in subsampled.values())
    assert cfg["data"]["kendall_tau"] == [0.0, 0.5]
    assert cfg["models"]["coxph"] == {
        "alpha": 1e-4, "ties": "breslow", "n_iter": 100, "tol": 1e-9,
    }
    assert cfg["models"]["dvfm"]["latent_dims"] == [1]
    assert cfg["evaluation"]["n_time_points"] == 100
    assert cfg["evaluation"]["save_latent_recovery"] is True
    assert cfg["evaluation"]["save_event_distribution_plots"] is False
    assert cfg["evaluation"]["compute_oracle_joint_survival_ise"] is True
    assert cfg["data"]["censoring_rates"] == "original"
    assert cfg["resources"] == {
        "cores": 4, "memory": "25g", "walltime": "06:00:00",
        "gpu_walltime": "02:00:00", "partition": "gpu-short",
        "account": "c2i-colon",
    }
    assert cfg["compute"] == {"device": "cuda", "torch_num_threads": 4}
    assert cfg["split"]["stratify"] == "time_event"
    assert cfg["split"]["validation_fraction"] == 0.10
    assert cfg["split"]["test_fraction"] == 0.20
    assert cfg["seeds"] == list(range(10))
    assert cfg["models"]["dvfm"]["primary_checkpoint"] == (
        "best_validation_elbo_post_warmup"
    )
    assert cfg["models"]["dvfm"]["scale_link"] == "exp"
    assert set(cfg["models"]["enabled"]) == {
        "coxph", "deepsurv", "mtlr", "clayton_aft", "hacsurv_2d",
        "bayesian_cox_gamma_frailty", "dvfm", "rsf",
    }


def test_semi_synthetic_oracle_tuning_winners_are_promoted():
    cfg = load_config(ROOT / "configs" / "semi_synthetic.yaml")
    expected_datasets = {
        item["name"] for item in semisynthetic_datasets(cfg["data"])
    }
    assert set(cfg["models"]["legacy_ipcw_tuning_not_for_final_runs"]) == expected_datasets
    assert set(cfg["models"]["tuned_by_dataset"]) == expected_datasets
    assert cfg["evaluation"]["primary_metrics"] == ["oracle_ibs"]
    assert cfg["evaluation"]["secondary_sensitivity_metrics"] == ["ibs_ipcw"]
    assert "rsf" in cfg["models"]["enabled"]
    assert "gbsa" not in cfg["models"]["enabled"]

    churn = _models_for_dataset(cfg, "churn")
    assert churn["dvfm"]["encoder_hidden"] == [64, 32]
    assert churn["dvfm"]["decoder_hidden"] == [32, 64]
    assert churn["hacsurv_2d"]["learning_rate"] == 0.0003
    assert churn["hacsurv_2d"]["copula_learning_rate"] == 0.0003
    assert churn["rsf"]["n_estimators"] == 300
    assert churn["gbsa"]["n_estimators"] == 500
    assert churn["mtlr"] == {
        "epochs": 200,
        "batch_size": 64,
        "early_stopping_patience": None,
        "bins": 200,
        "hidden_dims": [16],
        "dropout": 0.25,
        "learning_rate": 0.003,
        "weight_decay": 0.001,
    }

    with (ROOT / "configs" / "semi_synthetic_tuning.yaml").open(encoding="utf-8") as handle:
        tuning = yaml.safe_load(handle)
    assert tuning["tuning"]["selection_metric"] == "IBS Oracle"
    assert set(tuning["tuning"]["search_spaces"]["rsf"]) == {
        "n_estimators", "max_depth", "min_samples_split",
        "min_samples_leaf", "max_features",
    }
    assert set(tuning["tuning"]["search_spaces"]["gbsa"]) == {
        "n_estimators", "learning_rate", "max_depth", "min_samples_split",
        "min_samples_leaf", "max_features", "subsample",
    }
