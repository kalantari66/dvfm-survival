from pathlib import Path

from dvfm.config import load_config


def test_reference_parameters_are_not_shortened():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "reference_original.yaml")
    dvfm = cfg["models"]["dvfm"]
    assert dvfm["epochs"] == 200
    assert dvfm["lr"] == 0.001
    assert dvfm["batch_size"] == 64
    assert dvfm["latent_dim"] == 20
    assert dvfm["mc_samples"] == 100
    assert dvfm["warmup_epochs"] == 50
    assert cfg["models"]["deepsurv"]["epochs"] == 200
    assert cfg["models"]["mtlr"]["epochs"] == 200
    assert cfg["models"]["mtlr"]["bins"] == 200
    assert cfg["synthetic"]["n_samples"] == 10000
    assert cfg["split"]["test_size"] == 0.30
    assert cfg["evaluation"]["n_time_points"] == 1000
