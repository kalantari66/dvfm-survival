"""Functional tests only; these are not performance benchmarks."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from dvfm.metrics import compute_ipcw_brier_ibs, compute_oracle_brier_ibs
from dvfm.baselines import ClaytonWeibullAFT
from dvfm.model import DVFM, SurvivalDataset
from dvfm.prediction import predict_survival_curves
from dvfm.training import train_dvfm
from dvfm.synthetic import (
    generate_clayton_aft_data,
    generate_clayton_gamma_frailty,
    generate_copula_data,
    generate_gaussian_shared_frailty,
)


def test_generator_shapes():
    X, time, event, true_t, true_c = generate_copula_data(
        n_samples=128, n_features=5, copula_type="clayton", theta=1.0, seed=7
    )
    assert X.shape == (128, 5)
    assert time.shape == event.shape == true_t.shape == true_c.shape == (128,)
    assert set(np.unique(event)).issubset({0, 1})


def test_dvfm_forward_prediction_and_metrics():
    X, time, event, true_t, _ = generate_copula_data(
        n_samples=128, n_features=5, copula_type="clayton", theta=1.0, seed=8
    )
    loader = DataLoader(SurvivalDataset(X, time, event), batch_size=32, shuffle=False)
    model = DVFM(input_dim=5, latent_dim=20)
    x, t, e = next(iter(loader))
    outputs = model(x, t, e)
    loss, _, _ = model.loss_function(*outputs, t, e)
    assert torch.isfinite(loss)

    grid = np.linspace(0, float(np.max(time) * 1.2), 20)
    curves = predict_survival_curves(model, X[:8], grid, loader, n_samples=2)
    assert curves.shape == (8, 20)
    assert np.all(np.isfinite(curves))
    _, oracle = compute_oracle_brier_ibs(curves, grid, true_t[:8])
    _, ipcw, _ = compute_ipcw_brier_ibs(curves, grid, time[:8], event[:8])
    assert np.isfinite(oracle)
    assert np.isfinite(ipcw)


def test_dvfm_post_warmup_elbo_checkpoint_respects_minimum_epoch():
    X, time, event, _, _ = generate_copula_data(
        n_samples=96, n_features=3, copula_type="clayton", theta=1.0, seed=31
    )
    loader = DataLoader(SurvivalDataset(X, time, event), batch_size=32, shuffle=False)
    model = DVFM(input_dim=3, latent_dim=1, encoder_hidden=[8], decoder_hidden=[8])
    artifacts = train_dvfm(
        model, loader, loader, n_epochs=3, warmup_epochs=2,
        checkpoint_min_epoch=2, return_artifacts=True,
    )
    assert len(artifacts["history"]) == 3
    assert artifacts["best_validation_elbo_epoch"] in {2, 3}
    assert set(artifacts["final_state"]) == set(artifacts["best_validation_elbo_state"])


def test_clayton_prediction_uses_model_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClaytonWeibullAFT(n_features=3).to(device)
    curves = model.predict_survival(
        np.zeros((4, 3), dtype=np.float32),
        np.linspace(0.0, 2.0, 5),
    )
    assert curves.shape == (4, 5)
    assert np.all(np.isfinite(curves))


def test_gaussian_frailty_generator_is_reproducible_and_hits_targets():
    kwargs = dict(n_samples=4000, n_features=3, kendall_tau=0.5, censoring_rate=0.25,
                  dgp_seed=11, sampling_seed=12, calibration_samples=20_000)
    first = generate_gaussian_shared_frailty(**kwargs)
    second = generate_gaussian_shared_frailty(**kwargs)
    np.testing.assert_array_equal(first.observed_time, second.observed_time)
    np.testing.assert_array_equal(first.event, second.event)
    assert abs(first.empirical_conditional_kendall_tau - 0.5) < 0.03
    assert abs(first.achieved_censoring_rate - 0.25) < 1 / len(first.event) + 1e-12


def test_clayton_gamma_generator_exposes_true_frailty_and_hits_targets():
    kwargs = dict(n_samples=5000, n_features=3, kendall_tau=0.5, censoring_rate=0.5,
                  dgp_seed=21, sampling_seed=22)
    first = generate_clayton_gamma_frailty(**kwargs)
    second = generate_clayton_gamma_frailty(**kwargs)
    np.testing.assert_array_equal(first.true_z, second.true_z)
    np.testing.assert_array_equal(first.event, second.event)
    assert abs(first.empirical_conditional_kendall_tau - 0.5) < 0.03
    assert abs(first.achieved_censoring_rate - 0.5) < 1 / len(first.event) + 1e-12
    assert abs(float(first.true_z.mean())) < 1e-6
    assert abs(float(first.true_z.std()) - 1.0) < 1e-6


def test_clayton_aft_likelihood_matches_its_exact_generator():
    sample = generate_clayton_aft_data(n_samples=10_000, n_features=3,
                                       clayton_theta=2.0, seed=19)
    model = ClaytonWeibullAFT(n_features=3)
    with torch.no_grad():
        model.beta_t.copy_(torch.as_tensor(sample.beta_event, dtype=torch.float32))
        model.beta_c.copy_(torch.as_tensor(sample.beta_censor, dtype=torch.float32))
        model.log_shape_t.fill_(np.log(sample.shape_event))
        model.log_shape_c.fill_(np.log(sample.shape_censor))
        model.raw_theta.fill_(np.log(np.expm1(sample.clayton_theta - 1e-4)))
    x_aug = np.column_stack([sample.X, np.ones(len(sample.X), dtype=np.float32)])
    true_nll = model.neg_log_lik(torch.as_tensor(x_aug, dtype=torch.float32),
                                 torch.as_tensor(sample.observed_time, dtype=torch.float32),
                                 torch.as_tensor(sample.event, dtype=torch.float32))
    with torch.no_grad():
        model.beta_t.add_(0.75)
        model.beta_c.sub_(0.75)
        wrong_nll = model.neg_log_lik(torch.as_tensor(x_aug, dtype=torch.float32),
                                      torch.as_tensor(sample.observed_time, dtype=torch.float32),
                                      torch.as_tensor(sample.event, dtype=torch.float32))
    assert torch.isfinite(true_nll)
    assert true_nll < wrong_nll
