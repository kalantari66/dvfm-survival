"""Functional tests only; these are not performance benchmarks."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from dvfm.metrics import compute_ipcw_brier_ibs, compute_oracle_brier_ibs
from dvfm.baselines import ClaytonWeibullAFT
from dvfm.model import DVFM, SurvivalDataset
from dvfm.hacsurv import HACSurv2D
from dvfm.joint_metrics import (
    JointSurvivalEvaluation,
    oracle_joint_survival_ise,
)
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


def test_dvfm_rejects_all_numerically_invalid_checkpoint_epochs():
    X, time, event, _, _ = generate_copula_data(
        n_samples=96, n_features=3, copula_type="clayton", theta=1.0, seed=32
    )
    loader = DataLoader(SurvivalDataset(X, time, event), batch_size=32, shuffle=False)
    model = DVFM(input_dim=3, latent_dim=1, encoder_hidden=[8], decoder_hidden=[8])
    with np.testing.assert_raises_regex(RuntimeError, "No finite validation ELBO"):
        train_dvfm(
            model, loader, loader, n_epochs=2, warmup_epochs=1,
            checkpoint_min_epoch=1, numerical_failure_threshold=1e-12,
            return_artifacts=True,
        )


def test_dvfm_supports_dropout_and_adam_weight_decay():
    rng = np.random.default_rng(23)
    X = rng.normal(size=(96, 3)).astype(np.float32)
    time = rng.uniform(0.2, 2.0, size=96).astype(np.float32)
    event = rng.integers(0, 2, size=96).astype(np.float32)
    loader = DataLoader(
        SurvivalDataset(X, time, event), batch_size=32, shuffle=False
    )
    model = DVFM(
        input_dim=3, latent_dim=1, encoder_hidden=[8], decoder_hidden=[8],
        dropout=0.1,
    )
    artifacts = train_dvfm(
        model, loader, loader, n_epochs=2, warmup_epochs=1,
        checkpoint_min_epoch=1, numerical_failure_threshold=100.0,
        weight_decay=1e-4, return_artifacts=True,
    )
    assert any(isinstance(layer, torch.nn.Dropout) for layer in model.encoder.network)
    assert any(isinstance(layer, torch.nn.Dropout) for layer in model.decoder.network)
    assert artifacts["best_validation_elbo_epoch"] in {1, 2}


def test_hacsurv_2d_has_finite_likelihood_gradients_and_monotone_survival():
    torch.manual_seed(41)
    model = HACSurv2D(
        input_dim=3, hidden_size=8, hidden_survival=8,
        generator_samples=20, inverse_iterations=100, inverse_tolerance=1e-6,
    ).double()
    x = torch.randn(32, 3, dtype=torch.float64)
    time = torch.linspace(0.2, 2.0, 32, dtype=torch.float64)
    event = torch.arange(32).remainder(2).double()
    model.generator.resample(20)
    loss = -model.log_likelihood(x, time, event)
    loss.backward()
    generator_gradients = [
        parameter.grad for parameter in model.generator.parameters()
        if parameter.grad is not None
    ]
    assert torch.isfinite(loss)
    assert generator_gradients
    assert all(torch.isfinite(value).all() for value in generator_gradients)
    with torch.no_grad():
        early = model.event_survival(x, torch.full((32,), 0.25, dtype=torch.float64))
        late = model.event_survival(x, torch.full((32,), 2.5, dtype=torch.float64))
    assert torch.all(early >= late)


def test_oracle_joint_survival_ise_is_zero_for_truth():
    event_grid = np.linspace(0.0, 2.0, 5)
    censor_grid = np.linspace(0.0, 3.0, 6)
    truth = np.linspace(1.0, 0.0, 30).reshape(1, 5, 6)
    evaluation = JointSurvivalEvaluation(
        X=np.zeros((1, 2)), event_grid=event_grid,
        censor_grid=censor_grid, truth=truth,
    )
    assert oracle_joint_survival_ise(truth, evaluation) == 0.0


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
