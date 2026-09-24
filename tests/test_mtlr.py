import numpy as np
import torch

from sota.mtlr import (
    MTLR,
    encode_mtlr_targets,
    interval_probabilities,
    mtlr_neg_log_likelihood,
    mtlr_survival,
    train_mtlr,
)


def test_event_and_censored_target_encoding():
    boundaries = np.array([1.0, 2.0, 3.0])
    target = encode_mtlr_targets(
        np.array([0.5, 1.5, 2.0, 4.0]),
        np.array([1, 0, 1, 0]),
        boundaries,
    )
    expected = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 1, 1],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ], dtype=torch.float32)
    torch.testing.assert_close(target, expected)


def test_likelihood_matches_manual_event_and_censoring_calculation():
    probabilities = torch.tensor([
        [0.1, 0.2, 0.3, 0.4],
        [0.1, 0.2, 0.3, 0.4],
    ], dtype=torch.float64)
    logits = probabilities.log()
    target = torch.tensor([
        [0, 1, 0, 0],
        [0, 1, 1, 1],
    ], dtype=torch.float64)
    expected = (-np.log(0.2) - np.log(0.2 + 0.3 + 0.4)) / 2
    torch.testing.assert_close(
        mtlr_neg_log_likelihood(logits, target),
        torch.tensor(expected, dtype=torch.float64),
    )


def test_loss_and_gradients_are_finite_and_model_uses_lower_triangular_coding():
    model = MTLR(3, 5, hidden_dims=[4], dropout=0.1)
    assert torch.equal(model.G, torch.tril(torch.ones(4, 5)))
    features = torch.randn(6, 3)
    targets = encode_mtlr_targets(
        np.array([0.2, 0.6, 1.2, 1.8, 2.5, 3.5]),
        np.array([1, 0, 1, 0, 1, 0]),
        np.array([0.5, 1.0, 2.0, 3.0]),
    )
    loss = mtlr_neg_log_likelihood(model(features), targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_interval_probabilities_sum_to_one_and_survival_is_valid():
    logits = torch.randn(8, 6)
    probabilities = interval_probabilities(logits)
    survival = mtlr_survival(logits)
    torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(8))
    torch.testing.assert_close(survival[:, 0], torch.ones(8))
    assert ((survival >= 0) & (survival <= 1)).all()
    assert (torch.diff(survival, dim=1) <= 1e-7).all()


def test_train_mtlr_output_shapes_match_runner_and_curves_are_valid():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(30, 3)).astype(np.float32)
    times = rng.uniform(0.1, 4.0, size=30)
    events = rng.random(30) > 0.35
    grid = np.linspace(0.0, 4.0, 13)

    risk, median, curves = train_mtlr(
        X[:24], times[:24], events[:24], X[24:],
        num_bins=6, n_epochs=2, batch_size=8, lr=1e-3,
        hidden_dims=[5], dropout=0.0, weight_decay=1e-4,
        device="cuda", eval_time_points=grid,
    )

    assert risk.shape == (6,)
    assert median.shape == (6,)
    assert curves.shape == (6, len(grid))
    assert np.isfinite(risk).all() and np.isfinite(median).all()
    assert np.isfinite(curves).all()
    np.testing.assert_allclose(curves[:, 0], 1.0)
    assert ((curves >= 0.0) & (curves <= 1.0)).all()
    assert (np.diff(curves, axis=1) <= 1e-10).all()
