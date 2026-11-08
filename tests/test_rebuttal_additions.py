"""Tests for the rebuttal additions; every default path must stay unchanged."""

import numpy as np
import pytest
import torch
from scipy.stats import kendalltau
from torch.utils.data import DataLoader

from dvfm.model import DVFM, TwoTrunkDecoder
from dvfm.prediction import (
    aggregate_posterior_mixture,
    predict_survival_curves,
    predict_survival_from_mixture,
)
from experiments.runner import fit_dvfm
from utility.data import SurvivalData, SurvivalDataset
from utility.metrics import (
    JointSurvivalEvaluation,
    learned_conditional_kendall_tau,
    predict_dvfm_joint_survival,
)
from utility.synthetic import (
    generate_gaussian_shared_frailty,
    generate_stratified_gaussian_shared_frailty,
)


def _toy(n=64, d=3, seed=0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=generator)
    time = torch.rand(n, generator=generator) + 0.2
    event = (torch.rand(n, generator=generator) > 0.4).float()
    return x, time, event


def _reference_predict_survival_curves(model, X, time_points, train_loader, n_samples, device):
    """The pre-refactor implementation, kept verbatim as the reference."""
    model.eval().to(device)
    mus, logvars = [], []
    with torch.no_grad():
        for x_batch, t_batch, event_batch in train_loader:
            mu, logvar = model.encoder(x_batch, t_batch, event_batch)
            mus.append(mu)
            logvars.append(logvar)
    all_mus = torch.cat(mus)
    all_stds = torch.exp(0.5 * torch.cat(logvars))
    x_tensor = torch.as_tensor(X, dtype=torch.float32, device=device)
    grid = torch.as_tensor(time_points, dtype=torch.float32, device=device)
    result = torch.zeros((len(X), len(time_points)), device=device)
    with torch.no_grad():
        for _ in range(n_samples):
            indices = torch.randint(0, len(all_mus), (len(X),), device=device)
            z = all_mus[indices] + all_stds[indices] * torch.randn_like(all_mus[indices])
            shape, scale, _, _ = model.decoder(x_tensor, z)
            result += torch.exp(-((grid[None, :] / scale[:, None]) ** shape[:, None]))
    return (result / n_samples).cpu().numpy()


def test_refactored_aggregate_posterior_prediction_is_bit_identical():
    torch.manual_seed(3)
    x, time, event = _toy()
    model = DVFM(3, 1, encoder_hidden=[8], decoder_hidden=[8], scale_link="exp")
    grid = np.linspace(0.0, 1.5, 7)
    outputs = []
    for predictor in (_reference_predict_survival_curves, predict_survival_curves):
        loader = DataLoader(SurvivalDataset(x.numpy(), time.numpy(), event.numpy()),
                            batch_size=16, shuffle=True)
        torch.manual_seed(11)
        outputs.append(predictor(model, x.numpy()[:10], grid, loader, 20, "cpu"))
    np.testing.assert_array_equal(outputs[0], outputs[1])


def test_cached_mixture_is_the_encoded_training_set_and_composes_exactly():
    torch.manual_seed(5)
    x, time, event = _toy()
    model = DVFM(3, 1, encoder_hidden=[8], decoder_hidden=[8])
    grid = np.linspace(0.0, 1.5, 7)
    loader = DataLoader(SurvivalDataset(x.numpy(), time.numpy(), event.numpy()),
                        batch_size=16, shuffle=False)
    # Creating a DataLoader iterator draws from the torch RNG even without
    # shuffling, so both paths are seeded before the training set is encoded.
    torch.manual_seed(13)
    mu, std = aggregate_posterior_mixture(model, loader)
    cached = predict_survival_from_mixture(model, x.numpy()[:10], grid, mu, std, 20)
    torch.manual_seed(13)
    uncached = predict_survival_curves(model, x.numpy()[:10], grid, loader, 20)
    np.testing.assert_array_equal(cached, uncached)
    with torch.no_grad():
        direct_mu, direct_logvar = model.eval().encoder(x, time, event)
    torch.testing.assert_close(mu, direct_mu)
    torch.testing.assert_close(std, torch.exp(0.5 * direct_logvar))


def _joint_evaluation(n=4, d=3):
    grid = np.linspace(0.0, 1.0, 5)
    return JointSurvivalEvaluation(
        np.random.default_rng(0).normal(size=(n, d)), grid, grid, np.zeros((n, 5, 5))
    )


def test_joint_survival_default_still_samples_the_prior():
    torch.manual_seed(7)
    model = DVFM(3, 1, encoder_hidden=[8], decoder_hidden=[8]).eval()
    evaluation = _joint_evaluation()
    kwargs = dict(mc_samples=50, batch_size=2, seed=4, device=torch.device("cpu"))
    default = predict_dvfm_joint_survival(model, evaluation, **kwargs)
    generator = torch.Generator().manual_seed(4)
    manual = []
    with torch.no_grad():
        for start in range(0, 4, 2):
            x = torch.as_tensor(evaluation.X[start:start + 2], dtype=torch.float32)
            z = torch.randn((2 * 50, 1), generator=generator)
            st, sc, ct, cc = model.decoder(x.repeat_interleave(50, dim=0), z)
            grid = torch.as_tensor(evaluation.event_grid, dtype=torch.float32)
            event = torch.exp(-(grid[None] / sc[:, None]).pow(st[:, None])).reshape(2, 50, -1)
            censor = torch.exp(-(grid[None] / cc[:, None]).pow(ct[:, None])).reshape(2, 50, -1)
            manual.append((torch.einsum("bme,bmc->bec", event, censor) / 50).numpy())
    np.testing.assert_allclose(default, np.concatenate(manual), rtol=1e-6)


def test_joint_survival_mixture_with_a_point_mass_is_deterministic():
    torch.manual_seed(7)
    model = DVFM(3, 1, encoder_hidden=[8], decoder_hidden=[8]).eval()
    evaluation = _joint_evaluation()
    point = (np.full((3, 1), 0.7), np.zeros((3, 1)))
    kwargs = dict(mc_samples=20, batch_size=2, device=torch.device("cpu"), mixture=point)
    first = predict_dvfm_joint_survival(model, evaluation, seed=1, **kwargs)
    second = predict_dvfm_joint_survival(model, evaluation, seed=2, **kwargs)
    np.testing.assert_allclose(first, second, rtol=1e-6)


def test_independent_two_trunk_routes_each_latent_to_one_margin_only():
    torch.manual_seed(9)
    decoder = TwoTrunkDecoder(3, 2, hidden_dims=[8], latent_routing="independent").eval()
    x = torch.randn(16, 3)
    z = torch.randn(16, 2)
    base = decoder(x, z)
    moved_censor_latent = decoder(x, torch.stack((z[:, 0], z[:, 1] + 3.0), dim=1))
    moved_event_latent = decoder(x, torch.stack((z[:, 0] + 3.0, z[:, 1]), dim=1))
    torch.testing.assert_close(base[0], moved_censor_latent[0])
    torch.testing.assert_close(base[1], moved_censor_latent[1])
    torch.testing.assert_close(base[2], moved_event_latent[2])
    torch.testing.assert_close(base[3], moved_event_latent[3])
    assert not torch.allclose(base[1], moved_event_latent[1])
    assert not torch.allclose(base[3], moved_censor_latent[3])


def test_shared_two_trunk_feeds_both_latents_to_both_margins():
    torch.manual_seed(9)
    decoder = TwoTrunkDecoder(3, 2, hidden_dims=[8], latent_routing="shared").eval()
    x, z = torch.randn(16, 3), torch.randn(16, 2)
    for column in (0, 1):
        moved = z.clone()
        moved[:, column] += 3.0
        out_base, out_moved = decoder(x, z), decoder(x, moved)
        assert not torch.allclose(out_base[1], out_moved[1])
        assert not torch.allclose(out_base[3], out_moved[3])


def test_two_trunk_validation_and_penalty_gradients():
    with pytest.raises(ValueError):
        DVFM(3, 1, decoder_architecture="two_trunk_independent")
    with pytest.raises(ValueError):
        DVFM(3, 2, decoder_architecture="two_trunk_shared", shape_mode="global")
    with pytest.raises(ValueError):
        DVFM(3, 2, decoder_architecture="three_trunk")
    torch.manual_seed(1)
    model = DVFM(3, 2, encoder_hidden=[8], decoder_hidden=[8],
                 decoder_architecture="two_trunk_independent")
    x, time, event = _toy()
    outputs = model(x, time, event)
    loss, _, _ = model.loss_function(*outputs, time, event)
    penalty, loading, _, gate = model.regularization_terms()
    (loss + penalty).backward()
    assert torch.isfinite(loss) and loading.item() > 0 and gate.item() == 1.0
    for trunk in (model.decoder.event_trunk, model.decoder.censor_trunk):
        assert torch.isfinite(trunk[0].weight.grad).all()


def test_independent_two_trunk_has_no_decoded_dependence():
    torch.manual_seed(2)
    model = DVFM(3, 2, encoder_hidden=[8], decoder_hidden=[16],
                 decoder_architecture="two_trunk_independent", scale_link="exp").eval()
    with torch.no_grad():
        for trunk in (model.decoder.event_trunk, model.decoder.censor_trunk):
            trunk[0].weight[:, 3:] *= 50.0  # make the latent dominate both margins
    tau = learned_conditional_kendall_tau(model, np.zeros(3), n_samples=6000, seed=0)
    assert abs(tau) < 0.04


def test_fit_dvfm_passes_the_decoder_architecture_and_defaults_to_single():
    rng = np.random.default_rng(0)
    data = SurvivalData(rng.normal(size=(80, 3)).astype(np.float32),
                        rng.uniform(0.2, 2.0, 80), rng.integers(0, 2, 80), ["a", "b", "c"])
    settings = {"latent_dim": 2, "batch_size": 32, "epochs": 3, "learning_rate": 1e-3,
                "beta_max": 1.0, "warmup_epochs": 1, "free_bits": 0.0,
                "checkpoint_min_epoch": 1, "encoder_hidden": [8], "decoder_hidden": [8]}
    model, _, _, _ = fit_dvfm(data, data, {**settings, "decoder_architecture": "two_trunk_shared"}, "cpu")
    assert isinstance(model.decoder, TwoTrunkDecoder)
    model, _, _, _ = fit_dvfm(data, data, settings, "cpu")
    assert model.decoder_architecture == "single"


def test_stratified_frailty_shares_draws_and_switches_dependence_by_stratum():
    common = dict(n_samples=20_000, n_features=4, kendall_tau=0.5, censoring_rate=0.5,
                  dgp_seed=3, sampling_seed=4, calibration_samples=20_000)
    plain = generate_gaussian_shared_frailty(**common)
    stratified = generate_stratified_gaussian_shared_frailty(**common)
    np.testing.assert_array_equal(plain.X, stratified.X)
    np.testing.assert_array_equal(plain.true_z, stratified.true_z)
    assert stratified.frailty_loading == pytest.approx(plain.frailty_loading)
    dependent = stratified.X[:, 0] >= 0.0
    np.testing.assert_array_equal(stratified.dependent_stratum, dependent)
    assert np.all(stratified.subject_loading[~dependent] == 0.0)
    assert abs(stratified.achieved_censoring_rate - 0.5) < 0.01
    assert abs(stratified.stratum_conditional_kendall_tau["independent"]) < 0.03
    assert abs(stratified.stratum_conditional_kendall_tau["dependent"] - 0.5) < 0.03
    independent_rows = ~dependent
    residual_e = np.log(stratified.event_time) - stratified.X @ stratified.beta_event
    assert abs(kendalltau(residual_e[independent_rows], stratified.true_z[independent_rows]).statistic) < 0.03
