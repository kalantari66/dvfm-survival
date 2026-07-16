import numpy as np
import torch
from torch.utils.data import DataLoader

from dvfm.metrics import compute_ipcw_brier_ibs, compute_oracle_brier_ibs
from dvfm.model import DVFM, SurvivalDataset
from dvfm.prediction import predict_survival_curves
from dvfm.synthetic import generate_copula_data


def test_generator_shapes():
    X, time, event, true_t, true_c = generate_copula_data(
        n_samples=128, n_features=5, copula_type="clayton", theta=1.0, seed=7
    )
    assert X.shape == (128, 5)
    assert time.shape == event.shape == true_t.shape == true_c.shape == (128,)
    assert set(np.unique(event)).issubset({0, 1})


def test_dvfm_forward_and_metrics():
    X, time, event, true_t, _ = generate_copula_data(
        n_samples=128, n_features=5, copula_type="clayton", theta=1.0, seed=8
    )
    loader = DataLoader(SurvivalDataset(X, time, event), batch_size=32, shuffle=False)
    model = DVFM(input_dim=5, latent_dim=3)
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
