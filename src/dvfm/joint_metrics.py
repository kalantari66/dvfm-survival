"""Oracle joint-distribution evaluation for synthetic event/censoring data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class JointSurvivalEvaluation:
    X: np.ndarray
    event_grid: np.ndarray
    censor_grid: np.ndarray
    truth: np.ndarray


def _joint_from_conditional(event_survival, censor_survival):
    return np.einsum(
        "bme,bmc->bec", event_survival, censor_survival, optimize=True
    ) / event_survival.shape[1]


def prepare_joint_survival_evaluation(
    generated, mechanism: str, test_X: np.ndarray,
    train_event_time: np.ndarray, train_censor_time: np.ndarray,
    *, n_time_points: int, max_quantile: float, n_subjects: int,
    dgp_samples: int, seed: int,
) -> JointSurvivalEvaluation:
    """Evaluate the known DGP joint survival on held-out covariates."""
    rng = np.random.default_rng(seed)
    count = min(int(n_subjects), len(test_X))
    indices = np.sort(rng.choice(len(test_X), size=count, replace=False))
    X = np.asarray(test_X[indices], dtype=np.float64)
    event_grid = np.linspace(
        0.0, float(np.quantile(train_event_time, max_quantile)), int(n_time_points)
    )
    censor_grid = np.linspace(
        0.0, float(np.quantile(train_censor_time, max_quantile)), int(n_time_points)
    )
    batch_size = 32
    surfaces = []
    mechanism = str(mechanism).lower()
    for start in range(0, count, batch_size):
        xb = X[start:start + batch_size]
        if mechanism == "gaussian_shared_frailty":
            latent = rng.normal(size=int(dgp_samples))
            event_scale = np.exp(
                xb @ generated.beta_event[:, None]
                + generated.frailty_loading * latent[None, :]
            )
            censor_scale = np.exp(
                xb @ generated.beta_censor[:, None] + generated.censor_intercept
                + generated.frailty_loading * latent[None, :]
            )
            event_hazard = (
                event_grid[None, None, :] / event_scale[:, :, None]
            ) ** generated.shape_event
            censor_hazard = (
                censor_grid[None, None, :] / censor_scale[:, :, None]
            ) ** generated.shape_censor
            event_survival = np.exp(-event_hazard)
            censor_survival = np.exp(-censor_hazard)
        elif mechanism == "clayton_gamma_frailty":
            frailty = rng.gamma(
                shape=1.0 / generated.clayton_theta,
                scale=1.0, size=int(dgp_samples),
            )
            event_scale = np.exp(xb @ generated.beta_event)[:, None]
            censor_scale = np.exp(
                xb @ generated.beta_censor + generated.censor_intercept
            )[:, None]
            event_hazard = (
                event_grid[None, :] / event_scale
            ) ** generated.shape_event
            censor_hazard = (
                censor_grid[None, :] / censor_scale
            ) ** generated.shape_censor
            event_survival = np.exp(
                -frailty[None, :, None]
                * np.expm1(np.minimum(
                    generated.clayton_theta * event_hazard[:, None, :], 50.0
                ))
            )
            censor_survival = np.exp(
                -frailty[None, :, None]
                * np.expm1(np.minimum(
                    generated.clayton_theta * censor_hazard[:, None, :], 50.0
                ))
            )
        else:
            raise ValueError(f"Unsupported joint-survival DGP: {mechanism}")
        surfaces.append(_joint_from_conditional(event_survival, censor_survival))
    truth = np.concatenate(surfaces, axis=0)
    return JointSurvivalEvaluation(X, event_grid, censor_grid, truth)


def predict_dvfm_joint_survival(
    model, evaluation: JointSurvivalEvaluation, *, mc_samples: int,
    batch_size: int, seed: int, device: torch.device,
) -> np.ndarray:
    """Compute E_z[S_T(t|x,z) S_C(c|x,z)] under the DVFM prior."""
    predictions = []
    dtype = next(model.parameters()).dtype
    generator = torch.Generator(device=device.type).manual_seed(int(seed))
    model.eval()
    with torch.no_grad():
        for start in range(0, len(evaluation.X), int(batch_size)):
            x = torch.as_tensor(
                evaluation.X[start:start + int(batch_size)],
                dtype=dtype, device=device,
            )
            batch = len(x)
            repeated_x = x.repeat_interleave(int(mc_samples), dim=0)
            z = torch.randn(
                (batch * int(mc_samples), model.latent_dim),
                dtype=dtype, device=device, generator=generator,
            )
            shape_t, scale_t, shape_c, scale_c = model.decoder(repeated_x, z)
            event_grid = torch.as_tensor(
                evaluation.event_grid, dtype=dtype, device=device
            )
            censor_grid = torch.as_tensor(
                evaluation.censor_grid, dtype=dtype, device=device
            )
            event_survival = torch.exp(-(
                event_grid[None, :] / scale_t[:, None]
            ).pow(shape_t[:, None])).reshape(batch, int(mc_samples), -1)
            censor_survival = torch.exp(-(
                censor_grid[None, :] / scale_c[:, None]
            ).pow(shape_c[:, None])).reshape(batch, int(mc_samples), -1)
            joint = torch.einsum(
                "bme,bmc->bec", event_survival, censor_survival
            ) / int(mc_samples)
            predictions.append(joint.cpu().numpy())
    return np.concatenate(predictions, axis=0)


def predict_hacsurv_joint_survival(
    model, evaluation: JointSurvivalEvaluation, *, generator_samples: int,
    batch_size: int, device: torch.device,
) -> np.ndarray:
    """Evaluate HACSurv's learned survival copula and neural margins."""
    predictions = []
    dtype = next(model.parameters()).dtype
    event_grid = torch.as_tensor(
        evaluation.event_grid, dtype=dtype, device=device
    )
    censor_grid = torch.as_tensor(
        evaluation.censor_grid, dtype=dtype, device=device
    )
    model.eval()
    with torch.no_grad():
        for start in range(0, len(evaluation.X), int(batch_size)):
            x = torch.as_tensor(
                evaluation.X[start:start + int(batch_size)],
                dtype=dtype, device=device,
            )
            predictions.append(model.joint_survival_grid(
                x, event_grid, censor_grid, int(generator_samples)
            ).cpu().numpy())
    return np.concatenate(predictions, axis=0)


def oracle_joint_survival_ise(
    prediction: np.ndarray, evaluation: JointSurvivalEvaluation
) -> float:
    """Mean subject-specific 2D integrated squared error, normalized by area."""
    squared = np.square(np.asarray(prediction) - evaluation.truth)
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    over_censor = integrate(squared, evaluation.censor_grid, axis=2)
    over_both = integrate(over_censor, evaluation.event_grid, axis=1)
    area = max(
        float(evaluation.event_grid[-1] * evaluation.censor_grid[-1]), 1e-12
    )
    return float(np.mean(over_both / area))


__all__ = [
    "JointSurvivalEvaluation", "oracle_joint_survival_ise",
    "predict_dvfm_joint_survival", "predict_hacsurv_joint_survival",
    "prepare_joint_survival_evaluation",
]
