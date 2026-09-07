"""Authorized minimal port of HACSurv's bivariate single-event model.

Upstream: https://github.com/Raymvp/HACSurv
Reference revision: da945141058cf6ccd5cb175002d6bc35b6e9bd9a
Original authors: Xin Liu, Weijia Zhang, and Min-Ling Zhang (AISTATS 2025).

This module contains only the components required for ``HACSurv_2D``: the
monotone neural survival margins, stochastic mixture-of-exponentials
Archimedean generator, inverse-generator autograd operation, observed-data
likelihood, and a validation-checkpointed training adapter. The port is included
with permission from the repository owner and adapted to avoid global dtype and
device state.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function, grad
from torch.utils.data import DataLoader, TensorDataset
from utility.metrics import hacsurv_kendall_tau


class PositiveLinear(nn.Module):
    """Linear layer with elementwise-squared, hence nonnegative, weights."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.log_weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        nn.init.xavier_uniform_(self.log_weight)
        self.log_weight.data.abs_().sqrt_()
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.log_weight)
            bound = np.sqrt(1.0 / np.sqrt(fan_in))
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(inputs, self.log_weight.square(), self.bias)


def _network(input_dim: int, widths: list[int], positive: bool) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for index, width in enumerate(widths):
        layer = PositiveLinear if positive else nn.Linear
        layers.append(layer(previous, width, bias=True))
        if index < len(widths) - 1 or not positive:
            layers.append(nn.Tanh())
        previous = width
    return nn.Sequential(*layers)


class NeuralDensityEstimator(nn.Module):
    """HACSurv monotone-time neural survival margin."""

    def __init__(self, input_dim: int, hidden_size: int, hidden_survival: int):
        super().__init__()
        self.embedding = _network(input_dim, [hidden_size] * 3, positive=False)
        self.outcome = _network(
            hidden_size + 1, [hidden_survival] * 3 + [1], positive=True
        )

    def forward(
        self, x: torch.Tensor, horizon: torch.Tensor, *, gradient: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        embedded = self.embedding(x)
        time = horizon.detach().clone().requires_grad_(gradient)
        logits = self.outcome(torch.cat((embedded, time.reshape(-1, 1)), dim=1))
        cdf = torch.sigmoid(logits)
        density = grad(cdf.sum(), time, create_graph=True)[0] if gradient else None
        return 1.0 - cdf.squeeze(1), density

    def survival(self, x: torch.Tensor, horizon: torch.Tensor) -> torch.Tensor:
        return self.forward(x, horizon, gradient=False)[0]


class StochasticGenerator(nn.Module):
    """Completely monotone generator phi(t)=E[exp(-M t)]."""

    def __init__(self, samples: int = 100):
        super().__init__()
        self.mixing_network = nn.Sequential(
            nn.Linear(1, 10), nn.LeakyReLU(0.2),
            nn.Linear(10, 10), nn.LeakyReLU(0.2), nn.Linear(10, 1),
        )
        self.samples = int(samples)
        self.mixing_rates: torch.Tensor | None = None

    def resample(self, samples: int | None = None) -> None:
        count = self.samples if samples is None else int(samples)
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        uniforms = torch.rand((count, 1), device=device, dtype=dtype)
        self.mixing_rates = torch.exp(self.mixing_network(uniforms).reshape(-1))

    def _rates(self) -> torch.Tensor:
        if self.mixing_rates is None:
            self.resample()
        assert self.mixing_rates is not None
        return self.mixing_rates

    def derivative(self, t: torch.Tensor, order: int = 0) -> torch.Tensor:
        shape = t.shape
        flat = t.reshape(-1)
        rates = self._rates()
        values = (-rates[None, :]).pow(order) * torch.exp(
            -flat[:, None] * rates[None, :]
        )
        return values.mean(dim=1).reshape(shape)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.derivative(t, order=0)


def _newton_inverse(
    generator: StochasticGenerator, y: torch.Tensor, max_iter: int, tol: float
) -> torch.Tensor:
    estimate = torch.zeros_like(y)
    residual = torch.full_like(y, float("inf"))
    for _ in range(int(max_iter)):
        value = generator(estimate)
        derivative = generator.derivative(estimate, order=1)
        residual = value - y
        if torch.max(torch.abs(residual)) <= tol:
            break
        if not torch.all(torch.isfinite(derivative)):
            raise FloatingPointError("HACSurv generator derivative is non-finite")
        estimate = torch.clamp(estimate - residual / derivative, min=0.0)
    if torch.max(torch.abs(residual)) > max(float(tol), 1e-6):
        raise RuntimeError("HACSurv generator inversion did not converge")
    return estimate


class GeneratorInverse(nn.Module):
    def __init__(self, generator: StochasticGenerator):
        super().__init__()
        self.generator = generator

    class _FastInverse(Function):
        @staticmethod
        def forward(ctx, y, optimum, value_at_optimum, generator):
            ctx.save_for_backward(y, optimum, value_at_optimum)
            ctx.generator = generator
            return optimum

        @staticmethod
        def backward(ctx, gradient):
            y, optimum, value_at_optimum = ctx.saved_tensors
            generator = ctx.generator
            with torch.enable_grad():
                z = GeneratorInverse._FastInverse.apply(
                    y, optimum, value_at_optimum, generator
                )
                derivative = torch.autograd.grad(
                    generator(z).sum(), z, create_graph=True
                )[0]
                return gradient / derivative, None, -gradient / derivative, None

    def forward(
        self, y: torch.Tensor, *, max_iter: int = 200, tol: float = 1e-8
    ) -> torch.Tensor:
        with torch.no_grad():
            optimum = _newton_inverse(self.generator, y, max_iter, tol)
        optimum_graph = optimum.detach().clone().requires_grad_(True)
        value = self.generator(optimum_graph)
        return self._FastInverse.apply(y, optimum_graph, value, self.generator)


class HACSurv2D(nn.Module):
    """Bivariate HACSurv model for one event and dependent censoring."""

    def __init__(
        self, input_dim: int, hidden_size: int = 32,
        hidden_survival: int = 32, generator_samples: int = 100,
        inverse_iterations: int = 200, inverse_tolerance: float = 1e-8,
    ):
        super().__init__()
        self.event_margin = NeuralDensityEstimator(
            input_dim, hidden_size, hidden_survival
        )
        self.censor_margin = NeuralDensityEstimator(
            input_dim, hidden_size, hidden_survival
        )
        self.generator = StochasticGenerator(generator_samples)
        self.generator_inverse = GeneratorInverse(self.generator)
        self.inverse_iterations = int(inverse_iterations)
        self.inverse_tolerance = float(inverse_tolerance)

    def log_likelihood(
        self, x: torch.Tensor, time: torch.Tensor, event: torch.Tensor
    ) -> torch.Tensor:
        eps = torch.finfo(x.dtype).eps ** 0.5
        event_survival, event_density = self.event_margin(x, time, gradient=True)
        censor_survival, censor_density = self.censor_margin(x, time, gradient=True)
        assert event_density is not None and censor_density is not None
        y = torch.stack(
            [event_survival.clamp(eps, 1.0 - eps),
             censor_survival.clamp(eps, 1.0 - eps)], dim=1
        )
        inverse = self.generator_inverse(
            y, max_iter=self.inverse_iterations, tol=self.inverse_tolerance
        )
        copula = self.generator(inverse.sum(dim=1))
        partials = torch.autograd.grad(
            copula.sum(), y, create_graph=True
        )[0].clamp_min(eps)
        event_term = torch.log(event_density.clamp_min(eps)) + torch.log(partials[:, 0])
        censor_term = torch.log(censor_density.clamp_min(eps)) + torch.log(partials[:, 1])
        return torch.where(event > 0.5, event_term, censor_term).mean()

    def event_survival(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return self.event_margin.survival(x, time)

    def joint_survival_grid(
        self, x: torch.Tensor, event_grid: torch.Tensor,
        censor_grid: torch.Tensor, generator_samples: int,
    ) -> torch.Tensor:
        """Evaluate C_phi(S_T(t|x), S_C(c|x)) on a two-dimensional grid."""
        self.generator.resample(int(generator_samples))
        event_survival = torch.stack([
            self.event_margin.survival(x, point.expand(len(x)))
            for point in event_grid
        ], dim=1)
        censor_survival = torch.stack([
            self.censor_margin.survival(x, point.expand(len(x)))
            for point in censor_grid
        ], dim=1)
        eps = torch.finfo(x.dtype).eps ** 0.5
        event_inverse = _newton_inverse(
            self.generator, event_survival.clamp(eps, 1.0 - eps),
            self.inverse_iterations, self.inverse_tolerance,
        )
        censor_inverse = _newton_inverse(
            self.generator, censor_survival.clamp(eps, 1.0 - eps),
            self.inverse_iterations, self.inverse_tolerance,
        )
        return self.generator(
            event_inverse[:, :, None] + censor_inverse[:, None, :]
        )

def _as_loader(X, time, event, batch_size: int, shuffle: bool, seed: int, dtype):
    dataset = TensorDataset(
        torch.as_tensor(X, dtype=dtype), torch.as_tensor(time, dtype=dtype),
        torch.as_tensor(event, dtype=dtype),
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
    )


def _objective(model, loader, device, validation_samples):
    model.eval()
    weighted, count = 0.0, 0
    for x, time, event in loader:
        x, time, event = x.to(device), time.to(device), event.to(device)
        model.generator.resample(validation_samples)
        likelihood = model.log_likelihood(x, time, event)
        loss = -likelihood
        if not torch.isfinite(loss):
            return float("inf")
        weighted += float(loss.detach().cpu()) * len(x)
        count += len(x)
    return weighted / max(count, 1)


def fit_hacsurv_2d(
    X_train, time_train, event_train, X_validation, time_validation,
    event_validation, X_test, time_points, *, epochs=1000, batch_size=512,
    learning_rate=1e-4, copula_learning_rate=1e-4,
    copula_start_epoch=200, early_stopping_patience=100,
    minimum_epochs=0, checkpoint_min_epoch=0,
    generator_samples=200, validation_generator_samples=500,
    hidden_size=32, hidden_survival=32, inverse_iterations=200,
    inverse_tolerance=1e-8, scale_regularization=1.0,
    numerical_failure_threshold=100.0, dtype="float64", seed=0, device="cpu",
):
    """Fit HACSurv-2D and return marginal event survival plus diagnostics."""
    torch_dtype = torch.float64 if str(dtype) == "float64" else torch.float32
    train_loader = _as_loader(
        X_train, time_train, event_train, int(batch_size), True, int(seed), torch_dtype
    )
    validation_loader = _as_loader(
        X_validation, time_validation, event_validation,
        int(batch_size), False, int(seed), torch_dtype,
    )
    model = HACSurv2D(
        int(np.asarray(X_train).shape[1]), int(hidden_size), int(hidden_survival),
        int(generator_samples), int(inverse_iterations), float(inverse_tolerance),
    ).to(device=device, dtype=torch_dtype)
    margin_optimizer = torch.optim.AdamW(
        list(model.event_margin.parameters()) + list(model.censor_margin.parameters()),
        lr=float(learning_rate),
    )
    copula_optimizer = torch.optim.AdamW(
        model.generator.parameters(), lr=float(copula_learning_rate)
    )
    best_loss, best_epoch, best_state = float("inf"), None, None
    no_improvement, history = 0, []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        total, count = 0.0, 0
        for x, time, event in train_loader:
            x, time, event = x.to(device), time.to(device), event.to(device)
            margin_optimizer.zero_grad()
            copula_optimizer.zero_grad()
            model.generator.resample(int(generator_samples))
            loss = -model.log_likelihood(x, time, event)
            loss = loss + float(scale_regularization) * (
                model.generator._rates().mean() - 1.0
            ).square() / len(x)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite HACSurv loss at epoch {epoch}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if not torch.isfinite(norm):
                raise FloatingPointError(f"Non-finite HACSurv gradient at epoch {epoch}")
            margin_optimizer.step()
            if epoch > int(copula_start_epoch):
                copula_optimizer.step()
            total += float(loss.detach().cpu()) * len(x)
            count += len(x)
        training_loss = total / max(count, 1)
        validation_loss = _objective(
            model, validation_loader, device, int(validation_generator_samples)
        )
        valid = bool(
            np.isfinite(training_loss) and np.isfinite(validation_loss)
            and abs(training_loss) <= float(numerical_failure_threshold)
            and abs(validation_loss) <= float(numerical_failure_threshold)
        )
        history.append({
            "epoch": epoch, "train_negative_log_likelihood": training_loss,
            "validation_negative_log_likelihood": validation_loss,
            "copula_trainable": epoch > int(copula_start_epoch),
            "numerical_valid": valid,
        })
        if (
            epoch >= int(checkpoint_min_epoch)
            and valid and validation_loss < best_loss
        ):
            best_loss, best_epoch = validation_loss, epoch
            best_state = deepcopy({
                key: value.detach().cpu() for key, value in model.state_dict().items()
            })
            no_improvement = 0
        else:
            no_improvement += 1
        if (
            epoch >= int(minimum_epochs)
            and int(early_stopping_patience) > 0
            and no_improvement >= int(early_stopping_patience)
        ):
            break
    if best_state is None:
        raise RuntimeError("HACSurv produced no valid validation checkpoint")
    model.load_state_dict(best_state)
    model.to(device=device, dtype=torch_dtype).eval()
    X_test_tensor = torch.as_tensor(X_test, dtype=torch_dtype, device=device)
    curves = np.empty((len(X_test_tensor), len(time_points)), dtype=float)
    with torch.no_grad():
        for index, point in enumerate(np.asarray(time_points, dtype=float)):
            horizon = torch.full(
                (len(X_test_tensor),), float(point), dtype=torch_dtype, device=device
            )
            curves[:, index] = model.event_survival(
                X_test_tensor, horizon
            ).detach().cpu().numpy()
    curves = np.minimum.accumulate(np.clip(curves, 0.0, 1.0), axis=1)
    return curves, {
        "checkpoint": "best_validation_log_likelihood",
        "checkpoint_epoch": int(best_epoch),
        "checkpoint_validation_negative_log_likelihood": float(best_loss),
        "learned_conditional_kendall_tau": hacsurv_kendall_tau(model),
        "epochs_completed": len(history),
    }, history, model


__all__ = ["HACSurv2D", "fit_hacsurv_2d"]
