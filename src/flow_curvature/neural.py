"""Optional CPU PyTorch baseline; importing the core package does not need torch."""

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn

from .coupling import couple
from .data import Array
from .fields import Coupling, Velocity


@dataclass(frozen=True)
class TrainingConfig:
    steps: int = 2000
    batch_size: int = 128
    hidden_width: int = 64
    learning_rate: float = 1e-3
    seed: int = 42

    def __post_init__(self) -> None:
        if min(self.steps, self.batch_size, self.hidden_width) < 1:
            raise ValueError("Training steps, batch size, and width must be positive.")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")


class VelocityMLP(nn.Module):
    """Same time-conditioned architecture for both coupling objectives."""

    def __init__(self, dimension: int = 2, hidden_width: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dimension + 1, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, dimension),
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((x, t), dim=-1))


def train_velocity(
    source: Array, target: Array, coupling: Coupling, config: TrainingConfig
) -> tuple[VelocityMLP, Array]:
    """Fit ||v_theta(t, (1-t)x0+t*x1) - (x1-x0)||^2 on seeded batches.

    Uses local RNGs and forked torch state so notebook calls do not reset global
    random streams. CPU execution avoids device-specific nondeterminism.
    """
    if (
        source.ndim != 2
        or source.shape != target.shape
        or 0 in source.shape
        or not np.all(np.isfinite(source))
        or not np.all(np.isfinite(target))
    ):
        raise ValueError("Training endpoints must have equal, nonempty, finite 2D shapes.")
    if coupling not in ("independent", "ot"):
        raise ValueError(f"Unknown coupling: {coupling}")
    rng = np.random.default_rng(config.seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        model = VelocityMLP(source.shape[1], config.hidden_width)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    losses = np.empty(config.steps, dtype=np.float64)
    model.train()
    for step in range(config.steps):
        x0 = source[rng.integers(len(source), size=config.batch_size)]
        x1 = target[rng.integers(len(target), size=config.batch_size)]
        x0, x1 = couple(x0, x1, coupling, rng)
        start = torch.as_tensor(x0, dtype=torch.float32)
        end = torch.as_tensor(x1, dtype=torch.float32)
        t = torch.as_tensor(rng.random((config.batch_size, 1)), dtype=torch.float32)
        prediction = model(t, (1 - t) * start + t * end)
        loss = torch.mean((prediction - (end - start)) ** 2)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at step {step}.")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses[step] = loss.item()
    model.eval()
    return model, losses


def as_velocity(model: VelocityMLP) -> Velocity:
    """Adapt a CPU model to the integrator's (scalar time, NumPy batch) API."""
    def velocity(t: float, x: Array) -> Array:
        with torch.no_grad():
            points = torch.as_tensor(x, dtype=torch.float32)
            time = torch.full((len(points), 1), t, dtype=points.dtype)
            return model(time, points).numpy().astype(np.float64)

    return velocity
