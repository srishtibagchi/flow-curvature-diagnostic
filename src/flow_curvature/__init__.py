"""Trajectory-curvature diagnostics and flow-matching baselines."""

from .data import GaussianPair, sample_endpoints
from .diagnostics import trajectory_diagnostics
from .fields import gaussian_velocity
from .integration import integrate

__all__ = [
    "GaussianPair",
    "gaussian_velocity",
    "integrate",
    "sample_endpoints",
    "trajectory_diagnostics",
]
