"""Deterministic fixed-step RK4 integration with a common evaluation grid."""

import numpy as np

from .data import Array
from .fields import Velocity


def integrate(velocity: Velocity, initial: Array, times: Array) -> Array:
    """Return positions with shape (time, particle, dimension)."""
    initial = np.asarray(initial, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if initial.ndim != 2 or 0 in initial.shape or not np.all(np.isfinite(initial)):
        raise ValueError("initial must be a nonempty finite (particle, dimension) array.")
    if (
        times.ndim != 1
        or times.size < 3
        or not np.all(np.isfinite(times))
        or np.any(np.diff(times) <= 0)
        or times[0] < 0
        or times[-1] > 1
    ):
        raise ValueError("times must contain at least three increasing values in [0, 1].")

    def evaluate(t: float, x: Array) -> Array:
        result = np.asarray(velocity(t, x), dtype=np.float64)
        if result.shape != x.shape or not np.all(np.isfinite(result)):
            raise ValueError("Velocity returned a wrong-shaped or nonfinite array.")
        return result

    paths = np.empty((len(times), *initial.shape), dtype=np.float64)
    paths[0] = initial
    for i, dt in enumerate(np.diff(times)):
        t, x = times[i], paths[i]
        midpoint = float(t + dt / 2)
        k1 = evaluate(float(t), x)
        k2 = evaluate(midpoint, x + dt * k1 / 2)
        k3 = evaluate(midpoint, x + dt * k2 / 2)
        k4 = evaluate(float(times[i + 1]), x + dt * k3)
        paths[i + 1] = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return paths
