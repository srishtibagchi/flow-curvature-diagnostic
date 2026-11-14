"""Geometric bending separated from time-parameterization acceleration."""

import numpy as np

from .data import Array


def trajectory_diagnostics(
    paths: Array, times: Array, speed_tolerance: float = 1e-8
) -> dict[str, Array]:
    """Compute one value per particle on a uniform time grid.

    kappa = ||a - proj_v(a)|| / ||v||^2. The principal diagnostic is bending
    energy, integral kappa^2 ds, approximated at interior points. Curvature is
    undefined at zero speed: those samples are excluded and their fraction is
    returned explicitly. A wholly stationary path has zero bending by convention.
    Acceleration energy includes tangential acceleration and is not curvature.
    """
    paths = np.asarray(paths, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if (
        paths.ndim != 3
        or 0 in paths.shape
        or times.ndim != 1
        or paths.shape[0] != times.size
        or times.size < 3
        or not np.all(np.isfinite(paths))
        or not np.all(np.isfinite(times))
    ):
        raise ValueError("Expected finite paths (time, particle, dimension) and matching times.")
    steps = np.diff(times)
    if np.any(steps <= 0) or not np.allclose(steps, steps[0], rtol=1e-7, atol=1e-12):
        raise ValueError("Diagnostics require a strictly increasing uniform time grid.")
    if not np.isfinite(speed_tolerance) or speed_tolerance <= 0:
        raise ValueError("speed_tolerance must be finite and positive.")
    dt = steps[0]
    velocity = (paths[2:] - paths[:-2]) / (2 * dt)
    acceleration = (paths[2:] - 2 * paths[1:-1] + paths[:-2]) / dt**2
    speed_squared = np.sum(velocity**2, axis=-1)
    moving = speed_squared > speed_tolerance**2
    projection = np.divide(
        np.sum(acceleration * velocity, axis=-1),
        speed_squared,
        out=np.zeros_like(speed_squared),
        where=moving,
    )
    normal_acceleration = acceleration - projection[..., None] * velocity
    curvature = np.divide(
        np.linalg.norm(normal_acceleration, axis=-1),
        speed_squared,
        out=np.zeros_like(speed_squared),
        where=moving,
    )
    speed = np.sqrt(speed_squared)
    weights = np.where(moving, speed * dt, 0)
    arc_length = weights.sum(axis=0)
    mean_curvature = np.divide(
        (curvature * weights).sum(axis=0),
        arc_length,
        out=np.zeros(paths.shape[1]),
        where=arc_length > 0,
    )
    length = np.linalg.norm(np.diff(paths, axis=0), axis=-1).sum(axis=0)
    chord = np.linalg.norm(paths[-1] - paths[0], axis=-1)
    return {
        "bending_energy": (curvature**2 * weights).sum(axis=0),
        "mean_curvature": mean_curvature,
        "acceleration_energy": np.sum(acceleration**2, axis=(0, 2)) * dt,
        "path_length": length,
        "chord_length": chord,
        "length_excess": np.maximum(length - chord, 0),
        "stationary_fraction": 1 - moving.mean(axis=0),
    }


def summarize(diagnostics: dict[str, Array]) -> dict[str, dict[str, float]]:
    """Aggregate without discarding the per-particle diagnostics saved on disk."""
    return {
        name: {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)),
        }
        for name, values in diagnostics.items()
    }
