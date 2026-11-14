from typing import Callable, Literal

import numpy as np

from .data import Array, GaussianPair

Coupling = Literal["independent", "ot"]
Velocity = Callable[[float, Array], Array]


def gaussian_velocity(pair: GaussianPair, coupling: Coupling) -> Velocity:
    """Returns the exact marginal velocity field E[X1-X0 | (1-t)X0+tX1=x] 
    for the selected Gaussian coupling using either OT or independent coupling.

    ``ot`` uses the population Monge map, not a finite minibatch assignment.
    Both conditional interpolation schemes are linear, but their *marginal*
    ODE trajectories need not be straight under independent coupling.
    """
    if coupling not in ("independent", "ot"):
        raise ValueError(f"Unknown coupling: {coupling}")
    
    identity = np.eye(2)
    matrix = pair.transport_matrix
    covariance = pair.target_covariance
    mean = np.asarray(pair.target_mean)

    def velocity(t: float, x: Array) -> Array:
        if not np.isfinite(t) or not 0 <= t <= 1:
            raise ValueError("Time must be finite and in [0, 1].")
        
        if coupling == "ot":
            interpolation = (1 - t) * identity + t * matrix
            gain = np.linalg.solve(interpolation.T, (matrix - identity).T).T
        else:
            marginal_covariance = (1 - t) ** 2 * identity + t**2 * covariance
            cross_covariance = t * covariance - (1 - t) * identity
            gain = np.linalg.solve(marginal_covariance.T, cross_covariance.T).T
        return mean + (x - t * mean) @ gain.T

    return velocity
