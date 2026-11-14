import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

class GaussianPair:
    """2-dimensional standard normal source and a translated, anisotropic Gaussian target.

    The OT map is `x -> target_mean + A @ x`.
    Dimensions and parameters are fixed for initial testing.
    """

    target_mean: tuple[float, float] = (2.0, 1.0)
    target_stds: tuple[float, float] = (0.5, 1.8)
    rotation_angle: float = 0.6

    def __post_init__(self) -> None:
        if len(self.target_mean) != 2 or len(self.target_stds) != 2:
            raise ValueError("The Gaussian baseline must be two-dimensional.")
        values = (*self.target_mean, *self.target_stds, self.rotation_angle)
        if not np.all(np.isfinite(values)) or min(self.target_stds) <= 0:
            raise ValueError("Parameters must be finite and scales positive.")

    @property
    def transport_matrix(self) -> Array:
        cosine, sine = np.cos(self.rotation_angle), np.sin(self.rotation_angle)
        rotation = np.array([[cosine, -sine], 
                             [sine, cosine]])
        return rotation @ np.diag(self.target_stds) @ rotation.T

    @property
    def target_covariance(self) -> Array:
        A = self.transport_matrix
        return A @ A.T

    def transport(self, source: Array) -> Array:
        return source @ self.transport_matrix.T + np.asarray(self.target_mean)


def sample_endpoints(
    pair: GaussianPair, n_samples: int, seed: int
) -> tuple[Array, Array]:
    """Draw independent endpoints from source and target distributions, without 
    encoding OT pairings in the data."""

    if n_samples < 1:
        raise ValueError("n_samples must be positive.")
    
    rng = np.random.default_rng(seed)
    source = rng.standard_normal((n_samples, 2))
    target = pair.transport(rng.standard_normal((n_samples, 2)))
    return source, target
