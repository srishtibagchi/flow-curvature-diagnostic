"""Couplings for conditional flow matching with linear interpolation."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .data import Array
from .fields import Coupling


def couple(
    source: Array, target: Array, method: Coupling, rng: np.random.Generator
) -> tuple[Array, Array]:
    """Pair equal-size batches; OT solves the exact discrete quadratic assignment.

    Minibatch OT is not population OT. Its approximation error depends on batch
    size and should not be interpreted as a guarantee of straight learned paths.
    """
    if (
        source.ndim != 2
        or source.shape != target.shape
        or 0 in source.shape
        or not np.all(np.isfinite(source))
        or not np.all(np.isfinite(target))
    ):
        raise ValueError("Endpoint batches must have equal, finite, nonempty 2D shapes.")
    if method == "independent":
        return source, target[rng.permutation(len(target))]
    if method == "ot":
        rows, columns = linear_sum_assignment(cdist(source, target, metric="sqeuclidean"))
        return source[rows], target[columns]
    raise ValueError(f"Unknown coupling: {method}")
