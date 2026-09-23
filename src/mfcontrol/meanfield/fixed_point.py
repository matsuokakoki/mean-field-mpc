from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def fixed_point_tail(
    rho: float,
    d: int = 2,
    tolerance: float = 1e-12,
    max_terms: int = 10_000,
) -> NDArray[np.float64]:
    """Return q_0,...,q_K, stopping after the first tail below tolerance."""
    if not 0 <= rho < 1:
        raise ValueError("rho must satisfy 0 <= rho < 1")
    if d < 1:
        raise ValueError("d must be positive")
    if tolerance <= 0 or max_terms < 1:
        raise ValueError("invalid truncation settings")
    values = [1.0]
    if rho == 0:
        return np.asarray([1.0, 0.0], dtype=float)
    for k in range(1, max_terms + 1):
        if d == 1:
            value = rho**k
        else:
            exponent = (d**k - 1) / (d - 1)
            log_value = exponent * math.log(rho)
            value = 0.0 if log_value < math.log(np.finfo(float).tiny) else math.exp(log_value)
        values.append(value)
        if value < tolerance:
            return np.asarray(values, dtype=float)
    raise RuntimeError("mean-field tail did not reach the configured tolerance")


def recurrence_residual(tail: NDArray[np.float64], rho: float, d: int) -> float:
    """Maximum normalized stationary-recurrence residual on represented terms."""
    if len(tail) < 3:
        return 0.0
    residuals = [
        abs(rho * (tail[k - 1] ** d - tail[k] ** d) - (tail[k] - tail[k + 1])) for k in range(1, len(tail) - 1)
    ]
    return max(residuals, default=0.0)
