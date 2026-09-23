from __future__ import annotations

import math

import numpy as np


def reactive_targets(
    observed_rates: np.ndarray,
    service_rate: float,
    n: int,
    minimum: int,
    ramp: int,
    rho_target: float = 0.70,
    initial: int | None = None,
) -> np.ndarray:
    previous = minimum if initial is None else int(np.clip(initial, minimum, n))
    targets: list[int] = []
    for rate in observed_rates:
        desired = int(np.clip(math.ceil(float(rate) / (rho_target * service_rate)), minimum, n))
        previous = int(np.clip(desired, previous - ramp, previous + ramp))
        targets.append(previous)
    return np.asarray(targets, dtype=int)
