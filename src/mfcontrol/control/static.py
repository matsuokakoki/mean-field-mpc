from __future__ import annotations

import math

import numpy as np


def static_capacity(
    training_rates: np.ndarray,
    service_rate: float,
    n: int,
    minimum: int,
    rho_target: float = 0.75,
) -> int:
    if len(training_rates) == 0 or service_rate <= 0:
        raise ValueError("training rates and service rate are required")
    desired = math.ceil(float(np.quantile(training_rates, 0.95)) / (rho_target * service_rate))
    return int(np.clip(desired, minimum, n))
