from __future__ import annotations

import numpy as np


def calibrate_beta(
    actual_log: np.ndarray,
    predicted_log: np.ndarray,
    predicted_std: np.ndarray,
    epsilon: float = 1e-8,
    clip: tuple[float, float] = (0.0, 6.0),
) -> float:
    if not (len(actual_log) == len(predicted_log) == len(predicted_std)):
        raise ValueError("calibration arrays must have equal length")
    if len(actual_log) == 0:
        raise ValueError("calibration data cannot be empty")
    z = (actual_log - predicted_log) / np.maximum(predicted_std, epsilon)
    return float(np.clip(np.quantile(z, 0.9), *clip))
