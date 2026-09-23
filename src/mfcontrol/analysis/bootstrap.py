from __future__ import annotations

import numpy as np


def bootstrap_interval(values: np.ndarray, seed: int = 2026, draws: int = 5000) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        raise ValueError("bootstrap values cannot be empty")
    if len(array) == 1:
        return float(array[0]), float(array[0])
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(draws, len(array)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def paired_bootstrap_difference(
    first: np.ndarray, second: np.ndarray, seed: int = 2026, draws: int = 5000
) -> tuple[float, float, float]:
    if len(first) != len(second):
        raise ValueError("paired arrays must have equal length")
    differences = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    low, high = bootstrap_interval(differences, seed, draws)
    return float(differences.mean()), low, high
