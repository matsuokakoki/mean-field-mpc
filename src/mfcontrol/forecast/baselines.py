from __future__ import annotations

import numpy as np


def last_value(counts: np.ndarray, decision_index: np.ndarray) -> np.ndarray:
    indices = np.maximum(0, decision_index.astype(int) - 1)
    return np.asarray(counts[indices], dtype=float)


def seasonal_naive(counts: np.ndarray, decision_index: np.ndarray) -> np.ndarray:
    indices = np.maximum(0, decision_index.astype(int) - 288)
    return np.asarray(counts[indices], dtype=float)


def ewma_forecast(counts: np.ndarray, decision_index: np.ndarray, alpha: float) -> np.ndarray:
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    state = float(counts[0])
    states = np.empty(len(counts), dtype=float)
    states[0] = state
    for index in range(1, len(counts)):
        state = alpha * float(counts[index - 1]) + (1 - alpha) * state
        states[index] = state
    return np.asarray(states[decision_index.astype(int)], dtype=float)
