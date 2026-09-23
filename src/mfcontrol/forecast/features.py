from __future__ import annotations

import numpy as np
import pandas as pd


def feature_frame(counts: np.ndarray, horizon: int) -> pd.DataFrame:
    """Build causal features at decision index t for target t+horizon."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    series = pd.Series(np.asarray(counts, dtype=float))
    frame = pd.DataFrame(index=np.arange(len(series)))
    for lag in (1, 2, 3, 6, 12, 24, 288):
        frame[f"log_count_lag_{lag}"] = np.log1p(series.shift(lag))
    history = series.shift(1)
    for window in (3, 12, 72):
        frame[f"rolling_mean_{window}"] = history.rolling(window, min_periods=window).mean()
    for window in (12, 72):
        frame[f"rolling_std_{window}"] = history.rolling(window, min_periods=window).std(ddof=0)
    phase = 2 * np.pi * (frame.index.to_numpy() % 288) / 288
    frame["time_sin"] = np.sin(phase)
    frame["time_cos"] = np.cos(phase)
    frame["decision_index"] = frame.index
    frame["target_index"] = frame.index + horizon
    target = series.shift(-horizon)
    frame["target_count"] = target
    frame["target_log"] = np.log1p(target)
    return frame.dropna().reset_index(drop=True)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"decision_index", "target_index", "target_count", "target_log"}
    return [column for column in frame.columns if column not in excluded]
