from __future__ import annotations

from typing import Any

import numpy as np


def latency_metrics(waiting_times: np.ndarray, response_times: np.ndarray, min_jobs_for_tail: int) -> dict[str, Any]:
    """Return explicit missing tail metrics when a realized workload is too small."""
    waits = np.asarray(waiting_times, dtype=float)
    responses = np.asarray(response_times, dtype=float)
    if len(waits) != len(responses):
        raise ValueError("waiting and response arrays must be aligned")
    if len(waits) == 0:
        return {
            "metric_status": "insufficient_data",
            "metric_reason": "no_generated_jobs",
            "mean_wait": np.nan,
            "p95_wait": np.nan,
            "p99_wait": np.nan,
            "mean_response": np.nan,
            "p95_response": np.nan,
        }
    result: dict[str, Any] = {
        "metric_status": "ok",
        "metric_reason": "",
        "mean_wait": float(np.mean(waits)),
        "mean_response": float(np.mean(responses)),
    }
    if len(waits) < min_jobs_for_tail:
        result.update(
            {
                "metric_status": "insufficient_data",
                "metric_reason": f"fewer_than_{min_jobs_for_tail}_generated_jobs",
                "p95_wait": np.nan,
                "p99_wait": np.nan,
                "p95_response": np.nan,
            }
        )
        return result
    result.update(
        {
            "p95_wait": float(np.quantile(waits, 0.95)),
            "p99_wait": float(np.quantile(waits, 0.99)),
            "p95_response": float(np.quantile(responses, 0.95)),
        }
    )
    return result
