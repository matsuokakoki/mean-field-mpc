from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mfcontrol.sim.jsqd import Job


@dataclass(frozen=True)
class Workload:
    jobs: list[Job]
    expected_counts: np.ndarray
    realized_counts: np.ndarray
    scale_factor: float
    weighted_event_approximation: bool


def intensity_floor(training_counts: np.ndarray, fraction: float = 0.01) -> float:
    mean_count = float(np.mean(training_counts))
    if mean_count <= 0 or fraction <= 0:
        raise ValueError("positive training count and floor fraction are required")
    return fraction * mean_count


def scale_factor(
    training_counts: np.ndarray,
    mean_service: float,
    n: int,
    bin_seconds: float,
    target_utilization: float,
) -> float:
    mean_count = float(np.mean(training_counts))
    if mean_count <= 0 or mean_service <= 0:
        raise ValueError("positive training count and service mean are required")
    return n * target_utilization * bin_seconds / (mean_count * mean_service)


def generate_workload(
    raw_counts: np.ndarray,
    scale: float,
    bin_seconds: float,
    service_samples: np.ndarray,
    mean_service: float,
    service_model: str,
    seed: int,
    max_events_per_bin: int = 0,
) -> Workload:
    rng = np.random.default_rng(seed)
    expected = np.maximum(0.0, np.asarray(raw_counts, dtype=float) * scale)
    realized = rng.poisson(expected)
    jobs: list[Job] = []
    approximate = False
    for bin_index, full_count in enumerate(realized):
        simulated_count = int(full_count)
        weight = 1.0
        if max_events_per_bin > 0 and simulated_count > max_events_per_bin:
            approximate = True
            weight = simulated_count / max_events_per_bin
            simulated_count = max_events_per_bin
        times = bin_index * bin_seconds + rng.uniform(0.0, bin_seconds, simulated_count)
        if service_model == "exponential_matched":
            services = rng.exponential(mean_service, simulated_count)
        elif service_model == "empirical":
            if len(service_samples) == 0:
                raise ValueError("empirical service samples cannot be empty")
            services = rng.choice(service_samples, size=simulated_count, replace=True)
        else:
            raise ValueError(f"unknown service model: {service_model}")
        services = np.maximum(np.asarray(services) * weight, np.finfo(float).eps)
        jobs.extend(Job(float(time), float(service)) for time, service in zip(times, services, strict=True))
    jobs.sort(key=lambda job: job.demand_time)
    return Workload(jobs, expected, realized, scale, approximate)
