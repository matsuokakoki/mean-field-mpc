from __future__ import annotations

import math

from mfcontrol.meanfield.fixed_point import fixed_point_tail


def expected_jobs(rho: float, d: int = 2, tolerance: float = 1e-12, max_terms: int = 10_000) -> float:
    return float(fixed_point_tail(rho, d, tolerance, max_terms)[1:].sum())


def mean_system_time(
    arrival_rate_per_server: float,
    service_rate: float,
    d: int = 2,
    tolerance: float = 1e-12,
    max_terms: int = 10_000,
) -> float:
    if arrival_rate_per_server < 0 or service_rate <= 0:
        raise ValueError("rates must be nonnegative and service rate positive")
    if arrival_rate_per_server <= 1e-15:
        return 1.0 / service_rate
    rho = arrival_rate_per_server / service_rate
    if rho >= 1:
        return math.inf
    return expected_jobs(rho, d, tolerance, max_terms) / arrival_rate_per_server


def predicted_wait(
    total_rate: float,
    capacity: int,
    service_rate: float,
    d: int = 2,
    rho_max: float = 0.995,
    infeasible_penalty: float = 1e6,
) -> float:
    if total_rate < 0 or capacity < 1 or service_rate <= 0:
        raise ValueError("invalid rate, capacity, or service rate")
    if total_rate <= 1e-15:
        return 0.0
    per_server = total_rate / capacity
    rho = per_server / service_rate
    if rho >= rho_max:
        overload = max(0.0, rho - rho_max)
        return infeasible_penalty * (1.0 + overload * overload)
    return max(0.0, mean_system_time(per_server, service_rate, d) - 1.0 / service_rate)
