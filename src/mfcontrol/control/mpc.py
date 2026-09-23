from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mfcontrol.meanfield.metrics import predicted_wait


@dataclass(frozen=True)
class MPCSettings:
    n: int
    minimum: int
    ramp: int
    d: int
    service_rate: float
    mean_service: float
    alpha: float
    queue_weight: float = 1.0
    switching_weight: float = 0.05
    rho_max: float = 0.995
    activation_delay_bins: int = 1
    capacity_step: int = 1


def _candidates(previous: int, settings: MPCSettings) -> list[int]:
    low = max(settings.minimum, previous - settings.ramp)
    high = min(settings.n, previous + settings.ramp)
    candidates = list(range(low, high + 1, settings.capacity_step))
    for boundary in (low, high, previous):
        if boundary not in candidates:
            candidates.append(boundary)
    return sorted(set(candidates))


def choose_mpc_action(forecast_rates: np.ndarray, previous: int, settings: MPCSettings) -> int:
    if len(forecast_rates) == 0:
        raise ValueError("at least one forecast horizon is required")
    delay = settings.activation_delay_bins
    initial_pipeline = tuple([previous] * delay)
    # Only the first action is used by receding-horizon control.  Keeping an
    # entire path for every dynamic-programming state makes the result
    # unnecessarily expensive without affecting either the objective or its
    # lexicographic tie-break at the first action.
    states: dict[tuple[int, tuple[int, ...]], tuple[float, int | None]] = {(previous, initial_pipeline): (0.0, None)}
    wait_cache: dict[tuple[float, int], float] = {}
    for stage_index, rate in enumerate(forecast_rates):
        next_states: dict[tuple[int, tuple[int, ...]], tuple[float, int | None]] = {}
        for (last, pipeline), (cost, first_action) in states.items():
            if stage_index > 0 and first_action is None:
                raise AssertionError("a continued dynamic-programming state needs an initial action")
            for candidate in _candidates(last, settings):
                effective = candidate if delay == 0 else pipeline[0]
                new_pipeline = () if delay == 0 else (*pipeline[1:], candidate)
                cache_key = (float(rate), effective)
                queue_delay = wait_cache.get(cache_key)
                if queue_delay is None:
                    queue_delay = predicted_wait(
                        float(rate), effective, settings.service_rate, settings.d, settings.rho_max
                    )
                    wait_cache[cache_key] = queue_delay
                stage = (
                    settings.queue_weight * queue_delay / settings.mean_service
                    + settings.alpha * candidate / settings.n
                    + settings.switching_weight * abs(candidate - last) / settings.n
                )
                key = (candidate, new_pipeline)
                chosen_first_action = candidate if first_action is None else first_action
                proposal = (cost + stage, chosen_first_action)
                existing = next_states.get(key)
                if existing is None:
                    next_states[key] = proposal
                elif existing[1] is None:
                    raise AssertionError("dynamic-programming state needs an initial action")
                elif proposal[0] < existing[0] or (proposal[0] == existing[0] and proposal[1] < existing[1]):
                    next_states[key] = proposal
        states = next_states
    _, best_first_action = min(states.values())
    if best_first_action is None:
        raise AssertionError("non-empty forecast horizon must choose an action")
    return best_first_action


def mpc_targets(
    forecast_matrix: np.ndarray,
    settings: MPCSettings,
    initial: int | None = None,
) -> np.ndarray:
    previous = settings.minimum if initial is None else int(np.clip(initial, settings.minimum, settings.n))
    targets: list[int] = []
    for forecasts in forecast_matrix:
        previous = choose_mpc_action(np.asarray(forecasts, dtype=float), previous, settings)
        targets.append(previous)
    return np.asarray(targets, dtype=int)
