from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Job:
    demand_time: float
    service_time: float


@dataclass(frozen=True)
class CapacityEvent:
    time: float
    target: int


@dataclass
class SimulationResult:
    waiting_times: NDArray[np.float64]
    response_times: NDArray[np.float64]
    assigned_servers: NDArray[np.int64]
    queue_lengths_at_arrival: NDArray[np.int64]
    requested_trace: list[tuple[float, int]]
    effective_trace: list[tuple[float, int]]


def simulate_jsqd(
    jobs: list[Job],
    n_servers: int,
    d: int = 2,
    seed: int = 0,
    capacity_events: list[CapacityEvent] | None = None,
    initial_active: int | None = None,
    activation_delay: float = 0.0,
    prevalidated_sorted: bool = False,
) -> SimulationResult:
    """Simulate FCFS JSQ(d) with warming and non-routable draining servers."""
    if n_servers < 1 or d < 1 or activation_delay < 0:
        raise ValueError("invalid simulator settings")
    if prevalidated_sorted:
        ordered_jobs = jobs
    else:
        ordered_jobs = sorted(jobs, key=lambda job: job.demand_time)
        if any(job.service_time <= 0 for job in ordered_jobs):
            raise ValueError("service times must be positive")
    active_count = n_servers if initial_active is None else initial_active
    if not 1 <= active_count <= n_servers:
        raise ValueError("initial_active must be within server bounds")
    rng = np.random.default_rng(seed)
    queues: list[deque[float]] = [deque() for _ in range(n_servers)]
    state = np.full(n_servers, "inactive", dtype=object)
    state[:active_count] = "active"
    active_ids = list(range(active_count))
    pending_activation: list[tuple[float, int]] = []
    events = sorted(capacity_events or [], key=lambda event: event.time)
    event_idx = 0
    requested = active_count
    requested_trace = [(0.0, requested)]
    effective_trace = [(0.0, active_count)]
    waits: list[float] = []
    responses: list[float] = []
    assignments: list[int] = []
    queue_lengths: list[int] = []

    def prune(server: int, now: float) -> None:
        while queues[server] and queues[server][0] <= now:
            queues[server].popleft()
        if state[server] == "draining" and not queues[server]:
            state[server] = "inactive"

    def apply_target(now: float, target: int) -> None:
        nonlocal requested
        for server in range(n_servers):
            if state[server] == "draining":
                prune(server, now)
        target = int(np.clip(target, 1, n_servers))
        requested = target
        requested_trace.append((now, target))
        routable_or_warming = [i for i in range(n_servers) if state[i] in {"active", "warming"}]
        if target > len(routable_or_warming):
            candidates = [i for i in range(n_servers) if state[i] == "inactive"]
            for server in candidates[: target - len(routable_or_warming)]:
                state[server] = "warming"
                pending_activation.append((now + activation_delay, server))
        elif target < len(routable_or_warming):
            warming = [i for i in routable_or_warming if state[i] == "warming"]
            active = [i for i in routable_or_warming if state[i] == "active"]
            remove_count = len(routable_or_warming) - target
            to_remove = warming[:remove_count]
            if len(to_remove) < remove_count:
                to_remove.extend(active[-(remove_count - len(to_remove)) :])
            for server in to_remove:
                if state[server] == "active":
                    active_ids.remove(server)
                state[server] = "draining" if queues[server] else "inactive"
            effective_trace.append((now, int(np.sum(state == "active"))))

    for job in ordered_jobs:
        now = job.demand_time
        while event_idx < len(events) and events[event_idx].time <= now:
            apply_target(events[event_idx].time, events[event_idx].target)
            event_idx += 1
        for ready_time, server in list(pending_activation):
            if ready_time <= now and state[server] == "warming":
                state[server] = "active"
                active_ids.append(server)
                pending_activation.remove((ready_time, server))
                effective_trace.append((ready_time, int(np.sum(state == "active"))))
        if not active_ids:
            raise RuntimeError("no active server available for an admitted job")
        if d == 1 or len(active_ids) == 1:
            server = active_ids[int(rng.integers(len(active_ids)))]
            prune(server, now)
        elif d == 2:
            first_index = int(rng.integers(len(active_ids)))
            second_index = int(rng.integers(len(active_ids) - 1))
            if second_index >= first_index:
                second_index += 1
            first = active_ids[first_index]
            second = active_ids[second_index]
            prune(first, now)
            prune(second, now)
            first_length = len(queues[first])
            second_length = len(queues[second])
            if first_length < second_length:
                server = first
            elif second_length < first_length:
                server = second
            else:
                server = first if int(rng.integers(2)) == 0 else second
        else:
            sampled = rng.choice(np.asarray(active_ids), size=min(d, len(active_ids)), replace=False)
            lengths = []
            for raw_server in sampled:
                server = int(raw_server)
                prune(server, now)
                lengths.append(len(queues[server]))
            best = sampled[np.flatnonzero(np.asarray(lengths) == min(lengths))]
            server = int(rng.choice(best))
        queue_lengths.append(len(queues[server]))
        start = max(now, queues[server][-1] if queues[server] else now)
        completion = start + job.service_time
        queues[server].append(completion)
        waits.append(start - now)
        responses.append(completion - now)
        assignments.append(server)
    return SimulationResult(
        waiting_times=np.asarray(waits),
        response_times=np.asarray(responses),
        assigned_servers=np.asarray(assignments, dtype=np.int64),
        queue_lengths_at_arrival=np.asarray(queue_lengths, dtype=np.int64),
        requested_trace=requested_trace,
        effective_trace=effective_trace,
    )
