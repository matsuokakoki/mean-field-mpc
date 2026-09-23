from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mfcontrol.meanfield.fixed_point import fixed_point_tail


def _stationary_tail(
    n: int,
    rho: float,
    d: int,
    seed: int,
    warmup: float,
    measurement: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    queues: list[deque[float]] = [deque() for _ in range(n)]
    total_rate = n * rho
    arrivals: list[tuple[float, float]] = []
    time = 0.0
    end = warmup + measurement
    while time < end:
        time += float(rng.exponential(1.0 / total_rate))
        if time <= end:
            arrivals.append((time, float(rng.exponential(1.0))))
    sample_times = np.linspace(warmup, end, max(200, int(measurement * 2)))
    samples: list[np.ndarray] = []
    arrival_idx = 0
    sample_idx = 0
    while arrival_idx < len(arrivals) or sample_idx < len(sample_times):
        next_arrival = arrivals[arrival_idx][0] if arrival_idx < len(arrivals) else np.inf
        next_sample = sample_times[sample_idx] if sample_idx < len(sample_times) else np.inf
        now = min(next_arrival, next_sample)
        if next_sample <= next_arrival:
            lengths = np.empty(n, dtype=int)
            for server, queue in enumerate(queues):
                while queue and queue[0] <= now:
                    queue.popleft()
                lengths[server] = len(queue)
            samples.append(lengths)
            sample_idx += 1
            continue
        service = arrivals[arrival_idx][1]
        sampled = rng.choice(n, size=min(d, n), replace=False)
        sampled_lengths: list[int] = []
        for raw_server in sampled:
            server = int(raw_server)
            while queues[server] and queues[server][0] <= now:
                queues[server].popleft()
            sampled_lengths.append(len(queues[server]))
        best = sampled[np.flatnonzero(np.asarray(sampled_lengths) == min(sampled_lengths))]
        chosen = int(rng.choice(best))
        start = max(now, queues[chosen][-1] if queues[chosen] else now)
        queues[chosen].append(start + service)
        arrival_idx += 1
    matrix = np.vstack(samples)
    max_k = max(1, int(matrix.max()))
    return np.asarray([(matrix >= k).mean() for k in range(max_k + 1)])


def _validation_row(
    arguments: tuple[int, float, int, int, float, float, float, int, str],
) -> dict[str, float | int | str]:
    n, rho, d, seed, warmup, measurement, tolerance, tail_max, profile = arguments
    empirical = _stationary_tail(n, rho, d, seed, warmup, measurement)
    theoretical = fixed_point_tail(rho, d, tolerance, tail_max)
    length = max(len(empirical), len(theoretical))
    empirical = np.pad(empirical, (0, length - len(empirical)))
    theoretical = np.pad(theoretical, (0, length - len(theoretical)))
    difference = np.abs(empirical[1:] - theoretical[1:])
    return {
        "profile": profile,
        "n": n,
        "rho": rho,
        "d": d,
        "seed": seed,
        "l1_tail_error": float(difference.sum()),
        "max_tail_error": float(difference.max(initial=0.0)),
        "jobs_per_server_error": float(abs(empirical[1:].sum() - theoretical[1:].sum())),
    }


def run_meanfield_validation(config: dict[str, Any], output_dir: Path) -> pd.DataFrame:
    settings = config["meanfield"]
    tasks = [
        (
            int(n),
            float(rho),
            int(d),
            int(seed),
            float(settings["warmup_time"]),
            float(settings["measurement_time"]),
            float(settings["tail_tolerance"]),
            int(settings["tail_max"]),
            str(config["profile"]),
        )
        for n, rho, d, seed in product(
            settings["n_values"], settings["rho_values"], settings["d_values"], config["seeds"]
        )
    ]
    workers = int(settings.get("workers", 1))
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_validation_row, tasks))
    else:
        rows = [_validation_row(task) for task in tasks]
    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "meanfield_convergence.csv", index=False)
    figure_dir = Path("figures")
    (figure_dir / "png").mkdir(parents=True, exist_ok=True)
    (figure_dir / "pdf").mkdir(parents=True, exist_ok=True)
    summary = frame.groupby(["n", "rho", "d"], as_index=False)["l1_tail_error"].mean()
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for (rho, d), group in summary.groupby(["rho", "d"]):
        ax.plot(group["n"], group["l1_tail_error"], marker="o", label=f"rho={rho}, d={d}")
    ax.set(xlabel="servers N", ylabel="L1 queue-tail error", xscale="log", yscale="log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(figure_dir / extension / f"fig04_meanfield_convergence.{extension}", dpi=180)
    plt.close(fig)
    selected_n = int(max(settings["n_values"]))
    selected_rho = float(max(settings["rho_values"]))
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for d in settings["d_values"]:
        empirical = _stationary_tail(
            selected_n,
            selected_rho,
            int(d),
            int(config["seeds"][0]),
            float(settings["warmup_time"]),
            float(settings["measurement_time"]),
        )
        theoretical = fixed_point_tail(selected_rho, int(d))
        ax.plot(np.arange(1, len(empirical)), empirical[1:], marker="o", label=f"empirical d={d}")
        ax.plot(
            np.arange(1, len(theoretical)),
            theoretical[1:],
            linestyle="--",
            label=f"fixed point d={d}",
        )
    ax.set(xlabel="queue-tail index k", ylabel="fraction with length at least k", yscale="log")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(figure_dir / extension / f"fig03_meanfield_fixed_point.{extension}", dpi=180)
    plt.close(fig)
    return frame
