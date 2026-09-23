from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mfcontrol.analysis.metrics import latency_metrics
from mfcontrol.control.static import static_capacity
from mfcontrol.experiments.controllers import (
    ALL_CONTROLLERS,
    _select_alphas,
    _summary,
    _targets_for_period,
)
from mfcontrol.progress import progress
from mfcontrol.sim.arrivals import generate_workload, intensity_floor, scale_factor
from mfcontrol.sim.jsqd import CapacityEvent, simulate_jsqd

SENSITIVITY_CONTROLLERS = (
    "reactive",
    "gp_mean_mf_mpc",
    "gp_ucb_mf_mpc",
    "oracle_mf_mpc",
)


def _setting_rows(
    config: dict[str, Any],
    predictions: pd.DataFrame,
    counts: np.ndarray,
    train_durations: np.ndarray,
    test_durations: np.ndarray,
    experiment: str,
    n: int,
    delay: int,
    service_model: str,
    controllers: tuple[str, ...],
    scenario: str = "bursty",
) -> list[dict[str, Any]]:
    bin_seconds = float(config["data"]["bin_seconds"])
    train_end = int(0.6 * len(counts))
    mean_service = float(np.mean(train_durations))
    service_rate = 1.0 / mean_service
    floor = intensity_floor(counts[:train_end])
    scaling = scale_factor(
        counts[:train_end] + floor,
        mean_service,
        n,
        bin_seconds,
        float(config["control"]["target_mean_utilization"]),
    )
    validation = predictions[predictions["split"] == "validation"]
    test = predictions[predictions["split"] == "test"]
    selected, _, _ = _select_alphas(config, validation, scaling, bin_seconds, service_rate, n, delay, floor)
    decisions, targets = _targets_for_period(
        config, test, scaling, bin_seconds, service_rate, n, delay, selected, floor
    )
    evaluation_bins = int(config["simulation"]["evaluation_bins"])
    decisions = decisions[:evaluation_bins]
    targets = {name: values[: len(decisions)] for name, values in targets.items()}
    minimum = max(2, int(np.ceil(float(config["control"]["min_fraction"]) * n)))
    fixed = static_capacity(
        (counts[:train_end] + floor) * scaling / bin_seconds,
        service_rate,
        n,
        minimum,
        float(config["control"]["rho_static"]),
    )
    targets["static"] = np.full(len(decisions), fixed, dtype=int)
    raw_counts = counts[decisions] + floor
    slo = 2 * float(np.median(train_durations))
    tasks = [
        (
            config,
            raw_counts,
            scaling,
            bin_seconds,
            test_durations,
            mean_service,
            service_model,
            n,
            delay,
            targets,
            controllers,
            experiment,
            service_rate,
            slo,
            scenario,
            int(seed),
        )
        for seed in config["seeds"]
    ]
    workers = int(config["simulation"].get("workers", 1))
    logger = progress()
    if logger is not None:
        for task in tasks:
            logger.work_start(scenario, int(task[-1]), len(config["seeds"]), controller=experiment)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_sensitivity_seed_rows, task) for task in tasks]
            batches = []
            for future in as_completed(futures):
                batches.append(future.result())
                if logger is not None:
                    logger.work_complete("results/controller_metrics_per_seed.csv")
    else:
        batches = []
        for task in tasks:
            batches.append(_sensitivity_seed_rows(task))
            if logger is not None:
                logger.work_complete("results/controller_metrics_per_seed.csv")
    return [row for batch in batches for row in batch]


def _sensitivity_seed_rows(arguments: tuple[Any, ...]) -> list[dict[str, Any]]:
    (
        config,
        raw_counts,
        scaling,
        bin_seconds,
        test_durations,
        mean_service,
        service_model,
        n,
        delay,
        targets,
        controllers,
        experiment,
        service_rate,
        slo,
        scenario,
        seed,
    ) = arguments
    workload = generate_workload(
        raw_counts,
        scaling,
        bin_seconds,
        test_durations,
        mean_service,
        service_model,
        seed,
        int(config["simulation"].get("max_events_per_bin", 0)),
    )
    rows: list[dict[str, Any]] = []
    for controller in controllers:
        target = targets[controller]
        events = [CapacityEvent(index * bin_seconds, int(value)) for index, value in enumerate(target[1:], start=1)]
        result = simulate_jsqd(
            workload.jobs,
            n,
            int(config["control"]["d"]),
            seed,
            events,
            initial_active=int(target[0]),
            activation_delay=delay * bin_seconds,
            prevalidated_sorted=True,
        )
        waits = result.waiting_times
        responses = result.response_times
        utilization = workload.realized_counts / bin_seconds / (target * service_rate)
        metrics = latency_metrics(waits, responses, int(config["simulation"]["min_jobs_for_tail_metric"]))
        rows.append(
            {
                "profile": config["profile"],
                "config_id": f"{experiment}-n{n}-d{delay}-{service_model}",
                "experiment": experiment,
                "scenario": scenario,
                "controller": controller,
                "seed": seed,
                "n": n,
                "delay_bins": delay,
                "service_model": service_model,
                "jobs": len(waits),
                **metrics,
                "mean_resource_fraction": float(np.mean(target) / n),
                "server_time_resource_cost": float(np.sum(target) * bin_seconds),
                "scaling_actions": int(np.count_nonzero(np.diff(target))),
                "mean_absolute_target_change": float(np.mean(np.abs(np.diff(target)))) if len(target) > 1 else 0.0,
                "overload_fraction": float(np.mean(utilization >= 1.0)),
                "slo_violation_fraction": float(np.mean(waits > slo)),
            }
        )
    return rows


def run_sensitivity_experiments(config: dict[str, Any], output_dir: Path) -> pd.DataFrame:
    all_predictions = pd.read_csv(output_dir / "forecast_predictions.csv")
    predictions = all_predictions[all_predictions["scenario"] == "bursty"]
    dataset = np.load(Path("data/processed/bursty.npz"))
    counts = np.asarray(dataset["counts"], dtype=float)
    train_durations = np.asarray(dataset["durations_train"], dtype=float)
    test_durations = np.asarray(dataset["durations_test"], dtype=float)
    if len(test_durations) == 0:
        test_durations = train_durations
    base_n = int(config["control"]["n"])
    base_delay = int(config["control"]["activation_delay_bins"])
    rows: list[dict[str, Any]] = []
    for delay in (0, 1, 2):
        rows.extend(
            _setting_rows(
                config,
                predictions,
                counts,
                train_durations,
                test_durations,
                "delay_sensitivity",
                base_n,
                delay,
                "empirical",
                SENSITIVITY_CONTROLLERS,
            )
        )
    n_values = (50, 100, 200) if config["profile"] == "paper" else (max(10, base_n // 2), base_n, base_n * 2)
    for n in n_values:
        rows.extend(
            _setting_rows(
                config,
                predictions,
                counts,
                train_durations,
                test_durations,
                "many_server_sensitivity",
                n,
                base_delay,
                "empirical",
                SENSITIVITY_CONTROLLERS,
            )
        )
    for service_model in ("exponential_matched", "empirical"):
        rows.extend(
            _setting_rows(
                config,
                predictions,
                counts,
                train_durations,
                test_durations,
                "model_mismatch",
                base_n,
                base_delay,
                service_model,
                ALL_CONTROLLERS,
            )
        )
    # The high-volume EWMA/reactive comparison is the prespecified robustness
    # check for the strict matched-budget claim.  Only activation delay is
    # perturbed to keep the additional simulation cost modest.
    high_predictions = all_predictions[all_predictions["scenario"] == "high_volume"]
    high_dataset = np.load(Path("data/processed/high_volume.npz"))
    high_counts = np.asarray(high_dataset["counts"], dtype=float)
    high_train_durations = np.asarray(high_dataset["durations_train"], dtype=float)
    high_test_durations = np.asarray(high_dataset["durations_test"], dtype=float)
    if len(high_test_durations) == 0:
        high_test_durations = high_train_durations
    for delay in (0, 2):
        rows.extend(
            _setting_rows(
                config,
                high_predictions,
                high_counts,
                high_train_durations,
                high_test_durations,
                "high_volume_delay_robustness",
                base_n,
                delay,
                "empirical",
                ("reactive", "ewma_mf_mpc"),
                scenario="high_volume",
            )
        )
    frame = pd.DataFrame(rows)
    primary = pd.read_csv(output_dir / "controller_metrics_per_seed.csv")
    combined = pd.concat([primary[primary["experiment"] == "primary"], frame], ignore_index=True)
    combined.to_csv(output_dir / "controller_metrics_per_seed.csv", index=False)
    summary = _summary(combined)
    summary.to_csv(output_dir / "controller_summary.csv", index=False)
    _figures(summary)
    return frame


def _figures(summary: pd.DataFrame) -> None:
    definitions = (
        ("delay_sensitivity", "delay_bins", "fig10_delay_sensitivity", "activation delay (bins)"),
        ("many_server_sensitivity", "n", "fig11_many_server_sensitivity", "servers N"),
        ("model_mismatch", "service_model", "fig12_model_mismatch", "service model"),
        ("high_volume_delay_robustness", "delay_bins", "fig13_high_volume_delay", "activation delay (bins)"),
    )
    for experiment, x_column, filename, x_label in definitions:
        subset = summary[(summary["experiment"] == experiment) & (summary["metric"] == "p95_wait")]
        fig, ax = plt.subplots(figsize=(6.8, 4.0))
        for controller, group in subset.groupby("controller"):
            group = group.sort_values(x_column)
            ax.plot(
                group[x_column].astype(str) if x_column == "service_model" else group[x_column],
                group["mean"],
                marker="o",
                label=str(controller).replace("_mf_mpc", ""),
            )
        ax.set(xlabel=x_label, ylabel="p95 simulated waiting time (s)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(Path("figures") / extension / f"{filename}.{extension}", dpi=180)
        plt.close(fig)
