from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mfcontrol.analysis.bootstrap import bootstrap_interval
from mfcontrol.analysis.metrics import latency_metrics
from mfcontrol.control.mpc import MPCSettings, mpc_targets
from mfcontrol.control.reactive import reactive_targets
from mfcontrol.control.static import static_capacity
from mfcontrol.progress import progress
from mfcontrol.sim.arrivals import generate_workload, intensity_floor, scale_factor
from mfcontrol.sim.jsqd import CapacityEvent, simulate_jsqd

PREDICTIVE = (
    "seasonal_mf_mpc",
    "ewma_mf_mpc",
    "gp_mean_mf_mpc",
    "gp_ucb_mf_mpc",
    "oracle_mf_mpc",
)
ALL_CONTROLLERS = ("static", "reactive", *PREDICTIVE)


def _common_decisions(frame: pd.DataFrame) -> np.ndarray:
    horizon_count = frame["horizon"].nunique()
    counts = frame.groupby("decision_index")["horizon"].nunique()
    return counts[counts == horizon_count].index.to_numpy(dtype=int)


def _forecast_matrix(frame: pd.DataFrame, method: str, floor: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    ordered_horizons = sorted(frame["horizon"].unique())
    decisions = _common_decisions(frame)
    matrix = np.empty((len(decisions), len(ordered_horizons)), dtype=float)
    for column_index, horizon in enumerate(ordered_horizons):
        subset = frame[frame["horizon"] == horizon].set_index("decision_index").loc[decisions]
        if method == "seasonal_mf_mpc":
            values = subset["seasonal_naive"].to_numpy()
        elif method == "ewma_mf_mpc":
            values = subset["ewma"].to_numpy()
        elif method == "gp_mean_mf_mpc":
            values = subset["gp_mean"].to_numpy()
        elif method == "gp_ucb_mf_mpc":
            values = np.maximum(
                0.0,
                np.expm1(
                    subset["gp_mean_log"].to_numpy() + subset["beta"].to_numpy() * subset["gp_std_log"].to_numpy()
                ),
            )
        elif method == "oracle_mf_mpc":
            values = subset["actual"].to_numpy()
        else:
            raise ValueError(f"unknown predictive controller: {method}")
        matrix[:, column_index] = values + floor
    return decisions, matrix


def _mpc_settings(config: dict[str, Any], service_rate: float, alpha: float, n: int, delay: int) -> MPCSettings:
    control = config["control"]
    return MPCSettings(
        n=n,
        minimum=max(2, int(np.ceil(float(control["min_fraction"]) * n))),
        ramp=max(1, int(np.ceil(float(control["ramp_fraction"]) * n))),
        d=int(control["d"]),
        service_rate=service_rate,
        mean_service=1.0 / service_rate,
        alpha=alpha,
        queue_weight=float(control["queue_weight"]),
        switching_weight=float(control["switching_weight"]),
        rho_max=float(control["rho_max"]),
        activation_delay_bins=delay,
        capacity_step=int(control["capacity_step"]),
    )


def _targets_for_period(
    config: dict[str, Any],
    frame: pd.DataFrame,
    scale: float,
    bin_seconds: float,
    service_rate: float,
    n: int,
    delay: int,
    alpha_by_controller: dict[str, float] | None = None,
    floor: float = 0.0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    decisions = _common_decisions(frame)
    control = config["control"]
    minimum = max(2, int(np.ceil(float(control["min_fraction"]) * n)))
    ramp = max(1, int(np.ceil(float(control["ramp_fraction"]) * n)))
    observed = (
        frame[frame["horizon"] == min(frame["horizon"])]
        .set_index("decision_index")
        .loc[decisions, "last_value"]
        .to_numpy()
    )
    observed_rates = (observed + floor) * scale / bin_seconds
    targets: dict[str, np.ndarray] = {
        "reactive": reactive_targets(
            observed_rates,
            service_rate,
            n,
            minimum,
            ramp,
            float(control["rho_reactive"]),
        )
    }
    for method in PREDICTIVE:
        matrix_decisions, raw_matrix = _forecast_matrix(frame, method, floor)
        if not np.array_equal(matrix_decisions, decisions):
            raise AssertionError("forecast decision grids do not align")
        alpha = 0.1 if alpha_by_controller is None else alpha_by_controller[method]
        targets[method] = mpc_targets(
            raw_matrix * scale / bin_seconds,
            _mpc_settings(config, service_rate, alpha, n, delay),
        )
    return decisions, targets


def _select_alphas(
    config: dict[str, Any],
    validation: pd.DataFrame,
    scale: float,
    bin_seconds: float,
    service_rate: float,
    n: int,
    delay: int,
    floor: float = 0.0,
) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if "split" not in validation or not bool((validation["split"] == "validation").all()):
        raise AssertionError("controller alpha selection must receive validation rows only")
    decisions, base = _targets_for_period(config, validation, scale, bin_seconds, service_rate, n, delay, floor=floor)
    del decisions
    budget = float(np.mean(base["reactive"]) / n)
    search = config["control"]["alpha_search"]
    tolerance = float(search["relative_tolerance"])
    selected: dict[str, float] = {}
    pareto: list[dict[str, Any]] = []
    matches: dict[str, dict[str, Any]] = {}
    for method in PREDICTIVE:
        candidates: dict[float, float] = {}
        _, raw_matrix = _forecast_matrix(validation, method, floor)
        initial = np.geomspace(float(search["min_alpha"]), float(search["max_alpha"]), int(search["initial_points"]))
        alpha_values = {float(value) for value in config["control"]["alpha_grid"]}
        alpha_values.update(float(value) for value in initial)
        for alpha in sorted(alpha_values):
            target = mpc_targets(
                raw_matrix * scale / bin_seconds,
                _mpc_settings(config, service_rate, float(alpha), n, delay),
            )
            resource = float(np.mean(target) / n)
            candidates[float(alpha)] = resource
            pareto.append(
                {
                    "controller": method,
                    "alpha": float(alpha),
                    "validation_resource_fraction": resource,
                    "reference_budget": budget,
                    "selection_source": "validation_alpha_search",
                }
            )
        for _ in range(int(search["max_refinements"])):
            ordered = sorted(candidates.items())
            brackets = [
                (left, right)
                for left, right in zip(ordered, ordered[1:], strict=False)
                if (left[1] - budget) * (right[1] - budget) <= 0 and left[0] != right[0]
            ]
            if not brackets:
                break
            left, right = min(brackets, key=lambda pair: min(abs(pair[0][1] - budget), abs(pair[1][1] - budget)))
            midpoint = float(np.sqrt(left[0] * right[0]))
            if midpoint in candidates:
                break
            target = mpc_targets(
                raw_matrix * scale / bin_seconds,
                _mpc_settings(config, service_rate, midpoint, n, delay),
            )
            resource = float(np.mean(target) / n)
            candidates[midpoint] = resource
            pareto.append(
                {
                    "controller": method,
                    "alpha": midpoint,
                    "validation_resource_fraction": resource,
                    "reference_budget": budget,
                    "selection_source": "validation_alpha_refinement",
                }
            )
            if _relative_budget_difference(resource, budget) <= tolerance:
                break
        alpha, resource = min(candidates.items(), key=lambda pair: abs(pair[1] - budget))
        difference = _relative_budget_difference(resource, budget)
        selected[method] = alpha
        matches[method] = {
            "selected_alpha": alpha,
            "validation_resource_fraction": resource,
            "reactive_reference_budget": budget,
            "relative_budget_difference": difference,
            "within_tolerance": difference <= tolerance,
            "tolerance": tolerance,
            "status": "matched" if difference <= tolerance else "unmatched",
            "selection_partition": "validation",
        }
    return selected, pareto, matches


def _relative_budget_difference(resource: float, budget: float) -> float:
    return abs(resource - budget) / max(abs(budget), np.finfo(float).eps)


def _pareto_simulation_rows(
    config: dict[str, Any],
    scenario: str,
    split: str,
    frame: pd.DataFrame,
    raw_counts: np.ndarray,
    service_samples: np.ndarray,
    mean_service: float,
    scale: float,
    n: int,
    delay: int,
    selected: dict[str, float],
    floor: float,
) -> list[dict[str, Any]]:
    """Simulate every discrete alpha using paired seeds for one data split."""
    bin_seconds = float(config["data"]["bin_seconds"])
    service_rate = 1.0 / mean_service
    decisions = _common_decisions(frame)
    evaluation_bins = int(config["simulation"]["evaluation_bins"])
    decisions = decisions[:evaluation_bins]
    counts = raw_counts[decisions] + floor
    targets_by_setting: dict[tuple[str, float], np.ndarray] = {}
    for method in PREDICTIVE:
        matrix_decisions, matrix = _forecast_matrix(frame, method, floor)
        if not np.array_equal(matrix_decisions[: len(decisions)], decisions):
            raise AssertionError("Pareto forecast decision grids do not align")
        matrix = matrix[: len(decisions)]
        alpha_values = {*(float(value) for value in config["control"]["alpha_grid"]), float(selected[method])}
        for raw_alpha in sorted(alpha_values):
            alpha = float(raw_alpha)
            targets_by_setting[(method, alpha)] = mpc_targets(
                matrix * scale / bin_seconds,
                _mpc_settings(config, service_rate, alpha, n, delay),
            )
    tasks = [
        (
            config,
            scenario,
            split,
            counts,
            service_samples,
            mean_service,
            scale,
            n,
            delay,
            selected,
            targets_by_setting,
            int(seed),
        )
        for seed in config["seeds"]
    ]
    workers = int(config["simulation"].get("workers", 1))
    logger = progress()
    if logger is not None:
        for task in tasks:
            logger.work_start(scenario, int(task[-1]), len(config["seeds"]), controller="pareto")
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_pareto_seed_rows, task) for task in tasks]
            batches = []
            for future in as_completed(futures):
                batches.append(future.result())
                if logger is not None:
                    logger.work_complete("results/pareto_operating_points.csv")
    else:
        batches = []
        for task in tasks:
            batches.append(_pareto_seed_rows(task))
            if logger is not None:
                logger.work_complete("results/pareto_operating_points.csv")
    return [row for batch in batches for row in batch]


def _pareto_seed_rows(arguments: tuple[Any, ...]) -> list[dict[str, Any]]:
    (
        config,
        scenario,
        split,
        counts,
        service_samples,
        mean_service,
        scale,
        n,
        delay,
        selected,
        targets_by_setting,
        seed,
    ) = arguments
    bin_seconds = float(config["data"]["bin_seconds"])
    workload = generate_workload(
        counts,
        scale,
        bin_seconds,
        service_samples,
        mean_service,
        "empirical",
        seed,
        int(config["simulation"].get("max_events_per_bin", 0)),
    )
    rows: list[dict[str, Any]] = []
    for (method, alpha), target in targets_by_setting.items():
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
        metrics = latency_metrics(
            result.waiting_times,
            result.response_times,
            int(config["simulation"]["min_jobs_for_tail_metric"]),
        )
        rows.append(
            {
                "profile": config["profile"],
                "config_id": f"{config['profile']}-pareto-n{n}-d{delay}",
                "scenario": scenario,
                "split": split,
                "controller": method,
                "alpha": alpha,
                "seed": seed,
                "n": n,
                "delay_bins": delay,
                "jobs": len(result.waiting_times),
                "mean_resource_fraction": float(np.mean(target) / n),
                "metric_status": metrics["metric_status"],
                "metric_reason": metrics["metric_reason"],
                "p95_wait": metrics["p95_wait"],
                "p99_wait": metrics["p99_wait"],
                "selected_on_validation": bool(alpha == selected[method]),
                "weighted_event_approximation": workload.weighted_event_approximation,
            }
        )
    return rows


def _primary_seed_rows(arguments: tuple[Any, ...]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    """Run all controller variants for one paired workload seed."""
    (
        config,
        scenario,
        counts,
        scaling,
        bin_seconds,
        service_samples,
        mean_service,
        targets,
        service_rate,
        n,
        delay,
        median_train_service,
        seed,
    ) = arguments
    workload = generate_workload(
        counts,
        scaling,
        bin_seconds,
        service_samples,
        mean_service,
        "empirical",
        seed,
        int(config["simulation"].get("max_events_per_bin", 0)),
    )
    metric_rows: list[dict[str, Any]] = []
    queue_rows: list[dict[str, Any]] = []
    for controller in ALL_CONTROLLERS:
        target = targets[controller]
        capacity_events = [
            CapacityEvent(index * bin_seconds, int(value)) for index, value in enumerate(target[1:], start=1)
        ]
        result = simulate_jsqd(
            workload.jobs,
            n,
            int(config["control"]["d"]),
            seed,
            capacity_events,
            initial_active=int(target[0]),
            activation_delay=delay * bin_seconds,
            prevalidated_sorted=True,
        )
        waits = result.waiting_times
        responses = result.response_times
        actual_rates = workload.realized_counts / bin_seconds
        utilization = actual_rates / (target * service_rate)
        metrics = latency_metrics(waits, responses, int(config["simulation"]["min_jobs_for_tail_metric"]))
        metric_rows.append(
            {
                "profile": config["profile"],
                "config_id": config["profile"],
                "experiment": "primary",
                "scenario": scenario,
                "controller": controller,
                "seed": seed,
                "n": n,
                "delay_bins": delay,
                "service_model": "empirical",
                "jobs": len(waits),
                **metrics,
                "mean_resource_fraction": float(np.mean(target) / n),
                "server_time_resource_cost": float(np.sum(target) * bin_seconds),
                "scaling_actions": int(np.count_nonzero(np.diff(target))),
                "mean_absolute_target_change": float(np.mean(np.abs(np.diff(target)))) if len(target) > 1 else 0.0,
                "overload_fraction": float(np.mean(utilization >= 1.0)),
                "slo_violation_fraction": float(np.mean(waits > 2 * median_train_service)),
            }
        )
        queue_values, queue_counts = np.unique(result.queue_lengths_at_arrival, return_counts=True)
        queue_rows.extend(
            {
                "profile": config["profile"],
                "scenario": scenario,
                "controller": controller,
                "seed": seed,
                "queue_length_at_arrival": int(queue_length),
                "jobs": int(job_count),
                "fraction": (
                    float(job_count / len(result.queue_lengths_at_arrival))
                    if len(result.queue_lengths_at_arrival)
                    else np.nan
                ),
            }
            for queue_length, job_count in zip(queue_values, queue_counts, strict=True)
        )
        if scenario == "bursty" and seed == int(config["seeds"][0]) and controller == "gp_ucb_mf_mpc":
            _capacity_figure(counts, target, result, bin_seconds)
    return metric_rows, queue_rows, workload.weighted_event_approximation


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = ["experiment", "scenario", "controller", "n", "delay_bins", "service_model"]
    metric_columns = [
        "mean_wait",
        "p95_wait",
        "p99_wait",
        "mean_response",
        "p95_response",
        "mean_resource_fraction",
        "slo_violation_fraction",
        "overload_fraction",
    ]
    for key, group in frame.groupby(group_columns):
        base = dict(zip(group_columns, key, strict=True))
        for metric in metric_columns:
            values = group[metric].dropna().to_numpy()
            if len(values) == 0:
                low = high = np.nan
            else:
                low, high = bootstrap_interval(values, seed=2026)
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "mean": float(np.mean(values)) if len(values) else np.nan,
                    "median": float(np.median(values)) if len(values) else np.nan,
                    "ci95_low": low,
                    "ci95_high": high,
                    "seeds": len(values),
                    "total_seeds": len(group),
                }
            )
    return pd.DataFrame(rows)


def run_controller_experiments(config: dict[str, Any], output_dir: Path) -> pd.DataFrame:
    predictions = pd.read_csv(output_dir / "forecast_predictions.csv")
    bin_seconds = float(config["data"]["bin_seconds"])
    n = int(config["control"]["n"])
    delay = int(config["control"]["activation_delay_bins"])
    evaluation_bins = int(config["simulation"]["evaluation_bins"])
    rows: list[dict[str, Any]] = []
    queue_distribution_rows: list[dict[str, Any]] = []
    selected_payload: dict[str, Any] = {}
    alpha_candidate_rows: list[dict[str, Any]] = []
    pareto_rows: list[dict[str, Any]] = []
    approximation_used = False
    for scenario in ("high_volume", "bursty", "diurnal"):
        dataset = np.load(Path("data/processed") / f"{scenario}.npz")
        counts = np.asarray(dataset["counts"], dtype=float)
        train_end = int(0.6 * len(counts))
        train_durations = np.asarray(dataset["durations_train"], dtype=float)
        validation_durations = np.asarray(dataset["durations_validation"], dtype=float)
        test_durations = np.asarray(dataset["durations_test"], dtype=float)
        validation_service_samples = validation_durations if len(validation_durations) else train_durations
        test_service_samples = test_durations if len(test_durations) else train_durations
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
        scenario_predictions = predictions[predictions["scenario"] == scenario]
        validation = scenario_predictions[scenario_predictions["split"] == "validation"]
        test = scenario_predictions[scenario_predictions["split"] == "test"]
        selected, pareto, selected_budget_checks = _select_alphas(
            config, validation, scaling, bin_seconds, service_rate, n, delay, floor
        )
        selected_payload[scenario] = {
            "alpha": selected,
            "budget_match": selected_budget_checks,
            "beta": float(test["beta"].iloc[0]),
            "service_rate": service_rate,
            "scale_factor": scaling,
            "intensity_floor_count_per_bin": floor,
            "empirical_service_source": {
                "validation": "validation"
                if len(validation_durations)
                else "training fallback (empty validation split)",
                "test": "test" if len(test_durations) else "training fallback (empty test split)",
            },
        }
        alpha_candidate_rows.extend({"scenario": scenario, **row} for row in pareto)
        pareto_rows.extend(
            _pareto_simulation_rows(
                config,
                scenario,
                "validation",
                validation,
                counts,
                validation_service_samples,
                mean_service,
                scaling,
                n,
                delay,
                selected,
                floor,
            )
        )
        pareto_rows.extend(
            _pareto_simulation_rows(
                config,
                scenario,
                "test",
                test,
                counts,
                test_service_samples,
                mean_service,
                scaling,
                n,
                delay,
                selected,
                floor,
            )
        )
        decisions, targets = _targets_for_period(
            config, test, scaling, bin_seconds, service_rate, n, delay, selected, floor
        )
        if len(decisions) > evaluation_bins:
            decisions = decisions[:evaluation_bins]
            targets = {name: values[:evaluation_bins] for name, values in targets.items()}
        raw_test_counts = counts[decisions] + floor
        minimum = max(2, int(np.ceil(float(config["control"]["min_fraction"]) * n)))
        static = static_capacity(
            (counts[:train_end] + floor) * scaling / bin_seconds,
            service_rate,
            n,
            minimum,
            float(config["control"]["rho_static"]),
        )
        targets["static"] = np.full(len(decisions), static, dtype=int)
        median_train_service = float(np.median(train_durations))
        primary_tasks = [
            (
                config,
                scenario,
                raw_test_counts,
                scaling,
                bin_seconds,
                test_service_samples,
                mean_service,
                targets,
                service_rate,
                n,
                delay,
                median_train_service,
                int(seed),
            )
            for seed in config["seeds"]
        ]
        workers = int(config["simulation"].get("workers", 1))
        logger = progress()
        if logger is not None:
            for task in primary_tasks:
                logger.work_start(scenario, int(task[-1]), len(config["seeds"]), controller="all_primary")
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_primary_seed_rows, task) for task in primary_tasks]
                primary_batches = []
                for future in as_completed(futures):
                    primary_batches.append(future.result())
                    if logger is not None:
                        logger.work_complete("results/controller_metrics_per_seed.csv")
        else:
            primary_batches = []
            for task in primary_tasks:
                primary_batches.append(_primary_seed_rows(task))
                if logger is not None:
                    logger.work_complete("results/controller_metrics_per_seed.csv")
        for metric_rows, queues, weighted_approximation in primary_batches:
            rows.extend(metric_rows)
            queue_distribution_rows.extend(queues)
            approximation_used = approximation_used or weighted_approximation
    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "controller_metrics_per_seed.csv", index=False)
    pd.DataFrame(queue_distribution_rows).to_csv(output_dir / "queue_length_distribution.csv", index=False)
    summary = _summary(frame)
    summary.to_csv(output_dir / "controller_summary.csv", index=False)
    pd.DataFrame(alpha_candidate_rows).to_csv(output_dir / "alpha_selection_candidates.csv", index=False)
    pareto_frame = pd.DataFrame(pareto_rows)
    pareto_frame.to_csv(output_dir / "pareto_operating_points.csv", index=False)
    hyperparameters_path = output_dir / "selected_hyperparameters.json"
    existing = json.loads(hyperparameters_path.read_text(encoding="utf-8"))
    existing["controllers"] = selected_payload
    existing["weighted_event_approximation"] = approximation_used
    hyperparameters_path.write_text(json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8")
    _primary_figure(summary)
    _pareto_figure(pareto_frame)
    return frame


def _capacity_figure(raw_counts: np.ndarray, targets: np.ndarray, result: Any, bin_seconds: float) -> None:
    x = np.arange(len(raw_counts))
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8.0, 5.2), sharex=True)
    top.plot(x, raw_counts, color="black", linewidth=0.8, label="trace shape plus train-fixed floor")
    top.set_ylabel("trace count")
    bottom.step(x, targets, where="post", label="requested capacity")
    effective_times = np.asarray([time / bin_seconds for time, _ in result.effective_trace])
    effective_values = np.asarray([value for _, value in result.effective_trace])
    bottom.step(effective_times, effective_values, where="post", label="effective activation events")
    bottom.set(xlabel="test 5-minute bin", ylabel="servers")
    bottom.legend(fontsize=8)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(Path("figures") / extension / f"fig07_capacity_trace.{extension}", dpi=180)
    plt.close(fig)


def _primary_figure(summary: pd.DataFrame) -> None:
    subset = summary[(summary["experiment"] == "primary") & (summary["metric"] == "p95_wait")]
    controllers = list(ALL_CONTROLLERS)
    scenarios = ["high_volume", "bursty", "diurnal"]
    x = np.arange(len(controllers))
    fig, axes = plt.subplots(1, len(scenarios), figsize=(11.4, 3.9), sharey=False)
    for axis, scenario in zip(axes, scenarios, strict=True):
        group = subset[subset["scenario"] == scenario].set_index("controller").loc[controllers]
        means = group["mean"].to_numpy()
        errors = np.vstack([means - group["ci95_low"].to_numpy(), group["ci95_high"].to_numpy() - means])
        axis.bar(x, means, yerr=errors, capsize=2)
        axis.set_yscale("log")
        axis.set_title(scenario)
        axis.set_xticks(x, [name.replace("_mf_mpc", "") for name in controllers], rotation=45, ha="right", fontsize=7)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("p95 simulated waiting time (s, log scale)")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(Path("figures") / extension / f"fig08_primary_controller_results.{extension}", dpi=180)
    plt.close(fig)


def _pareto_figure(frame: pd.DataFrame) -> None:
    summary = (
        frame.groupby(["scenario", "split", "controller", "alpha", "selected_on_validation"], as_index=False)[
            ["mean_resource_fraction", "p95_wait", "p99_wait"]
        ]
        .mean()
        .sort_values(["scenario", "split", "controller", "alpha"])
    )
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.4), sharex=False, sharey="row")
    for column, scenario in enumerate(("high_volume", "bursty", "diurnal")):
        for row, metric in enumerate(("p95_wait", "p99_wait")):
            axis = axes[row, column]
            subset = summary[summary["scenario"] == scenario]
            for (split, controller), group in subset.groupby(["split", "controller"]):
                axis.scatter(
                    group["mean_resource_fraction"],
                    group[metric],
                    s=np.where(group["selected_on_validation"], 42, 16),
                    marker="o" if split == "validation" else "x",
                    alpha=0.75,
                    label=f"{split}: {str(controller).replace('_mf_mpc', '')}",
                )
            axis.set(title=scenario, yscale="log")
            if row == 1:
                axis.set_xlabel("mean resource fraction")
            if column == 0:
                axis.set_ylabel(f"{metric.replace('_', ' ')} (s)")
            axis.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, fontsize=7)
    fig.subplots_adjust(bottom=0.18, hspace=0.28, wspace=0.25)
    for extension in ("png", "pdf"):
        fig.savefig(Path("figures") / extension / f"fig09_pareto.{extension}", dpi=180)
    plt.close(fig)
