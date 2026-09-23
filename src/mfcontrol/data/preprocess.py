from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pandas as pd

from mfcontrol.data.azure import iter_trace_chunks, locate_trace


class ScenarioStat(TypedDict):
    app: str
    events: int
    burst: float
    autocorr: float


def scenario_is_eligible(diagnostics: dict[str, float | int], rules: dict[str, Any]) -> tuple[bool, list[str]]:
    """Apply precommitted split-level data-quality rules without controller outcomes."""
    checks = {
        "train_events": ("min_train_events", diagnostics["train_events"]),
        "validation_events": ("min_validation_events", diagnostics["validation_events"]),
        "test_events": ("min_test_events", diagnostics["test_events"]),
        "validation_nonzero_bins": ("min_validation_nonzero_bins", diagnostics["validation_nonzero_bins"]),
        "test_nonzero_bins": ("min_test_nonzero_bins", diagnostics["test_nonzero_bins"]),
        "train_service_samples": ("min_train_service_samples", diagnostics["train_service_samples"]),
        "train_nonzero_bins": ("min_train_nonzero_bins", diagnostics["train_nonzero_bins"]),
        "train_unique_positive_counts": (
            "min_train_unique_positive_counts",
            diagnostics["train_unique_positive_counts"],
        ),
    }
    failures = [name for name, (rule, value) in checks.items() if float(value) < float(rules[rule])]
    for split in ("validation", "test"):
        fraction = float(diagnostics[f"{split}_nonzero_fraction"])
        if fraction < float(rules[f"min_{split}_nonzero_fraction"]):
            failures.append(f"{split}_nonzero_fraction")
    return not failures, failures


def _service_stat_row(scenario: str, split: str, values: np.ndarray) -> dict[str, str | float | int]:
    if len(values) == 0:
        return {
            "scenario": scenario,
            "split": split,
            "observations": 0,
            **{
                column: np.nan
                for column in (
                    "mean_seconds",
                    "median_seconds",
                    "std_seconds",
                    "coefficient_of_variation",
                    "p90_seconds",
                    "p95_seconds",
                    "p99_seconds",
                    "max_seconds",
                )
            },
        }
    mean_duration = float(np.mean(values))
    return {
        "scenario": scenario,
        "split": split,
        "observations": len(values),
        "mean_seconds": mean_duration,
        "median_seconds": float(np.median(values)),
        "std_seconds": float(np.std(values)),
        "coefficient_of_variation": float(np.std(values) / mean_duration),
        "p90_seconds": float(np.quantile(values, 0.90)),
        "p95_seconds": float(np.quantile(values, 0.95)),
        "p99_seconds": float(np.quantile(values, 0.99)),
        "max_seconds": float(np.max(values)),
    }


def _valid_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    app = frame["app"].astype("string")
    func = frame["func"].astype("string")
    end = pd.to_numeric(frame["end_timestamp"], errors="coerce")
    duration = pd.to_numeric(frame["duration"], errors="coerce")
    invalid_id = app.isna() | func.isna() | (app.str.len() == 0) | (func.str.len() == 0)
    invalid_end = ~np.isfinite(end)
    invalid_duration = ~np.isfinite(duration) | (duration <= 0)
    valid = ~(invalid_id | invalid_end | invalid_duration)
    cleaned = pd.DataFrame(
        {
            "app": app[valid].astype(str),
            "func": func[valid].astype(str),
            "end_timestamp": end[valid].astype(float),
            "duration": duration[valid].astype(float),
        }
    )
    cleaned["demand_timestamp"] = cleaned["end_timestamp"] - cleaned["duration"]
    finite_demand = np.isfinite(cleaned["demand_timestamp"])
    removed_demand = int((~finite_demand).sum())
    cleaned = cleaned[finite_demand]
    return cleaned, {
        "invalid_id": int(invalid_id.sum()),
        "invalid_end_timestamp": int((invalid_end & ~invalid_id).sum()),
        "invalid_duration": int((invalid_duration & ~invalid_id & ~invalid_end).sum()),
        "invalid_demand_timestamp": removed_demand,
    }


def _synthetic_scenarios(config: dict[str, Any], processed_dir: Path) -> dict[str, Any]:
    rng = np.random.default_rng(int(config["seed"]))
    bins = int(config["data"].get("synthetic_bins", 720))
    t = np.arange(bins)
    shapes = {
        "high_volume": 30 + 8 * np.sin(2 * np.pi * t / 288),
        "bursty": 14 + 55 * ((t % 97) < 5) + 5 * np.sin(2 * np.pi * t / 48),
        "diurnal": 18 + 14 * (1 + np.sin(2 * np.pi * (t - 60) / 288)),
    }
    scenarios: dict[str, Any] = {}
    service_rows: list[dict[str, str | float | int]] = []
    for index, (name, rate) in enumerate(shapes.items()):
        counts = rng.poisson(np.maximum(rate, 0.2))
        durations = rng.lognormal(mean=-0.1 + index * 0.1, sigma=0.8, size=int(counts.sum()))
        train_bins = int(0.6 * bins)
        validation_bins = int(0.8 * bins)
        train_jobs = int(counts[:train_bins].sum())
        validation_jobs = int(counts[train_bins:validation_bins].sum())
        np.savez_compressed(
            processed_dir / f"{name}.npz",
            counts=counts,
            durations=durations,
            durations_train=durations[:train_jobs],
            durations_validation=durations[train_jobs : train_jobs + validation_jobs],
            durations_test=durations[train_jobs + validation_jobs :],
        )
        scenarios[name] = {
            "app": f"synthetic-{name}",
            "training_events": int(counts[: int(0.6 * bins)].sum()),
            "synthetic": True,
        }
        for split, values in {
            "train": durations[:train_jobs],
            "validation": durations[train_jobs : train_jobs + validation_jobs],
            "test": durations[train_jobs + validation_jobs :],
        }.items():
            service_rows.append(_service_stat_row(name, split, np.asarray(values, dtype=float)))
    Path("artifacts").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(service_rows).to_csv("artifacts/service_time_statistics.csv", index=False)
    return scenarios


def prepare_data(config: dict[str, Any], raw_dir: Path = Path("data/raw")) -> dict[str, Any]:
    processed_dir = Path("data/processed")
    artifact_dir = Path("artifacts")
    processed_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if bool(config["data"].get("force_synthetic_fixture", False)):
        scenarios = _synthetic_scenarios(config, processed_dir)
        payload = {"source": "synthetic_smoke_fixture", "scenarios": scenarios}
        (artifact_dir / "scenario_selection.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
    try:
        trace = locate_trace(raw_dir)
    except FileNotFoundError:
        if not bool(config["data"].get("synthetic_if_missing", False)):
            raise
        scenarios = _synthetic_scenarios(config, processed_dir)
        payload = {"source": "synthetic_smoke_fixture", "scenarios": scenarios}
        (artifact_dir / "scenario_selection.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    total = 0
    removed: defaultdict[str, int] = defaultdict(int)
    minimum = np.inf
    maximum = -np.inf
    for raw in iter_trace_chunks(trace):
        total += len(raw)
        clean, reasons = _valid_rows(raw)
        for reason, count in reasons.items():
            removed[reason] += count
        if not clean.empty:
            minimum = min(minimum, float(clean["demand_timestamp"].min()))
            maximum = max(maximum, float(clean["demand_timestamp"].max()))
    train_end = minimum + 0.6 * (maximum - minimum)
    bin_seconds = int(config["data"]["bin_seconds"])
    app_bins: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    app_events: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    app_duration_samples: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    validation_end = minimum + 0.8 * (maximum - minimum)
    bins_total = int(np.ceil((maximum - minimum) / bin_seconds)) + 1
    train_bins = max(1, int(np.ceil((train_end - minimum) / bin_seconds)))
    validation_bins = max(1, int(np.ceil((validation_end - train_end) / bin_seconds)))
    test_bins = max(1, bins_total - train_bins - validation_bins)
    for raw in iter_trace_chunks(trace):
        clean, _ = _valid_rows(raw)
        clean = clean.copy()
        clean["bin"] = np.floor((clean["demand_timestamp"] - minimum) / bin_seconds).astype(int)
        clean["split"] = np.where(
            clean["demand_timestamp"] < train_end,
            "train",
            np.where(clean["demand_timestamp"] < validation_end, "validation", "test"),
        )
        grouped = clean.groupby(["app", "bin"]).size()
        for key, raw_count in grouped.items():
            if not isinstance(key, tuple) or len(key) != 2:
                raise RuntimeError("Expected an (app, bin) group key")
            app, bin_index = str(key[0]), int(str(key[1]))
            count = int(str(raw_count))
            app_bins[app][bin_index] += count
        for (app, split), group in clean.groupby(["app", "split"]):
            app_events[str(app)][str(split)] += int(len(group))
            app_duration_samples[str(app)][str(split)] += int(len(group))
    rules = dict(config["data"]["scenario_eligibility"])
    stats: list[ScenarioStat] = []
    diagnostics_by_app: dict[str, dict[str, float | int | list[str] | bool]] = {}
    for app, bins in app_bins.items():
        counts = np.zeros(bins_total, dtype=float)
        for bin_index, count in bins.items():
            if 0 <= bin_index < bins_total:
                counts[bin_index] = count
        train_counts = counts[:train_bins]
        validation_counts = counts[train_bins : train_bins + validation_bins]
        test_counts = counts[train_bins + validation_bins :]
        diagnostics: dict[str, float | int] = {
            "train_events": int(app_events[app]["train"]),
            "validation_events": int(app_events[app]["validation"]),
            "test_events": int(app_events[app]["test"]),
            "validation_nonzero_bins": int(np.count_nonzero(validation_counts)),
            "test_nonzero_bins": int(np.count_nonzero(test_counts)),
            "validation_nonzero_fraction": float(np.count_nonzero(validation_counts) / validation_bins),
            "test_nonzero_fraction": float(np.count_nonzero(test_counts) / test_bins),
            "train_service_samples": int(app_duration_samples[app]["train"]),
            "train_nonzero_bins": int(np.count_nonzero(train_counts)),
            "train_unique_positive_counts": int(len(np.unique(train_counts[train_counts > 0]))),
        }
        eligible, failures = scenario_is_eligible(diagnostics, rules)
        diagnostics_by_app[app] = {**diagnostics, "eligible": eligible, "ineligible_reasons": failures}
        if not eligible:
            continue
        mean = float(train_counts.mean())
        burst = float(np.quantile(train_counts, 0.99) / (mean + 1e-9))
        autocorr = float(pd.Series(train_counts).autocorr(lag=288)) if len(train_counts) > 288 else np.nan
        stats.append(
            {
                "app": app,
                "events": int(app_events[app]["train"]),
                "burst": burst,
                "autocorr": autocorr,
            }
        )
    if len(stats) < 3:
        raise RuntimeError("Fewer than three applications satisfy precommitted scenario eligibility rules")
    high = max(stats, key=lambda row: int(row["events"]))
    bursty = max(
        (row for row in stats if row["app"] != high["app"]),
        key=lambda row: (float(row["burst"]), int(row["events"])),
    )
    diurnal_candidates = [
        row
        for row in stats
        if row["app"] not in {high["app"], bursty["app"]}
        and np.isfinite(float(row["autocorr"]))
        and float(row["autocorr"]) > 0
    ]
    if not diurnal_candidates:
        raise RuntimeError("No distinct eligible application has positive lag-288 autocorrelation")
    diurnal = max(diurnal_candidates, key=lambda row: float(row["autocorr"]))
    selected = {"high_volume": high, "bursty": bursty, "diurnal": diurnal}
    selected_apps = {str(row["app"]): name for name, row in selected.items()}
    counts_by_name = {name: np.zeros(bins_total, dtype=np.int64) for name in selected}
    durations_by_name: dict[str, dict[str, list[float]]] = {
        name: {"train": [], "validation": [], "test": []} for name in selected
    }
    for raw in iter_trace_chunks(trace):
        clean, _ = _valid_rows(raw)
        subset = clean[clean["app"].isin(selected_apps)]
        for raw_app, group in subset.groupby("app"):
            name = selected_apps[str(raw_app)]
            indices = np.floor((group["demand_timestamp"].to_numpy() - minimum) / bin_seconds).astype(int)
            counts_by_name[name] += np.bincount(indices, minlength=bins_total)[:bins_total]
            train_group = group[group["demand_timestamp"] < train_end]
            validation_group = group[
                (group["demand_timestamp"] >= train_end) & (group["demand_timestamp"] < validation_end)
            ]
            test_group = group[group["demand_timestamp"] >= validation_end]
            durations_by_name[name]["train"].extend(train_group["duration"].astype(float).tolist())
            durations_by_name[name]["validation"].extend(validation_group["duration"].astype(float).tolist())
            durations_by_name[name]["test"].extend(test_group["duration"].astype(float).tolist())
    service_rows: list[dict[str, str | float | int]] = []
    for name in selected:
        split_arrays = {
            split: np.asarray(durations_by_name[name][split], dtype=float) for split in ("train", "validation", "test")
        }
        np.savez_compressed(
            processed_dir / f"{name}.npz",
            counts=counts_by_name[name],
            durations=np.concatenate(
                [np.asarray(durations_by_name[name][split], dtype=float) for split in ("train", "validation", "test")]
            ),
            durations_train=split_arrays["train"],
            durations_validation=split_arrays["validation"],
            durations_test=split_arrays["test"],
        )
        for split, values in split_arrays.items():
            service_rows.append(_service_stat_row(name, split, values))
    pd.DataFrame(service_rows).to_csv(artifact_dir / "service_time_statistics.csv", index=False)
    official_payload: dict[str, Any] = {
        "source": "official_azure_2021",
        "time_bounds": {"minimum": minimum, "train_end": train_end, "maximum": maximum},
        "filtering": {
            "original_rows": total,
            "removed": dict(removed),
            "final_rows": total - sum(removed.values()),
        },
        "eligibility_rules": rules,
        "eligible_application_count": len(stats),
        "scenario_diagnostics": {name: diagnostics_by_app[str(value["app"])] for name, value in selected.items()},
        "scenarios": selected,
    }
    (artifact_dir / "scenario_selection.json").write_text(
        json.dumps(official_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return official_payload
