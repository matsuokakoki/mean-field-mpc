from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mfcontrol.analysis.metrics import latency_metrics
from mfcontrol.analysis.reporting import _paired_effect, _sentence
from mfcontrol.data.preprocess import scenario_is_eligible
from mfcontrol.experiments.controllers import _relative_budget_difference, _select_alphas
from mfcontrol.experiments.forecasting import _metric_rows
from mfcontrol.progress import ProgressLogger


def test_scenario_eligibility_rejects_empty_test_partition() -> None:
    rules = {
        "min_train_events": 50,
        "min_validation_events": 20,
        "min_test_events": 20,
        "min_validation_nonzero_bins": 4,
        "min_test_nonzero_bins": 4,
        "min_validation_nonzero_fraction": 0.02,
        "min_test_nonzero_fraction": 0.02,
        "min_train_service_samples": 50,
        "min_train_nonzero_bins": 12,
        "min_train_unique_positive_counts": 3,
    }
    diagnostics = {
        "train_events": 100,
        "validation_events": 30,
        "test_events": 0,
        "validation_nonzero_bins": 5,
        "test_nonzero_bins": 0,
        "validation_nonzero_fraction": 0.1,
        "test_nonzero_fraction": 0.0,
        "train_service_samples": 100,
        "train_nonzero_bins": 20,
        "train_unique_positive_counts": 4,
    }
    eligible, reasons = scenario_is_eligible(diagnostics, rules)
    assert not eligible
    assert {"test_events", "test_nonzero_bins", "test_nonzero_fraction"} <= set(reasons)


def test_insufficient_jobs_are_not_encoded_as_zero_tail_latency() -> None:
    metrics = latency_metrics(np.asarray([]), np.asarray([]), min_jobs_for_tail=10)
    assert metrics["metric_status"] == "insufficient_data"
    assert np.isnan(metrics["p95_wait"])
    small = latency_metrics(np.asarray([0.0, 0.0]), np.asarray([1.0, 1.0]), min_jobs_for_tail=10)
    assert small["metric_status"] == "insufficient_data"
    assert np.isnan(small["p95_wait"])


def _validation_frame() -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for decision in range(10):
        for horizon in (1, 2, 3):
            rows.append(
                {
                    "split": "validation",
                    "horizon": horizon,
                    "decision_index": decision,
                    "last_value": 4.0,
                    "seasonal_naive": 4.0,
                    "ewma": 4.0,
                    "gp_mean": 4.0,
                    "gp_mean_log": np.log1p(4.0),
                    "gp_std_log": 0.1,
                    "beta": 1.0,
                    "actual": 999999.0,
                    "test_only_latency": -999999.0,
                }
            )
    return pd.DataFrame(rows)


def test_alpha_search_is_validation_only_and_records_tolerance() -> None:
    config = {
        "control": {
            "min_fraction": 0.1,
            "ramp_fraction": 0.5,
            "d": 2,
            "queue_weight": 1.0,
            "switching_weight": 0.05,
            "rho_max": 0.995,
            "capacity_step": 1,
            "rho_reactive": 0.7,
            "alpha_grid": [0.01, 0.1, 1.0],
            "alpha_search": {
                "min_alpha": 0.001,
                "max_alpha": 10.0,
                "initial_points": 5,
                "max_refinements": 3,
                "relative_tolerance": 0.5,
            },
        }
    }
    selected, _, matches = _select_alphas(config, _validation_frame(), 1.0, 1.0, 1.0, 20, 0)
    assert set(selected)
    assert all(match["selection_partition"] == "validation" for match in matches.values())
    assert all("actual" not in match for match in matches.values())
    assert _relative_budget_difference(1.0, 1.0) == 0.0


def test_gp_ucb_metrics_are_uncertainty_metrics_not_point_forecast_ranking() -> None:
    frame = pd.DataFrame(
        {
            "split": ["test", "test"],
            "scenario": ["fixture", "fixture"],
            "horizon": [1, 1],
            "actual": [2.0, 3.0],
            "last_value": [2.0, 2.0],
            "seasonal_naive": [2.0, 2.0],
            "ewma": [2.0, 2.0],
            "gp_mean": [2.0, 2.0],
            "gp_mean_log": np.log1p([2.0, 2.0]),
            "gp_std_log": [0.5, 0.5],
        }
    )
    rows = {row["method"]: row for row in _metric_rows(frame, beta=1.0)}
    assert np.isnan(rows["gp_ucb"]["mae"])
    assert rows["gp_ucb"]["underprediction_rate"] >= 0.0
    assert rows["gp_ucb"]["upper_coverage"] <= 1.0


def test_progress_writes_atomic_state_and_failure_events_without_rng_side_effect(tmp_path: Path) -> None:
    np.random.seed(123)
    expected = np.random.random()
    np.random.seed(123)
    logger = ProgressLogger(2, tmp_path)
    logger.stage_start("controllers", 1, total_units=2)
    logger.work_start("fixture", 1, 2, "gp_ucb")
    logger.work_complete("results/example.csv")
    logger.fail("fixture failure")
    logger.close()
    assert np.random.random() == expected
    payload = json.loads((tmp_path / "artifacts/progress.json").read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["last_completed_artifact"] == "results/example.csv"
    events = (tmp_path / "logs/events.jsonl").read_text(encoding="utf-8")
    assert "work_unit_start" in events and "error" in events
    assert list((tmp_path / "logs").glob("reproduce_*.log"))


def test_paired_ewma_effect_uses_seed_pairs_and_human_sign_wording() -> None:
    rows = []
    for seed, reactive, ewma in ((1, 100.0, 60.0), (2, 200.0, 100.0)):
        rows.extend(
            [
                {
                    "experiment": "primary", "scenario": "high_volume", "controller": "reactive", "seed": seed,
                    "p95_wait": reactive, "mean_resource_fraction": 0.50,
                },
                {
                    "experiment": "primary", "scenario": "high_volume", "controller": "ewma_mf_mpc", "seed": seed,
                    "p95_wait": ewma, "mean_resource_fraction": 0.505,
                },
            ]
        )
    effect = _paired_effect(pd.DataFrame(rows), "high_volume", "ewma_mf_mpc", "reactive")
    assert effect["seeds"] == 2
    assert effect["point_estimate_percent_reduction"] == 45.0
    assert np.isclose(effect["relative_resource_mismatch"], 0.01)
    sentence = _sentence(effect, "EWMA", "reactive")
    assert "reduced" in sentence and "45.0%" in sentence
