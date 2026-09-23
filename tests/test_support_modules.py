from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mfcontrol.analysis.bootstrap import bootstrap_interval, paired_bootstrap_difference
from mfcontrol.config import config_hash, load_config
from mfcontrol.control.reactive import reactive_targets
from mfcontrol.control.static import static_capacity
from mfcontrol.forecast.baselines import ewma_forecast, last_value, seasonal_naive
from mfcontrol.forecast.calibration import calibrate_beta
from mfcontrol.sim.arrivals import generate_workload, intensity_floor, scale_factor


def test_baselines_are_causal_and_deterministic() -> None:
    counts = np.arange(400, dtype=float)
    decisions = np.asarray([288, 300, 399])
    np.testing.assert_array_equal(last_value(counts, decisions), [287, 299, 398])
    np.testing.assert_array_equal(seasonal_naive(counts, decisions), [0, 12, 111])
    first = ewma_forecast(counts, decisions, 0.3)
    second = ewma_forecast(counts, decisions, 0.3)
    np.testing.assert_array_equal(first, second)
    with pytest.raises(ValueError):
        ewma_forecast(counts, decisions, 0.0)


def test_simple_capacity_policies_validate_and_respect_bounds() -> None:
    rates = np.asarray([0.0, 5.0, 100.0])
    np.testing.assert_array_equal(reactive_targets(rates, 1.0, 10, 2, 3), [2, 5, 8])
    assert static_capacity(np.asarray([1.0, 2.0, 3.0]), 1.0, 10, 2) == 4
    with pytest.raises(ValueError):
        static_capacity(np.asarray([]), 1.0, 10, 2)


def test_calibration_and_bootstrap_edge_cases() -> None:
    actual = np.asarray([0.0, 1.0, 2.0])
    predicted = np.asarray([0.0, 0.0, 0.0])
    std = np.ones(3)
    assert calibrate_beta(actual, predicted, std) == pytest.approx(1.8)
    assert bootstrap_interval(np.asarray([4.0])) == (4.0, 4.0)
    difference, low, high = paired_bootstrap_difference(np.asarray([2.0, 3.0]), np.asarray([1.0, 1.0]), draws=100)
    assert difference == pytest.approx(1.5)
    assert low <= difference <= high
    with pytest.raises(ValueError):
        calibrate_beta(actual, predicted[:2], std)
    with pytest.raises(ValueError):
        bootstrap_interval(np.asarray([]))
    with pytest.raises(ValueError):
        paired_bootstrap_difference(np.asarray([1.0]), np.asarray([]))


def test_workload_generation_exact_and_weighted_modes() -> None:
    counts = np.asarray([2.0, 4.0])
    assert intensity_floor(counts) == pytest.approx(0.03)
    scaling = scale_factor(counts, 2.0, 4, 10.0, 0.5)
    assert scaling == pytest.approx(10 / 3)
    exact = generate_workload(counts, 1.0, 10.0, np.asarray([1.0, 2.0]), 1.5, "empirical", 8)
    assert exact.realized_counts.sum() == len(exact.jobs)
    assert not exact.weighted_event_approximation
    weighted = generate_workload(np.asarray([100.0]), 1.0, 10.0, np.asarray([2.0]), 2.0, "exponential_matched", 8, 5)
    assert len(weighted.jobs) == 5
    assert weighted.weighted_event_approximation
    assert all(job.service_time > 0 for job in weighted.jobs)
    with pytest.raises(ValueError):
        scale_factor(np.asarray([0.0]), 1.0, 2, 10.0, 0.5)
    with pytest.raises(ValueError):
        intensity_floor(np.asarray([0.0]))
    with pytest.raises(ValueError):
        generate_workload(np.asarray([1.0]), 1.0, 1.0, np.asarray([]), 1.0, "empirical", 1)
    with pytest.raises(ValueError):
        generate_workload(np.asarray([1.0]), 1.0, 1.0, np.asarray([1.0]), 1.0, "invalid", 1)


def test_config_loading_and_hash(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("profile: smoke\n", encoding="utf-8")
    assert load_config(config) == {"profile": "smoke"}
    assert len(config_hash(config)) == 64
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("- not\n- a mapping\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(invalid)
