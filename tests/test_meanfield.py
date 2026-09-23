import math

import numpy as np
import pytest

from mfcontrol.meanfield.fixed_point import fixed_point_tail, recurrence_residual
from mfcontrol.meanfield.metrics import expected_jobs, mean_system_time, predicted_wait


def test_d1_is_geometric() -> None:
    tail = fixed_point_tail(0.7, d=1, tolerance=1e-8)
    expected = np.asarray([0.7**k for k in range(len(tail))])
    np.testing.assert_allclose(tail, expected)


def test_d2_satisfies_recurrence_and_is_monotone() -> None:
    tail = fixed_point_tail(0.9, d=2)
    assert recurrence_residual(tail, 0.9, 2) < 1e-12
    assert np.all(np.diff(tail) <= 0)


def test_low_and_near_capacity_are_stable() -> None:
    assert expected_jobs(1e-10, 2) == pytest.approx(1e-10)
    value = expected_jobs(0.994, 2)
    assert math.isfinite(value) and value > 0
    assert predicted_wait(99.4, 100, 1.0, 2) < 1e6
    assert predicted_wait(99.6, 100, 1.0, 2) >= 1e6


def test_zero_rate_limit_and_validation() -> None:
    assert mean_system_time(0.0, 2.0) == pytest.approx(0.5)
    assert predicted_wait(0.0, 3, 2.0) == 0.0
    with pytest.raises(ValueError):
        fixed_point_tail(1.0)
