import numpy as np

from mfcontrol.control.mpc import MPCSettings, choose_mpc_action, mpc_targets
from mfcontrol.control.reactive import reactive_targets


def settings() -> MPCSettings:
    return MPCSettings(
        n=20,
        minimum=2,
        ramp=4,
        d=2,
        service_rate=1.0,
        mean_service=1.0,
        alpha=0.1,
        activation_delay_bins=1,
    )


def test_mpc_is_bounded_ramped_and_deterministic() -> None:
    forecasts = np.asarray([[3.0, 5.0, 7.0], [12.0, 9.0, 4.0]])
    first = mpc_targets(forecasts, settings(), initial=5)
    second = mpc_targets(forecasts, settings(), initial=5)
    np.testing.assert_array_equal(first, second)
    assert np.all((first >= 2) & (first <= 20))
    assert abs(first[0] - 5) <= 4
    assert abs(first[1] - first[0]) <= 4


def test_high_forecast_increases_capacity() -> None:
    low = choose_mpc_action(np.asarray([0.1, 0.1, 0.1]), 5, settings())
    high = choose_mpc_action(np.asarray([30.0, 30.0, 30.0]), 5, settings())
    assert high >= low


def test_reactive_respects_ramp() -> None:
    targets = reactive_targets(np.asarray([0.0, 100.0, 0.0]), 1.0, 20, 2, 3, initial=2)
    assert np.all(np.abs(np.diff(np.r_[2, targets])) <= 3)
