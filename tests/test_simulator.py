import numpy as np
import pytest

from mfcontrol.sim.jsqd import CapacityEvent, Job, simulate_jsqd


def test_one_server_deterministic_toy() -> None:
    result = simulate_jsqd([Job(0.0, 2.0), Job(1.0, 1.0)], 1, seed=4)
    np.testing.assert_allclose(result.waiting_times, [0.0, 1.0])
    np.testing.assert_allclose(result.response_times, [2.0, 2.0])


def test_seed_reproducibility_and_invariants() -> None:
    jobs = [Job(i * 0.1, 0.8) for i in range(40)]
    first = simulate_jsqd(jobs, 5, d=2, seed=9)
    second = simulate_jsqd(jobs, 5, d=2, seed=9)
    np.testing.assert_array_equal(first.assigned_servers, second.assigned_servers)
    assert np.all(first.waiting_times >= 0)
    assert np.all(first.response_times >= 0.8)


def test_activation_delay_blocks_warming_server() -> None:
    jobs = [Job(0.1, 1.0), Job(0.2, 1.0), Job(2.0, 1.0)]
    result = simulate_jsqd(
        jobs,
        2,
        d=2,
        seed=3,
        initial_active=1,
        activation_delay=1.0,
        capacity_events=[CapacityEvent(0.0, 2)],
    )
    assert result.assigned_servers[0] == 0
    assert result.assigned_servers[1] == 0
    assert result.assigned_servers[2] in {0, 1}


def test_bad_service_rejected() -> None:
    with pytest.raises(ValueError):
        simulate_jsqd([Job(0.0, 0.0)], 1)


def test_scale_down_cancels_warming_before_active_capacity() -> None:
    result = simulate_jsqd(
        [Job(2.0, 1.0)],
        2,
        d=2,
        seed=3,
        initial_active=1,
        activation_delay=1.0,
        capacity_events=[CapacityEvent(0.0, 2), CapacityEvent(0.5, 1)],
    )
    assert result.assigned_servers.tolist() == [0]
    assert all(active_count == 1 for _, active_count in result.effective_trace)
