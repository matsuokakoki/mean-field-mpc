import numpy as np

from mfcontrol.forecast.features import feature_columns, feature_frame


def test_future_change_does_not_change_past_features() -> None:
    counts = np.arange(500, dtype=float)
    original = feature_frame(counts, 2)
    changed = counts.copy()
    changed[450:] += 10_000
    modified = feature_frame(changed, 2)
    columns = feature_columns(original)
    past_original = original[original["decision_index"] < 450].set_index("decision_index")[columns]
    past_modified = modified[modified["decision_index"] < 450].set_index("decision_index")[columns]
    np.testing.assert_allclose(past_original, past_modified)


def test_target_is_strictly_future() -> None:
    frame = feature_frame(np.arange(500, dtype=float), 3)
    assert np.all(frame["target_index"] == frame["decision_index"] + 3)
