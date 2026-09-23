import numpy as np
import pytest

from mfcontrol.forecast.calibration import calibrate_beta
from mfcontrol.forecast.gp import fit_gp, predict_gp


def test_gp_std_is_finite_nonnegative() -> None:
    x = np.linspace(0, 1, 30)[:, None]
    model = fit_gp(x, np.sin(x[:, 0]), max_train=30, seed=1, restarts=0)
    _, std = predict_gp(model, x)
    assert np.all(np.isfinite(std))
    assert np.all(std >= 0)


def test_calibration_empirical_quantile() -> None:
    actual = np.asarray([0.0, 1.0, 2.0])
    beta = calibrate_beta(actual, np.zeros(3), np.ones(3))
    assert beta == pytest.approx(1.8)
