from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler


@dataclass
class GPModel:
    scaler: StandardScaler
    regressor: GaussianProcessRegressor
    sampled_indices: NDArray[np.int64]


def fit_gp(
    features: NDArray[np.float64],
    targets: NDArray[np.float64],
    max_train: int,
    seed: int,
    restarts: int,
) -> GPModel:
    if len(features) != len(targets) or len(features) == 0:
        raise ValueError("features and targets must be nonempty and aligned")
    rng = np.random.default_rng(seed)
    if len(features) > max_train:
        sampled = np.sort(rng.choice(len(features), size=max_train, replace=False)).astype(np.int64)
    else:
        sampled = np.arange(len(features), dtype=np.int64)
    scaler = StandardScaler().fit(features[sampled])
    scaled = scaler.transform(features[sampled])
    dimensions = features.shape[1]
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(dimensions), length_scale_bounds=(1e-2, 1e3), nu=1.5
    ) + WhiteKernel(noise_level=0.05, noise_level_bounds=(1e-6, 10.0))
    regressor = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-8,
        normalize_y=True,
        n_restarts_optimizer=restarts,
        random_state=seed,
    )
    regressor.fit(scaled, targets[sampled])
    return GPModel(scaler, regressor, sampled)


def predict_gp(model: GPModel, features: NDArray[np.float64]) -> tuple[np.ndarray, np.ndarray]:
    mean, std = model.regressor.predict(model.scaler.transform(features), return_std=True)
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std < 0):
        raise FloatingPointError("GP returned invalid predictive moments")
    return np.asarray(mean), np.asarray(std)
