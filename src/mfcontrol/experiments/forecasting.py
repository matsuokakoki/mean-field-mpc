from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mfcontrol.forecast.baselines import ewma_forecast, last_value, seasonal_naive
from mfcontrol.forecast.calibration import calibrate_beta
from mfcontrol.forecast.features import feature_columns, feature_frame
from mfcontrol.forecast.gp import fit_gp, predict_gp

SCENARIOS = ("high_volume", "bursty", "diurnal")


def _split(target_index: np.ndarray, length: int) -> np.ndarray:
    train_end = int(0.6 * length)
    validation_end = int(0.8 * length)
    return np.where(
        target_index < train_end,
        "train",
        np.where(target_index < validation_end, "validation", "test"),
    )


def _metric_rows(predictions: pd.DataFrame, beta: float) -> list[dict[str, Any]]:
    test = predictions[predictions["split"] == "test"]
    rows: list[dict[str, Any]] = []
    for method, column in {
        "last_value": "last_value",
        "seasonal_naive": "seasonal_naive",
        "ewma": "ewma",
        "gp_mean": "gp_mean",
    }.items():
        error = test[column].to_numpy() - test["actual"].to_numpy()
        rows.append(
            {
                "scenario": str(test["scenario"].iloc[0]),
                "horizon": int(test["horizon"].iloc[0]),
                "method": method,
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "stabilized_mape": float(np.mean(np.abs(error) / (test["actual"].to_numpy() + 1.0))),
                "upper_coverage": np.nan,
                "underprediction_rate": np.nan,
                "average_log_width": np.nan,
                "normalized_upper_width": np.nan,
            }
        )
    upper = np.maximum(0.0, np.expm1(test["gp_mean_log"] + beta * test["gp_std_log"]))
    rows.append(
        {
            "scenario": str(test["scenario"].iloc[0]),
            "horizon": int(test["horizon"].iloc[0]),
            "method": "gp_ucb",
            "mae": np.nan,
            "rmse": np.nan,
            "stabilized_mape": np.nan,
            "upper_coverage": float(np.mean(test["actual"].to_numpy() <= upper)),
            "underprediction_rate": float(np.mean(test["actual"].to_numpy() > upper)),
            "average_log_width": float(np.mean(beta * test["gp_std_log"].to_numpy())),
            "normalized_upper_width": float(
                np.mean((upper - test["gp_mean"].to_numpy()) / (test["actual"].to_numpy() + 1.0))
            ),
        }
    )
    return rows


def run_forecast_evaluation(config: dict[str, Any], output_dir: Path) -> pd.DataFrame:
    prediction_frames: list[pd.DataFrame] = []
    metadata: dict[str, Any] = {"scenarios": {}, "warnings": []}
    for scenario_index, scenario in enumerate(SCENARIOS):
        dataset = np.load(Path("data/processed") / f"{scenario}.npz")
        counts = np.asarray(dataset["counts"], dtype=float)
        horizon_frames: list[pd.DataFrame] = []
        validation_actual: list[np.ndarray] = []
        validation_mean: list[np.ndarray] = []
        validation_std: list[np.ndarray] = []
        scenario_meta: dict[str, Any] = {"horizons": {}}
        for horizon in config["forecast"]["horizons"]:
            frame = feature_frame(counts, int(horizon))
            columns = feature_columns(frame)
            target_indices = frame["target_index"].to_numpy(dtype=int)
            splits = _split(target_indices, len(counts))
            train_mask = splits == "train"
            seed = int(config["seed"]) + scenario_index * 10 + int(horizon)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = fit_gp(
                    frame.loc[:, columns].to_numpy(dtype=float)[train_mask],
                    frame["target_log"].to_numpy(dtype=float)[train_mask],
                    int(config["forecast"]["max_train"]),
                    seed,
                    int(config["forecast"]["gp_restarts"]),
                )
            metadata["warnings"].extend(str(item.message) for item in caught)
            mean_log, std_log = predict_gp(model, frame.loc[:, columns].to_numpy(dtype=float))
            decision_indices = frame["decision_index"].to_numpy(dtype=int)
            predictions = pd.DataFrame(
                {
                    "scenario": scenario,
                    "horizon": int(horizon),
                    "decision_index": decision_indices,
                    "target_index": target_indices,
                    "split": splits,
                    "actual": frame["target_count"].to_numpy(dtype=float),
                    "actual_log": frame["target_log"].to_numpy(dtype=float),
                    "last_value": last_value(counts, decision_indices),
                    "seasonal_naive": seasonal_naive(counts, target_indices),
                    "ewma": ewma_forecast(counts, decision_indices, float(config["forecast"]["ewma_alpha"])),
                    "gp_mean_log": mean_log,
                    "gp_std_log": std_log,
                    "gp_mean": np.maximum(0.0, np.expm1(mean_log)),
                }
            )
            validation = predictions[predictions["split"] == "validation"]
            if validation.empty or not bool((validation["split"] == "validation").all()):
                raise AssertionError("beta calibration must use validation rows only")
            validation_actual.append(validation["actual_log"].to_numpy())
            validation_mean.append(validation["gp_mean_log"].to_numpy())
            validation_std.append(validation["gp_std_log"].to_numpy())
            scenario_meta["horizons"][str(horizon)] = {
                "sampled_training_row_indices": model.sampled_indices.tolist(),
                "fitted_kernel": str(model.regressor.kernel_),
            }
            horizon_frames.append(predictions)
        beta = calibrate_beta(
            np.concatenate(validation_actual),
            np.concatenate(validation_mean),
            np.concatenate(validation_std),
        )
        scenario_meta["beta"] = beta
        metadata["scenarios"][scenario] = scenario_meta
        for frame in horizon_frames:
            frame["beta"] = beta
            prediction_frames.append(frame)
    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_predictions.to_csv(output_dir / "forecast_predictions.csv", index=False)
    metric_rows: list[dict[str, Any]] = []
    for (_, _), group in all_predictions.groupby(["scenario", "horizon"]):
        metric_rows.extend(_metric_rows(group, float(group["beta"].iloc[0])))
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "forecast_metrics.csv", index=False)
    (output_dir / "selected_hyperparameters.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    _forecast_figures(all_predictions, metrics)
    return metrics


def _forecast_figures(predictions: pd.DataFrame, metrics: pd.DataFrame) -> None:
    png = Path("figures/png")
    pdf = Path("figures/pdf")
    png.mkdir(parents=True, exist_ok=True)
    pdf.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 6.2), sharex=False)
    for axis, scenario in zip(axes, SCENARIOS, strict=True):
        counts = np.load(Path("data/processed") / f"{scenario}.npz")["counts"]
        axis.plot(counts, linewidth=0.7)
        axis.set_ylabel(scenario)
    axes[-1].set_xlabel("5-minute bin")
    fig.tight_layout()
    for extension, directory in (("png", png), ("pdf", pdf)):
        fig.savefig(directory / f"fig01_workload_scenarios.{extension}", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for scenario in SCENARIOS:
        durations = np.load(Path("data/processed") / f"{scenario}.npz")["durations"]
        values = np.sort(np.asarray(durations, dtype=float))
        probabilities = np.arange(1, len(values) + 1) / len(values)
        ax.plot(values, probabilities, linewidth=1.0, label=scenario)
    ax.set(xlabel="service duration (s, log scale)", ylabel="empirical CDF", xscale="log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    for extension, directory in (("png", png), ("pdf", pdf)):
        fig.savefig(directory / f"fig02_service_distributions.{extension}", dpi=180)
    plt.close(fig)
    burst = predictions[
        (predictions["scenario"] == "bursty") & (predictions["horizon"] == 1) & (predictions["split"] == "test")
    ].copy()
    difference = np.abs(np.diff(burst["actual"].to_numpy(), prepend=burst["actual"].iloc[0]))
    width = min(144, len(burst))
    scores = np.convolve(difference, np.ones(width), mode="valid")
    start = int(np.argmax(scores)) if len(scores) else 0
    window = burst.iloc[start : start + width]
    upper = np.maximum(0.0, np.expm1(window["gp_mean_log"] + window["beta"] * window["gp_std_log"]))
    x = np.arange(len(window))
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    ax.plot(x, window["actual"], label="observed", linewidth=1.0)
    ax.plot(x, window["gp_mean"], label="GP mean", linewidth=1.0)
    ax.fill_between(x, window["gp_mean"], upper, alpha=0.25, label="validation-calibrated upper scenario")
    ax.set(xlabel="5-minute bin in objective burst window", ylabel="invocation count")
    ax.legend(fontsize=8)
    fig.tight_layout()
    for extension, directory in (("png", png), ("pdf", pdf)):
        fig.savefig(directory / f"fig05_gp_forecast_burst.{extension}", dpi=180)
    plt.close(fig)
    point_metrics = metrics[metrics["method"] != "gp_ucb"]
    summary = point_metrics.groupby("method", as_index=False)[["mae"]].mean().sort_values("mae")
    ucb = metrics[metrics["method"] == "gp_ucb"]
    fig, (ax, uncertainty_axis) = plt.subplots(1, 2, figsize=(9.0, 3.8))
    ax.bar(summary["method"], summary["mae"])
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylabel("mean test MAE across scenarios/horizons")
    uncertainty_axis.bar(
        ["coverage", "underprediction", "norm. width"],
        [
            float(ucb["upper_coverage"].mean()),
            float(ucb["underprediction_rate"].mean()),
            float(ucb["normalized_upper_width"].mean()),
        ],
    )
    uncertainty_axis.tick_params(axis="x", rotation=25)
    uncertainty_axis.set_ylabel("GP-UCB uncertainty metric")
    fig.tight_layout()
    for extension, directory in (("png", png), ("pdf", pdf)):
        fig.savefig(directory / f"fig06_forecast_metrics.{extension}", dpi=180)
    plt.close(fig)
