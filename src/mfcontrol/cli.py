from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from mfcontrol.analysis.reporting import build_report
from mfcontrol.config import load_config
from mfcontrol.data.azure import download_trace
from mfcontrol.data.preprocess import prepare_data
from mfcontrol.experiments.controllers import run_controller_experiments
from mfcontrol.experiments.forecasting import run_forecast_evaluation
from mfcontrol.experiments.meanfield_validation import run_meanfield_validation
from mfcontrol.experiments.sensitivity import run_sensitivity_experiments
from mfcontrol.progress import ProgressLogger, active_progress
from mfcontrol.provenance import build_manifest


def _pipeline_fingerprint(config_path: Path) -> str:
    digest = hashlib.sha256()
    for path in [config_path, Path("uv.lock"), *sorted(Path("src").rglob("*.py"))]:
        digest.update(path.as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _run_cached(
    stage: str,
    fingerprint: str,
    outputs: list[Path],
    action: Callable[[], Any],
    output_dir: Path = Path("results"),
) -> None:
    cache_path = output_dir / ".stage_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    if cache.get(stage) == fingerprint and all(path.exists() for path in outputs):
        print(f"cache hit: {stage}")
        return
    action()
    missing = [path.as_posix() for path in outputs if not path.exists()]
    if missing:
        raise RuntimeError(f"stage {stage} did not create required outputs: {missing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache[stage] = fingerprint
    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mfcontrol")
    sub = parser.add_subparsers(dest="command", required=True)
    data = sub.add_parser("data")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    for name in ("download", "prepare"):
        command = data_sub.add_parser(name)
        command.add_argument("--config", default="configs/paper.yaml")
    validate = sub.add_parser("validate")
    validate_sub = validate.add_subparsers(dest="validate_command", required=True)
    meanfield = validate_sub.add_parser("meanfield")
    meanfield.add_argument("--config", default="configs/paper.yaml")
    forecast = sub.add_parser("forecast")
    forecast_sub = forecast.add_subparsers(dest="forecast_command", required=True)
    evaluate = forecast_sub.add_parser("evaluate")
    evaluate.add_argument("--config", default="configs/paper.yaml")
    experiment = sub.add_parser("experiment")
    experiment_sub = experiment.add_subparsers(dest="experiment_command", required=True)
    controllers = experiment_sub.add_parser("controllers")
    controllers.add_argument("--config", default="configs/paper.yaml")
    sensitivity = experiment_sub.add_parser("sensitivity")
    sensitivity.add_argument("--config", default="configs/paper.yaml")
    report = sub.add_parser("report")
    report_sub = report.add_subparsers(dest="report_command", required=True)
    report_build = report_sub.add_parser("build")
    report_build.add_argument("--config", default="configs/paper.yaml")
    reproduce = sub.add_parser("reproduce")
    reproduce.add_argument("--profile", choices=("smoke", "paper"), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "data":
        config = load_config(args.config)
        if args.data_command == "download":
            print(download_trace(config))
        else:
            print(prepare_data(config))
        return
    if args.command == "validate":
        config = load_config(args.config)
        frame = run_meanfield_validation(config, Path("results"))
        print(frame.groupby(["n", "rho", "d"])["l1_tail_error"].mean())
        return
    if args.command == "forecast":
        config = load_config(args.config)
        metrics = run_forecast_evaluation(config, Path("results"))
        print(metrics.to_string(index=False))
        return
    if args.command == "experiment":
        config = load_config(args.config)
        if args.experiment_command == "controllers":
            metrics = run_controller_experiments(config, Path("results"))
        else:
            metrics = run_sensitivity_experiments(config, Path("results"))
        print(metrics.groupby(["scenario", "controller"])["p95_wait"].mean())
        return
    if args.command == "report":
        config_path = Path(args.config)
        config = load_config(config_path)
        build_report(config, config_path, Path("results"))
        print("paper/report.pdf")
        return
    config_path = Path("configs") / f"{args.profile}.yaml"
    config = load_config(config_path)
    fingerprint = _pipeline_fingerprint(config_path)
    stages: list[tuple[str, list[Path], Callable[[], Any]]] = []
    if args.profile == "paper":
        stages.append(("download", [Path("data/raw/download_manifest.json")], lambda: download_trace(config)))
    stages.extend(
        [
            (
                "prepare",
                [Path("artifacts/scenario_selection.json"), Path("data/processed/bursty.npz")],
                lambda: prepare_data(config),
            ),
            (
                "meanfield",
                [Path("results/meanfield_convergence.csv"), Path("figures/pdf/fig04_meanfield_convergence.pdf")],
                lambda: run_meanfield_validation(config, Path("results")),
            ),
            (
                "forecast",
                [Path("results/forecast_metrics.csv"), Path("results/forecast_predictions.csv")],
                lambda: run_forecast_evaluation(config, Path("results")),
            ),
            (
                "controllers",
                [
                    Path("results/controller_metrics_per_seed.csv"),
                    Path("results/pareto_operating_points.csv"),
                    Path("results/queue_length_distribution.csv"),
                ],
                lambda: run_controller_experiments(config, Path("results")),
            ),
            (
                "sensitivity",
                [Path("results/controller_summary.csv"), Path("figures/pdf/fig12_model_mismatch.pdf")],
                lambda: run_sensitivity_experiments(config, Path("results")),
            ),
        ]
    )
    if args.profile == "paper":
        stages.append(
            (
                "report",
                [Path("paper/report.pdf"), Path("results/result_manifest.json")],
                lambda: build_report(config, config_path, Path("results")),
            )
        )
    else:
        stages.append(
            (
                "manifest",
                [Path("results/result_manifest.json")],
                lambda: build_manifest(config, config_path, Path("results")),
            )
        )
    logger = ProgressLogger(len(stages))
    seed_count = len(config["seeds"])
    expected_units = {
        "download": 1,
        "prepare": 1,
        "meanfield": len(config["meanfield"]["n_values"])
        * len(config["meanfield"]["rho_values"])
        * len(config["meanfield"]["d_values"])
        * seed_count,
        "forecast": 9,
        "controllers": 9 * seed_count,
        "sensitivity": 8 * seed_count,
        "report": 1,
        "manifest": 1,
    }
    try:
        with active_progress(logger):
            for index, (stage, outputs, action) in enumerate(stages, start=1):
                logger.stage_start(stage, index, total_units=expected_units[stage])
                _run_cached(stage, fingerprint, outputs, action)
                logger.stage_complete(outputs[-1].as_posix())
        logger.complete()
    except Exception as error:
        logger.fail(f"{type(error).__name__}: {error}")
        logger.close()
        raise
    print(f"Completed currently implemented stages for {args.profile}")
