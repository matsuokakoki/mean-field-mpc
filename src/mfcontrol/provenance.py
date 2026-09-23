from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mfcontrol.config import config_hash


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str | None:
    completed = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, check=False, text=True)
    return completed.stdout.strip() or None


def build_manifest(config: dict[str, Any], config_path: Path, output_dir: Path) -> dict[str, Any]:
    raw_manifest_path = Path("data/raw/download_manifest.json")
    raw = (
        json.loads(raw_manifest_path.read_text(encoding="utf-8"))
        if raw_manifest_path.exists()
        else {"source_url": "synthetic smoke fixture"}
    )
    scenarios_path = Path("artifacts/scenario_selection.json")
    scenarios = json.loads(scenarios_path.read_text(encoding="utf-8"))
    if scenarios.get("source") == "synthetic_smoke_fixture":
        raw = {"source_url": "deterministic synthetic smoke fixture"}
    hyperparameters_path = output_dir / "selected_hyperparameters.json"
    hyperparameters = json.loads(hyperparameters_path.read_text(encoding="utf-8"))
    dependencies = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"]
    }
    manifest = {
        "run_timestamp_utc": datetime.now(UTC).isoformat(),
        "profile": config["profile"],
        "git_commit": _git_commit(),
        "code_version": "0.1.0",
        "python_version": platform.python_version(),
        "dependencies": dict(sorted(dependencies.items(), key=lambda item: item[0].lower())),
        "docker_image_id": "not exposed; Docker socket is intentionally not mounted",
        "uv_lock_sha256": _sha256(Path("uv.lock")),
        "config_file": config_path.as_posix(),
        "config_sha256": config_hash(config_path),
        "configuration": config,
        "raw_dataset": {
            "source_url": raw.get("source_url"),
            "archive_sha256": raw.get("archive_sha256"),
            "extracted_sha256": raw.get("extracted_sha256"),
            "row_count": raw.get("row_count"),
        },
        "scenario_selection": scenarios,
        "seeds": config["seeds"],
        "controller_alpha_and_gp_beta": hyperparameters,
        "scientific_assertions": {
            "scenario_selection_partition": "train only",
            "gp_fit_partition": "train only",
            "beta_selection_partition": "validation only",
            "alpha_selection_partition": "validation only",
            "test_tuning_permitted": False,
            "oracle_future_access_isolated": True,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
