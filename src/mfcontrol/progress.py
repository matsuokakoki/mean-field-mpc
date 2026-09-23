from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class ProgressLogger:
    """Bind-mounted, atomic progress and append-only event logging.

    This class deliberately owns no scientific random-number generator.
    """

    def __init__(self, total_stages: int, root: Path = Path(".")) -> None:
        now = datetime.now(UTC)
        stamp = now.strftime("%Y%m%dT%H%M%SZ")
        self.root = root
        self.logs = root / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.log_path = self.logs / f"reproduce_{stamp}.log"
        self.events_path = self.logs / "events.jsonl"
        self.progress_path = root / "artifacts" / "progress.json"
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.total_stages = total_stages
        self.state: dict[str, Any] = {
            "status": "starting",
            "stage": None,
            "stage_index": 0,
            "total_stages": total_stages,
            "scenario": None,
            "controller": None,
            "seed": None,
            "total_seeds": None,
            "completed_units": 0,
            "total_units": 0,
            "pipeline_start_time": now.isoformat(),
            "stage_start_time": None,
            "last_heartbeat": now.isoformat(),
            "elapsed_seconds": 0.0,
            "last_completed_artifact": None,
        }
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._write_progress()
        self.event("pipeline_start", message="pipeline initialized")
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._thread.start()

    def _timestamp(self) -> str:
        return datetime.now(UTC).isoformat()

    def _write_progress(self) -> None:
        with self._lock:
            self.state["elapsed_seconds"] = round(time.monotonic() - self.started, 3)
            self.state["last_heartbeat"] = self._timestamp()
            temporary = self.progress_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(temporary, self.progress_path)

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(30.0):
            self.event("heartbeat", message="pipeline still running")

    def event(self, event_type: str, message: str = "", **fields: Any) -> None:
        with self._lock:
            self.state.update({key: value for key, value in fields.items() if value is not None})
            self.state["elapsed_seconds"] = round(time.monotonic() - self.started, 3)
            self.state["last_heartbeat"] = self._timestamp()
            event = {
                "timestamp": self.state["last_heartbeat"],
                "event_type": event_type,
                "stage": self.state.get("stage"),
                "scenario": self.state.get("scenario"),
                "controller": self.state.get("controller"),
                "seed": self.state.get("seed"),
                "total_seeds": self.state.get("total_seeds"),
                "completed_units": self.state.get("completed_units"),
                "total_units": self.state.get("total_units"),
                "elapsed_seconds": self.state["elapsed_seconds"],
                "message": message,
            }
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
                handle.flush()
            clock = datetime.now().strftime("%H:%M:%S")
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"[{clock}] {event_type.upper()} {message}\n")
                handle.flush()
            temporary = self.progress_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(temporary, self.progress_path)

    def stage_start(self, stage: str, stage_index: int, total_units: int = 0) -> None:
        self.event(
            "stage_start",
            message=f"START {stage}",
            status="running",
            stage=stage,
            stage_index=stage_index,
            stage_start_time=self._timestamp(),
            completed_units=0,
            total_units=total_units,
            scenario=None,
            controller=None,
            seed=None,
            total_seeds=None,
        )

    def work_start(self, scenario: str, seed: int, total_seeds: int, controller: str | None = None) -> None:
        self.event(
            "work_unit_start",
            message=f"scenario={scenario} controller={controller or 'all'} seed={seed}/{total_seeds}",
            scenario=scenario,
            controller=controller,
            seed=seed,
            total_seeds=total_seeds,
        )

    def work_complete(self, artifact: str | None = None) -> None:
        self.event(
            "work_unit_complete",
            message="work unit complete",
            completed_units=int(self.state.get("completed_units", 0)) + 1,
            last_completed_artifact=artifact,
        )

    def stage_complete(self, artifact: str | None = None) -> None:
        self.event("stage_complete", message=f"COMPLETE {self.state.get('stage')}", last_completed_artifact=artifact)

    def fail(self, message: str) -> None:
        self.event("error", message=message, status="failed")

    def complete(self) -> None:
        self.event("pipeline_complete", message="pipeline completed", status="completed")
        self._stop.set()
        self._thread.join(timeout=1.0)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)


_ACTIVE: ProgressLogger | None = None


@contextmanager
def active_progress(logger: ProgressLogger) -> Iterator[None]:
    global _ACTIVE
    old = _ACTIVE
    _ACTIVE = logger
    try:
        yield
    finally:
        _ACTIVE = old


def progress() -> ProgressLogger | None:
    return _ACTIVE
