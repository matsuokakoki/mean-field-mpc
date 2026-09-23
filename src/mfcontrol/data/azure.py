from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

EXPECTED_COLUMNS = ("app", "func", "end_timestamp", "duration")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def locate_trace(raw_dir: Path) -> Path:
    candidates = [
        path for path in raw_dir.rglob("*") if path.is_file() and path.suffix.lower() in {".csv", ".txt", ".tsv"}
    ]
    if not candidates:
        raise FileNotFoundError("No extracted CSV/TXT/TSV trace file found")
    return max(candidates, key=lambda path: path.stat().st_size)


def iter_trace_chunks(path: Path, chunksize: int = 500_000) -> Iterator[pd.DataFrame]:
    sample = path.read_bytes()[:8192].decode("utf-8", errors="replace")
    separator = "\t" if sample.count("\t") > sample.count(",") else ","
    reader = pd.read_csv(path, sep=separator, chunksize=chunksize, low_memory=False)
    for frame in reader:
        frame.columns = [str(column).strip().lower() for column in frame.columns]
        missing = set(EXPECTED_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"Trace is missing columns: {sorted(missing)}")
        yield frame.loc[:, list(EXPECTED_COLUMNS)]


def download_trace(config: dict[str, Any], raw_dir: Path = Path("data/raw")) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    source_url = str(config["data"]["source_url"])
    archive = raw_dir / "AzureFunctionsInvocationTraceForTwoWeeksJan2021.rar"
    errors: list[str] = []
    if not archive.exists():
        for attempt_number in range(3):
            try:
                with requests.get(source_url, stream=True, timeout=(30, 180)) as response:
                    response.raise_for_status()
                    with archive.open("wb") as handle:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                handle.write(chunk)
                break
            except requests.RequestException as error:
                errors.append(f"attempt {attempt_number + 1}: {error}")
        else:
            (raw_dir / "download_error.json").write_text(
                json.dumps({"source_url": source_url, "errors": errors}, indent=2),
                encoding="utf-8",
            )
            raise RuntimeError("Official Azure archive download failed after 3 attempts")
    manifest: dict[str, Any] = {
        "source_url": source_url,
        "downloaded_utc": datetime.now(UTC).isoformat(),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256_file(archive),
    }
    manifest_path = raw_dir / "download_manifest.json"
    extraction_tool = "existing verified extraction"
    previous_manifest: dict[str, Any] | None = None
    if manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous_manifest.get("archive_sha256") != manifest["archive_sha256"]:
            raise RuntimeError("Existing Azure archive hash differs from its recorded manifest")
        trace = locate_trace(raw_dir)
    else:
        extraction_tool = "unar"
        extraction = subprocess.run(["unar", "-f", "-o", str(raw_dir), str(archive)], check=False)
        if extraction.returncode != 0:
            extraction_tool = "official 7-Zip 26.02 fallback after unar RAR5 failure"
            subprocess.run(["7zz", "x", "-y", f"-o{raw_dir}", str(archive)], check=True)
        trace = locate_trace(raw_dir)
    rows = 0
    schema: dict[str, str] = {}
    for chunk in iter_trace_chunks(trace):
        rows += len(chunk)
        if not schema:
            schema = {str(column): str(dtype) for column, dtype in chunk.dtypes.items()}
    extracted_hash = sha256_file(trace)
    if previous_manifest is not None and previous_manifest.get("extracted_sha256") != extracted_hash:
        raise RuntimeError("Existing extracted trace hash differs from its recorded manifest")
    manifest.update(
        {
            "extracted_file": trace.relative_to(raw_dir).as_posix(),
            "extracted_sha256": extracted_hash,
            "row_count": rows,
            "schema": schema,
            "extraction_tool": extraction_tool,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return trace
