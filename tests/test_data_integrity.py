from __future__ import annotations

import json
from pathlib import Path

import pytest

from mfcontrol.data.azure import download_trace, sha256_file


def test_existing_download_is_revalidated(tmp_path: Path) -> None:
    archive = tmp_path / "AzureFunctionsInvocationTraceForTwoWeeksJan2021.rar"
    trace = tmp_path / "trace.csv"
    archive.write_bytes(b"archive fixture")
    trace.write_text("app,func,end_timestamp,duration\na,f,2.0,1.0\n", encoding="utf-8")
    manifest = {
        "archive_sha256": sha256_file(archive),
        "extracted_sha256": sha256_file(trace),
    }
    (tmp_path / "download_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    config = {"data": {"source_url": "https://invalid.example/not-used.rar"}}
    assert download_trace(config, tmp_path) == trace
    trace.write_text("app,func,end_timestamp,duration\na,f,3.0,1.0\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="extracted trace hash"):
        download_trace(config, tmp_path)
