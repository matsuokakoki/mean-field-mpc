#!/bin/sh
set -eu
uv run python -m mfcontrol report build --config "${1:-configs/paper.yaml}"

