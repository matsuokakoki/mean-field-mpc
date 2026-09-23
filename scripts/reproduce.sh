#!/bin/sh
set -eu
uv run python -m mfcontrol reproduce --profile "${1:-paper}"

