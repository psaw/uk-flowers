#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

mode="${1:-all}"
case "$mode" in
  all|smoke|quality) ;;
  *) echo "Usage: bash scripts/pilot-check.sh [all|smoke|quality]" >&2; exit 2 ;;
esac

export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.cache/uv}"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export XDG_CACHE_HOME="$PWD/.cache"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export UV_NO_PROGRESS=1

uv sync --locked --no-default-groups --group check --group dev --python 3.13
uv run --no-sync python scripts/pilot-check.py "$mode"
