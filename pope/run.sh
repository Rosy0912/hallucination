#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-$PWD/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then echo 'Run bash setup.sh first, or set PYTHON to an existing verified environment.' >&2; exit 2; fi
if [[ $# -eq 0 ]]; then set -- all; fi
exec "$PYTHON" -u pipeline.py "$@"
