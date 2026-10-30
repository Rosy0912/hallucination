#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-python3.11}"
"$BOOTSTRAP_PYTHON" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements.txt
echo 'Setup complete. On an allocated CUDA GPU: bash run.sh all'
