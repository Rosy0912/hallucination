#!/usr/bin/env bash
#SBATCH --partition=RTX4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=04:00:00
set -euo pipefail
cd "${BENCH_ROOT:?Set BENCH_ROOT}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
unset PYTHONPATH
PY="${BENCH_PYTHON:-$PWD/legacy/.venv/bin/python}"
srun "$PY" legacy/experiment/bind_allocated_gpu.py --audit "logs/gpu-${SLURM_JOB_ID}.json" --expected-count 1 --min-free-mib 22000 -- "$PY" -u "$@"
