#!/usr/bin/env bash
#SBATCH --partition=RTX4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=06:00:00
# Output path is set by submit.sh.
set -euo pipefail
cd "${AOD_PROJECT_ROOT:?Run through acceptance/bin/submit.sh}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
unset PYTHONPATH
PY="$PWD/.venv/bin/python"
srun "$PY" experiment/bind_allocated_gpu.py --audit "acceptance/logs/gpu-${SLURM_JOB_ID}.json" --expected-count 1 --min-free-mib 22000 -- "$PY" -u "$@"
