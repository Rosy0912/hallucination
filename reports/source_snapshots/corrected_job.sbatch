#!/usr/bin/env bash
#SBATCH --partition=RTX3090,RTX4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=04:00:00
set -euo pipefail
TASK_ROOT=/home/jmzhang/Workspace/hallucination-20261030
PYTHON="$TASK_ROOT/legacy/.venv/bin/python"
export PYTHON OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 NLTK_DATA="$TASK_ROOT/data/nltk_data"
unset PYTHONPATH
srun "$PYTHON" "$TASK_ROOT/legacy/experiment/bind_allocated_gpu.py" --audit "$TASK_ROOT/logs/corrected-gpu-${SLURM_JOB_ID}.json" --expected-count 1 --min-free-mib 22000 -- bash "$TASK_ROOT/corrected_task.sh" "$1"
