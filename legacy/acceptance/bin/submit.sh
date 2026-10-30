#!/usr/bin/env bash
set -euo pipefail
AOD_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
export AOD_PROJECT_ROOT
cd "$AOD_PROJECT_ROOT"
mkdir -p acceptance/logs acceptance/runs
case "${1:-}" in
  pope) mode="${2:-full}"; run="acceptance/runs/pope-${mode}-$(date +%Y%m%d-%H%M%S)-$$"; sbatch --chdir="$AOD_PROJECT_ROOT" --output="$AOD_PROJECT_ROOT/acceptance/logs/%j.log" --exclude=node4 --job-name=aod-accept-pope acceptance/bin/job.sbatch acceptance/bin/evaluate.py pope --mode "$mode" --out "$run" ;;
  train-pope) run="acceptance/runs/train-pope-$(date +%Y%m%d-%H%M%S)-$$"; sbatch --chdir="$AOD_PROJECT_ROOT" --output="$AOD_PROJECT_ROOT/acceptance/logs/%j.log" --exclude=node4 --job-name=aod-retrain-pope acceptance/bin/job.sbatch acceptance/bin/retrain.py pope --out "$run" ;;
  internvl-pope) model="${1%-pope}"; run="acceptance/runs/${1}-$(date +%Y%m%d-%H%M%S)-$$"; sbatch --chdir="$AOD_PROJECT_ROOT" --output="$AOD_PROJECT_ROOT/acceptance/logs/%j.log" --exclude=node4 --job-name="aod-${1}" acceptance/bin/job.sbatch acceptance/bin/verify_aod.py --config "acceptance/configs/${model}_pope.json" --out "$run" ;;
  llava-hb) run="acceptance/runs/llava-hb-$(date +%Y%m%d-%H%M%S)-$$"; sbatch --chdir="$AOD_PROJECT_ROOT" --output="$AOD_PROJECT_ROOT/acceptance/logs/%j.log" --exclude=node4 --job-name=aod-llava-hb acceptance/bin/job.sbatch experiment/verify_hb_aod_gt.py --source results/llava_hb_imageonly --experiment results/llava_hb_imageonly/aod_gt_balanced_v13 --model llava --out "$run" ;;
  train-llava-hb) run="acceptance/runs/train-llava-hb-$(date +%Y%m%d-%H%M%S)-$$"; sbatch --chdir="$AOD_PROJECT_ROOT" --output="$AOD_PROJECT_ROOT/acceptance/logs/%j.log" --exclude=node4 --job-name=aod-train-llava-hb acceptance/bin/job.sbatch experiment/hb_aod_gt.py --source results/llava_hb_imageonly --out "$run" --layers 1,4,8,12,16,20,24,28,31,32 --normalization standard --width 64 --balanced --f1-guard ;;
  hb|train-hb) run="acceptance/runs/internvl-${1}-$(date +%Y%m%d-%H%M%S)-$$"; extra=(); if [[ "$1" == train-hb ]]; then extra=(--retrain); fi; sbatch --chdir="$AOD_PROJECT_ROOT" --output="$AOD_PROJECT_ROOT/acceptance/logs/%j.log" --exclude=node4 --job-name=aod-accept-iv3-hb acceptance/bin/job.sbatch experiment/internvl_verify.py hb ${extra[@]+"${extra[@]}"} --out "$run" ;;
  *) echo 'Usage: bash acceptance/bin/submit.sh pope [full|cached] | internvl-pope | hb | llava-hb | train-pope | train-hb | train-llava-hb'; exit 2 ;;
esac
echo "PROJECT_ROOT=$AOD_PROJECT_ROOT"
echo "WORKSPACE=$AOD_PROJECT_ROOT/acceptance/AOD验收.code-workspace"
echo "RESULT_DIR=$AOD_PROJECT_ROOT/$run"
echo "提交成功不等于已经运行。PENDING 时结果目录和日志可能尚未生成；请用 squeue -j 作业号 查看状态。"
echo "终端 cd 不会切换 VS Code 左侧工作区，请打开上述 WORKSPACE。"
