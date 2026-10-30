#!/usr/bin/env bash
set -euo pipefail
R=/home/jmzhang/Workspace/hallucination-20261030
B="$1"
cd "$R/standalone-corrected/$B"
case "$B" in
pope) bash run.sh all --model-dir "$R/legacy/models/llava-1.5-7b-hf" --coco-root "$R/legacy/data/images";;
hallusionbench) bash run.sh all --model-dir "$R/legacy/models/llava-1.5-7b-hf" --hb-root "$R/legacy/data/images";;
amber)
 bash run.sh prepare --model-dir "$R/legacy/models/llava-1.5-7b-hf" --amber-root "$R/data/AMBER"
 "$PYTHON" -u run.py --data-root data/AMBER/data --image-root data/AMBER/image --model models/llava-1.5-7b-hf --out results/corrected/features --images 200 --extract-only
 "$PYTHON" -u "$R/optimize_amber.py" --package "$PWD" --cache results/corrected/features/features.pt --model models/llava-1.5-7b-hf --image-root data/AMBER/image --data-root data/AMBER/data --out results/corrected/optimization
 ;;
chair)
 bash run.sh prepare --model-dir "$R/legacy/models/llava-1.5-7b-hf" --coco-root "$R/legacy/data/images" --annotations-root "$R/data/coco/annotations"
 bash run.sh extract
 "$PYTHON" -u "$R/optimize_chair.py" --package "$PWD" --cache results/default/features/features.pt --old-run "$R/runs/chair-50-s42" --model models/llava-1.5-7b-hf --data-root data --annotations data/coco/annotations --out results/corrected/optimization
 ;;
esac
