# HallusionBench

LLaVA-1.5-7B，951 个实际带图片样本，582/173/196 题固定划分；同一个图片编辑组不跨划分。
本次对 196 题重新图片推理，验证 Base、隐藏状态、AOD 预测与原冻结记录一致。

```bash
python hallusionbench/run.py evaluate --out runs/hallusionbench
python hallusionbench/run.py train --out runs/hallusionbench-retrain
python hallusionbench/run.py extract --out runs/hallusionbench-new-features
python hallusionbench/evaluate.py reports/hallusionbench/predictions.jsonl
```

原选中模型为 L12 的 seed 42/43/44 分类头集成。训练命令保留原验证集选择流程，输出到新目录；
原权重保留在 `legacy/results/llava_hb_imageonly/aod_gt_balanced_v13`。
主指标为二元 Accuracy；报告同时保存原分组指标。它不是包含无图片问题的官方全量成绩，
也不调用 GPT judge。
