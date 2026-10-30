# POPE

LLaVA-1.5-7B，MSCOCO popular，1800/600/600 题按图片隔离为 train/val/test。
本次重新读取全部 600 张逐题引用图片进行 Base 与冻结 AOD head 推理。

```bash
python pope/run.py evaluate --out runs/pope
python pope/run.py train --out runs/pope-retrain
python pope/run.py extract --out runs/pope-new-features
python pope/evaluate.py reports/pope/predictions.jsonl
```

训练入口重复原 seed=42、L13、100 epoch 的 projective recipe 并重新做图片评测。
测试 F1、FPR 直接由逐题 0/1 预测重算。AOD checkpoint、阈值、来源缓存 SHA 均在
`legacy/acceptance/configs/pope.json` 和 `legacy/results/llava_separate/pope_popular/gt_sweep`。
这是 GT 监督分类头实验，不能称为与 CHAIR 相同的对比解码。

新缓存的语义身份可以通过 `tools/validate_reextraction.py pope /path/to/features.pt` 检查；
原冻结复跑仍需要原缓存 SHA，不会自动放宽检查。
