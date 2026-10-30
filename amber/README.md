# AMBER（判别任务）

官方共有 14,216 条判别问题。加载器兼容官方 list annotations，并映射为 existence、attribute、relation。
本轮用 seed=42 选择 200 张图，前 160 张训练、后 40 张测试。同图的所有问题留在同一划分。
标签只用于一致性方向训练和评分，模型输入只有图片与问题。

```bash
python amber/run.py --data-root data/AMBER/data --image-root data/AMBER/image \
  --model legacy/models/llava-1.5-7b-hf --out runs/amber --images 200
python amber/evaluate.py runs/amber/predictions.jsonl --output runs/amber/recomputed.json
```

运行顺序：真实图片特征提取 → 仅训练图片上的一致性 AOD（L24、5 epochs、seed=42）→
测试图片 Base 与 CD 对比（alpha=1、beta=0.5、APC=0.1）。参数预先固定，不使用测试结果选参。
当 Yes/No 都被 APC 屏蔽时，按原实现回退到 positive-pass logits，并逐题记录该事件。

`--images 0` 可以使用全部图片，但仍是内部 train/test 划分，不是零样本官方全量榜单。
输出包括完整 manifest、训练 checkpoint、逐题预测、分类型混淆矩阵及 protocol。
