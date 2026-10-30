# CHAIR（COCO 图片描述）

本轮从原 POPE 测试图片池固定抽取 50 张（seed=42），与训练方向的 POPE train 图片不重叠。
同图分别生成 Base/AOD 描述，最多 128 个新 token，greedy decoding。

```bash
python chair/run.py --pope-features legacy/results/llava_separate/pope_popular/features.pt \
  --data-root legacy/data --model legacy/models/llava-1.5-7b-hf --out runs/chair --images 50
export NLTK_DATA="$PWD/data/nltk_data"
python chair/evaluate.py runs/chair/predictions.jsonl \
  --annotations data/coco/annotations --output runs/chair/metrics.json
```

方向用 POPE **训练集的一致性标签**重新训练，L24、5 epochs；不复用 POPE GT 分类头。
干预使用 AOD-main 的 CD 与 APC，无 KV cache，忠实保留“仅当前最后 token 干预”的原实现语义。

评分使用原始 CHAIR 的分词/复合词/同义词规则，以实例分割与参考描述的对象并集为真值。
CHAIRi 按对象**提及次数**统计，不按去重类别数统计；CHAIRs 为包含至少一次幻觉的描述比例。
报告单位为百分数，越低越好。如果没有任何对象提及，CHAIRi 为 null，避免伪造 0 分。
逐图命中的对象、幻觉对象和原描述都会保留。这是 50 图流程验证，不代表 COCO 全量结果。
