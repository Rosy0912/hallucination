# 来源与修改

## 用户提供的 AOD 代码

- `legacy/experiment`、`legacy/upstream`、`legacy/acceptance` 与两个 LLaVA 结果目录来自
  `06_源码与运行包/AOD-evaluation-20260929`。继承包内 MIT LICENSE。
- `src/aod`、`cli`、`scripts` 来自用户提供的 `AOD-main`，继承其 MIT LICENSE。
- 修正 `src/aod/data/amber.py`：接受官方 list 格式，按 ID 连接，映射官方细分类别，拒绝空集/重复 ID。
- AMBER 评测按官方 `inference.py` 以 No 为正类；同时保存精确值和官方舍入/epsilon 值，保留 Yes-positive 辅助结果。
- 新的 benchmark 入口、训练/离线评分包装器、样本划分、检查和文档位于四个 benchmark 目录与 `tools/`。
- POPE 重训练增加显式 drift 记录和 strict 选项：跨硬件新训练可保存真实成绩，冻结权重复跑仍严格校验。
- 原提供的简化 CHAIR 评分代码保留供比对，不用于报告中的 CHAIR 指标。

## Benchmark 数据与评分

- [AMBER 官方仓库](https://github.com/junyangwang0410/AMBER)：官方 query/annotations 和官方 Google Drive 图片。
  原仓库 Apache-2.0；本仓库不重新分发图片。这里只运行 discriminative 子集。
- [CHAIR 原始实现](https://github.com/LisaAnne/Hallucination/blob/master/utils/chair.py)：
  `chair/official_algorithm.py` 保留其分词、同义词、多词消歧、标注合并规则，适配 Python 3、相对资源路径，
  去除逐标注日志及与 CHAIR 无关的 BLEU/CIDEr 等代码。评分输出包装在 `chair/evaluate.py`。
- [CHAIR 同义词](https://github.com/LisaAnne/Hallucination/blob/master/data/synonyms.txt)：`chair/synonyms.txt`。
- [Pattern English inflection](https://github.com/clips/pattern/blob/master/pattern/text/en/inflect.py)：
  `chair/pattern_singular.py` 仅提取原单数化规则、函数及依赖常量，避免安装完整 Pattern 的无关模块；
  算法不变，版权说明保留，BSD 许可证见 `chair/PATTERN_LICENSE.txt`。
- [POPE](https://github.com/RUCAIBox/POPE)、[HallusionBench](https://github.com/tianyi-lab/HallusionBench)、
  [COCO](https://cocodataset.org/)：使用用户提供的固定数据快照/分组及官方 COCO 2014 标注。

`legacy` 的历史配置/预测用于复跑一致性校验；`reports` 中标为 2026-10-30 的结果来自本次新作业。
不存在保证 AOD 优于 Base 的筛选条件；新任务的原始指标如实保留。
