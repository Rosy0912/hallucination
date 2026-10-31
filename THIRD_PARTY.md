# 来源与修改

## 用户提供的 AOD 代码

- `legacy/experiment`、`legacy/upstream`、`legacy/acceptance`来自
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

`legacy` 保留历史来源代码。发布内容不包含历史实验成绩；运行产生的完整输出保存在使用者本地。
参数选择使用验证集，最终测试不参与参数排名。

## 本次诊断与训练改版

- 固定 LLaVA tokenizer 的裸 Yes/No token 不等于生成时词首 token；四个独立入口改用已核验的 3869/1939，并重新提取正确性标签。
- tools/scalar_direction.py 实现一个明确的 AOD 训练变体：单一方向与 GRL 残差对抗保持不变，投影分支改为标量 MLP；训练集均值/全局 RMS 标准化、类别平衡损失、GRL 权重 0.1，验证 AUROC 选 epoch。其后仍用验证集实际推理选干预参数。
- HallusionBench 公共镜像固定 revision cd417161857aefb23d878d42cf1bb53aa9dd646f，并校验字节 SHA256 和全部问题/标签。
- AMBER 标注固定官方仓库 commit 534babf6bbfcce2e735c26289dedfb21cef3c939，并校验两个 JSON 文件的 SHA256。
- CHAIR 默认增加描述领域 AOD：由训练图片的 Base 描述采集物体首个子词之前的 L24 状态，以训练图标注监督一致性；16 张验证图选参后，用另外 84 张验证图确认，再进行冻结测试。投影对比解码和训练标准差定标的加法干预作为固定候选，均保留内容覆盖门槛。此前问答领域训练入口通过 `--aod-variant question` 保留。
