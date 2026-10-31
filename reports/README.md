# 运行记录

当前采用修正后的 LLaVA 词首答案 token（3869/1939）。所有旧 token 结果单独放在 legacy_token_scoring/，不用于证明当前方法提升。

**补充核验：Lumia 的 CHAIRs、CHAIRi 均退化，当前 CHAIR 流程未稳定优于 Base；见 [完整补充结果](https://github.com/Rosy0912/hallucination/tree/main/reports/supplementary/lumia-recovered)。**

| Benchmark | 已核验范围 | Base | AOD | 状态 |
|---|---|---:|---:|---|
| POPE | 600 题，F1 | 84.0764 | 92.5990 | fresh features → train → 实图测试完成 |
| HallusionBench | 196 题，Accuracy | 44.3878 | 58.1633 | fresh features → train → 实图测试完成 |
| AMBER | 200 图 / 2,836 题，No-positive F1 | 85.4705 | 87.2829 | 独立包完整流程成功；本地重算一致 |
| CHAIR | 50 图，CHAIRi（越低越好） | 14.3959 | 13.7203 | CHAIRs 56→56；点估计改善，区间包含 0 |

HallusionBench 的 F1 同时从 55.1440 降至 48.7500，不能把准确率提升描述为所有指标提升。

corrected/ 中保存逐题预测、训练/验证选择和执行记录。uncertainty.json 用图片/问题组为抽样单位做配对 bootstrap，避免把同一图片的多个问题当成完全独立样本。这些是固定小规模划分实验，不是官方全量榜单结果。

AMBER/CHAIR 的标量探针是明确的训练改版。验证 AUROC 或验证 benchmark 提升不替代最终测试；最终配置必须由验证集冻结，然后只按冻结配置核验测试数据。

AMBER 总体提升，但 Existence 子项下降；CHAIRi 点估计改善、CHAIRs 持平，且区间包含 0。完整分项与质量指标见 [AMBER 记录](corrected/amber/README.md)、[CHAIR 记录](corrected/chair/README.md)。

## Lumia 恢复连接后的补充核验

此前提交的 Lumia 最终作业现已取回并独立重算。相同测试样本上，AMBER F1 为 85.4113 → 86.8885；CHAIRs 为 56 → 58，CHAIRi 为 13.6598 → 14.0306，CHAIR 发生退化。Lumia 的 CHAIR 验证选中 original5/alpha=2/beta=0.5，PJLab 选中 scalar/alpha=1/beta=0.5：训练及验证选择均独立进行，不是仅改变 GPU 的受控比较。当前 CHAIR 流程尚未稳定优于 Base。既有 PJLab 记录保持原样，补充失败结果同样公开，见 [Lumia 核验记录](https://github.com/Rosy0912/hallucination/tree/main/reports/supplementary/lumia-recovered)。
