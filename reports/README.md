# 运行记录

当前采用修正后的 LLaVA 词首答案 token（3869/1939）。所有旧 token 结果单独放在 legacy_token_scoring/，不用于证明当前方法提升。

| Benchmark | 已核验范围 | Base | AOD | 状态 |
|---|---|---:|---:|---|
| POPE | 600 题，F1 | 84.0764 | 92.5990 | fresh features → train → 实图测试完成 |
| HallusionBench | 196 题，Accuracy | 44.3878 | 58.1633 | fresh features → train → 实图测试完成 |
| AMBER | 200 张独立测试图片 | — | — | 最终测试尚待核验 |
| CHAIR | 50 张独立测试图片 | — | — | 最终测试尚待核验 |

HallusionBench 的 F1 同时从 55.1440 降至 48.7500，不能把准确率提升描述为所有指标提升。

corrected/ 中保存逐题预测、训练/验证选择和执行记录。uncertainty.json 用图片/问题组为抽样单位做配对 bootstrap，避免把同一图片的多个问题当成完全独立样本。这些是固定小规模划分实验，不是官方全量榜单结果。

AMBER/CHAIR 的标量探针是明确的训练改版。验证 AUROC 或验证 benchmark 提升不替代最终测试；最终配置必须由验证集冻结，然后只按冻结配置核验测试数据。
