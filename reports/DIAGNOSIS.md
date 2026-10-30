# AMBER / CHAIR 修正记录

## 已确认的问题

1. 原包装脚本用裸字符串 `Yes` / `No` 的 token（8241 / 3782）取分数。固定 LLaVA tokenizer 在实际回答词首使用 3869 / 1939。现场生成检查保存在 [token-audit.log](diagnostics/token-audit.log)。四个独立包改用同一个经断言检查的答案映射，重新提取特征和一致性标签；旧缓存与旧成绩不能用于证明本次提升。
2. AMBER 原先把全词表 APC 用在 Yes/No 子空间，旧运行的 575 条记录全部进入回退。判别任务现在明确采用 Yes/No 条件分数及其对比组合；CHAIR 自由生成仍保留全词表 APC。两项协议分别记录，不混用。
3. 原一致性探针出现偏向多数类的诊断现象。新候选保留秩一方向与残差对抗分支，使用训练集均值/全局 RMS、标量一致性探针、类别平衡损失和梯度裁剪。标量探针避免对秩一投影做 LayerNorm 时丢失幅度信息的风险。这是明确的训练改版，并非原论文默认实现。


PJLab 从头重训后再次观察到探针退化（这部分是验证集的一致性分类诊断，不是最终 benchmark 成绩）：

| 数据与探针 | 验证 AUROC | 平衡准确率 | 预测为一致的比例 |
|---|---:|---:|---:|
| AMBER normalized5 | 0.7185 | 0.5000 | 100% |
| AMBER scalar | 0.8764 | 0.7683 | 53.06% |
| CHAIR 问答特征 original5 | 0.5392 | 0.5000 | 100% |
| CHAIR 问答特征 scalar | 0.8955 | 0.8606 | 68.75% |

完整数值见 [AMBER 探针诊断](diagnostics/amber-probes-pjlab.json)、[CHAIR 探针诊断](diagnostics/chair-probes-pjlab.json)。类别不平衡时，全部预测为多数类也能得到较高普通准确率，因此这里同时检查平衡准确率、AUROC 和输出分布。

## 如何检验修改

原训练变体保留为验证候选，与标量变体按相同划分比较。AMBER 用 128 张训练图片、32 张验证图片；CHAIR 用 POPE train 的 1,800 条记录训练，用 16 张 validation 图片比较生成。训练集、验证集、最终测试图片不重叠。

AMBER 在验证集选择准确率提高且官方 No-positive F1 不下降的配置。CHAIR 要求验证 CHAIRs、CHAIRi 同时降低，正确唯一对象数量至少保留 Base 的 90%，平均生成 token 至少保留 Base 的 80%。这些保护条件用于验证选择，最终测试指标照实报告。

每次运行先写出 `search_plan.json`，然后保存所有验证候选及 `selection.json`，最后才执行冻结配置的测试推理。没有合格候选时停止该次最终测试，保留失败记录。验证 AUROC、验证 benchmark 提升均不能替代最终测试结果。

## 复现与执行记录

四项修正结果均已核验；AMBER、CHAIR 在 PJLab 从独立包重新提取、训练和评测成功。AMBER 总体准确率和 F1 均提高；CHAIRi 点估计改善，CHAIRs 持平，CHAIR 区间仍包含 0。本次训练变体同时改变了归一化、探针和损失权重，不能把全部增益归因于某一个改动。最终成绩以 [运行汇总](README.md) 和 `corrected/` 中的逐题预测及执行记录为准。

模型 revision 固定为 `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`。备用下载通道仅在模型文件逐一匹配该 revision 的 Git/LFS 哈希时接受；不会为了运行通畅更换底座权重。
