# CHAIR：从公开数据到 AOD 训练与评测

本目录自带源码、依赖及数据准备脚本，不需要父目录、作者服务器、预训练 AOD 权重或已有特征。本文列出已验证的主要指标摘录；运行后的完整记录保存在本地。

## 一次运行

Linux、Python 3.11、已分配的 NVIDIA CUDA GPU（24GB 显存），建议至少 32GB 内存和 35GB 可用磁盘。

```bash
curl -fL https://raw.githubusercontent.com/Rosy0912/hallucination/main/packages/chair.zip -o chair.zip
unzip chair.zip
cd chair
bash setup.sh
bash run.sh all
```

首次自动准备固定 revision 的 LLaVA-1.5-7B、COCO 图片、实例及描述标注，然后训练、验证、冻结配置并评测。集群用户先申请 GPU 资源；脚本本身不申请算力。

## 默认训练与评测协议

默认 `--aod-variant caption` 使用描述领域的一致性方向：

1. 从固定 POPE train 图片中选 60 张，让 Base 生成描述。用这些训练图片的 COCO 实例和参考描述标注判断生成物体是否有依据，提取物体首个子词生成前的 L24 隐藏状态。
2. 训练标量一致性探针及残差对抗分支。使用训练集中心化、全局 RMS、类别平衡损失、GRL 权重 0.1，最多 100 epochs；在 16 张验证图上按 AUROC 选择 epoch。
3. 在同一批 16 张验证图上比较投影对比解码及加法干预，共 8 组固定参数。投影方式使用 α∈{0.5,1,2,-1}、β=0.5；加法方式沿训练正确性定向的向量移动，幅度为训练投影标准差乘 α。
4. 冻结一个候选，在另外 84 张验证图上独立确认。选参和确认都要求 CHAIRs、CHAIRi 同时降低，描述长度、正确物体数及覆盖率分别至少保留 Base 的 90%。未通过时保存记录，不启动最终测试。
5. 通过确认后，在固定随机种子选出的 200 张 COCO val2014 图片上比较 Base/AOD。这些图排除整个原 POPE 图片集合，不参与训练或选参。

Base/AOD 均使用 `Please describe this image in detail.`、greedy、最多 128 tokens。生成不读取待测图片的标注；标注只用于训练监督及离线评分。测试不参与候选排名。本方法是明确的 AOD 训练与干预变体，不声称原论文默认设置或官方全量榜单复现。

CHAIRs/CHAIRi 越低越好。评分保留原始 CHAIR 的同义词、多词消歧、物体提及计数及实例/参考描述标注合并规则，并额外保存生成长度和正确物体覆盖率。

## 分步运行

```bash
bash run.sh prepare
bash run.sh train
bash run.sh validate
bash run.sh evaluate

# 另起一轮，保留全部旧输出
bash run.sh all --run-dir results/run2
```

`train` 包括描述生成、特征提取及方向训练；`extract` 是该阶段的别名。`validate` 包括选参和独立确认；`evaluate` 只使用冻结后的候选做最终推理与评分。`prepare` 同时准备最终评测图片，可在联网机器上执行后将整个目录复制到离线计算节点。

默认输出在 `results/default/`：`caption-training/`、`selection.json`、`confirmation.json`、`predictions.jsonl`、`metrics.json` 及各阶段原始记录。配置、特征、checkpoint 与源文件哈希用于防止错误复用缓存。未完成的预测或改变过的配置应使用新的 `--run-dir`。

`--images 1..200` 控制最终测试图数；较小值只适合流程检查。默认协议固定 128 tokens。

## 复用公开资源

```bash
bash run.sh all --model-dir /path/to/llava-1.5-7b-hf \
  --coco-root /path/to/val2014 \
  --annotations-root /path/to/coco/annotations
```

模型 revision 固定为 `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`。已有环境可用 `PYTHON=/path/to/python bash run.sh all`。依赖见 requirements.txt；setup.sh 使用官方 CUDA 12.8 安装源安装 PyTorch 2.10.0，再安装其余依赖。

Hugging Face 访问受限时可设置 `HF_ENDPOINT`。大文件也可使用逐文件核验固定 Git/LFS 哈希的通道：`AOD_MODEL_SOURCE=modelscope bash run.sh prepare`。

## 问答领域方向入口

保留此前以 POPE 1800 题正确性训练方向的流程：

```bash
bash run.sh all --aod-variant question --run-dir results/question-direction
```

该流程在 16 张验证图选参、50 张固定测试图评测，`--images` 范围为 1..50。它与默认描述领域流程的训练目标和划分不同，应分别保存输出。二分类提取使用固定 tokenizer 的词首 Yes/No token 3869/1939，不应复用旧 token 协议的缓存。


## 可选：AOD 加物体核验与重写

这一组合变体先生成 AOD 描述，再用图像条件下的 POPE 物体判别头检查描述提到的物体。仅当同时存在判定支持和不支持的物体时，使用支持的物体列表重新生成。它包含额外监督和提示词步骤，应与纯 AOD 分开报告。

```bash
bash run.sh all --aod-variant verified --run-dir results/verified --test-images 400
```

分步运行时，对 `prepare`、`train`、`validate`、`evaluate` 都保留同一组 `--aod-variant verified --run-dir results/verified --test-images 400` 参数。首次从公开 POPE 数据提取 L13 特征并训练 projective GT 判别头，α=0；同时训练描述领域的一致性方向，不下载作者训练好的权重。默认描述方向入口保持不变。

该变体固定 AOD 解码 α=2、β=0.5、APC=0.1，在原验证集的 16 张图上比较物体判别 logit 阈值 −1、0、0.5。选定后使用 80 张新图确认；仅通过确认才运行最多 400 张冻结测试图。新图排除完整 POPE 图片池及默认描述流程预留的 200 张图；种子和逐图清单均保存在本地 plan.json。开发集和确认集都要求同时优于 Base 与未加核验的 AOD，且至少保留 Base 的 90% 描述长度、正确物体数和物体覆盖率。

推理仅使用图像、生成文本、固定物体词表和已训练判别头，不读取待测图片的标注。评分保持原 CHAIR 规则。每阶段保留原 Base、未加核验的 AOD、使用相同核验上下文的 Base 重写、组合方案四组输出，另存每个物体的判别分数和重写触发记录；共用上下文的 Base 是组件对照，不是独立训练的另一种方法。不会用测试结果重新挑阈值，也不会自动删除未提升的本地记录。

所有生成均为 greedy、最多 128 tokens。输出位于所选 run-dir 的 `caption-training/`、`verifier-features/`、`verifier-training/`、`development/`、`confirmation/`、`test/` 及冻结配置 JSON；本节后附主要指标摘录，完整记录由脚本保存在本地。

## 已验证的主要指标：物体核验组合（2026-10-31）

本节为已验证的主要指标摘录，不是完整指标表，也不表示所有指标或子项均提升。模型为 LLaVA-1.5-7B；参数由验证集选择后冻结，以下为固定测试划分上的实测结果，不是官方全量榜单。完整评测记录由运行脚本保存在本地。

本表对应 `--aod-variant verified` 的 400 张冻结测试图：排除原 POPE 的全部 500 张图、此前描述流程预留的 200 张图和本轮 80 张确认图。下列未加核验 AOD 为同一批 400 张图上的组件对照；默认 `caption` 入口使用另一套测试划分，不应直接混用数值。

| 指标（越低越好） | Base | 未加核验 AOD | AOD＋物体核验＋重写 | 组合相对 Base（百分点） |
|---|---:|---:|---:|---:|
| CHAIRs ↓ | 52.00 | 41.50 | **36.75** | -15.25 |
| CHAIRi ↓ | 15.32 | 11.69 | **10.49** | -4.83 |

组合方法包含额外的 POPE 监督物体判别头及提示词重写，不应将全部收益归为纯 AOD。400 张图按图像配对 bootstrap（2000 次，seed=42）：相对 Base 和未加核验 AOD，两项差值的 95% 区间均低于 0；这是单次冻结运行的区间，不是跨训练种子的保证。该运行通过了预先设定的内容保留门槛。

```bash
bash run.sh all --aod-variant verified --run-dir results/verified --test-images 400
```
