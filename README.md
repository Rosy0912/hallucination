# LLaVA-1.5-7B：四个独立可运行的幻觉 benchmark

每个目录都包含完整代码和依赖。**只下载一个 benchmark，就能准备公开数据、提取特征、训练 AOD、比较 Base/AOD 并重算指标**，不需要作者服务器、父目录源码或已有特征。

| Benchmark | 独立源码 | 独立压缩包 |
|---|---|---|
| POPE | [pope/](pope/) | [pope.zip](packages/pope.zip) |
| HallusionBench | [hallusionbench/](hallusionbench/) | [hallusionbench.zip](packages/hallusionbench.zip) |
| AMBER | [amber/](amber/) | [amber.zip](packages/amber.zip) |
| CHAIR | [chair/](chair/) | [chair.zip](packages/chair.zip) |

## 运行方式

Linux、Python 3.11、已分配的 NVIDIA CUDA GPU（24GB 显存）、建议每个独立包预留 35GB 磁盘。以 POPE 为例：

```bash
curl -fL https://raw.githubusercontent.com/Rosy0912/hallucination/main/packages/pope.zip -o pope.zip
unzip pope.zip
cd pope
bash setup.sh
bash run.sh all
```

把 pope 换成另一个 benchmark 即可。也可以 clone 本仓库，进入相应子目录后执行最后两条命令。数据和底座权重首次自动下载。联网机器可先执行 `bash run.sh prepare`，再将整个目录连同 data/、models/ 搬到计算节点。

Hugging Face 网络不可用时，可自行选择可访问的端点，例如 `HF_ENDPOINT=https://hf-mirror.com bash run.sh prepare`；模型固定 revision，HallusionBench 镜像还校验 SHA256 和全部问题/标签。AMBER 图片由官方 Google Drive 发布，受其访问与下载配额限制。

## 协议与范围

| Benchmark | 划分与方法 |
|---|---|
| POPE | COCO popular 3000 题，1800/600/600；L13 GT 监督 projective 分类头，验证集选 epoch/阈值 |
| HallusionBench | 有图片的 951 题，582/173/196；GT 监督 AOD 分类头，按分组划分，验证集选层与参数 |
| AMBER | 判别任务；默认 200 图提取池，128 图训练、32 图验证；另取未参与旧实验的 200 图测试 |
| CHAIR | 60 张训练图的生成描述训练一致性方向；16 张验证图选参、另外 84 张验证图独立确认；排除原 POPE 图片集合后的 200 张图作冻结测试，greedy 128 tokens |

前两项继承验收包的 GT 分类头；后两项是 AOD 一致性方向及内部干预/对比解码，**不是完全相同的算法变体**。这里提供透明的训练、选参、评测流程，不称为四项官方全量榜单复现。

AMBER 以官方 No-positive F1 为主，同时列准确率及分项。CHAIR 使用原始算法的 Python 3 适配，合并实例标注与参考描述，按提及次数计算 CHAIRi。CHAIR 越低越好，附描述长度和正确物体覆盖率。

CHAIR 另外提供可选的图像物体核验与描述重写组合入口，完整训练及运行方式见 [CHAIR README](chair/README.md#可选aod-加物体核验与重写)。该入口与默认 AOD 分开保留。

本仓库发布可执行代码、数据准备方式、评测协议及已验证的主要指标摘录。运行后在本地 results/ 中生成逐题预测、完整指标及选参记录。

## 已验证的主要指标摘录（2026-10-31）

模型均为 LLaVA-1.5-7B。下表只列已验证的主要指标，不是完整指标表，不表示所有指标或子项均提升；四项采用的方法变体及固定子集不同，具体协议见对应目录。

| Benchmark | 测试范围 | 指标 | Base → 改进方案 |
|---|---|---|---:|
| [POPE](pope/README.md) | 600 题 | F1 ↑ | 84.08 → **92.60** |
| [HallusionBench](hallusionbench/README.md) | 196 题 | Accuracy ↑ | 44.39 → **58.16** |
| [AMBER](amber/README.md) | 200 图 / 2836 题 | No-positive F1 ↑ | 85.47 → **87.28** |
| [CHAIR](chair/README.md) | 400 图 | CHAIRs / CHAIRi ↓ | 52.00 / 15.32 → **36.75 / 10.49** |

CHAIR 行对应可选的 `--aod-variant verified --test-images 400`：AOD＋额外监督的物体核验＋重写，并非默认纯描述方向入口；同批图片上的未加核验 AOD 对照见 CHAIR README。以上为冻结测试结果，不是四项官方全量榜单，也不保证其他硬件或配置取得相同数值。

## 重要修正与可复现性

2026-10-30 检查发现旧包装脚本读取了不带词首空格的 Yes/No token（8241/3782），而固定 LLaVA tokenizer 实际生成词首 token（3869/1939）。四个独立包现使用经过现场生成核验的 token，并重新生成训练标签。运行新代码时需重新提取特征，不能复用旧 token 协议的缓存。

模型固定为 `llava-hf/llava-1.5-7b-hf@b234b804b114d9e37bb655e11cbbb5f5e971b7a9`。验证环境为 Python 3.11、PyTorch 2.10.0+cu128、Transformers 4.57.0。GPU 数值差异会影响训练和验证集选出的配置；脚本保留实际结果，不以历史分数作为通过条件。

legacy/ 仅保存历史来源和旧协议。新读者从四个独立目录开始。来源、许可证和改动见 [THIRD_PARTY.md](THIRD_PARTY.md)。

模型大文件连接超时时，可使用经过逐文件哈希核验的公开镜像：`AOD_MODEL_SOURCE=modelscope bash run.sh prepare`。镜像只提供下载通道；所有文件必须匹配本包固定 Hugging Face revision 的 Git/LFS 哈希，内容变化会报错。
