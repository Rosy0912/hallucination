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

## 本次实测

**补充核验：Lumia 的 CHAIRs、CHAIRi 均退化，当前 CHAIR 流程未稳定优于 Base；见 [完整补充结果](https://github.com/Rosy0912/hallucination/tree/main/reports/supplementary/lumia-recovered)。**



| Benchmark / 指标 | Base | AOD 变体 |
|---|---:|---:|
| POPE F1 ↑（600 题） | 84.0764 | 92.5990 |
| HallusionBench Accuracy ↑（196 题） | 44.3878 | 58.1633 |
| AMBER No-positive F1 ↑（200 图 / 2,836 题） | 85.4705 | 87.2829 |
| CHAIRi ↓（50 图） | 14.3959 | 13.7203 |

CHAIRs 为 56.00 → 56.00，且 CHAIR 小样本置信区间包含 0；HallusionBench F1 与 AMBER Existence 子项下降。指标定义、全部分项、区间和运行证据见 [reports/](reports/)，原因分析见 [修正记录](reports/DIAGNOSIS.md)。这是固定划分的实测结果，不是四项所有指标均提升的结论。

## 协议与范围

| Benchmark | 划分与方法 |
|---|---|
| POPE | COCO popular 3000 题，1800/600/600；L13 GT 监督 projective 分类头，验证集选 epoch/阈值 |
| HallusionBench | 有图片的 951 题，582/173/196；GT 监督 AOD 分类头，按分组划分，验证集选层与参数 |
| AMBER | 判别任务；默认 200 图提取池，128 图训练、32 图验证；另取未参与旧实验的 200 图测试 |
| CHAIR | POPE 1800 题训练一致性方向；16 张 validation 图片选参数；50 张未参与旧实验的 test 图片评测，greedy 128 tokens |

前两项继承验收包的 GT 分类头；后两项是 AOD 一致性方向及内部干预/对比解码，**不是完全相同的算法变体**。这里提供透明的训练、选参、评测流程，不称为四项官方全量榜单复现。

AMBER 以官方 No-positive F1 为主，同时列准确率及分项。CHAIR 使用原始算法的 Python 3 适配，合并实例标注与参考描述，按提及次数计算 CHAIRi。CHAIR 越低越好，附描述长度和正确物体覆盖率。

修正版结果、逐题输出、划分、选参记录和 任务完成状态保存在 [reports/](reports/)。

## 重要修正与可复现性

2026-10-30 检查发现旧包装脚本读取了不带词首空格的 Yes/No token（8241/3782），而固定 LLaVA tokenizer 实际生成词首 token（3869/1939）。四个独立包现使用经过现场生成核验的 token，并重新生成训练标签。旧成绩仅保留在 [历史协议记录](reports/legacy_token_scoring/)，不能拿错误的 Base 分数证明涨点。

模型固定为 `llava-hf/llava-1.5-7b-hf@b234b804b114d9e37bb655e11cbbb5f5e971b7a9`。验证环境为 Python 3.11、PyTorch 2.10.0+cu128、Transformers 4.57.0。GPU 数值差异会影响训练和验证集选出的配置；脚本保留实际结果，不以历史分数作为通过条件。

legacy/ 仅保存历史来源和旧协议。新读者从四个独立目录开始。来源、许可证和改动见 [THIRD_PARTY.md](THIRD_PARTY.md)。

模型大文件连接超时时，可使用经过逐文件哈希核验的公开镜像：`AOD_MODEL_SOURCE=modelscope bash run.sh prepare`。镜像只提供下载通道；所有文件必须匹配本包固定 Hugging Face revision 的 Git/LFS 哈希，内容变化会报错。

## Lumia 恢复连接后的补充核验

此前提交的 Lumia 最终作业现已取回并独立重算。相同测试样本上，AMBER F1 为 85.4113 → 86.8885；CHAIRs 为 56 → 58，CHAIRi 为 13.6598 → 14.0306，CHAIR 发生退化。Lumia 的 CHAIR 验证选中 original5/alpha=2/beta=0.5，PJLab 选中 scalar/alpha=1/beta=0.5：训练及验证选择均独立进行，不是仅改变 GPU 的受控比较。当前 CHAIR 流程尚未稳定优于 Base。既有 PJLab 记录保持原样，补充失败结果同样公开，见 [Lumia 核验记录](https://github.com/Rosy0912/hallucination/tree/main/reports/supplementary/lumia-recovered)。
