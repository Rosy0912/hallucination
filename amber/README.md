# AMBER：独立下载，从公开数据到训练和评测

这个目录自带完整源码依赖，不需要父目录、作者服务器、旧特征缓存或 AOD 权重。

## 最短运行方式

Linux、Python 3.11、NVIDIA CUDA GPU（24GB 显存）、建议 32GB 以上内存和 35GB 可用磁盘。

```bash
curl -fL https://raw.githubusercontent.com/Rosy0912/hallucination/main/packages/amber.zip -o amber.zip
unzip amber.zip
cd amber
bash setup.sh
bash run.sh all
```

也可 clone 仓库后进入 amber/ 执行最后两条命令。all 完成公开资源下载、特征提取、AOD 训练、Base/AOD 推理、评分。脚本不会申请 GPU，集群用户先获得 Slurm 等资源分配。

## 实验协议

仅 discriminative。默认 seed=42 选择 200 图提取池：其中 128 图训练、32 图验证，原来的 40 图不用于选参或最终确认。另从池外抽取 200 图作最终测试。比较原始 MLP 探针的两种训练配置及标量一致性探针变体，在验证集选择 L24 干预参数；冻结后才测试。标量变体以训练集均值/全局 RMS 处理特征，使用 GRL 权重 0.1，并按验证 AUROC 选择 100 epochs 内的 checkpoint；这是明确的训练改版，不声称完全复现论文默认参数。主指标为官方 No-positive F1，附准确率与各类结果。二选一推理使用 Yes/No 条件分数，不采用全词表 APC。

--images 控制提取池（5..800，默认 200）；--test-images 控制池外最终测试图片数（默认 200，0 表示所有剩余图片）。例如：bash run.sh all --test-images 0 --run-dir results/all-holdout。

结果：`results/default/experiment/metrics.json`。相邻目录保留划分、训练权重、选参记录和逐题预测。选参只用验证集，测试不参与训练或参数排名。若没有候选通过验证门槛，程序保存 selection.json 并报告未通过；不会制造正向结果或反复查看测试集。

所有二分类分数使用经实际生成核验的词首答案 token 3869/1939。旧包装脚本的裸 Yes/No token 8241/3782 不适用于固定 tokenizer，不能使用旧缓存冒充修正版数据。

## 分步执行

```bash
bash run.sh prepare
bash run.sh extract
bash run.sh train
bash run.sh evaluate
# 独立保留另一轮实验
bash run.sh all --run-dir results/run2
```

train 包括验证选参及冻结后的测试推理；evaluate 独立重算指标。失败留下未完成目录时，用新的 run-dir；不要把未完成输出视为成功。

## 下载与已有资源

首次自动下载固定 revision 的 LLaVA-1.5-7B。可传 `--model-dir /path/to/llava-1.5-7b-hf` 复用公开底座。
AMBER 官方数据可用 --amber-root /path/to/AMBER 复用，目录含 data/ 和 image/。
这些选项都是可选的，不需要任何旧训练结果。

若计算节点无法联网，先在联网机器执行 prepare，再复制整个目录连同 data/、models/。
Hugging Face 可按网络情况设置 HF_ENDPOINT，例如 `HF_ENDPOINT=https://hf-mirror.com bash run.sh prepare`。AMBER 官方 Google Drive 可能受网络或配额限制。

已有验证环境可运行：`PYTHON=/path/to/python bash run.sh all`。setup.sh 使用 [PyTorch 官方 CUDA 12.8 源](https://pytorch.org/get-started/previous-versions/#v2100)安装 2.10.0，然后安装锁定依赖；实际计算环境为 PyTorch 2.10.0+cu128、Transformers 4.57.0。模型 revision：b234b804b114d9e37bb655e11cbbb5f5e971b7a9。

本包是透明、可执行的子集实验，不是官方全量榜单复现。跨 GPU 的数值和验证选择可能不同，指标以实际输出为准。根仓库 reports/ 保留运行证据，但本目录运行不依赖 reports/。

模型大文件连接超时时，可使用经过逐文件哈希核验的公开镜像：`AOD_MODEL_SOURCE=modelscope bash run.sh prepare`。镜像只提供下载通道；所有文件必须匹配本包固定 Hugging Face revision 的 Git/LFS 哈希，内容变化会报错。

## 已核验的固定划分结果

单张 A100 80GB、Python 3.10.20、固定依赖环境中，从本包完整入口重新提取、训练、选参并评测成功。200 张独立测试图片、2,836 题：准确率 80.1834 → 83.9915，官方 No-positive F1 85.4705 → 87.2829。验证选中 scalar、alpha=-2、beta=0.5。总体提高，但 Existence 子项 F1 95.4715 → 94.4026，有所下降。所有候选、逐题输出、独立重算与图片分组 bootstrap 见 [公开运行记录](https://github.com/Rosy0912/hallucination/tree/main/reports/corrected/amber)。这是本次确认结果，不要求跨硬件逐位相同。

## Lumia 恢复连接后的补充核验

此前提交的 Lumia 最终作业现已取回并独立重算。相同测试样本上，AMBER F1 为 85.4113 → 86.8885；CHAIRs 为 56 → 58，CHAIRi 为 13.6598 → 14.0306，CHAIR 发生退化。Lumia 的 CHAIR 验证选中 original5/alpha=2/beta=0.5，PJLab 选中 scalar/alpha=1/beta=0.5：训练及验证选择均独立进行，不是仅改变 GPU 的受控比较。当前 CHAIR 流程尚未稳定优于 Base。既有 PJLab 记录保持原样，补充失败结果同样公开，见 [Lumia 核验记录](https://github.com/Rosy0912/hallucination/tree/main/reports/supplementary/lumia-recovered)。
