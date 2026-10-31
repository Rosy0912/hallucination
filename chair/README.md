# CHAIR：独立下载，从公开数据到训练和评测

这个目录自带完整源码依赖，不需要父目录、作者服务器、旧特征缓存或 AOD 权重。

## 最短运行方式

Linux、Python 3.11、NVIDIA CUDA GPU（24GB 显存）、建议 32GB 以上内存和 35GB 可用磁盘。

```bash
curl -fL https://raw.githubusercontent.com/Rosy0912/hallucination/main/packages/chair.zip -o chair.zip
unzip chair.zip
cd chair
bash setup.sh
bash run.sh all
```

也可 clone 仓库后进入 chair/ 执行最后两条命令。all 完成公开资源下载、特征提取、AOD 训练、Base/AOD 推理、评分。脚本不会申请 GPU，集群用户先获得 Slurm 等资源分配。

## 实验协议

提取 POPE 的 L24 特征，用训练集 1800 题的正确性标签训练方向。在 16 张 POPE validation 图片上选参数，随后在 50 张此前未用于调试的 test 图片上确认效果。Base/AOD 使用同一提示、greedy、最多 128 token。CHAIRs/CHAIRi 越低越好，同时记录描述长度、正确物体覆盖率；验证阶段要求二者降低且内容覆盖通过门槛。

--images 控制最终测试图片数（1..50）；--max-new-tokens 默认 128。减少这两个参数只适合流程检查，不可与完整报告成绩混用。评分使用原始 CHAIR 规则：实例/参考描述标注并集、原同义词与 Pattern 单数化、物体提及次数。

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
COCO val2014 图片可用 --coco-root /path/to/val2014 复用。 标注可用 --annotations-root /path/to/coco/annotations 复用。
这些选项都是可选的，不需要任何旧训练结果。

若计算节点无法联网，先在联网机器执行 prepare，再复制整个目录连同 data/、models/。
Hugging Face 可按网络情况设置 HF_ENDPOINT，例如 `HF_ENDPOINT=https://hf-mirror.com bash run.sh prepare`。AMBER 官方 Google Drive 可能受网络或配额限制。

已有验证环境可运行：`PYTHON=/path/to/python bash run.sh all`。setup.sh 使用 [PyTorch 官方 CUDA 12.8 源](https://pytorch.org/get-started/previous-versions/#v2100)安装 2.10.0，然后安装锁定依赖；实际计算环境为 PyTorch 2.10.0+cu128、Transformers 4.57.0。模型 revision：b234b804b114d9e37bb655e11cbbb5f5e971b7a9。

本包是透明、可执行的子集实验，不是官方全量榜单复现。跨 GPU 的数值和验证选择可能不同，指标以实际输出为准。本包不附预先计算的实验成绩，运行后在本地保存预测、指标及完整选参记录。

## 训练改版说明

候选包括原始初始化 5 epochs、先归一化初始化 5 epochs、类别平衡 30 epochs，以及标量一致性探针 AOD（训练集中心化/全局 RMS、GRL 0.1、最多 100 epochs，验证 AUROC 选 epoch）。标量探针避免在单一投影上用 LayerNorm 削弱幅度信息。最终仍通过真实描述生成和同一套 CHAIR/内容门槛选择；探针 AUROC 提高本身不作为 benchmark 涨点证据。这些训练改版明确记录，不当作论文默认设置复现。

模型大文件连接超时时，可使用经过逐文件哈希核验的公开镜像：`AOD_MODEL_SOURCE=modelscope bash run.sh prepare`。镜像只提供下载通道；所有文件必须匹配本包固定 Hugging Face revision 的 Git/LFS 哈希，内容变化会报错。
