# POPE：独立下载，从公开数据到训练和评测

这个目录自带完整源码依赖，不需要父目录、作者服务器、旧特征缓存或 AOD 权重。

## 最短运行方式

Linux、Python 3.11、NVIDIA CUDA GPU（24GB 显存）、建议 32GB 以上内存和 35GB 可用磁盘。

```bash
curl -fL https://raw.githubusercontent.com/Rosy0912/hallucination/main/packages/pope.zip -o pope.zip
unzip pope.zip
cd pope
bash setup.sh
bash run.sh all
```

也可 clone 仓库后进入 pope/ 执行最后两条命令。all 完成公开资源下载、特征提取、AOD 训练、Base/AOD 推理、评分。脚本不会申请 GPU，集群用户先获得 Slurm 等资源分配。

## 实验协议

COCO popular 固定 3000 题，按图片划分 1800 训练、600 验证、600 测试。提取 L13 特征；训练 GT Yes/No 监督的 projective 分类头；验证集选 epoch 和阈值；重新做 600 道真实图片推理。



结果：`results/default/evaluation/metrics.json`。相邻目录保留划分、训练权重、选参记录和逐题预测。选参只用验证集，测试不参与训练或参数排名。

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

train 训练并保存验证选择；evaluate 用真实图片复核测试结果。失败留下未完成目录时，用新的 run-dir；不要把未完成输出视为成功。

## 下载与已有资源

首次自动下载固定 revision 的 LLaVA-1.5-7B。可传 `--model-dir /path/to/llava-1.5-7b-hf` 复用公开底座。
COCO val2014 图片可用 --coco-root /path/to/val2014 复用。
这些选项都是可选的，不需要任何旧训练结果。

若计算节点无法联网，先在联网机器执行 prepare，再复制整个目录连同 data/、models/。
Hugging Face 可按网络情况设置 HF_ENDPOINT，例如 `HF_ENDPOINT=https://hf-mirror.com bash run.sh prepare`。AMBER 官方 Google Drive 可能受网络或配额限制。

已有验证环境可运行：`PYTHON=/path/to/python bash run.sh all`。setup.sh 使用 [PyTorch 官方 CUDA 12.8 源](https://pytorch.org/get-started/previous-versions/#v2100)安装 2.10.0，然后安装锁定依赖；实际计算环境为 PyTorch 2.10.0+cu128、Transformers 4.57.0。模型 revision：b234b804b114d9e37bb655e11cbbb5f5e971b7a9。

本包是透明、可执行的子集实验，不是官方全量榜单复现。跨 GPU 的数值和验证选择可能不同，指标以实际输出为准。根仓库 reports/ 保留运行证据，但本目录运行不依赖 reports/。

模型大文件连接超时时，可使用经过逐文件哈希核验的公开镜像：`AOD_MODEL_SOURCE=modelscope bash run.sh prepare`。镜像只提供下载通道；所有文件必须匹配本包固定 Hugging Face revision 的 Git/LFS 哈希，内容变化会报错。
