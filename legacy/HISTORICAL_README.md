# LLaVA-1.5-7B 幻觉评测：四个可执行 benchmark

本仓库整理用户提供的两份 AOD 代码，并在 Lumia 的 Slurm GPU 上重新执行。
模型固定为 **LLaVA-1.5-7B**，比较 Base 与 AOD。四个入口分别是
[`pope/`](pope/)、[`hallusionbench/`](hallusionbench/)、[`amber/`](amber/)、[`chair/`](chair/)。
本次范围为“先跑通”，**不是四项官方全量榜单复现**。

## 实验口径

| Benchmark | 本轮范围 | AOD 实现 | 主要指标 |
|---|---|---|---|
| POPE | MSCOCO popular，固定 600 题测试集 | 原验收包 L13、GT Yes/No 监督的 projective 分类头 | F1、FPR、Accuracy |
| HallusionBench | 有图片样本，固定 196 题测试集 | 原验收包 L12、三个种子的 GT 监督分类头集成 | Accuracy、F1、组别指标 |
| AMBER | seed=42 抽取 200 张图片，160 张训练/40 张测试；仅判别任务 | 当前 AOD-main 的 L24 一致性方向训练及 CD 解码 | 总体及 existence/attribute/relation 分项 F1、Accuracy、FPR |
| CHAIR | POPE 测试图片中 seed=42 抽取 50 张，每图最多生成 128 token | POPE **训练集**的一致性方向，L24 CD 解码 | CHAIRs、CHAIRi（越低越好） |

**两份来源代码使用不同的 AOD 变体。** 前两项是保留原验收协议的分类头实验；
后两项是隐藏状态方向干预和对比解码。它们不能被描述为完全相同的干预算法。
特别是旧 POPE checkpoint 不是 CHAIR 的生成方向，CHAIR 会单独训练一个一致性方向。
原 POPE/HallusionBench 划分曾在历史研究中反复评估，本次是冻结配置复跑，不是独立盲测。

## 环境

Linux、NVIDIA CUDA GPU、Python 3.11。实际验证环境见 `requirements.txt`；GPU 调度信息和
逐题输出见 [`reports/`](reports/)。模型 revision 为
`llava-hf/llava-1.5-7b-hf@b234b804b114d9e37bb655e11cbbb5f5e971b7a9`。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader -d data/nltk_data punkt_tab
export NLTK_DATA="$PWD/data/nltk_data"
```

如果使用已有 AOD 服务器资源：

```bash
python tools/attach_resources.py \
  --source-root /path/to/AOD-repro-20260928 \
  --python-env /path/to/verified/python-environment
```

这会连接模型、图片和两个原始特征缓存，不复制大型模型或覆盖已有文件。
原服务器的已验证环境是 `AOD-repro-20260928/acceptance/releases/AOD-evaluation-20260929/.venv`。
不要把系统 `python` 与这个环境混用。

## 没有原服务器资源时

1. 下载上述固定 revision 的模型到 `legacy/models/llava-1.5-7b-hf`。
   原始提取脚本检查 `verified_manifest.json`；参考 `tools/prepare_model.py`。
2. 下载 COCO val2014 图片及 HallusionBench 官方图片，运行
   `python tools/prepare_images.py --coco-val2014 /path/to/val2014 --hallusionbench-images /path/to/hallusion_bench`。
3. 使用 `pope/run.py extract`、`hallusionbench/run.py extract` 生成新缓存。固定题目与分组在
   `legacy/resources/*_manifest.json`。新缓存存放到对应 `legacy/results/.../features.pt`。
4. 原验收程序绑定原缓存 SHA256；重提取后序列化字节可能不同，不能冒充原缓存。
   请用 `tools/validate_reextraction.py` 检查题目/预测并明确采用新训练流程；原冻结复跑需要原缓存。
5. AMBER/CHAIR 的数据下载：另装 `pip install gdown`，执行 `bash tools/download_data.sh`。
   若计算服务器无法访问 Google Drive，可在本机执行后传输 `data/AMBER`。

## 四个入口

所有命令在仓库根目录运行。GPU 命令必须在已分配 GPU 的计算节点执行，输出目录应为新目录。

```bash
python pope/run.py evaluate --out runs/pope
python hallusionbench/run.py evaluate --out runs/hallusionbench
python amber/run.py --data-root data/AMBER/data --image-root data/AMBER/image \
  --model legacy/models/llava-1.5-7b-hf --out runs/amber --images 200
python chair/run.py --pope-features legacy/results/llava_separate/pope_popular/features.pt \
  --data-root legacy/data --model legacy/models/llava-1.5-7b-hf --out runs/chair --images 50
python chair/evaluate.py runs/chair/predictions.jsonl \
  --annotations data/coco/annotations --output runs/chair/metrics.json
```

AMBER 的 `--images 0` 使用全部图片，仍按图片划分 80%/20%；不能据此称为官方全量测试。
CHAIR 的样本池为 POPE held-out COCO 图片，默认 50 张，不是官方固定榜单集合。

Lumia 提交示例（`tools/job.sbatch` 会校验实际分配的 GPU）：

```bash
mkdir -p logs
export BENCH_ROOT="$PWD"
export BENCH_PYTHON="$PWD/legacy/.venv/bin/python"
sbatch --chdir="$PWD" --output="$PWD/logs/amber-%j.log" tools/job.sbatch \
  amber/run.py --data-root data/AMBER/data --image-root data/AMBER/image \
  --model legacy/models/llava-1.5-7b-hf --out runs/amber --images 200
```

`sacct` 的 COMPLETED/0:0、预测数量及最终指标文件需同时验证。提交成功不等于实验完成。

## 重新训练与离线评测

```bash
python pope/run.py train --out runs/pope-retrain
python hallusionbench/run.py train --out runs/hallusionbench-retrain
python pope/evaluate.py reports/pope/predictions.jsonl
python hallusionbench/evaluate.py reports/hallusionbench/predictions.jsonl
python amber/evaluate.py reports/amber/predictions.jsonl
python chair/evaluate.py reports/chair/predictions.jsonl \
  --ground-truth reports/chair/ground_truth.json --output runs/chair-rescored.json
python -m unittest discover -s tests -v
```

AMBER 和 CHAIR 的 `run.py` 已包括方向训练。CHAIR 评分使用原始 CHAIR 规则的 Python 3
适配：实例标注与参考描述取并集、使用原始同义词和 Pattern 单数化、按对象提及次数计数。
`src/aod/data/chair.py` 保留原提供的简化实现供溯源，**本轮正式评分不使用它**。

模型、图片、特征缓存、凭证不进 Git。小型 AOD 权重、冻结配置、运行证据、逐题预测及重算脚本保留。
第三方来源与修改见 [`THIRD_PARTY.md`](THIRD_PARTY.md)。
