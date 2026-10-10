# HorizonKV / HERMES 实验代码

本仓库保留原始 HERMES、DP 逐头预算与 recency alpha/k、v8、timestamp-v8、current-ourtop 和 HorizonKV，并提供统一实验入口与冻结配置。下面从克隆代码开始介绍环境配置和运行方法。原论文、作者信息及其他模型/数据集入口见 [UPSTREAM_README.md](UPSTREAM_README.md)，原始 [MIT License](LICENSE) 保留。

## 1. 运行条件与克隆代码

参考配置为 **Linux + NVIDIA A100 40GB + Qwen2.5-VL-7B-Instruct**，使用 Python 3.12、PyTorch 2.9.1、torchvision 0.24.1、CUDA 13.0、Transformers 4.57.1 和 FlashAttention 2.8.3。其他显卡的显存需求取决于视频长度、分辨率和预算；本次没有验证其他显卡或 LLaVA 环境。

先准备 Git、Conda/Miniconda、NVIDIA 驱动和 C++ 编译器。下面的 cu130 源码构建流程需要 **CUDA Toolkit 13.0 的 nvcc**；驱动通常需要 R580 或更新版本，详见 [NVIDIA 兼容要求](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)。PyTorch 的 pip 安装包不会提供 nvcc。

```bash
git clone https://github.com/STARRY-2026/HorizonKV.git
cd HorizonKV

# 检查显卡、驱动和本机 CUDA 编译器
nvidia-smi
nvcc --version
```

后面的命令都在仓库根目录执行。若没有 nvcc，请先配置对应的 CUDA Toolkit；`nvidia-smi` 显示的 CUDA 上限不等于本机 Toolkit 版本。

## 2. 创建环境并安装依赖

```bash
conda create -n hermes-release python=3.12 -y
conda activate hermes-release

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-runtime-qwen.txt
python -m pip install packaging ninja psutil

# 以本机 PyTorch / CUDA 编译 FlashAttention
FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=4 \
  python -m pip install flash-attn==2.8.3 --no-build-isolation --no-cache-dir

python -m pip check
```

FlashAttention 编译可能耗时较长；主机内存不足时将 `MAX_JOBS=4` 改为 `MAX_JOBS=1`。这里显式要求源码构建：依据 [FlashAttention 2.8.3 的安装脚本](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/setup.py)，其预编译 wheel 推断会把 CUDA 13 归入 cu12，需要避免自动选用另一 CUDA 家族的 wheel。安装命令依据现有实测环境和官方源码整理，尚未完成全新环境重装验证。

使用 `requirements-runtime-qwen.txt`；旧 `requirements_qwen.txt` / `requirements_llava.txt` 作为历史资料保留。其他 CUDA wheel 选择、ABI 和独立 DSCache 环境见 [RUNTIME_SETUP.md](RUNTIME_SETUP.md)。

## 3. 准备模型与视频

### 下载 Qwen2.5-VL-7B 模型

```bash
hf download Qwen/Qwen2.5-VL-7B-Instruct \
  --local-dir models/Qwen2.5-VL-7B-Instruct
```

`hf` 命令由已固定的 `huggingface-hub` 提供。模型目录应包含配置、processor/tokenizer 文件及完整权重分片。也可以复用已有模型，在下一步修改 `MODEL_DIR`。模型来源：[官方 Qwen checkpoint](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)。

### 下载 StreamingBench 视频

仓库已包含标注和小型策略表，**模型权重及视频需要单独下载**。本项目使用 [StreamingBench 官方数据](https://huggingface.co/datasets/mjuicem/StreamingBench) 的实时视觉理解视频。已有视频时可直接跳过下载并修改 `VIDEO_DIR`；否则可以下载对应分包并解压：

```bash
hf download mjuicem/StreamingBench --repo-type dataset \
  --include 'Real-Time Visual Understanding_*.zip' \
  --local-dir datasets/StreamingBench

mkdir -p datasets/StreamingBench/videos
for archive in datasets/StreamingBench/Real-Time\ Visual\ Understanding_*.zip; do
  python -m zipfile -e "$archive" datasets/StreamingBench/videos
done
```

这会下载全部实时视觉理解视频分包，压缩包合计约 132 GB，解压另需空间。保留视频原文件名；视频可以位于 `VIDEO_DIR` 下的子目录中。

## 4. 设置本机路径并检查环境

```bash
# 如果模型/视频放在其他位置，修改这两个路径
export MODEL_DIR="$PWD/models/Qwen2.5-VL-7B-Instruct"
export VIDEO_DIR="$PWD/datasets/StreamingBench/videos"

# 将 half 标注中的旧机器路径改为本机视频路径，保留问题与顺序
python tools/relocate_annotation.py \
  --input data/streamingbench/streamingbench_realtime_half.json \
  --video-root "$VIDEO_DIR" \
  --output data/streamingbench/streamingbench_portable.json
export ANNO="$PWD/data/streamingbench/streamingbench_portable.json"

# 检查 CUDA、依赖、FlashAttention、DP/v8/HorizonKV 策略表、模型和视频
python tools/check_environment.py --method all \
  --model-path "$MODEL_DIR" --anno-path "$ANNO" --check-video-decode
```

重定位工具递归按文件名查找视频，缺失或有多个同名文件时会报错；输出标注已存在时也会拒绝覆盖，重新生成请换输出文件名。环境检查全部 `PASS` 后再进行 GPU 推理。这里只检查资源和导入，下面的真实运行才会执行模型与压缩内核。

只想检查代码包里的策略表、不准备模型或 GPU时，可以单独运行：

```bash
python tools/check_environment.py --assets-only --skip-model --method all
```

## 5. 先运行一个小实验

建议先用 `--smoke`：从指定标注选择一道位于 64–120 秒的真实问题，运行视频编码、KV 压缩、回答生成及本地评测。以下四个命令分别运行 HERMES、DP、v8 和 HorizonKV；可先只运行自己需要的方法。

```bash
# 原始 HERMES，K1000
python tools/run_experiment.py \
  --method hermes --budget 1000 \
  --model-path "$MODEL_DIR" --annotation "$ANNO" --video-root "$VIDEO_DIR" \
  --device 0 --smoke --output-dir outputs/hermes-k1000-smoke

# DP 预算 + 逐头 recency alpha/k，K1000
python tools/run_experiment.py \
  --method dp --budget 1000 \
  --model-path "$MODEL_DIR" --annotation "$ANNO" --video-root "$VIDEO_DIR" \
  --device 0 --smoke --output-dir outputs/dp-k1000-smoke

# Native v8 / MemHeadKV，K1000
python tools/run_experiment.py \
  --method v8 --budget 1000 \
  --model-path "$MODEL_DIR" --annotation "$ANNO" --video-root "$VIDEO_DIR" \
  --device 0 --smoke --output-dir outputs/v8-k1000-smoke

# HorizonKV，使用冻结的 K6000 配置
python tools/run_experiment.py \
  --method horizonkv --budget 6000 \
  --model-path "$MODEL_DIR" --annotation "$ANNO" --video-root "$VIDEO_DIR" \
  --device 0 --smoke --output-dir outputs/horizonkv-k6000-smoke
```

`--device 0` 选择第 0 张物理 GPU，可按需修改。**每次运行都使用新的或空的 `--output-dir`**；重新运行时换目录名。运行器会自动处理 `--video-root` 下的视频路径，保留的旧脚本不适合作为新机器的首次运行入口。

若只想预检参数、视频路径和预算而不推理，可以先执行：

```bash
python tools/run_experiment.py \
  --method dp --budget 1000 \
  --model-path "$MODEL_DIR" --annotation "$ANNO" --video-root "$VIDEO_DIR" \
  --device 0 --smoke --dry-run --output-dir outputs/dp-k1000-preflight
```

`--dry-run` 不加载模型或使用 GPU，但会写入标注和 manifest；实际运行必须换到另一个输出目录。

## 6. 运行完整实验与选择方法

小实验通过后，将 `--smoke` 改为 `--full`，同时换输出目录。例如运行 DP 的 half 实验：

```bash
python tools/run_experiment.py \
  --method dp --budget 1000 \
  --model-path "$MODEL_DIR" --annotation "$ANNO" --video-root "$VIDEO_DIR" \
  --device 0 --full --output-dir outputs/dp-k1000-half-full
```

这里 `--full` 表示跑完**所给标注**：上面的 half 标注为 **250 个视频 / 1255 道问题**。若要运行仓库提供的全部 realtime 标注（498 个视频 / 2495 道问题），使用：

```bash
python tools/run_experiment.py \
  --method hermes --budget 1000 \
  --model-path "$MODEL_DIR" \
  --annotation data/streamingbench/streamingbench_realtime.json \
  --video-root "$VIDEO_DIR" --device 0 --full \
  --output-dir outputs/hermes-k1000-realtime-full
```

通过修改 `--method`、`--budget` 和输出目录运行其他保留方法：

| `--method` | 方法 | 已包含冻结配置支持的预算 | 分辨率 |
|---|---|---|---|
| `hermes` | 原始 dense HERMES | 任意整数 ≥64 | Native |
| `uniform` | 与 DP 匹配的 uniform ragged 对照 | 任意整数 ≥64 | Native |
| `dp` | 精确 DP 逐头容量 + recency alpha/k | 100 / 250 / 500 / 800 / 1000 | Native |
| `v8` / `memheadkv` | Native v8 逐头预算分配 | 1000 | Native |
| `timestamp-v8` | v8 预算 + timestamp eviction | 1000 | Fixed448 |
| `current-ourtop` | 重新校准的 timestamp 变体 | 1000 | Fixed448 |
| `horizonkv` | 独立 Recent + Memory HorizonKV | 6000 | Native |

DP 的匹配实现对照使用 `uniform`。比较实验请统一标注、分辨率和预算；Fixed448 对照需显式加 `--frame-canvas-size 448`。HorizonKV K6000 对比原始 HERMES 时，也应使用 HERMES K6000。详细配置、first150 划分与历史运行入口见 [EXPERIMENTS.md](EXPERIMENTS.md)。

## 7. 输出、主要结果与验证范围

每次成功运行的输出目录包含：

- `results.csv`：逐题预测。
- `eval_results.txt`：本地 StreamingBench 多选题评测结果，无需 API key。
- `inference.log`：推理和压缩日志。
- `experiment_manifest.json`：运行参数、标注、策略表哈希、预算及完成状态。

缺少输入、子进程失败、预测不完整或 smoke 未发生实际压缩时，运行器会返回失败。遇到问题先查看 `inference.log` 和 manifest。

[主要实验结果](experiment_results/README.md) 附有 **14 项已完成的历史结果**及原始逐题 CSV、评测报告和来源；汇总表为 [main_results.csv](experiment_results/main_results.csv)。未完成的 current-ourtop 输出和最新 HorizonKV 的小规模 pilot 没有冒充完整主实验。

已在现有 **A100 40GB + Qwen2.5-VL-7B** 环境验证真实视频 smoke，完成编码、实际 KV 压缩、生成和评测；85 项 CPU 测试通过。验证细节见 [RUNTIME_VALIDATION.md](RUNTIME_VALIDATION.md)。本次没有重跑全量准确率实验，也没有完成全新环境重装验证。

如需运行策略与入口测试：

```bash
python -m pip install -r requirements-runtime-analysis.txt
python -m pytest tests -q -p no:cacheprovider
```

## 代码与来源

核心目录包括 `inference/`、`video_qa/`、`head_analysis/`、`eval/`、`tools/`、`scripts/`、`experiments/`、`tests/` 和 `docs/`。运行使用冻结 profile，不会在评测数据上重新校准；重新生成部分历史 profile 需要另备未上传的大型原始观察和 trace。`dscache/` 是独立对照；可选 ProtoKV / InfiniPot-V 适配器需要外部项目，统一入口关闭这些适配器。

基于原项目 `6020490` 及其本地最新源码整理，保留策略算法；发布改动提供可移植路径、可靠失败传播、统一入口和验证工具。逐文件来源见 [SNAPSHOT_MANIFEST.csv](SNAPSHOT_MANIFEST.csv)，导出说明见 [UPLOAD_PLAN.md](UPLOAD_PLAN.md)。
