# Runtime setup and required experiment resources

The release includes the HERMES implementation, local research methods and their frozen small policy inputs. Model weights and benchmark videos remain separate downloads. The original `requirements_qwen.txt` and `requirements_llava.txt` are retained as historical environment exports; use the new runtime requirements below for this release.

## Reference environment

The local Qwen environment inspected on 2026-10-10 uses Linux, Python 3.12.13, PyTorch **2.9.1+cu130**, torchvision **0.24.1+cu130**, CUDA build **13.0**, transformers **4.57.1**, tokenizers **0.22.2** and FlashAttention **2.8.3**. Both dense and variable-length FlashAttention functions must import successfully. HERMES and the frozen DP, timestamp v8 and HorizonKV profiles target `Qwen2.5-VL-7B-Instruct`; those extension profiles encode **28 layers × 4 physical KV heads**, and must not be reused for a different model shape.

The separately inspected LLaVA environment has PyTorch 2.9.1+cu128, CUDA build 12.8, transformers 4.57.1 and tokenizers 0.22.2. Its old requirements file points to an earlier transformers Git revision and does not describe the current environment. No FlashAttention distribution was present there during inspection. Retaining LLaVA source and model choices does not certify that separate environment or every historical experiment.

## Install a clean environment

Use one dedicated environment. Select the PyTorch CUDA wheel family that your NVIDIA driver and GPU support. The reference Qwen environment uses `cu130`:

For this cu130 source-build recipe, prepare Linux, Git/Conda, a compatible NVIDIA GPU, an R580 or newer driver, CUDA Toolkit **13.0** with `nvcc`, and a C++ compiler. Check `nvidia-smi` and `nvcc --version` first. PyTorch wheels include CUDA runtime dependencies but do not install the CUDA compiler. [NVIDIA driver requirements](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)

```bash
conda create -n hermes-release python=3.12 -y
conda activate hermes-release
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-runtime-qwen.txt
python -m pip install packaging ninja psutil
FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=4 \
  python -m pip install flash-attn==2.8.3 --no-build-isolation --no-cache-dir
python -m pip check
```

PyTorch also publishes the matching 2.9.1/0.24.1 pair for CUDA 12.8 and 12.6. Changing the wheel index is an alternative installation, and requires its own import and GPU smoke validation. The unsuffixed requirements pins accept a previously installed matching CUDA build; install torch/torchvision from the chosen official index first. Do not install the `nvidia-*-cu12` lines from the old environment export into the CUDA 13.0 environment: PyTorch resolves the corresponding CUDA runtime packages itself. [Official PyTorch version commands](https://pytorch.org/get-started/previous-versions/)

FlashAttention is deliberately installed separately because it depends on the installed PyTorch/CUDA/Python ABI. The explicit `FLASH_ATTENTION_FORCE_BUILD=TRUE` is based on the [v2.8.3 installer](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/setup.py): its prebuilt-wheel selection maps non-CUDA-11 PyTorch builds to cu12, including CUDA 13. For cu130, build against the local matching toolkit to avoid selecting that wheel. This source-build recipe has not been validated in a completely fresh environment. Lower `MAX_JOBS` to 1 if host memory is limited. Source compilation requires a working compatible CUDA toolkit (`nvcc`), a compiler and build resources; follow the project's instructions and keep the toolkit aligned with the selected PyTorch CUDA build. Rebuild/reinstall FlashAttention after changing PyTorch or CUDA. A successful `pip install` alone does not establish that the binary extension imports or its kernels work. [FlashAttention installation and supported GPUs](https://github.com/Dao-AILab/flash-attention#installation-and-features)

For policy unit tests, plotting and profile analysis, also install:

```bash
python -m pip install -r requirements-runtime-analysis.txt
python -m pytest tests -q -p no:cacheprovider
```

These runtime files intentionally exclude optional sibling projects, presentation services and independent DSCache requirements. `dscache/DSCache` has its own environment instructions. Existing `protokv` and `infinipot_v` integration hooks require those external projects when explicitly enabled; leave the corresponding environment switches unset for the included methods.

StreamingBench multiple-choice evaluation runs locally and requires no API service. The optional open-ended evaluator imports `openai==1.57.1`, included in the analysis requirements, but its judge service credentials/base URL must be configured separately. That optional service and evaluator have not been validated by the StreamingBench checks.

## Obtain the model and data

Download the official Qwen model to the conventional directory, or pass an alternate local directory through the release runner's `--model-path` argument:

```bash
hf download Qwen/Qwen2.5-VL-7B-Instruct \
  --local-dir models/Qwen2.5-VL-7B-Instruct
```

The required directory contains `config.json`, processor/tokenizer files, `model.safetensors.index.json` and every weight shard referenced by that index. For the inspected checkpoint this is five shards totaling about 16.6 GB. The preflight tool checks model type and head shape, loads the processor locally, verifies all shard filenames, parses safetensors headers and checks file sizes against tensor offsets without loading weights into RAM/GPU. Record the model repository revision when publishing experiment results. [Official Qwen model and download guidance](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)

Download the StreamingBench videos from its [official dataset](https://huggingface.co/datasets/mjuicem/StreamingBench). Follow that dataset's instructions to extract the video archives. Preserve filenames and relocate the packaged annotation to the directory containing the extracted videos:

```bash
python tools/relocate_annotation.py \
  --input data/streamingbench/streamingbench_realtime_half.json \
  --video-root /path/to/StreamingBench/videos \
  --output data/streamingbench/streamingbench_portable.json
```

The helper refuses missing or ambiguous video filenames and preserves annotation ordering/questions. The half annotation contains 250 videos; the full realtime annotation contains 498. Use the same explicit annotation for every method in a comparison. For a small dependency/forward smoke test, relocate `data/streamingbench/llava_smoke_1q_20261001.json`; it uses the same generic QA format and also works with Qwen. That short sample tests execution only and does not establish compression behavior or benchmark accuracy.

The upstream README lists the official sources and annotation files for OVO-Bench, RVS, VideoMME, MVBench and EgoSchema. Their videos must also be downloaded separately when reproducing those particular experiments. The packaged JSON files alone do not include the videos.

## Validate the environment and assets

Run from the repository root:

```bash
# No dependencies, no GPU and no model needed: validate all frozen profiles.
python tools/check_environment.py --assets-only --skip-model --method all

# CPU import checks plus model/annotation validation; no CUDA required.
python tools/check_environment.py --cpu-only --method all \
  --model-path /path/to/Qwen2.5-VL-7B-Instruct \
  --anno-path data/streamingbench/streamingbench_portable.json \
  --check-video-decode

# Require CUDA and importable dense/varlen FlashAttention before a GPU run.
python tools/check_environment.py --method all \
  --model-path /path/to/Qwen2.5-VL-7B-Instruct \
  --anno-path data/streamingbench/streamingbench_portable.json
```

Each check reports `PASS`/`FAIL`; any failed check exits with status 1. `--json` returns the same results as structured JSON. The tool never downloads weights, installs packages, prints environment variables or runs model inference. `--cpu-only` skips the CUDA requirement and imports an installed FlashAttention extension if present; `--assets-only` uses only the standard library. `--skip-model` is intended for source packaging checks and does not prove a runnable model installation.

Choose `--method hermes`, `--method dp --kv-size 500`, `--method timestamp_v8` or `--method horizonkv` to check the resources for one method. The default `--method all` checks every DP budget and the distinct frozen HorizonKV budget of 6000, regardless of the DP `--kv-size` value. It does not treat the HorizonKV budget as 1000.

The tool validates these frozen inputs through `configs/runtime_assets.json`:

| Method | Required included assets | Validation |
| --- | --- | --- |
| HERMES | `inference/`, `video_qa/`, model and annotation | Core imports; model/processor/video availability |
| DP + recency | Score and `recent_ratio_profile.csv` for K=100/250/500/800/1000 | Exactly 112 unique heads, recent+history=capacity, exact global budget, score residuals and runtime SparseMM mapping |
| Timestamp v8 | `results/myobservations/relocatev3/v8_blend3_scores.csv`, `v8_budget.csv` | 112 heads in both files; actual policy loader |
| HorizonKV | `.../horizonkv_pilot_10v_fixed896/pseudo_trend_memory_budget_6000_qwen_adaptive_recent2691.csv` | 112 heads, mean capacity 6000, recent capacity 2691 and actual policy loader |

The K800 score/recency pair was deterministically exported from the preserved `per_head_allocation.csv`, using the same mapping as other DP budgets. Its `runtime_export_provenance.json` records source hashes and export parameters. This fills an execution input gap; it does not imply a historical K800 DP QA run or an independently reoptimized allocation.

## Compute and validation limits

GPU inference uses FP16 weights and FlashAttention. Start with one process on one GPU; `--num_chunks` in the legacy launcher partitions the annotation across processes and does not shard a single model across GPUs. The reference smoke setup is an NVIDIA A100 with 40 GB memory. Required peak memory and runtime depend on video duration, sampled frames, frame resolution, `encode_chunk_size`, budget and method. A smaller GPU may require lower encoding batch sizes or a smaller model, but the frozen 7B research profiles are specific to that model.

Reserve space for model weights, extracted benchmark videos, generated QA results and any traces; traces can become much larger than the source package. A CUDA/import/profile preflight does not execute a FlashAttention kernel, video encoding, cache compression or answer generation. Run the documented release runner smoke test for each chosen method before a long benchmark, then use the matching frozen protocol and evaluate all results on the same dataset split. Full benchmark accuracy, every model variant and every historical analysis script require separate validation with their own inputs.
