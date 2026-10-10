# Runtime validation — 2026-10-10

The release was exercised on a **NVIDIA A100-PCIE-40GB** using local
**Qwen2.5-VL-7B-Instruct** weights and a real StreamingBench question.
The reference runtime was Python 3.12.13, PyTorch 2.9.1+cu130, torchvision
0.24.1+cu130, CUDA runtime 13.0, Transformers 4.57.1, tokenizers 0.22.2 and
FlashAttention 2.8.3. Inference uses FP16.

## Executed checks

- **85 CPU tests passed**, including policy behavior, frozen budgets, launcher
  input protection, child failure propagation and CSV merging.
- **48/48 environment checks passed**, including dense/varlen FlashAttention
  imports, CUDA, all five DP profile budgets, timestamp/Horizon profiles, model
  processor and shard headers, video paths and first-frame decoding.
- The GPU runs below each produced one prediction and an evaluation report,
  with **three actual cache-pruning events**. These were counted from emitted
  pruning messages, rather than patch-installation messages.

| Method | K | Actual compression events | Prediction / evaluation |
|---|---:|---:|---|
| Dense HERMES | 1000 | 3 | Completed |
| Uniform ragged control | 1000 | 3 | Completed |
| DP + head recency alpha/k | 100 | 3 | Completed |
| DP + head recency alpha/k | 800 | 3 | Completed |
| DP + head recency alpha/k | 1000 | 3 | Completed |
| Native v8 / MemHeadKV | 1000 | 3 | Completed |
| Timestamp v8, fixed448 | 1000 | 3 | Completed |
| Current-ourtop, fixed448 | 1000 | 3 | Completed |
| HorizonKV | 6000 | 3 | Completed |

Dense HERMES was first checked through `video_qa/run_infer.py`, including an
output path containing a space. The other runs used `tools/run_experiment.py`.
The earlier manifests predate the compression-count field; their logs were
independently checked. New smoke runs require compression evidence to succeed.
Machine-readable details and log/profile hashes are in
[`validation/smoke_summary.json`](validation/smoke_summary.json).
A clean export was additionally checked from another working directory through
the portable HERMES launcher and the released, relocated smoke annotation. It
returned 0, produced one prediction/evaluation and recorded three pruning
events. The clean export also passed all 85 CPU tests and eight standard-library
asset checks. Raw runtime logs and predictions remain outside the source package.

## Reproduce the bounded GPU smoke

The included [`validation/smoke_annotation.json`](validation/smoke_annotation.json)
contains the unchanged real question at 84 seconds from video ID 12,
`sample_158_real.mp4`, question index 1 (zero-based) in the first150 annotation.
Its clip is [0,84]; sampling at 0.5 FPS encodes 42 frames in chunks of 16+16+10.
Only the video path is made portable. Prepare that original video and the
external model, then run from this repository root:

```bash
python tools/run_experiment.py \
  --method hermes --budget 1000 \
  --model-path /path/to/Qwen2.5-VL-7B-Instruct \
  --annotation validation/smoke_annotation.json \
  --video-root /path/to/streamingbench/videos \
  --output-dir /path/to/new-output/hermes-smoke --device 0 --smoke
```

Use a new output directory and change `--method` for each retained protocol;
HorizonKV requires `--budget 6000`. See [EXPERIMENTS.md](EXPERIMENTS.md) for full
runs and matched controls, and [RUNTIME_SETUP.md](RUNTIME_SETUP.md) for setup.

## Scope of the evidence

This is an execution smoke test, **not a full-benchmark accuracy result**.
A completed run can answer incorrectly, as DP K100 did on this question.
DP K250/K500 profiles were checked on CPU but not run on GPU in this batch.
The full 150/250/498-video sets were not rerun. A fresh environment installation,
other Qwen sizes, LLaVA, DSCache, external adapters and open-ended judge services
have not been certified by these checks. Model weights and videos are external
inputs and are not bundled.

The released frozen profiles support inference; regenerating historical
calibration can require excluded raw traces. KV selection implementations
are preserved. Release fixes affect paths, process/error handling and experiment
packaging. The K800 runtime tables are a deterministic export of the existing
frozen allocation, not a new calibration fit.

Selected historical benchmark results are archived in [experiment_results/README.md](experiment_results/README.md), with predictions, reports and source hashes. They were checked during packaging and were not rerun as a full benchmark.
