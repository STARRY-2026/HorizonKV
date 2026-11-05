# OVO-Bench + StreamingBench unified launcher

This wrapper selects existing frozen implementations; it does not copy or replace them.

## Models and methods

`--model qwen25vl|llavaov` selects Qwen2.5-VL-7B-Instruct or LLaVA-OneVision-Qwen2-7B-OV. `--method` selects `adakv`, `pyramidkv`, `sparsemm`, `lukv`, `hermes`, or `fullkv`. HorizonKV is not connected. Both models are fixed to the warehouse checkpoints used by the existing OVO experiments; arbitrary checkpoint paths/backbones are not yet supported.

OVO-Bench is delegated to `20261102-ovobench-multimodel-unified-v1`. StreamingBench reuses the corresponding frozen model adapters. The LLaVA SparseMM and LU-KV assets were calibrated on Qwen2.5-VL and transferred; they are not LLaVA-calibrated. The LLaVA video feature assembly is a streaming adaptation and has not been proved equivalent to the original full-video template. LLaVA HERMES additionally records previous Q/A in `conv_history` to support its prediction step across StreamingBench's multiple questions.

Default setting is 0.5 fps, 16-frame outer chunks, FP16, K=500 visual budget except Full KV. Individual selectors may perform internal updates at a finer granularity. Different model families use their own visual processors and can have different numbers of tokens per frame; do not infer identical preprocessing from the shared options. Fixed canvas resolution is not imposed by this launcher.

## StreamingBench dataset and leakage

The 498-video, 2,495-question annotation is used as source. By default all methods use the same held-out 448-video / 2,244-question cohort, excluding 50 videos used by the current LU-KV budget profile. One question (video 182) is excluded because the correct answer occurs in two identical options and the original label does not uniquely identify a letter; the manifest records the exclusion. All other questions from each video are evaluated in chronological order. Choose `--cohort full` for the original 498-video set; LU-KV refuses overlapping calibration videos unless `--allow-calibration-overlap` is explicitly set. Full-cohort LU-KV is not an unbiased held-out evaluation.

## Examples

```
python run.py --benchmark streamingbench --model qwen25vl --method lukv \
  --gpus 0,1,2,3,4,5,6,7 --output-dir /path/to/new/output
python run.py --benchmark ovobench --model llavaov --method hermes \
  --gpus 0,1,2,3,4,5,6,7 --output-dir /path/to/new/output
python summarize.py /path/to/new/output
```

Add `--dry-run` to inspect configuration without launching inference. `--max-videos N` selects the first N videos of the chosen cohort. `--annotation /path/annotation.json` supplies an annotation with the same video/conversation schema. Missing source media paths are resolved by basename under `--video-root`; files must exist. Source annotations and frozen releases are never modified.

Each shard checkpoints completed videos. Rerun the identical command and output directory to resume; a partial video is repeated. Manifest and shard annotation mismatches stop resumption. StreamingBench writes `manifest.json`, per-shard `run.log` and CSV, and merged `results.csv` / `summary.json` after `summarize.py` verifies every question. Large per-update cache trace JSON is disabled. Avoid launching two workers for the same output directory concurrently.

The LLaVA runtime currently depends on `/tmp/llava-ovo-transformers440` and `/tmp/siglip-so400m-patch14-384`, as do the existing frozen OVO adapters; these may disappear when the container is rebuilt and must then be restored before use.

StreamingBench `micro_accuracy` is question-weighted and the summary also reports task and required-ability breakdowns; it is not OVO's RT/Back macro average. Compare metrics only when cohort and scoring protocol match.

## LLaVA Uniform (SnapKV-style)

`--model llavaov --method uniform` uses equal per-head K=500, a protected 32-token recent window, recent-query attention (GQA mean), average pooling with kernel 5, and per-head history Top-K. The text prefix is outside the visual budget. It is the existing streaming adaptation of the SnapKV selection rule, not a claim of an unmodified official implementation. Supports both benchmarks. It runs separately from the v1 queue. Qwen Uniform is not enabled in this v2.
