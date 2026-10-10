# Prospective Lifecycle Validation

## Question

After controlling for a token's within-frame admission importance, does a
head-specific remaining-lifecycle profile predict future reads better than a
global profile and invalid head/time controls?

The existing age-bin CSV cannot answer this question. It is a cross-sectional
snapshot at query time. This experiment follows the same source temporal patch
from admission through later observations.

## Trace contract

Run an uncompressed replay. Each incoming visual chunk supplies sampled visual
queries that read the complete preceding cache. Records contain:

- source and observation frame IDs, with age derived from `sample_fps`;
- physical layer and KV head;
- within-frame admission-rank bins;
- token count and mean value-vector L1 norm;
- attention mass and `attention * ||V||_1` effective-read mass;
- `full_cache`, which becomes false after any prune.

The analyzer rejects compressed records by default. The main label is the
strictly-future cumulative effective read of the same source patch, head, and
content bin. The read at the scoring observation itself is excluded.

## Calibration run

Use calibration videos disjoint from the final StreamingBench evaluation. The
two existing 20-video shards can be collected sequentially on one GPU:

```bash
export PYTHONPATH=/home/yujun/program/hermes
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PYTHON=/home/yujun/.conda/envs/hermes-qwen/bin/python

$PYTHON video_qa/run_infer.py \
  --model qwen2.5_vl_7b \
  --dataset streamingbench \
  --anno_path results/myobservations/temporal_topk_profile_50v_fixed448/new40_gpu0.json \
  --num_chunks 1 --devices 0 --sample_fps 0.5 \
  --kv_size 1000000 --compress_mode hermes \
  --kv_head_ragged_prefill true \
  --prospective_lifecycle_trace_path results/myobservations/prospective_lifecycle_n40/gpu0_{chunk_idx}.jsonl \
  --fixed_frame_canvas_size 448 --fixed_frame_pad_mode letterbox \
  --debug false --output_tag prospective-lifecycle-calibration-gpu0
```

Repeat with `new40_gpu1.json`, a `gpu1_{chunk_idx}.jsonl` trace path, and a
different output tag. `kv_size` is only a threshold; it does not preallocate a
million-token cache. The run aborts if the threshold is exceeded rather than
silently collecting survivor-biased data.

The one-video smoke replay produced a 196 MB trace in about 11 minutes on the
local A100, so plan for several hours and roughly 8 GB for 40 similar videos.

## Held-out decision

```bash
$PYTHON scripts/analyze_prospective_lifecycle.py \
  --trace results/myobservations/prospective_lifecycle_n40/gpu0_0.jsonl \
  --trace results/myobservations/prospective_lifecycle_n40/gpu1_0.jsonl \
  --output results/myobservations/prospective_lifecycle_n40/report.json
```

Videos, not observations, are split into train and test sets. The report gives
frame/content-unit AUC, NDCG, future top-k recall, and video-bootstrap confidence
intervals for these policies:

- content only;
- content plus global remaining lifecycle;
- content plus physical-head remaining lifecycle;
- within-layer shuffled head lifecycle;
- reversed-time head lifecycle.

Proceed to online lifecycle eviction only when the paired video-bootstrap 95%
confidence-interval lower bound is above zero for head-specific NDCG minus all
three controls: global, shuffled, and reversed-time. Otherwise the honest
conclusion is that the current head identity or lifecycle estimator does not
provide actionable prospective information.

## Pilot outcome

The six-video LOOV pilot produced a split result. Head-specific lifecycle
passed all controls for future visual-chunk readout, but failed for future
true-question readout: content-only NDCG was 0.82035 versus 0.78978 for the head
lifecycle. Weights from 0.01 through 1.0 did not repair the task-facing failure.
The current visual-read lifecycle is therefore a no-go for online eviction.
See `results/myobservations/prospective_lifecycle_pilot6/README.md`.
