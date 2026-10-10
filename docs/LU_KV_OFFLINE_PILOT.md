# LU-style offline video pilot

This experiment changes no online model or eviction code. The implementation is
`head_analysis/lu_kv_offline.py`; artifacts are under
`results/myobservations/lu_kv_offline_10v_20260913`.

## Frozen protocol

- Qwen2.5-VL-7B-Instruct, BF16, 0.5 FPS, fixed 448 letterboxed frames.
- Seed 20260913; ten videos from the existing 50-video calibration complement.
  Both IDs and video basenames are checked against the 250-video evaluation set.
- At most two questions per video with timestamps in [120, 240] seconds.
  This is a bounded-prefix pilot, not a representative long-video evaluation.
- Each prefix is encoded without any compression. Previously answered questions
  are **not** retained in either KV or conversation history. The question-free
  selector uses the default no-history local/global HERMES prompts.
- K=1000 and K=6000 mean global visual capacities of 112*K KV pairs. The
  protected recent 32 visual tokens count toward this capacity; text is separate.
- Minimum head capacity is `int((K-32)*0.2)+32`, matching the current sparsemm
  allocation floor. No role, timestamp, frame-protection or HE-voting variants.

## Measurement

1. Temporarily convert the full dense cache to ragged and call the existing
   `_ragged_query_scores` and `_default_hermes_scores_for_trace` functions.
   Neither observes the real question. No pruning occurs.
2. Rank unprotected tokens by the resulting score. Validate selected sets at
   both deployed budgets against `_select_ragged_keep_indices`. Stable-sort vs
   topk tie differences are recorded and cannot change the summed selector score.
3. Restore dense state and answer the actual MC question, using the runtime's
   16-token limit, greedy decoding and repetition penalty 1.1. Hooks observe
   each layer's last query position, including the prefill position predicting
   the first answer token. The unused forward after token 16 is excluded.
4. Compute physical-GQA token utility as
   `max_position sum_shared_query_heads attention * ||V @ Wo_query_head||_2`.
   Projection norms use float32 Gram matrices, not random projections.
5. Restore position and conversation snapshots. SHA-256 checks verify that
   selector and QA branches leave the video KV bytes unchanged.
6. Store token-level utility, selector score and rank, source token index,
   approximate source time and source chunk; raw attention tensors are not kept.

The primary utility is **raw projected utility, without intra-layer value
normalization**. This, physical-GQA aggregation, real video calibration and
fixed absolute budgets distinguish this pilot from a literal LU-KV reproduction.
The paper's attention-contribution target is a surrogate, not causal QA evidence.

## Allocation and evaluation

Utilities are summed over the actual selector's retained set. PAVA enforces
non-increasing marginal gains, forming the upper concave envelope of utility.
A heap allocates envelope segments under the exact global budget. The report
contains both actual original-curve utility and its relaxed upper bound; it
does not claim optimality for the unrelaxed integer problem or QA accuracy.

Each observation is solved separately. Budgets are averaged within a video,
then equally across training videos, with largest-remainder integer correction.
Every held-out video's table excludes **all** that video's questions. The table
trained on all ten videos is an export for subsequent independent evaluation,
not the table used to report leave-one-video-out retention.

Comparisons use Uniform, frozen v8 with exact-total correction, LU leave-one-video-
out, and an explicitly labeled per-observation allocation diagnostic. The latter
uses its own question utility and is **not deployable**. If a held-out prefix has
fewer tokens than a head's capacity, actual occupancy is clamped and reported;
unused capacity is not reassigned using question information.

Video bootstrap intervals describe these exploratory out-of-fold differences;
the folds share training data, and ten videos are insufficient for strong
generalization or publication-level significance claims. Frozen v8 is a historical
baseline, not retrained in each fold. No compressed-cache QA accuracy is measured.

## Commands

```bash
/home/yujun/.conda/envs/hermes-qwen/bin/python -m unittest tests.test_lu_kv_offline -v
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DECORD_EOF_RETRY_MAX=40960 \
  /home/yujun/.conda/envs/hermes-qwen/bin/python -u head_analysis/lu_kv_offline.py
/home/yujun/.conda/envs/hermes-qwen/bin/python head_analysis/lu_kv_offline.py --analyze-only
```

The initial smoke artifacts, which include the runtime's extra unused forward,
are archived in `smoke_initial/` and excluded from the final analysis.
