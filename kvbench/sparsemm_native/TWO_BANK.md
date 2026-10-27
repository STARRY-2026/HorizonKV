# DP dual-cache OVO-Bench comparison

The `dp_dual` arm runs a per-head recent sliding window plus attention-selected
history in the same native SparseMM bridge as the completed `native` and `dp`
arms. It is an opt-in policy; the original two selectors are preserved.

For head h, the CSV supplies total N, recent R and history H with R+H=N and
R>=32. The newest R visual tokens stay in chronological order. Tokens leaving
this window compete with surviving history for H slots, using the existing
SparseMM FP32 causal attention score, last-32 query observation window, GQA mean,
avgpool kernel5 and torch.sort ranking. Previously evicted tokens never return.
The initial text prefix is protected separately. No spare quota is moved between
heads or banks. Protect32 is included inside R, rather than added to N.

## Fixed-capacity primary allocation

`hermes/results/myobservations/two_bank_grid_dp_k500_floor375_recent32_20261020/per_head_allocation_fixed_capacity.csv`

All 112 head totals exactly match the previous bounded K=500 DP table, including
the 375..750 limits and 56,000 total visual KV slots. For those fixed totals, the
offline solver reoptimizes the recent/history split with R>=32 over the original
calibration curves. It does not just clamp the old recent value.

Recent totals 26,104 and history totals 29,896. The source calibration has 48
eligible videos / 68 observations from the original 50-video trace directory;
it uses no OVO-Bench labels. Offline history ranking is the original HERMES
ranking, so its retained full-cache utility remains a proxy for the online
SparseMM attention selector, not a question-answering score.

The adjacent `per_head_allocation.csv` globally reoptimizes head totals and is
an optional separate allocation experiment. The primary OVO runner rejects it
if any per-head total differs from the saved DP control.

## Matched warmup and benchmark protocol

The primary arm uses `sparsemm_two_bank_gate=legacy_global`: it retains the
completed experiments' exact short-cache bypass condition, based on K−32 and
the physical history before selection. During this shared warmup all candidates
are kept, even if an individual head exceeds its nominal quota. After the gate
activates, R/H quotas are enforced. This avoids changing compression timing
while comparing bank policies. The strict `per_head_capacity` gate is available
in the low-level API and config, but is a separate experiment and is rejected
by the primary three-way comparator.

Keep Qwen2.5-VL-7B-Instruct, BF16, 0.5 FPS, 151200 pixel cap, 16-frame ingestion,
two-frame temporal patches, greedy decoding, repetition penalty1.1 and
max_new_tokens16. The attention observation/protection window stays32 even when
the recent bank is larger.

The nine-task run must use `--dataset hermes9`; it preserves HERMES prompts,
MC completion prefix, targets and letter scoring (including query1303 targetA).
The forward-task run must use `--dataset forward3`; it preserves official
REC/SSR/CRR prompts and scoring. Their combined coverage is3035 queries, but
they are separate prompt/scoring protocols and cannot form a unified official
OVO score.

New outputs record full and selected query hashes, prompt manifests, allocation
hashes, processor files, dependency versions, grouped/debug settings and source
snapshots. Changed protocols or missing provenance on a nonempty run are
rejected on resume. Benchmark runs keep debug explanations disabled.

## Run and compare

From `/home/yujun/program/kvbench`, with the same `hermes-qwen` Python environment
and PYTHONPATH used by the existing runs:

```bash
python -m sparsemm_native.run_ovobench --arm dp_dual --dataset hermes9 \
  --output-root results/ovobench_dual_hermes9_k500_fps05_p151200_20261020 \
  --grouped --max-new 100000
python -m sparsemm_native.run_ovobench --arm dp_dual --dataset forward3 \
  --output-root results/ovobench_dual_forward3_k500_fps05_p151200_20261020 \
  --grouped --max-new 100000
python -m sparsemm_native.compare_ovobench_three_way \
  --baseline-root results/ovobench_hermes9_k500_fps05_p151200_20261019 \
  --dual-root results/ovobench_dual_hermes9_k500_fps05_p151200_20261020 \
  --output results/ovobench_dual_hermes9_k500_fps05_p151200_20261020/comparison
```

Use the corresponding forward3 paths for its comparison. The tool checks model,
dataset, prompt identity, scoring, frame counts, budget and head capacities, and
only reports matched-question accuracy. Old per-query timings and grouped
incremental timings have different accounting and should not be pooled as a
speed comparison.

Independently audit a dual output with:

```bash
python -m sparsemm_native.audit_two_bank \
  results/ovobench_dual_hermes9_k500_fps05_p151200_20261020/dp_dual
```

The trace audit verifies allocation, indices, bank counts, warmup and source
provenance. It cannot recompute attention scores without the original Q/K;
real scoring parity is covered separately by CPU selector tests.

## Completed implementation validation

`results/ovobench_two_bank_validation_20261020/report.md` records the completed
70-test CPU suite and17-query GPU regression (all12 tasks). Both old arms
reproduced saved answers exactly. The new dual arm gave identical grouped and
fresh answers, including odd/even endpoints and nonmonotonic question times.
Four independently audited dual traces contain4,368 layer records /17,472 head
records. These small samples verify implementation and protocol compatibility;
they do not establish full-dataset quality.

The generic StreamingBench CLI's legacy auditor does not support this policy;
it refuses the mode instead of labeling a dual-cache run as ordinary SparseMM.
