# HorizonKV-EGate unified-interface validation

2026-11-05, AI Studio container 55430687, physical PPU devices 4-7.
CLI method: `horizonkv-egate`. Experiment arm: `pae_online_e_gate_k500`.
These are GPU smoke and resume checks, not full benchmark evaluations.

| Backbone | Benchmark | Saved questions | Exit | Peak process RSS GiB | Peak GPU allocated GiB | Partial resume |
|---|---|---:|---:|---:|---:|---|
| Qwen2.5-VL | StreamingBench | 3 | 0 | 20.50 | 16.06 | Identical results |
| Qwen2.5-VL | OVO-Bench | 1 | 0 | 20.50 | 16.06 | Single question |
| LLaVA-OV | StreamingBench | 3 | 0 | 8.13 | 17.47 | Identical results |
| LLaVA-OV | OVO-Bench | 1 | 0 | 8.23 | 17.47 | Single question |

StreamingBench uses video 1's first three original questions at 6, 157 and
301 seconds, without altering the choices or ground truth. OVO uses the
existing real OJR smoke record's six-second visible prefix. This OVO prefix
check verifies execution and must not be interpreted as full-question accuracy.
An earlier single-question smoke also succeeded for all four combinations.

Both backbones reached 301 sampled frames and 75 history transfers. Separate
resume outputs retained only the first saved question, then continued from
that ledger. All result fields except elapsed seconds matched the uninterrupted
three-question runs. Earlier inputs in the incomplete video are replayed to
restore the same persistent state, while saved questions are not answered again.

Each answer checks the 500-visual-token cap per head and verifies that two-pass
answering leaves persistent video KV unchanged. Recent tokens are selected once
and shared by both passes, with the same suffix start. Models have no unloaded
meta parameters. Decoder comparison against the native text model passed with
FP16 FlashAttention/SDPA tolerance `atol=0.1, rtol=0.05`; bitwise parity is not
claimed. Native decoder errors were max/mean 0.21875/0.003727 for Qwen and
0.125/0.002948 for LLaVA.

Actual visual inputs in these runs:

- Qwen StreamingBench: ordinary grid `[1,24,42]`, 252 merged tokens; latest
  grid `[1,48,84]`, 1,008 merged tokens.
- Qwen OVO: 231 ordinary and 1,032 latest tokens for this aspect ratio.
- LLaVA: 384x384, 196 ordinary and 729 latest tokens; asserted at runtime.

At the 301-frame answer, both models used 28,636 history slots and 27,364 recent
slots, totaling 56,000 = 112 x 500. This is the occupancy of that sample; heads
with fewer available tokens are not padded to their cap.

RSS is Linux getrusage peak process RSS, including model loading. GPU figures
are torch.max_memory_allocated, not reserved memory or whole-device use.
There is no full-video CPU feature archive. These measurements cover this
301-frame sample, not all long videos or eight concurrent processes. FFmpeg
emitted mmco warnings; uninterrupted and resumed inference still exited
normally and produced identical result ledgers.

Twenty remote regression checks passed, including the original fourteen plus
allocation, gate routing, dynamic grid identity, irreversible per-head history
transfer, equal recent caches across passes, and partial-ledger resume checks.

The allocation CSV is unchanged:
`d7f644848b3521a18fdb73441e9a42052933141513b4753e273796539909a0d3`.
It has not been recalibrated for these inputs. LLaVA transfers the original
Qwen allocation. The original recent32 selector's softmax, causal mask, pooling,
latest32 protection, and history no-protection rules are retained.

This adapter introduces dynamic grids and captures raw keys at projection.
It retains EGate's own compact positions and four-frame memory boundaries;
Qwen ordinary paired units can split at a memory boundary. Numerical equality
with old fixed896 runs or DSCache's temporal/cache policy is not claimed.
Only small result CSV rows and an overwritten latest worker_status.json are
written; no per-token cache.json is read or written.

The six baseline source/asset fingerprint checks still pass. The previous 24
baseline GPU combinations were not rerun in this change; their existing results
are documented in GPU_SMOKE_REPORT.md. This change ran the four EGate
combinations and the two partial-resume cases.

Reproduce the checked three-question Qwen run with a new output directory:

```bash
python /mnt/aifordata2026-p/common/aifordata/541707/code/horizonkv/releases/20261105-qwen-llava-ovo-streaming-unified-v1/run.py \
  --benchmark streamingbench --model qwen25vl --method horizonkv-egate \
  --gpus 4 \
  --annotation /mnt/aifordata2026-p/common/aifordata/541707/experiments/horizonkv/unified-egate-gpu-smoke-55430687-20261105-02/inputs/streamingbench.json \
  --output-dir /mnt/aifordata2026-p/common/aifordata/541707/experiments/horizonkv/egate-new-smoke-output
```

Use `--model llavaov` for the checked LLaVA input path. Full benchmark runs
still require selecting the intended cohort and duplicate-answer/calibration
overlap policy. Machine-readable evidence and shard logs are under:

```text
/mnt/aifordata2026-p/common/aifordata/541707/experiments/horizonkv/unified-egate-gpu-smoke-55430687-20261105-02/report.json
```

The sibling output directories are `qwen25vl-streamingbench`,
`qwen25vl-ovobench`, `llavaov-streamingbench`, `llavaov-ovobench`, and
`{qwen25vl,llavaov}-streamingbench-resume`.
