# H2SVC reproduction status

This directory implements the paper's three per-head frame banks, static
10/40/50 routing, and Value-based Semantic Trajectory Curvature Eviction.
The paper budgets are **4, 16, and 64 sampled frames per head** for
instantaneous, short-term, and episodic heads. There is no K500 variant.

Qwen's unified input merges up to two sampled frames into one visual KV unit.
The unit is indivisible: FIFO and STCE evict both represented frames together.
LLaVA uses one frame per unit. Qwen's latest-frame answer replay follows the
existing project input protocol; the paired persistent KV is not split.

The baseline is **not yet inference-validated**. It needs a separately measured
head map for each backbone under `assets/qwen25vl_head_map.json` and
`assets/llavaov_head_map.json`, or an explicit `--head-map` path. Do not create
a random or copied map as a substitute for offline profiling. Maps must record
the calibration source and exactly one route for each of 28 × 4 KV heads.
`routing.py` freezes a map from measured V2V and T2V temporal attention
distances. `observe.py` now collects those measurements; a complete 50-video
observation has not yet run here.

## Offline observation

`observe.py` measures the paper's V2V and T2V temporal attention distances
without cache eviction. It uses the same 50 StreamingBench calibration videos
as the frozen LU-KV heldout split, 1 fps, at most the first 64 sampled frames
per video, and the paper's fixed descriptive prompt. V2V is averaged over
visual queries within each input unit, then over units and videos. T2V is
averaged over descriptive prompt and generated answer tokens, then videos.
Nonvisual keys remain in each attention softmax with zero distance. GQA query
heads are averaged within each KV head. Qwen's merged pair is observed as
one visual unit at its later frame index, matching the online KV identity.

The command accepts `--model qwen25vl|llavaov --model-path PATH --output-dir
PATH`; LLaVA additionally needs `--siglip PATH`. Set `PYTHONPATH` to include
the project root, `methods/_shared/kvbench`, and `backbones/llava_next` for
LLaVA. The tested LLaVA runtime also needs its pinned Transformers 4.40.2
directory first in `PYTHONPATH`. `--plan-only` checks and prints the video
selection. For a small smoke test, `--max-videos 1 --max-frames 2` is supported;
that partial run does **not** produce an inference-valid frozen map.

Each successful video is durably appended to `per_video.jsonl`. The script can
resume with the same plan. After all 50 videos it writes `measurements.json`
and `<model>_head_map.json`; only the completed map is accepted for inference.

The implementation uses mean Value vectors over a merged visual unit for the
STCE trajectory and preserves its oldest/newest units as eviction endpoints.
The supplied paper does not specify those two details at implementation level;
they are explicit reproduction choices, not verified author-code parity.

Code paths: `policy.py` has the banks and STCE; `state.py` assembles bounded
per-head ragged KV; `adapter.py` uses the unified visual input and answer
replay; `run.py`, `worker.py`, and `summarize.py` handle benchmark execution.
No model weights, video datasets, pinned runtime, or calibrated maps are
bundled here.
