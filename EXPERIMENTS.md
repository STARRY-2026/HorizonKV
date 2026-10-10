# Running the released HERMES experiments

The portable launcher runs the released source directly. It does not require the
original checkout, its Git history, its Conda path, or sibling projects. Download
the Qwen2.5-VL-7B model and StreamingBench videos separately, and install the
Qwen environment described in `README.md`. Model weights and videos are not
included in the GitHub source package.

## Retained protocols

The exact flags and input tables are frozen in
[`configs/experiments/protocols.json`](configs/experiments/protocols.json).
Every launcher run uses Qwen2.5-VL-7B, StreamingBench, `sample_fps=0.5`, and
`compress_mode=hermes`. The current model loader uses **FP16**
(`torch_dtype=torch.float16`), including where historical reports say BF16.

| `--method` | Runtime | Budgets | Input resolution |
|---|---|---|---|
| `hermes` | Original dense HERMES path, without head-budget/ragged patches | Any integer >=64 | Native processor |
| `uniform` | Uniform physical per-KV-head ragged control | Any integer >=64 | Native processor |
| `dp` | Frozen exact DP capacity + head recency alpha/k | 100, 250, 500, 800, 1000 | Native processor |
| `v8` / `memheadkv` | MemHeadKV native v8 head-budget allocation only | 1000 | Native processor |
| `timestamp-v8` | v8 allocation + timestamp v8 eviction | 1000 | Fixed448 letterbox |
| `current-ourtop` | Separately recalibrated v8 scores + timestamp eviction | 1000 | Fixed448 letterbox |
| `horizonkv` | Independent fixed RecentCache + MemoryCache HorizonKV selector | 6000 | Native processor |

`hermes` and `uniform` are different baselines. For the DP comparison, use
`uniform`: it has the same physical ragged implementation and SparseMM
`ratio=0.2, window=32, union_cap=1`, with uniform capacities and no head alpha/k.
The original dense HERMES path remains available independently.

MemHeadKV in the native experiment reports refers to allocation from
`v8_blend3_scores.csv`, without `timestamp_v8_policy` or a fixed448 canvas.
The later timestamp policy is named `timestamp-v8` here to keep those results
distinct. `current-ourtop` is another separately named timestamp variant.

DP recency parameters are `metric=recent_ratio`, `delta=0.1`, `tau=1`,
`alpha_min=0`, `alpha_max=1`, `k_delta=2`, `k_min=0`, `k_max=20`.
K250 intentionally uses its existing DP score CSV; the historical pilot shell
defaults to a uniform score file unless `EXPERIMENT_BUDGET_SCORES` is overridden.
K800 CSVs were exported deterministically from its existing frozen
`per_head_allocation.csv`, without fitting DP again. The export formula, source
SHA256, and agreement with historical K100/250/500/1000 exports are recorded in
`results/myobservations/two_bank_grid_dp_k800_floor600_20260928/runtime_export_provenance.json`.

HorizonKV uses the real `--horizonkv_policy` runtime with Recent=2691 per head,
EMA gamma=0.9, age penalty on, cold temporal coverage, original three probes,
and head-type score blending. Its frozen CSV totals 672000 slots, averaging
6000 across 112 physical KV heads. Other Horizon budgets are rejected instead
of silently reusing or rescaling this table.

## Offline preflight and real smoke run

Run from this repository root with your environment's Python:

```bash
python tools/run_experiment.py \
  --method dp --budget 1000 \
  --model-path /path/to/Qwen2.5-VL-7B-Instruct \
  --annotation data/streamingbench/streamingbench_realtime_half.json \
  --video-root /path/to/streamingbench/videos \
  --output-dir /path/to/new-output/dp-k1000-smoke \
  --device 0 --smoke --dry-run
```

Remove `--dry-run` and choose a **new output directory** to execute the same experiment. A successful preflight writes
`annotation.json` and `experiment_manifest.json` into the chosen output
directory. Every invocation, including a dry run, requires a new or empty output
directory. A source annotation's containing directory and a model directory (or
their ancestors) are rejected as output destinations. All staged input files
are written exclusively without overwriting existing files. Model loading, video decoding and CUDA are
never invoked by the dry run. The local model config, weight-shard presence,
selected video paths, all 112 profile heads, per-head budgets and recency-bank
identities are checked. An existing `PROTOKV_ENABLE` or `INFINIPOT_V_PUBLIC`
environment variable cannot implicitly activate an unbundled adapter: the
child environment sets both to `0` and sets `PYTHONPATH` to this source root.

Without `--full`, the default is one video and one **real question** whose
original `end_time` is in `[64,120]` seconds. The complete prefix from the
original sampling origin is retained. When no clip existed, the launcher adds
`clip=[0,end_time]` to avoid decoding the unused future suffix; it never changes
questions, timestamps or existing clips. This exercises a substantially longer
prefix than the older one-second smoke annotation. The completed smoke run
requires actual cache-pruning evidence in `inference.log`, records the count in
the manifest, and fails if no compression occurred; the CPU preflight cannot
prove GPU execution.
Use `--smoke-videos`, `--smoke-questions`, `--smoke-min-seconds` and
`--smoke-max-seconds` to adjust this bounded selection explicitly.

Run the other retained paths by changing `--method`, using a new output directory
for each. For HorizonKV also set `--budget 6000`. `--method memheadkv` is an
explicit alias of `--method v8`, recorded in the manifest.

For a fixed448 control against `timestamp-v8`, add `--frame-canvas-size 448` to
`uniform` or `hermes`; this explicit override is recorded separately from the
frozen protocol. Native and fixed448 scores should be compared using matched
input resolution. HorizonKV currently prohibits this override.

## Complete historical evaluation sets

Full runs must be requested explicitly with `--full`. Use the same annotation
and video root for every method in a comparison. The available historical sets
include:

| Annotation | Videos / questions | Historical usage |
|---|---|---|
| `results/myobservations/native_k500_first150/anno_first150.json` | 150 / 754 | K100, K500, K800 native comparisons |
| `results/myobservations/k250_cachecomp_score_pilot_20260923/anno_10v.json` | 10 / 50 | K250 pilot |
| `data/streamingbench/streamingbench_realtime_half_userpaths.json` | 250 / 1255 | K1000 first250 protocols |
| `data/streamingbench/streamingbench_realtime.json` | 498 / 2495 | Full released StreamingBench annotation |

The launcher relocates only the explicitly selected annotation and keeps its
record/question order. Machine-specific paths in these files are replaced in
the staged copy with `--video-root`; the source annotations are preserved.

Example full K500 DP comparison:

```bash
python tools/run_experiment.py \
  --method dp --budget 500 --full \
  --model-path /path/to/Qwen2.5-VL-7B-Instruct \
  --annotation results/myobservations/native_k500_first150/anno_first150.json \
  --video-root /path/to/streamingbench/videos \
  --output-dir /path/to/new-output/dp-k500-first150 --device 0
```

Repeat with `--method uniform` and a distinct output directory for the matched
control. Reproducing a historical evaluation also requires the same question
set, preprocessing, model revision and runtime versions; a smoke run verifies
execution and does not establish full-benchmark accuracy.

## Budget accounting and recorded outputs

DP budgets conserve exactly: totals are 11200, 28000, 56000, 89600 and 112000
for K100/250/500/800/1000 respectively. Uniform and HorizonKV also conserve
their declared totals exactly. The original dense HERMES implementation retains
`K-1` selected visual tokens plus one pooled summary KV in long-term layers;
the summary is included in its K-slot capacity.
The historical SparseMM rounding in the v8
K1000 file yields **112004**, and current-ourtop yields **112001**. These
historical parameters are preserved; the manifest reports the exact per-head
table and `rounding_delta` of +4 or +1. `--strict-budget` rejects either
historical difference. No strategy is silently rewritten to adjust its total.

`experiment_manifest.json` contains the exact argument list, FP16 dtype,
protocol and input hashes, source question indices, video/question counts,
budget table, resolution overrides, actual compression events and final result-row count. Inference logs
are saved to `inference.log`; predictions are saved to `results.csv` and the
normal evaluator writes `eval_results.txt`. Missing prediction rows make the
launcher fail even if a child unexpectedly exits successfully.

## Frozen inference versus recalibration

The launcher runs the included **frozen profiles**. It does not recalibrate on
evaluation questions. All profile CSVs needed by the listed protocols are
included. To verify/rebuild the K800 runtime export only:

```bash
python tools/export_frozen_dp_profiles.py
python tools/export_frozen_dp_profiles.py --write-k800
```

The original analysis, allocation, profiling and legacy experiment source
remains in `head_analysis/`, `scripts/` and `experiments/`, including recent32,
role, timestamp, LU and other variants. Regenerating historical calibration
profiles may require original trace/observation files excluded from this source
release. The legacy shell scripts contain old machine paths; use the portable
launcher for the protocols above, or pass the corresponding advanced arguments
directly to `video_qa/run_infer.py` with explicit `--model_path`, `--anno_path`
and `--output_dir`.

The optional DSCache source is retained separately. PROTOKV and INFINIPOT
adapters reference sibling projects and are not prerequisites for any protocol
in this document.

Run the release protocol checks with:

```bash
python -m pytest tests/test_run_experiment.py -q -p no:cacheprovider
```

Selected historical benchmark results are archived in [experiment_results/README.md](experiment_results/README.md), with predictions, reports and source hashes. They were checked during packaging and were not rerun as a full benchmark.
