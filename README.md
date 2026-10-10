# HERMES and head-wise video KV experiments

This release contains the current HERMES implementation and the retained local research methods, with frozen inference profiles and portable experiment entry points. Original authorship, paper citation and model/benchmark links are preserved in [UPSTREAM_README.md](UPSTREAM_README.md). Retain the original [MIT license](LICENSE).

## Set up the runtime

Follow [RUNTIME_SETUP.md](RUNTIME_SETUP.md) to install the reference Python 3.12 / PyTorch 2.9.1 / Transformers 4.57.1 / FlashAttention 2.8.3 environment. Use `requirements-runtime-qwen.txt`; the old Qwen/LLaVA requirements files are retained as historical exports.

Download **Qwen2.5-VL-7B-Instruct** and **StreamingBench videos** separately. The source package includes annotations and small profiles, not model weights or videos. Pass your own model and video directories explicitly; the portable launcher does not require the original server paths.

## Run an experiment

From this repository root, first check the local installation:

```bash
python tools/check_environment.py --method all \
  --model-path /path/to/Qwen2.5-VL-7B-Instruct
```

Run a real long-prefix smoke experiment, including video encoding, cache compression, answer generation and evaluation:

```bash
python tools/run_experiment.py \
  --method dp --budget 1000 \
  --model-path /path/to/Qwen2.5-VL-7B-Instruct \
  --annotation data/streamingbench/streamingbench_realtime_half.json \
  --video-root /path/to/streamingbench/videos \
  --output-dir /path/to/new-output/dp-k1000-smoke \
  --device 0 --smoke
```

The default smoke selects a real question after 64–120 seconds. Add `--dry-run` to check model files, video paths and frozen budgets without importing GPU libraries or running inference. Use a new output directory for each run.

| Method | Purpose | Frozen supported budgets |
|---|---|---|
| `hermes` | Original dense HERMES | Any integer >=64 |
| `uniform` | Matched uniform physical ragged control | Any integer >=64 |
| `dp` | Exact DP head capacities + recency alpha/k | 100, 250, 500, 800, 1000 |
| `v8` / `memheadkv` | Native v8 head-budget allocation | 1000 |
| `timestamp-v8` | v8 allocation + timestamp eviction, fixed448 | 1000 |
| `current-ourtop` | Recalibrated timestamp variant, fixed448 | 1000 |
| `horizonkv` | Independent Recent + Memory HorizonKV | 6000 |

Change `--method` and the output directory to run another method. For HorizonKV use `--budget 6000`. Use `uniform` as the matched implementation control for DP. For fixed448 timestamp comparisons, pass `--frame-canvas-size 448` to the control as well; explicit overrides are recorded in the manifest.

For a complete benchmark replace `--smoke` with `--full` and select the intended annotation. The included first150 split has 150 videos / 754 questions; the half split has 250 / 1255; the full annotation has 498 / 2495. Keep the same split, resolution and budget for comparisons. [EXPERIMENTS.md](EXPERIMENTS.md) gives exact protocols, full-run commands and legacy entry points.

Each run produces `experiment_manifest.json`, `inference.log`, `results.csv` and `eval_results.txt`. The manifest records profile hashes, selected questions, per-head budgets, runtime flags and result counts. Child failures, missing resources and incomplete predictions make the command fail.

## Included code and validation

The release retains `inference/`, `video_qa/`, **`head_analysis/`**, `eval/`, `scripts/`, `experiments/`, `tests/` and `docs/`. `head_analysis/` is required by the current inference entry point. Small runtime tables remain at their existing relative `results/` paths and are explicitly allowed by `.gitignore`; generated logs, weights, videos and unselected experiment outputs are excluded. Selected historical predictions and evaluation reports are archived separately in [`experiment_results/`](experiment_results/README.md).

The [main result appendix](experiment_results/README.md) contains 14 completed method/budget/split records: DP K100/250/500/1000 with uniform controls, native v8, timestamp v8 and historical HorizonKV v0.6 with their controls. Original CSVs/reports and merged-shard provenance are retained. Incomplete current-ourtop outputs and targeted latest-Horizon pilots are not presented as complete benchmark results.

Run the policy, launcher and failure-propagation checks with:

```bash
python -m pip install -r requirements-runtime-analysis.txt
python -m pytest tests -q -p no:cacheprovider
```

[RUNTIME_VALIDATION.md](RUNTIME_VALIDATION.md) records actual GPU smoke checks and their limits. Bounded execution checks do not establish full-benchmark accuracy or validate every historical analysis program/model variant. Installation in a completely fresh environment requires its own validation.

## Research and source provenance

Inference uses the included **frozen profiles**. Regenerating the original calibration may require large raw observations/traces excluded from the release. All legacy method source is preserved. Optional `dscache/` is an independent comparison baseline with separate setup; `protokv` and `infinipot_v` adapters require external sibling projects only when explicitly enabled. The portable runner disables those adapters for the retained protocols.

Base source inspected on 2026-10-10: local `main`, commit `6020490`, origin `https://gitee.com/charlottenaixu/hermes.git`, including current uncommitted and untracked source. Release changes add portable model/output paths, reliable process failure handling, environment checks, protocols and tests while preserving the KV strategy implementations. K800 runtime tables are deterministic exports of an already frozen allocation, with provenance beside the tables.

`SNAPSHOT_MANIFEST.csv` records source files and release changes. [UPLOAD_PLAN.md](UPLOAD_PLAN.md) explains the clean-history GitHub export. No original Git history is bundled.
