# Unified Qwen / LLaVA evaluation

Use the project-root `run.py` to select the backbone, benchmark and method. Baseline source is now vendored under `methods/`; the supported unified paths do not load method implementations from older release directories. Models, videos, annotations and Python/PPU runtime dependencies remain external.

```
methods/
  adakv/       policy.py; llava/ entry
  pyramidkv/   policy.py; llava/ entry
  sparsemm/    sparsemm_native/; llava/ entry; assets/head_scores.csv
  lukv/        sparsemm_lukv/; llava/ entry; assets/.../profile.json
  hermes/      qwen/ frozen source; llava/ decoder and bridge
  fullkv/      qwen/ frozen source; llava/ decoder
  horizonkv_egate/  own method: independent recent branches, per-head KV transfer, E/F gate
  h2svc/       reproduction in progress: paper frame budgets; paired Qwen eviction
  _shared/     pinned AdaKV/PyramidKV selectors, ragged cache, native streaming, shared LLaVA decoders
backbones/llava_next/  LLaVA project model source (weights not included)
core/                 unified planning and dispatch
qwen/, qwen_full2495/, llava/  input adapters and dataset workers
```

AdaKV and PyramidKV intentionally share their original selector implementation under `_shared/kvbench/sparsemm_baselines/core.py`. Their Qwen and LLaVA adapters dispatch through their respective `methods/<name>/policy.py` entries. SparseMM and LU-KV also reuse the common streaming and ragged-cache components. Model- and benchmark-specific input adapters remain separate from cache policies.

## Input and protocol

- Qwen: 1 fps, target area 448² for every sampled frame, with at most two frames per native video unit. Answering uses the persistent streaming KV without replaying the latest frame.
- LLaVA: 1 fps, 384×384 and 196 visual tokens per frame. Answering uses the persistent streaming KV without replaying the latest frame.
- Existing K500 compression methods: mean visual budget 500; Full KV: no visual eviction cap. H2SVC retains its paper frame budgets instead of using K500.
- Qwen full StreamingBench requires explicit `--accept-equivalent-choices`; this handles one duplicate-correct-text question. LLaVA AdaKV/PyramidKV do not yet support this scoring option.
- LU-KV retains the same frozen profile calibrated on 50 StreamingBench videos. Full-cohort overlap requires `--allow-calibration-overlap`; copying the profile does not recalibrate it.
- HorizonKV-EGate is available as `--method horizonkv-egate` for both backbones and both benchmarks. It has a fixed 500 visual-token cap per head, independent one/four-frame recent branches, direct selected-KV history transfer, and two-pass evidence gating. Its frozen budget is transferred without new calibration. Actual retained tokens can be below 500 when a branch contains fewer tokens. See `methods/horizonkv_egate/README.md` and `EGATE_VALIDATION.md`. Historical `llava/egate/` files are not the active implementation.
- H2SVC is registered as `--method h2svc`. Qwen and LLaVA each have a measured 50-video offline head map; end-to-end inference is not yet validated, so benchmark results must not be reported as completed. Its per-head budgets are 4/16/64 sampled frames; Qwen's paired visual KV units are evicted together. See `methods/h2svc/README.md`.

## Commands

```bash
python run.py --benchmark ovobench --model qwen25vl --method sparsemm --gpus 0,1 --output-dir /path/to/new/output --plan-only
python run.py --benchmark ovobench --model llavaov --method hermes --gpus 0,1 --output-dir /path/to/new/output --max-videos 1 --dry-run
python run.py --benchmark streamingbench --model qwen25vl --method lukv --cohort full --accept-equivalent-choices --allow-calibration-overlap --gpus 0,1 --output-dir /path/to/new/output
python run.py --benchmark streamingbench --model qwen25vl --method horizonkv-egate --gpus 4 --output-dir /path/to/new/egate-output --max-videos 1
python summarize.py /path/to/output
PYTHONPATH="$PWD/methods/_shared/kvbench:$PWD/methods/_shared/equivalent_choices:$PWD" python -m unittest discover -s tests -v
```

`--plan-only` prints the plan without loading datasets/models. `--dry-run` calls the backend's data/configuration checks without GPU inference. Successful actual runs record `launcher_plan.json`. For legacy or unfinished Qwen StreamingBench runs, select the original scoring backend explicitly with `summarize.py OUTPUT --backend qwen` or `--backend qwen_full2495`.

Use a new output directory after this source relocation: backend code paths/hashes and the method-source inventory are now different. Existing running evaluations continue using their original releases and outputs. Root pre-refactor entry scripts are preserved in `legacy_control/`.

## Provenance and external dependencies

`methods/horizonkv_egate/SOURCE.json` records the EGate adaptation origins and frozen allocation. `methods/SOURCE_MANIFEST.json` records original baseline source paths, original SHA-256 values, and vendored SHA-256 values. Path adaptations and the explicit AdaKV/PyramidKV dispatch are recorded. Original selector formulas, attention routines, scoring policies and budget assets are preserved.

This is a project-source bundle, not a bundled execution environment. You still need the model weights, dataset media/annotations, the pinned LLaVA Transformers environment, SigLIP weights, and the Qwen/PPU runtime dependencies. Standalone historical smoke/calibration/audit scripts retained with the frozen packages may reference their original calibration inputs; use the root CLI for current evaluations. See `ARCHITECTURE.md` for remaining work.
