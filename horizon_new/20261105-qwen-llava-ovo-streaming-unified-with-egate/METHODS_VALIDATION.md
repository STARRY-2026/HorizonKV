# Method source migration validation

- Six baseline method directories are present: AdaKV, PyramidKV, SparseMM, LU-KV, HERMES, Full KV.
- 12 control/bundle tests passed, including the 24 default dispatch combinations and every recorded source/asset hash.
- 24 model/benchmark/method backend dry-runs plus the full-2495 Qwen LU-KV protocol dry-run passed.
- AdaKV and PyramidKV kept identical selected indices against the original frozen selector on CPU, at layers 0/13/27.
- SparseMM head-score and LU-KV native50 profile hashes match the prior experiment assets; LU-KV allocations were read at three sequence lengths.
- All six LLaVA method imports and Qwen HERMES/Full-KV imports resolved to the bundled methods/backbone source; no older release's baseline module was imported in these checks.
- The full-2495 scoring variant now resolves within methods/_shared/equivalent_choices; the original unused override directory was moved under legacy_control/.
- LLaVA's missing temporary Transformers 4.40.2 environment was restored by linking the persistent runtime. Models, videos and runtime packages remain external.

No new GPU inference or full benchmark rerun was performed for this migration. These checks establish packaging, routing, assets and selector parity; end-to-end GPU parity is not claimed. Existing evaluations and their output directories were not switched to this source bundle.

## Subsequent GPU smoke

Real GPU smoke completed for all 24 backbone/benchmark/method combinations. See GPU_SMOKE_REPORT.md for results and the summary-file fixes.
