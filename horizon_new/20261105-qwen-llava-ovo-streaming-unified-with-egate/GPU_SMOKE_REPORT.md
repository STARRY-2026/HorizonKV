# Real GPU smoke validation

Code: `/mnt/aifordata2026-p/common/aifordata/541707/code/horizonkv/releases/20261105-qwen-llava-ovo-streaming-unified-v1`
Results: `/mnt/aifordata2026-p/common/aifordata/541707/experiments/horizonkv/unified-methods-gpu-smoke-55430687-20261105-012359`

Passed: 24/24. Each combination processed one original question at t=6s from its benchmark, using 1 fps and the current DSCache-style input adapter. PPU 4-7 were used; existing PPU 0-3 jobs were not changed.

| Method | Qwen OVO | Qwen Streaming | LLaVA OVO | LLaVA Streaming |
| --- | --- | --- | --- | --- |
| adakv | passed | passed | passed | passed |
| pyramidkv | passed | passed | passed | passed |
| sparsemm | passed | passed | passed | passed |
| lukv | passed | passed | passed | passed |
| hermes | passed | passed | passed | passed |
| fullkv | passed | passed | passed | passed |

All passed cases generated a parseable answer, saved one result row, exited with code 0 and completed summary validation. The initial Qwen StreamingBench PyramidKV summary used an obsolete filename; its inference had already succeeded. The summary code was fixed and summary validation rerun successfully. LLaVA OVO summary delegation was localized.

Historical input observations before the ordinary-only change: LLaVA ordinary frame 196 tokens and latest frame 729 tokens; this Qwen sample ordinary two-frame unit 231 tokens and latest high-resolution unit 1032 tokens. These observations do not validate the current input path. Qwen HERMES logs showed 514 cache entries including the 14-token text prefix; Qwen Full KV retained 1739 at answering in the OVO case. Cache restoration assertions passed where provided by the adapter.

This validates short-video execution and packaging. It does not establish benchmark accuracy, long-video memory bounds or numerical equivalence with DSCache temporal pairing.
