# LLaVA StreamingBench input adapter (experimental)

Supported methods with --input-profile dscache: adakv, pyramidkv, sparsemm, lukv, hermes, fullkv.
Qwen and OVO are not covered by this adapter. Original releases are unchanged.
All methods: 1 fps, one frame per update, LLaVA preprocessing 384x384, ordinary features pooled stride 2 (196 tokens), latest frame pooled stride 1 (729 tokens). The latest feature is used in a temporary replay for each answer; the baseline cache is restored afterward. SparseMM/LU-KV and HERMES replay their own compression policy; Full KV does not evict. This adapts input settings, not DSCache's independent cache construction.
LLaVA SparseMM/LU-KV use Qwen-calibrated assets (cross-model transfer). Full cohort LU-KV overlaps calibration and requires --allow-calibration-overlap. 1 fps/full-KV long-video memory needs separate measurement.
For the full 2495 questions, pass --accept-equivalent-choices; duplicate answer text is scored as correct for any matching answer letter.
Tests: short 2-question synthetic-time smoke on same source video for sparsemm/lukv/hermes/fullkv, with 196/729 token checks and cache identity/counter restoration checks, HERMES position restoration, Full KV head-length checks. These checks do not establish long-video or accuracy parity.

Example:
python run.py --benchmark streamingbench --model llavaov --method sparsemm --gpus 0 --input-profile dscache --output-dir /your/new/output
