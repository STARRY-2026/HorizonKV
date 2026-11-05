> Historical experiment record (2026-11-04). The current unified entry uses ordinary visual resolution for every frame and does not replay the latest frame at question time. See the project root README.

OVO LLaVA DSCache-style input integration, experimental.
Methods: AdaKV, PyramidKV, SparseMM, LU-KV, HERMES, Full KV.
Uses the StreamingBench-tested input adapters in OVO's existing annotation, per-video checkpoint and scoring runner.
Settings: 1 fps, per-frame updates, 384x384, normal 196 and latest 729 features.
Cache handling is baseline-specific; not DSCache cache algorithm.
GPU inference is not validated in this release. SigLIP and pinned transformers runtime must be restored.
Existing release and results unchanged. Qwen and HorizonKV not implemented here.
Static Python compilation and launcher dry-runs passed for all six methods, confirming 1 fps, one frame per update, and input profile in the manifest.
No model/PPU smoke or accuracy run has been completed in this release. Current container lacks the pinned LLaVA transformers environment and usable SigLIP model mount.
