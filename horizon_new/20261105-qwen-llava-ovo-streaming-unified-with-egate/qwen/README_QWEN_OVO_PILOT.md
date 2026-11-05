> Historical experiment record (2026-11-04). The current unified entry uses ordinary visual resolution for every frame and does not replay the latest frame at question time. See the project root README.

# Experimental Qwen input adapter: StreamingBench + OVO-Bench

Qwen2.5-VL, six baselines (AdaKV, PyramidKV, SparseMM, LU-KV, HERMES, Full KV). The opt-in input profile is --visual-input dscache448-video --sample-fps 1 --chunk-frames 2. Ordinary frames target 448² pixel area, and the latest frame in the answer path targets 896². The baseline's existing KV selection/compression policy is retained; temporary latest-frame replay uses cache copies and changes temporal grouping. This is not the DSCache caching algorithm or a proven full numerical reproduction of DSCache.

The OVO path reads the same local warehouse annotation and calls the same sample/MCQA evaluation helper used by the existing LLaVA OVO runner; it uses sample_fps=1 and chunk size 2. This locally established helper is not asserted to be identical to DSCache's own timestamp rounding and inclusion rules. Results have a unified_manifest, per-shard result CSV, and the existing OVO summarizer can validate and calculate metrics.

2026-11-04 validation: one unmodified OVO EPM record (video ID 8, question end time 7 seconds) was run on PPU 2/3 for each of six methods. Each produced one answer, logged a latest grid [1,48,84] (1008 visual tokens), passed its persistent-cache restoration check, exited zero, and passed the existing OVO summarizer. These one-question tests do not establish full-benchmark accuracy, long-video memory safety, temporal/input equivalence to the DSCache paper, or throughput. Full KV and snapshot copies may require substantial memory. No full OVO job was launched.

The StreamingBench code was copied from the independently tested pilot v3. Original releases and running LLaVA jobs were not altered.
