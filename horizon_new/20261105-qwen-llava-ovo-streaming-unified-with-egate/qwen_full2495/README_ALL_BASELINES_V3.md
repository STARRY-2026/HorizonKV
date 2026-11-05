# Experimental Qwen StreamingBench input adapter

Methods: AdaKV, PyramidKV, SparseMM, LU-KV, HERMES, Full KV. Ordinary frames target 448x448 pixels by area, latest frame targets 896x896, sampling 1 fps. Maximum 2 sampled frames per native video input. The input adapter is experimental and temporarily replays the last chunk for answering, preserving baseline-specific cache strategies; this is not a DSCache cache reproduction. Two-frame replay can use different temporal grouping and context. Deep cache copies add memory overhead. OVO-Bench and HorizonKV are not supported by this experiment.

AdaKV, PyramidKV, HERMES, Full KV each completed two-question smoke on 2026-11-04 with latest grid [1,48,84] and 1008 visual tokens, while restoring persistent cache values. SparseMM/LU-KV previously passed the corresponding two-question test in v2. No full, long-video, accuracy-parity, or real throughput validation has been completed. Do not infer final benchmark performance from smoke.
