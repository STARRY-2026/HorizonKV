# pyramidkv

`policy.py` is the method-specific entry. The original shared selector is in `methods/_shared/kvbench/sparsemm_baselines/core.py`, including its pinned upstream source. `llava/smoke.py` reuses the shared pinned LLaVA decoder.

Launch through the project root run.py for the current DSCache-style input adapters. The standalone historical smoke main functions retain their old test defaults and are not the unified evaluation entry.
