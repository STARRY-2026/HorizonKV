"""SparseMM static visual-head budgets around the shared native HF decoder."""

import hashlib
from pathlib import Path

import torch

from sparsemm_native.cache import SparseMMCache
from sparsemm_native.policy import (DEFAULT_SCORE_PATH, allocate_head_budgets,
                                    load_fixed_head_budgets, load_head_scores)
from sparsemm_native.two_bank import load_two_bank_budgets
from sparsemm_protokv.streaming import StreamingQwen


class SparseMMStreamingQwen(StreamingQwen):
    def __init__(self, model, processor, config, *args, asset_metadata=None, **kwargs):
        if config.method != "sparsemm":
            raise ValueError("SparseMMStreamingQwen requires method=sparsemm")
        text = model.language_model.config
        path = Path(config.sparsemm_score_path or DEFAULT_SCORE_PATH).expanduser().resolve()
        scores = load_head_scores(path, text.num_hidden_layers, text.num_attention_heads)
        self.recent_budgets = None
        if config.sparsemm_cache_policy == "two_bank":
            self.budgets, self.recent_budgets, self.allocation = load_two_bank_budgets(
                config.sparsemm_budget_path, text.num_hidden_layers,
                text.num_key_value_heads, config.budget, config.window_size,
                short_cache_gate=config.sparsemm_two_bank_gate)
        elif config.sparsemm_budget_path:
            self.budgets, self.allocation = load_fixed_head_budgets(
                config.sparsemm_budget_path, text.num_hidden_layers,
                text.num_key_value_heads, config.budget, config.window_size)
        else:
            self.budgets, self.allocation = allocate_head_budgets(
                scores, budget=config.budget, window_size=config.window_size,
                ratio=config.sparsemm_ratio, num_kv_heads=text.num_key_value_heads)
        self.asset_metadata = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                               "query_head_shape": list(scores.shape), "shape": list(self.budgets.shape)}
        if asset_metadata is not None:
            if (Path(asset_metadata["path"]).resolve() != path
                    or asset_metadata["sha256"] != self.asset_metadata["sha256"]
                    or asset_metadata["shape"] != list(self.budgets.shape)):
                raise ValueError("SparseMM asset provenance disagrees with the loaded scores/model shape")
            self.asset_metadata = dict(asset_metadata)
        elif path == DEFAULT_SCORE_PATH.resolve() or (path.parent / "manifest.json").is_file():
            from sparsemm_native.assets import asset_metadata as inspect_asset
            self.asset_metadata = inspect_asset(path)
        super().__init__(model, processor, config, *args, **kwargs)
        self.validation_checks.update(probe_restore_checks=0, probe_checks=[], selector_checks=[])
        self.timings["selector_seconds"] = 0.0

    def clear_cache(self):
        self.kv_cache = SparseMMCache(
            self.config, self.num_layers, self.num_kv_heads, self.head_dim, self.device, self.dtype,
            budgets=self.budgets, allocation=self.allocation, asset_metadata=self.asset_metadata,
            recent_budgets=self.recent_budgets,
        )
        self.visual_start_idx = 0
        self._next_position = self._seen_visual_tokens = self.total_processed_frames = 0
        self.last_encoded_frames = self._prefills = self._compressions = 0
        self.conv_history, self.last_metadata = [], []

    @torch.inference_mode()
    def forward_chunk(self, *args, **kwargs):
        cache = self.kv_cache
        cache.timing_enabled = self.timing_enabled
        before = cache.selector_seconds
        hidden = super().forward_chunk(*args, **kwargs)
        # This timer is a subset of language_video_seconds, not an extra stage.
        self.timings["selector_seconds"] += cache.selector_seconds - before
        if kwargs.get("compress", True) and cache.phase == "video":
            for layer in cache.last_metadata["layers"]:
                checks = layer["position_checks"]
                self.validation_checks["selector_checks"].append({
                    "video_id": getattr(self, "_active_video_id", None),
                    "prefill_index": self._prefills, "layer_idx": layer["layer_idx"],
                    "validation_passed": all(checks.values()), "compressed": layer["compressed"],
                    "bypassed_short_cache": layer["bypassed_short_cache"],
                    "before_head_lengths": layer["before_head_lengths"],
                    "head_lengths": layer["head_lengths"], "head_budgets": layer["head_budgets"],
                    "asset": self.asset_metadata, "allocation": self.allocation, **checks,
                })
        return hidden

    def predict_and_compress(self):
        """All selection runs inside each native visual patch; no probes."""
        return None


def load_model(model_path, config=None, sample_fps=0.5, trace_path=None,
               device="cuda", dtype="bfloat16", validate_state=True, video_max_pixels=None):
    if dtype not in ("bfloat16", "float16", "float32"):
        raise ValueError("Unsupported model dtype")
    if config is None or config.method != "sparsemm":
        raise ValueError("SparseMM load_model requires explicit method=sparsemm config")
    if video_max_pixels is not None and (not isinstance(video_max_pixels, int) or video_max_pixels < 3136):
        raise ValueError("SparseMM video pixel cap must be an integer >= 3136")
    from sparsemm_native.assets import asset_metadata
    metadata = asset_metadata(config.sparsemm_score_path or None)
    requested_model = Path(model_path).expanduser().resolve()
    if (Path(metadata["model_path"]).resolve() != requested_model
            or hashlib.sha256((requested_model / "config.json").read_bytes()).hexdigest()
            != metadata["model_config_sha256"]):
        raise ValueError("SparseMM visual scores were calibrated with a different model")
    from sparsemm_protokv.model_loader import load_pretrained
    model, processor = load_pretrained(model_path, device=device, dtype=dtype)
    return SparseMMStreamingQwen(model, processor, config, sample_fps, trace_path, validate_state,
                                asset_metadata=metadata, video_max_pixels=video_max_pixels)
