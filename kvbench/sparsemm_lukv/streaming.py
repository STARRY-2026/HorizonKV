"""LU static budgets plus direct visual-query attention in native SparseMM."""

import torch

from sparsemm_lukv.cache import LUKVCache
from sparsemm_lukv.policy import DEFAULT_PROFILE_PATH, load_budget_profile, load_budget_table
from sparsemm_protokv.streaming import StreamingQwen


class LUStreamingQwen(StreamingQwen):
    def __init__(self, model, processor, config, *args, **kwargs):
        if config.method != "lukv":
            raise ValueError("LUStreamingQwen requires method=lukv")
        text = model.language_model.config
        self.profile = None
        self.budgets = None
        if config.lukv_budget_path:
            self.budgets = load_budget_table(config.lukv_budget_path,
                num_layers=text.num_hidden_layers, num_kv_heads=text.num_key_value_heads,
                budget=config.budget, window=config.window_size)
        else:
            self.profile = load_budget_profile(config.lukv_profile_path or DEFAULT_PROFILE_PATH,
                num_layers=text.num_hidden_layers, num_kv_heads=text.num_key_value_heads,
                window=config.window_size)
        super().__init__(model, processor, config, *args, **kwargs)
        self.validation_checks.update(probe_restore_checks=0, probe_checks=[], selector_checks=[])
        self.timings["selector_seconds"] = 0.0

    def clear_cache(self):
        self.kv_cache = LUKVCache(self.config, self.num_layers, self.num_kv_heads,
            self.head_dim, self.device, self.dtype, budgets=self.budgets, profile=self.profile)
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
        # Selection runs inside the language forward; these timers overlap.
        self.timings["selector_seconds"] += cache.selector_seconds - before
        if kwargs.get("compress", True) and cache.phase == "video":
            for layer in cache.last_metadata["layers"]:
                checks = layer["position_checks"]
                self.validation_checks["selector_checks"].append({
                    "video_id": getattr(self, "_active_video_id", None),
                    "prefill_index": self._prefills, "layer_idx": layer["layer_idx"],
                    "validation_passed": all(checks.values()), "compressed": layer["compressed"],
                    "before_head_lengths": layer["before_head_lengths"],
                    "head_lengths": layer["head_lengths"], "head_budgets": layer["head_budgets"],
                    **({"allocation": layer["allocation"]} if "allocation" in layer else {}),
                    **checks,
                })
        return hidden

    def predict_and_compress(self):
        """The runner callback adds no probe or work; each native patch prunes."""
        return None


def load_model(model_path, config=None, sample_fps=0.5, trace_path=None,
               device="cuda", dtype="bfloat16", validate_state=True):
    from sparsemm_protokv.model_loader import load_pretrained
    model, processor = load_pretrained(model_path, device=device, dtype=dtype)
    return LUStreamingQwen(model, processor, config, sample_fps, trace_path, validate_state)
