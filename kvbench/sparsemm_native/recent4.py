"""Uniform two-temporal-unit recent caches with globally apportioned DP history.

The online sliding/attention selector is unchanged.  Quotas are recomputed
before a visual temporal unit enters the decoder, from its actual token count.
All mutable allocation state belongs to the cache so an odd-endpoint branch
can be discarded without changing its even-frame parent.
"""

import hashlib
from pathlib import Path

import torch

from sparsemm_native.cache import SparseMMCache
from sparsemm_native.streaming import SparseMMStreamingQwen


def allocate_recent4_budgets(history_weights, unit_token_counts, budget=500, window_size=32):
    """Return total/recent quotas and an exact integer allocation snapshot.

    A cold start uses the single available unit.  Otherwise recent is the sum
    of the last two units, including Qwen's duplicated odd final frame.  The
    remaining global slots use Hamilton apportionment with exact remainders;
    ties follow ascending layer/head index. Old per-head bounds do not apply.
    """
    if (not isinstance(history_weights, torch.Tensor) or history_weights.ndim != 2
            or not history_weights.numel() or history_weights.dtype not in (torch.int32, torch.int64)
            or bool((history_weights < 0).any())):
        raise ValueError("History weights must be a nonempty nonnegative integer matrix")
    if (type(budget) is not int or type(window_size) is not int
            or window_size < 1 or budget < window_size):
        raise ValueError("Mean budget must include the positive protected window")
    if (not isinstance(unit_token_counts, (list, tuple)) or not 1 <= len(unit_token_counts) <= 2
            or any(type(n) is not int or n < 1 for n in unit_token_counts)):
        raise ValueError("One or two positive integer temporal-unit sizes are required")
    weights = [int(value) for value in history_weights.detach().cpu().flatten().tolist()]
    denominator = sum(weights)
    if denominator < 1:
        raise ValueError("At least one positive historical weight is required")
    r = sum(unit_token_counts)
    if not window_size <= r <= budget:
        raise ValueError("Actual recent units must include protect32 and fit the global mean budget")
    target = len(weights) * budget
    pool = target - len(weights) * r
    parts = [divmod(pool * weight, denominator) for weight in weights]
    history = [quotient for quotient, _ in parts]
    leftover = pool - sum(history)
    ranking = sorted(range(len(weights)), key=lambda h: (-parts[h][1], h))
    for head in ranking[:leftover]:
        history[head] += 1
    h = torch.tensor(history, dtype=torch.long).reshape(history_weights.shape)
    recent = torch.full_like(h, r)
    total = recent + h
    if int(total.sum()) != target or bool((h[history_weights.detach().cpu() == 0] != 0).any()):
        raise AssertionError("Global apportionment did not conserve slots/zero weights")
    metadata = dict(
        method="dp_recent4_history_proportional",
        selector="recent_sliding_then_sparsemm_history_attention",
        head_budgets=total.tolist(), recent_budgets=recent.tolist(), history_budgets=h.tolist(),
        recent_unit_token_counts=list(unit_token_counts), recent_tokens_per_head=r,
        recent_temporal_units=2, cold_start_available_units=len(unit_token_counts),
        target_visual_slots=target, allocated_visual_slots=int(total.sum()),
        configured_mean_budget=budget, history_weight_sum=denominator,
        history_weights=history_weights.detach().cpu().tolist(),
        rounding="largest_remainder_layer_head_ties",
        protect_window_included_in_recent=True, protect_window=window_size,
        old_per_head_bounds_applied=False, unused_capacity_redistribution=False,
        evicted_history_recoverable=False,
    )
    return total, recent, metadata


class Recent4Cache(SparseMMCache):
    """A SparseMM two-bank cache whose current quotas travel with its state."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.recent_budgets is None:
            raise ValueError("Recent-four requires a two-bank history-weight table")
        self.history_weights = self.budgets - self.recent_budgets
        if int(self.history_weights.sum()) < 1:
            raise ValueError("The source allocation must contain historical weight")
        self.source_allocation = dict(self.allocation)
        self.recent4_unit_token_counts = []
        # No unit has entered yet. These nominal placeholders are replaced
        # atomically before the first visual forward, never used to prune it.
        self.budgets = torch.full_like(self.budgets, self.config.budget)
        self.recent_budgets = torch.full_like(self.budgets, self.config.window_size)
        self.allocation = dict(method="dp_recent4_uninitialized")

    def begin_visual_unit(self, token_count):
        units = (self.recent4_unit_token_counts + [token_count])[-2:]
        total, recent, snapshot = allocate_recent4_budgets(
            self.history_weights, units, self.config.budget, self.config.window_size)
        snapshot.update(
            short_cache_gate=self.config.sparsemm_two_bank_gate,
            history_weight_source=self.source_allocation.get("budget_path"),
            history_weight_sha256=self.source_allocation.get("budget_sha256"),
            warmup_rule="unchanged_sparsemm_legacy_global" if self.config.sparsemm_two_bank_gate == "legacy_global"
                        else "per_head_capacity",
        )
        # All validation completes before changing any cache state. Replacing
        # the dict also preserves earlier per-layer allocation snapshots.
        self.recent4_unit_token_counts = units
        self.budgets, self.recent_budgets, self.allocation = total, recent, snapshot


class Recent4StreamingQwen(SparseMMStreamingQwen):
    is_recent4 = True

    def __init__(self, model, processor, config, *args, **kwargs):
        if (model.config.model_type != "qwen2_5_vl"
                or int(model.config.vision_config.temporal_patch_size) != 2):
            raise ValueError("Recent-four is defined for Qwen2.5-VL two-frame temporal units")
        if config.sparsemm_cache_policy != "two_bank":
            raise ValueError("Recent-four requires the two-bank configuration")
        super().__init__(model, processor, config, *args, **kwargs)

    # Existing grouped execution swaps only kv_cache on an odd-endpoint
    # branch. Reading these attributes through that cache restores all quotas
    # and provenance together, without adding hidden bridge-side state.
    @property
    def budgets(self):
        cache = getattr(self, "kv_cache", None)
        return cache.budgets if cache is not None else self._seed_budgets

    @budgets.setter
    def budgets(self, value):
        self._seed_budgets = value

    @property
    def recent_budgets(self):
        cache = getattr(self, "kv_cache", None)
        return cache.recent_budgets if cache is not None else self._seed_recent_budgets

    @recent_budgets.setter
    def recent_budgets(self, value):
        self._seed_recent_budgets = value

    @property
    def allocation(self):
        cache = getattr(self, "kv_cache", None)
        return cache.allocation if cache is not None else self._seed_allocation

    @allocation.setter
    def allocation(self, value):
        self._seed_allocation = value

    def clear_cache(self):
        self.kv_cache = Recent4Cache(
            self.config, self.num_layers, self.num_kv_heads, self.head_dim, self.device, self.dtype,
            budgets=self._seed_budgets, recent_budgets=self._seed_recent_budgets,
            allocation=self._seed_allocation, asset_metadata=self.asset_metadata,
        )
        self.visual_start_idx = 0
        self._next_position = self._seen_visual_tokens = self.total_processed_frames = 0
        self.last_encoded_frames = self._prefills = self._compressions = 0
        self.conv_history, self.last_metadata = [], []

    @torch.inference_mode()
    def forward_chunk(self, inputs_embeds, positions, compress=True, **kwargs):
        if compress:
            self.kv_cache.begin_visual_unit(int(inputs_embeds.shape[1]))
        return super().forward_chunk(inputs_embeds, positions, compress=compress, **kwargs)


def load_model(model_path, config=None, sample_fps=0.5, trace_path=None,
               device="cuda", dtype="bfloat16", validate_state=True, video_max_pixels=None):
    if config is None or config.method != "sparsemm" or config.sparsemm_cache_policy != "two_bank":
        raise ValueError("Recent-four load_model requires an explicit SparseMM two-bank config")
    if dtype not in ("bfloat16", "float16", "float32"):
        raise ValueError("Unsupported model dtype")
    if video_max_pixels is not None and (type(video_max_pixels) is not int or video_max_pixels < 3136):
        raise ValueError("Video pixel cap must be an integer >= 3136")
    from sparsemm_native.assets import asset_metadata
    from sparsemm_protokv.model_loader import load_pretrained

    metadata = asset_metadata(config.sparsemm_score_path or None)
    requested = Path(model_path).expanduser().resolve()
    if (Path(metadata["model_path"]).resolve() != requested
            or hashlib.sha256((requested / "config.json").read_bytes()).hexdigest()
            != metadata["model_config_sha256"]):
        raise ValueError("SparseMM visual scores were calibrated with a different model")
    model, processor = load_pretrained(model_path, device=device, dtype=dtype)
    return Recent4StreamingQwen(
        model, processor, config, sample_fps, trace_path, validate_state,
        asset_metadata=metadata, video_max_pixels=video_max_pixels,
    )
