"""Fixed LU head capacities with direct SparseMM visual-attention selection."""

import time

import torch

from sparsemm_lukv.policy import attention_select_layer
from sparsemm_protokv.cache import StreamingCache


class LUKVCache(StreamingCache):
    def __init__(self, *args, budgets=None, profile=None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.config.method != "lukv" or (budgets is None) == (profile is None):
            raise ValueError("LU cache requires exactly one budget profile or fixed table")
        self.profile = profile
        if profile is not None:
            from sparsemm_lukv.profile import validate_profile
            validate_profile(profile, self.num_layers, self.num_kv_heads, self.config.window_size)
            budgets = torch.zeros((self.num_layers, self.num_kv_heads), dtype=torch.long)
        if tuple(budgets.shape) != (self.num_layers, self.num_kv_heads):
            raise ValueError("LU cache requires matching fixed layer/KV-head capacities")
        self.budgets = budgets
        self.allocation = None
        self.selector_seconds = 0.0
        self.timing_enabled = False

    def set_context(self, phase, positions, xy=None, time=0):
        super().set_context(phase, positions, xy=xy, time=time)
        if phase == "video" and self.profile is not None:
            from sparsemm_lukv.profile import allocate_profile
            self.budgets, self.allocation = allocate_profile(
                self.profile, self.seen_visual_tokens + positions.shape[1],
                self.config.budget, self.config.window_size,
            )

    @torch.inference_mode()
    def append_layer(self, layer_idx, k, v, raw_k=None, q=None):
        if self.phase != "video":
            return super().append_layer(layer_idx, k, v, raw_k=raw_k, q=q)
        if self._positions is None or self._context_done:
            raise RuntimeError("set_context must precede a native LU video forward")
        if not 0 <= layer_idx < self.num_layers or layer_idx in self._appended:
            raise ValueError("Each native layer must execute exactly once")
        expected = (1, self.num_kv_heads, self._positions.shape[1], self.head_dim)
        if k.shape != expected or v.shape != expected:
            raise ValueError(f"LU KV shape must be {expected}")
        if q is None or q.ndim != 4 or q.shape[0] != 1 or q.shape[2:] != k.shape[2:]:
            raise ValueError("Direct LU scoring requires current visual query projections")
        layer = self.layers[layer_idx]
        layer.append_chunk(k, v, self._positions)
        before_lengths = layer.head_lens.tolist()
        if self.timing_enabled and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        selected, metadata = attention_select_layer(
            [layer.get_segment(h)[0] for h in range(self.num_kv_heads)], q[0],
            self.budgets[layer_idx], prefix_tokens=self.prefix_tokens,
            window_size=self.config.window_size, pooling=self.config.pooling,
            kernel_size=self.config.kernel_size, gqa_reduce=self.config.gqa_reduce,
        )
        checks = dict(prefix_preserved=True, recent_preserved=True,
                      indices_unique=True, indices_in_range=True, positions_match_kept=True)
        expected_positions = []
        for head, keep in enumerate(selected):
            old_pos = layer.get_position_segment(head)
            length = before_lengths[head]
            prefix = min(self.prefix_tokens, length)
            recent = min(self.config.window_size, length - prefix)
            checks["indices_unique"] &= keep.unique().numel() == keep.numel()
            checks["indices_in_range"] &= not keep.numel() or (int(keep.min()) >= 0 and int(keep.max()) < length)
            checks["prefix_preserved"] &= torch.equal(keep[:prefix], torch.arange(prefix, device=self.device))
            if recent:
                checks["recent_preserved"] &= torch.equal(keep[-recent:], torch.arange(length - recent, length, device=self.device))
            expected_positions.append(old_pos.index_select(1, keep))
        layer.prune(selected)
        after_lengths = layer.head_lens.tolist()
        for head, expected_pos in enumerate(expected_positions):
            checks["positions_match_kept"] &= torch.equal(layer.get_position_segment(head), expected_pos)
            target = self.prefix_tokens + min(int(self.budgets[layer_idx, head]),
                                               before_lengths[head] - self.prefix_tokens)
            if after_lengths[head] != target:
                raise AssertionError("LU attention selection did not enforce the fixed head capacity")
        if not all(checks.values()):
            raise AssertionError("LU attention pruning failed prefix/recent/position checks")
        if self.timing_enabled:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.selector_seconds += time.perf_counter() - started
        metadata.update(method="lukv", selector="visual_attention", layer_idx=layer_idx,
                        query_len=k.shape[2], before_head_lengths=before_lengths,
                        head_lengths=after_lengths, head_budgets=self.budgets[layer_idx].tolist(),
                        prefix_tokens=self.prefix_tokens, window_size=self.config.window_size,
                        kept_indices=[keep.tolist() for keep in selected], position_checks=checks,
                        compressed=any(after < before for before, after in zip(before_lengths, after_lengths)))
        if self.allocation is not None:
            metadata["allocation"] = self.allocation
        self._pending_metadata[layer_idx] = metadata
        self._appended.add(layer_idx)
