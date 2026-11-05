"""Native streaming cache for SparseMM's calibrated static head quotas."""

import time

import torch

from sparsemm_native.policy import select_layer
from sparsemm_protokv.cache import StreamingCache


class SparseMMCache(StreamingCache):
    def __init__(self, *args, budgets=None, allocation=None, asset_metadata=None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.config.method != "sparsemm":
            raise ValueError("SparseMMCache requires method=sparsemm")
        if (not isinstance(budgets, torch.Tensor)
                or budgets.shape != (self.num_layers, self.num_kv_heads)
                or budgets.dtype not in (torch.int32, torch.int64)
                or bool((budgets < self.config.window_size).any())):
            raise ValueError("SparseMM static quotas must match layers/KV heads and protect the recent window")
        self.budgets = budgets.detach().cpu().clone()
        self.allocation = allocation or {}
        self.asset_metadata = asset_metadata or {}
        self.selector_seconds = 0.0
        self.timing_enabled = False

    @torch.inference_mode()
    def append_layer(self, layer_idx, k, v, raw_k=None, q=None):
        if self.phase != "video":
            return super().append_layer(layer_idx, k, v, raw_k=raw_k, q=q)
        if self._positions is None or self._context_done:
            raise RuntimeError("set_context must precede a native SparseMM video forward")
        if not 0 <= layer_idx < self.num_layers or layer_idx in self._appended:
            raise ValueError("Each native layer must append exactly once")
        expected = (1, self.num_kv_heads, self._positions.shape[1], self.head_dim)
        if k.shape != expected or v.shape != expected:
            raise ValueError(f"SparseMM KV shape must be {expected}")
        if q is None or q.ndim != 4 or q.shape[0] != 1 or q.shape[2:] != k.shape[2:]:
            raise ValueError("SparseMM selection requires current native visual queries")
        layer = self.layers[layer_idx]
        layer.append_chunk(k, v, self._positions)
        before = layer.head_lens.tolist()
        if self.timing_enabled and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        selected, metadata = select_layer(
            [layer.get_segment(h)[0] for h in range(self.num_kv_heads)], q[0],
            self.budgets[layer_idx], prefix_tokens=self.prefix_tokens,
            budget=self.config.budget, window_size=self.config.window_size,
            kernel_size=self.config.kernel_size, pooling=self.config.pooling,
            gqa_reduce=self.config.gqa_reduce, layer_idx=layer_idx,
        )
        checks = dict(prefix_preserved=True, recent_preserved=True,
                      indices_unique=True, indices_in_range=True, positions_match_kept=True)
        expected_positions = []
        for head, keep in enumerate(selected):
            length = before[head]
            recent = min(self.config.window_size, length - self.prefix_tokens)
            checks["indices_unique"] &= keep.unique().numel() == keep.numel()
            checks["indices_in_range"] &= (not keep.numel()
                                           or (int(keep.min()) >= 0 and int(keep.max()) < length))
            checks["prefix_preserved"] &= torch.equal(
                keep[:self.prefix_tokens], torch.arange(self.prefix_tokens, device=self.device))
            if recent:
                checks["recent_preserved"] &= torch.equal(
                    keep[-recent:], torch.arange(length - recent, length, device=self.device))
            expected_positions.append(layer.get_position_segment(head).index_select(1, keep))
        layer.prune(selected)
        after = layer.head_lens.tolist()
        bypass = metadata["bypassed_short_cache"]
        for head, expected_pos in enumerate(expected_positions):
            checks["positions_match_kept"] &= torch.equal(layer.get_position_segment(head), expected_pos)
            visual = before[head] - self.prefix_tokens
            recent = min(self.config.window_size, visual)
            history = min(visual - recent, int(self.budgets[layer_idx, head]) - self.config.window_size)
            expected_length = before[head] if bypass else self.prefix_tokens + history + recent
            if after[head] != expected_length:
                raise AssertionError("SparseMM pruning disagrees with its global gate/static head quota")
        if not all(checks.values()):
            raise AssertionError("SparseMM pruning failed prefix/recent/index/position validation")
        if self.timing_enabled:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.selector_seconds += time.perf_counter() - started
        metadata.update(
            query_len=k.shape[2], prefix_tokens=self.prefix_tokens,
            before_head_lengths=before, head_lengths=after,
            head_budgets=self.budgets[layer_idx].tolist(),
            kept_indices=[keep.tolist() for keep in selected], position_checks=checks,
            asset=self.asset_metadata, allocation=self.allocation,
            compressed=any(a < b for a, b in zip(after, before)),
        )
        self._pending_metadata[layer_idx] = metadata
        self._appended.add(layer_idx)
