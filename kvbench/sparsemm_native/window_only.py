"""Strict streaming visual KV window: two Qwen temporal units, no history.

Outgoing units are removed BEFORE the next unit enters any decoder layer.
The text prefix and absolute M-RoPE positions are preserved. Retained KV is
never recomputed, so representations can still contain earlier context.
"""
from dataclasses import dataclass

import torch

from sparsemm_protokv.cache import StreamingCache
from sparsemm_protokv.streaming import StreamingQwen


@dataclass
class WindowUnit:
    positions: torch.Tensor
    frame_start: int
    real_frames: int

    @property
    def tokens(self):
        return int(self.positions.shape[1])


class WindowOnlyCache(StreamingCache):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = []
        self.window_tokens = 0
        self.dropped_units = 0
        self.dropped_tokens_per_head = 0
        self._removed_this_unit = 0

    @torch.inference_mode()
    def begin_unit(self, positions, frame_start, real_frames):
        if not self._initialized or not self._context_done or self._query_saved is not None:
            raise RuntimeError('A new visual unit requires completed persistent video state')
        if positions.ndim != 2 or positions.shape[0] != 3 or positions.shape[1] < 1:
            raise ValueError('One nonempty [3,tokens] temporal unit is required')
        if type(frame_start) is not int or frame_start < 0 or real_frames not in (1, 2):
            raise ValueError('Invalid source frame descriptor')
        if self.units and frame_start <= self.units[-1].frame_start:
            raise ValueError('Video frame order must advance')
        old_tokens = sum(unit.tokens for unit in self.units)
        expected = self.prefix_tokens + old_tokens
        if any(layer.max_seqlen_k != expected
               or layer.flat_k.shape[0] != self.num_kv_heads * expected
               or not bool((layer.head_lens == expected).all()) for layer in self.layers):
            raise AssertionError('Previous persistent window has divergent per-head lengths')
        retained = self.units[-1:]
        retain_tokens = sum(unit.tokens for unit in retained)
        removed = old_tokens - retain_tokens
        if removed:
            # Every head has the same chronological window. This copies only
            # prefix and retained slices; no score, sort, or calibration data.
            p = self.prefix_tokens
            length = p + retain_tokens
            for layer in self.layers:
                key = layer.flat_k.reshape(self.num_kv_heads, expected, self.head_dim)
                value = layer.flat_v.reshape(self.num_kv_heads, expected, self.head_dim)
                pos = layer.flat_pos.reshape(3, self.num_kv_heads, expected)
                layer.flat_k = torch.cat((key[:, :p], key[:, -retain_tokens:]), dim=1).reshape(-1, self.head_dim).contiguous()
                layer.flat_v = torch.cat((value[:, :p], value[:, -retain_tokens:]), dim=1).reshape(-1, self.head_dim).contiguous()
                layer.flat_pos = torch.cat((pos[:, :, :p], pos[:, :, -retain_tokens:]), dim=2).reshape(3, -1).contiguous()
                layer.head_lens = layer.head_lens.new_full((self.num_kv_heads,), length)
                layer._refresh()
            self.dropped_units += len(self.units) - len(retained)
            self.dropped_tokens_per_head += removed
        self.units = retained + [WindowUnit(positions.detach().to(self.device, torch.float32).clone(),
                                             frame_start, real_frames)]
        self.window_tokens = sum(unit.tokens for unit in self.units)
        self._removed_this_unit = removed

    @torch.inference_mode()
    def append_layer(self, layer_idx, k, v, raw_k=None, q=None):
        if self.phase != 'video':
            return super().append_layer(layer_idx, k, v, raw_k=raw_k, q=q)
        if self._positions is None or self._context_done or layer_idx in self._appended:
            raise RuntimeError('Each visual layer must append exactly once')
        expected_shape = (1, self.num_kv_heads, self._positions.shape[1], self.head_dim)
        if k.shape != expected_shape or v.shape != expected_shape or not self.units:
            raise ValueError('Visual KV does not match the declared temporal unit')
        if not torch.equal(self._positions, self.units[-1].positions):
            raise AssertionError('The current visual positions do not match the window')
        layer = self.layers[layer_idx]
        old = layer.max_seqlen_k
        if old - self.prefix_tokens + k.shape[2] != self.window_tokens:
            raise AssertionError('The visual attention would exceed the two-unit window')
        layer.append_chunk(k, v, self._positions)
        expected = self.prefix_tokens + self.window_tokens
        if layer.max_seqlen_k != expected or not bool((layer.head_lens == expected).all()):
            raise AssertionError('A head retained KV outside its declared recent window')
        self._pending_metadata[layer_idx] = dict(
            method='recent_four_streaming_window_only', layer_idx=layer_idx,
            compressed=self._removed_this_unit > 0,
            evicted_tokens_per_head=self._removed_this_unit,
            retained_visual_tokens_per_head=self.window_tokens,
            before_head_lengths=[old + k.shape[2] + self._removed_this_unit] * self.num_kv_heads,
            head_lengths=[expected] * self.num_kv_heads,
            history_tokens_per_head=0, attention_scoring=False,
            eviction_timing='before_new_visual_attention',
        )
        self._appended.add(layer_idx)

    def state_tensors(self):
        state = super().state_tensors()
        for index, unit in enumerate(self.units):
            state[f'window_unit.{index}.positions'] = unit.positions
        return state

    def audit_window(self):
        if not self.units or len(self.units) > 2 or not self._context_done:
            raise AssertionError('A question requires one or two completed visual units')
        p, r = self.prefix_tokens, self.window_tokens
        prefix = torch.arange(p, dtype=torch.float32, device=self.device)[None].expand(3, -1)
        positions = torch.cat((prefix, *(unit.positions for unit in self.units)), dim=1)
        counts = []
        for layer in self.layers:
            if (not bool((layer.head_lens == p + r).all())
                    or layer.flat_k.shape[0] != self.num_kv_heads * (p + r)
                    or layer.flat_v.shape != layer.flat_k.shape
                    or not torch.equal(layer.flat_pos.reshape(3, self.num_kv_heads, p + r),
                                       positions[:, None].expand(-1, self.num_kv_heads, -1))):
                raise AssertionError('Per-head prefix/window positions or physical KV counts disagree')
            counts.append([r] * self.num_kv_heads)
        return dict(
            recent_visual_tokens_per_head=r, actual_per_head_visual_counts=counts,
            actual_visual_slots=r * self.num_layers * self.num_kv_heads,
            history_visual_tokens=0, prefix_tokens_per_head=p,
            recent_unit_tokens=[unit.tokens for unit in self.units],
            recent_frame_starts=[unit.frame_start for unit in self.units],
            recent_real_frames=sum(unit.real_frames for unit in self.units),
            recent_padded_frames=2 * len(self.units) - sum(unit.real_frames for unit in self.units),
            dropped_units=self.dropped_units, dropped_tokens_per_head=self.dropped_tokens_per_head,
            no_old_visual_positions=True, all_heads_same_window=True,
            eviction_timing='before_new_visual_attention',
        )


class WindowOnlyStreamingQwen(StreamingQwen):
    def __init__(self, model, processor, config, *args, **kwargs):
        if (model.config.model_type != 'qwen2_5_vl'
                or model.config.vision_config.temporal_patch_size != 2
                or config.method != 'sparsemm' or config.sparsemm_cache_policy != 'capacity'
                or config.sparsemm_budget_path):
            raise ValueError('Window-only requires Qwen2.5 two-frame units and no external quotas')
        super().__init__(model, processor, config, *args, **kwargs)

    def clear_cache(self):
        self.kv_cache = WindowOnlyCache(self.config, self.num_layers, self.num_kv_heads,
                                       self.head_dim, self.device, self.dtype)
        self.visual_start_idx = 0
        self._next_position = self._seen_visual_tokens = self.total_processed_frames = 0
        self.last_encoded_frames = self._prefills = self._compressions = 0
        self.conv_history, self.last_metadata = [], []

    @torch.inference_mode()
    def forward_chunk(self, inputs_embeds, positions, compress=True, **kwargs):
        if compress:
            # encode_video_chunk sets time to the number of real input frames,
            # so an odd endpoint remains one real frame plus Qwen padding.
            real_frames = int(kwargs.get('time', self.total_processed_frames + 2)) - self.total_processed_frames
            self.kv_cache.begin_unit(positions[:, 0], self.total_processed_frames, real_frames)
        return super().forward_chunk(inputs_embeds, positions, compress=compress, **kwargs)


def load_model(model_path, sample_fps=1.0, video_max_pixels=151200, **kwargs):
    from sparsemm_protokv.config import StreamingConfig
    from sparsemm_protokv.model_loader import load_pretrained
    model, processor = load_pretrained(model_path, device='cuda', dtype='bfloat16')
    # The shared bridge's dispatch tag enables the existing native Qwen path.
    # This cache never calls its SparseMM quota loader/attention selector; 500
    # is an unused compatibility field, not a target or maximum for the window.
    config = StreamingConfig(method='sparsemm', budget=500, budget_scope='visual')
    return WindowOnlyStreamingQwen(model, processor, config, sample_fps=sample_fps,
                                   video_max_pixels=video_max_pixels, **kwargs)
