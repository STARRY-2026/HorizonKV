"""HF Cache protocol for native attention with bounded streaming method state.

The prototype bank is shared across all layer/head groups, so its sequential
absorption loop runs once per visual patch rather than once per decoder layer.
Attention appends layer KV first; finish_video commits the online update after
all layers have completed the patch. Query KV is a temporary view of the state.
"""

from contextlib import contextmanager

import torch
from transformers.cache_utils import Cache

from protokv.full_state import FullProtoBank
from sparsemm_baselines.cache import RaggedLayer
from sparsemm_baselines.core import select_layer
from sparsemm_protokv.config import StreamingConfig


class _StreamingLayer(RaggedLayer):
    def get_positions(self, head):
        return self.get_position_segment(head)


def _tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


def _layer_snapshot(layer):
    return (*layer.snapshot(), layer.cu_klen, layer.max_seqlen_k)


def _layer_restore(layer, saved):
    (layer.flat_k, layer.flat_v, layer.flat_pos, layer.head_lens,
     layer.cu_klen, layer.max_seqlen_k) = saved


class StreamingCache(Cache):
    def __init__(self, config, num_layers, num_kv_heads, head_dim, device, dtype):
        if not isinstance(config, StreamingConfig):
            raise TypeError("config must be StreamingConfig")
        if any(not isinstance(n, int) or isinstance(n, bool) or n < 1
               for n in (num_layers, num_kv_heads, head_dim)):
            raise ValueError("Layer, head and dimension counts must be positive integers")
        self.config = config
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = torch.device(device)
        self.dtype = dtype
        layers = [_StreamingLayer(num_kv_heads, head_dim, self.device, dtype)
                  for _ in range(num_layers)]
        super().__init__(layers=layers)
        self.phase = "init"
        self.logical_tokens = 0
        self.prefix_tokens = 0
        self.seen_visual_tokens = 0
        self.prefills = 0
        self.compressions = 0
        self.native_attention_calls = 0
        self.query_attention_calls = 0
        self.query_pseudo_attention_calls = 0
        self.bank = None
        self.raw_near_keys = torch.empty((num_layers * num_kv_heads, 0, head_dim),
                                         device=self.device, dtype=dtype)
        self.xy = torch.empty((0, 2), device=self.device, dtype=torch.float32)
        self.query_bias = [None] * num_layers
        self.last_metadata = {}
        self._positions = None
        self._context_xy = None
        self._time = 0
        self._appended = set()
        self._pending_raw = {}
        self._pending_metadata = {}
        self._context_done = True
        self._query_saved = None
        self._initialized = False
        self._context_counted = False

    @property
    def is_compileable(self):
        return False

    @property
    def is_sliding(self):
        return [False] * self.num_layers

    def get_seq_length(self, layer_idx=0):
        if not 0 <= layer_idx < self.num_layers:
            return 0
        return self.logical_tokens

    def get_max_cache_shape(self, layer_idx=0):
        return -1

    def get_mask_sizes(self, cache_position, layer_idx):
        return self.layers[layer_idx].max_seqlen_k + cache_position.shape[0], 0

    def get_usable_length(self, new_seq_length, layer_idx=0):
        return self.layers[layer_idx].max_seqlen_k

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        raise RuntimeError("StreamingCache requires the native streaming attention patch")

    def to_legacy_cache(self):
        raise RuntimeError("Legacy KV tuples cannot represent the prototype bank")

    def reorder_cache(self, beam_idx):
        if beam_idx.numel() != 1 or int(beam_idx[0]) != 0:
            raise ValueError("Native streaming plugins currently support batch=1 and num_beams=1")

    def batch_repeat_interleave(self, repeats):
        if repeats != 1:
            raise ValueError("Native streaming plugins currently support batch=1 and num_beams=1")

    def batch_select_indices(self, indices):
        self.reorder_cache(indices)

    def set_context(self, phase, positions, xy=None, time=0):
        if phase not in ("init", "video", "query"):
            raise ValueError("phase must be init, video or query")
        if not self._context_done:
            raise RuntimeError("Previous forward has not completed all layers or finish_video")
        if positions.ndim != 2 or positions.shape[0] != 3 or positions.shape[1] < 1:
            raise ValueError("positions must be [3,Q] with Q > 0")
        if not bool(torch.isfinite(positions).all()):
            raise ValueError("positions must be finite")
        if self._query_saved is not None and phase != "query":
            raise RuntimeError("A query transaction cannot mutate streaming memory")
        if phase == "query" and self._query_saved is None:
            raise RuntimeError("Use cache.query(...) before forwarding query tokens")
        if phase == "video":
            if not self._initialized:
                raise RuntimeError("finish_init must precede video processing")
            if xy is None or xy.shape != (positions.shape[1], 2):
                raise ValueError("Video positions require xy [Q,2]")
            if not bool(torch.isfinite(xy).all()):
                raise ValueError("xy must be finite")
        if not isinstance(time, int) or isinstance(time, bool) or time < 0:
            raise ValueError("time must be a nonnegative integer")
        self.phase = phase
        self._positions = positions.to(device=self.device, dtype=torch.float32)
        self._context_xy = None if xy is None else xy.to(device=self.device, dtype=torch.float32)
        self._time = time
        self._appended = set()
        self._pending_raw = {}
        self._pending_metadata = {}
        self._context_done = False
        self._context_counted = False

    @torch.inference_mode()
    def append_layer(self, layer_idx, k, v, raw_k=None, q=None):
        if self._positions is None or self._context_done:
            raise RuntimeError("set_context must precede a forward")
        if not 0 <= layer_idx < self.num_layers or layer_idx in self._appended:
            raise ValueError("Each layer must append exactly once per forward")
        expected = (1, self.num_kv_heads, self._positions.shape[1], self.head_dim)
        if k.shape != expected or v.shape != expected:
            raise ValueError(f"KV shape must be {expected}")
        if self.phase == "video" and self.config.method == "protokv":
            if raw_k is None or raw_k.shape != expected:
                raise ValueError("ProtoKV video update requires pre-RoPE raw_k")
            self._pending_raw[layer_idx] = raw_k[0]
        if self.phase == "video" and self.config.method != "protokv":
            if q is None or q.ndim != 4 or q.shape[0] != 1 or q.shape[2:] != k.shape[2:]:
                raise ValueError("SparseMM selectors require query projections [1,Hq,Q,D]")
        layer = self.layers[layer_idx]
        layer.append_chunk(k, v, self._positions)
        if self.phase == "video" and self.config.method != "protokv":
            baseline = self.config.baseline_config()
            selected, metadata = select_layer(
                [layer.get_segment(h)[0] for h in range(self.num_kv_heads)], q[0], baseline,
                layer_idx=layer_idx, num_layers=self.num_layers,
                prefix_tokens=self.prefix_tokens if baseline.budget_scope == "visual" else 0,
                seen_tokens=self.seen_visual_tokens + self._positions.shape[1] + self.prefix_tokens,
            )
            layer.prune(selected)
            self._pending_metadata[layer_idx] = metadata
        elif self.phase == "query" and self.query_bias[layer_idx] is not None:
            zeros = self.query_bias[layer_idx].new_zeros((self.num_kv_heads, k.shape[2]))
            self.query_bias[layer_idx] = torch.cat((self.query_bias[layer_idx], zeros), dim=1)
        self._appended.add(layer_idx)
        if len(self._appended) == self.num_layers and self.phase != "video":
            self._context_done = True

    def finish_init(self, prefix_tokens):
        if self.phase != "init" or not self._context_done:
            raise RuntimeError("finish_init requires a completed initial text forward")
        if self._initialized:
            raise RuntimeError("Create a fresh StreamingCache for each video/request")
        if (not isinstance(prefix_tokens, int) or isinstance(prefix_tokens, bool)
                or prefix_tokens < 1 or self._positions.shape[1] != prefix_tokens):
            raise ValueError("prefix_tokens must match the initial text KV")
        if any(layer.max_seqlen_k != prefix_tokens for layer in self.layers):
            raise ValueError("Initial cache lengths differ from prefix_tokens")
        self.prefix_tokens = prefix_tokens
        self.logical_tokens = prefix_tokens
        if self.config.method == "protokv":
            proto = self.config.proto_config(prefix_tokens)
            self.bank = FullProtoBank(self.num_layers * self.num_kv_heads, self.head_dim,
                                      self.device, proto)
        self._initialized = True
        self._context_counted = True
        self._positions = None
        self._context_xy = None

    def advance_query(self, q_len):
        """Advance the logical generation cursor once the native forward returns."""
        if (self.phase != "query" or not self._context_done or self._context_counted
                or not isinstance(q_len, int) or isinstance(q_len, bool)
                or q_len != self._positions.shape[1]):
            raise RuntimeError("advance_query requires one completed, uncounted query forward")
        self.logical_tokens += q_len
        self._context_counted = True

    @torch.inference_mode()
    def finish_video(self):
        if self.phase != "video" or len(self._appended) != self.num_layers or self._context_done:
            raise RuntimeError("finish_video requires one completed video forward at every layer")
        q_len = self._positions.shape[1]
        compressed = False
        metadata = {}
        if self.config.method == "protokv":
            appended_raw = torch.cat([self._pending_raw[i] for i in range(self.num_layers)], dim=0)
            raw = torch.cat((self.raw_near_keys, appended_raw), dim=1)
            xy = torch.cat((self.xy, self._context_xy), dim=0)
            n = raw.shape[1]
            values, positions = [], None
            for layer in self.layers:
                for head in range(self.num_kv_heads):
                    _, value = layer.get_segment(head)
                    pos = layer.get_positions(head)[:, self.prefix_tokens:]
                    if value.shape[0] != self.prefix_tokens + n:
                        raise RuntimeError("ProtoKV near KV lengths diverged across heads")
                    if positions is None:
                        positions = pos
                    elif not torch.equal(positions, pos):
                        raise RuntimeError("ProtoKV near positions diverged across heads")
                    values.append(value[self.prefix_tokens:])
            outgoing = max(0, n - self.bank.cfg.near)
            if outgoing:
                self.bank.absorb(raw, torch.stack(values), xy, positions,
                                 outgoing=outgoing, time=self._time)
                keep = torch.cat((torch.arange(self.prefix_tokens, device=self.device),
                                  self.prefix_tokens + torch.arange(outgoing, n, device=self.device)))
                for layer in self.layers:
                    layer.prune([keep] * self.num_kv_heads)
                compressed = True
            self.raw_near_keys = raw[:, outgoing:].contiguous()
            self.xy = xy[outgoing:].contiguous()
            metadata = {
                "absorbed_tokens": self.bank.total_absorbed,
                "outgoing_tokens": outgoing,
                "active_prototypes": int(self.bank.active.sum()),
                "pq_ready_groups": int(self.bank.code_ready.sum()),
                "total_groups": self.num_layers * self.num_kv_heads,
                "near_tokens": self.raw_near_keys.shape[1],
                "near_capacity": self.bank.cfg.near,
                "prototype_capacity": self.bank.cfg.capacity,
                "pseudo_tokens_per_head": self.bank.cfg.capacity * self.bank.cfg.modes,
                "visual_budget": self.bank.cfg.budget,
            }
        else:
            metadata["layers"] = [self._pending_metadata[i] for i in range(self.num_layers)]
            compressed = any(item.get("compressed", False) for item in metadata["layers"])
        self.logical_tokens += q_len
        self.seen_visual_tokens += q_len
        self.prefills += 1
        self.compressions += int(compressed)
        self._context_done = True
        self._context_counted = True
        self._pending_raw = {}
        self._pending_metadata = {}
        metadata.update({"event": "video_prefill", "prefill": self.prefills,
                         "compressions": self.compressions, "compressed": compressed,
                         "input_tokens": q_len, "logical_tokens": self.logical_tokens,
                         "head_lengths": self.head_lengths(), **self.memory_stats()})
        self.last_metadata = metadata
        # Patch-local metadata is already copied into the near state. Do not
        # retain the outgoing patch's input tensors as unreported state.
        self._positions = None
        self._context_xy = None
        return metadata

    @contextmanager
    def query(self, rotate_pseudo=None):
        if not self._initialized or not self._context_done:
            raise RuntimeError("Queries require completed initial/video state")
        if self._query_saved is not None:
            raise RuntimeError("Nested query transactions are not supported")
        saved_layers = [_layer_snapshot(layer) for layer in self.layers]
        names = ("phase", "logical_tokens", "_positions", "_context_xy", "_time",
                 "_appended", "_pending_raw", "_pending_metadata", "_context_done",
                 "_context_counted", "query_bias")
        saved = {name: getattr(self, name) for name in names}
        self._query_saved = saved_layers
        try:
            self.phase = "query"
            self.query_bias = [None] * self.num_layers
            if self.config.method == "protokv":
                if rotate_pseudo is None:
                    raise ValueError("ProtoKV query requires a pseudo-key rotary function")
                pk, pv, anchor, active, mass = self.bank.pseudo()
                pk = rotate_pseudo(pk.to(self.dtype), anchor).to(device=self.device, dtype=self.dtype)
                pv = pv.to(device=self.device, dtype=self.dtype)
                expected = (self.num_layers * self.num_kv_heads,
                            self.bank.cfg.capacity * self.bank.cfg.modes, self.head_dim)
                if pk.shape != expected or pv.shape != expected:
                    raise ValueError("Rotated pseudo KV has an unexpected shape")
                for i, layer in enumerate(self.layers):
                    keys, values, pos, biases = [], [], [], []
                    for h in range(self.num_kv_heads):
                        group = i * self.num_kv_heads + h
                        k, v = layer.get_segment(h)
                        p = layer.get_positions(h)
                        keys.append(torch.cat((k[:self.prefix_tokens], pk[group], k[self.prefix_tokens:])))
                        values.append(torch.cat((v[:self.prefix_tokens], pv[group], v[self.prefix_tokens:])))
                        pos.append(torch.cat((p[:, :self.prefix_tokens], anchor[:, group],
                                              p[:, self.prefix_tokens:]), dim=1))
                        b = torch.where(active[group], mass[group].clamp_min(1).log(),
                                        torch.full_like(mass[group], -torch.inf))
                        biases.append(torch.cat((b.new_zeros(self.prefix_tokens), b,
                                                 b.new_zeros(k.shape[0] - self.prefix_tokens))))
                    layer.flat_k = torch.cat(keys).contiguous()
                    layer.flat_v = torch.cat(values).contiguous()
                    layer.flat_pos = torch.cat(pos, dim=1).contiguous()
                    layer.head_lens = layer.head_lens.new_tensor([key.shape[0] for key in keys])
                    layer._refresh()
                    self.query_bias[i] = torch.stack(biases)
            yield self
        finally:
            for layer, state in zip(self.layers, saved_layers):
                _layer_restore(layer, state)
            for name, value in saved.items():
                setattr(self, name, value)
            self._query_saved = None

    def head_lengths(self):
        return [layer.head_lens.tolist() for layer in self.layers]

    def snapshot(self):
        return [_layer_snapshot(layer) for layer in self.layers]

    def restore(self, saved):
        if len(saved) != self.num_layers:
            raise ValueError("Snapshot layer count differs from cache")
        for layer, state in zip(self.layers, saved):
            _layer_restore(layer, state)

    def nbytes(self):
        return sum(_tensor_bytes(t) for layer in self.layers for t in (layer.flat_k, layer.flat_v))

    def state_tensors(self):
        """Persistent tensors, even while a temporary query view is attached."""
        tensors = {}
        fields = ("flat_k", "flat_v", "flat_pos", "head_lens", "cu_klen")
        for i, layer in enumerate(self.layers):
            values = (getattr(layer, field) for field in fields) if self._query_saved is None else self._query_saved[i][:5]
            tensors.update({f"layers.{i}.{field}": value for field, value in zip(fields, values)})
        tensors["raw_near_keys"] = self.raw_near_keys
        tensors["xy"] = self.xy
        if self.bank is not None:
            tensors.update({f"bank.{name}": value for name, value in vars(self.bank).items()
                            if isinstance(value, torch.Tensor)})
        return tensors

    def memory_stats(self):
        state = self.state_tensors()
        persistent_kv = sum(_tensor_bytes(t) for name, t in state.items()
                            if name.startswith("layers.") and name.endswith((".flat_k", ".flat_v")))
        bank = sum(_tensor_bytes(t) for name, t in state.items() if name.startswith("bank."))
        raw = _tensor_bytes(self.raw_near_keys)
        metadata = sum(_tensor_bytes(t) for name, t in state.items()
                       if not name.startswith("bank.") and name != "raw_near_keys"
                       and not name.endswith((".flat_k", ".flat_v")))
        return {"persistent_kv_bytes": persistent_kv, "bank_bytes": bank,
                "raw_near_key_bytes": raw, "metadata_bytes": metadata,
                "persistent_total_bytes": persistent_kv + bank + raw + metadata,
                "query_context_kv_bytes": self.nbytes() if self._query_saved is not None else 0}
