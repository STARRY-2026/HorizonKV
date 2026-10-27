"""Native Qwen2 attention bridge for the LLaVA-OneVision text backbone."""

import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, apply_rotary_pos_emb

from sparsemm_baselines.adapter import ragged_attention
from sparsemm_protokv.qwen2_5 import biased_attention

_ORIGINAL = Qwen2Attention.forward


def attention_forward(self, hidden_states, position_embeddings, attention_mask,
                      past_key_values=None, cache_position=None, **kwargs):
    from sparsemm_protokv.cache import StreamingCache

    if not isinstance(past_key_values, StreamingCache):
        return _ORIGINAL(self, hidden_states, position_embeddings, attention_mask,
                         past_key_values=past_key_values, cache_position=cache_position,
                         **kwargs)
    cache = past_key_values
    if self.training or hidden_states.shape[0] != 1:
        raise ValueError("KVBench LLaVA streaming requires inference and batch_size=1")
    if attention_mask is not None:
        raise ValueError("KVBench constructs its own causal mask")
    batch, count, _ = hidden_states.shape
    q = self.q_proj(hidden_states).view(batch, count, -1, self.head_dim).transpose(1, 2)
    raw_k = self.k_proj(hidden_states).view(batch, count, -1, self.head_dim).transpose(1, 2)
    v = self.v_proj(hidden_states).view(batch, count, -1, self.head_dim).transpose(1, 2)
    if not all(torch.isfinite(t).all().item() for t in (q, raw_k, v)):
        raise FloatingPointError("Non-finite native LLaVA Q/K/V projection")
    q, k = apply_rotary_pos_emb(q, raw_k, *position_embeddings)
    layer = cache.layers[self.layer_idx]
    cache.native_attention_calls += 1
    if cache.phase == "query":
        cache.query_attention_calls += 1
    if cache.config.method == "protokv" and cache.phase == "query":
        attended = biased_attention(q, k, v, layer, cache.query_bias[self.layer_idx], self.scaling)
        if cache.bank is not None and bool(cache.bank.active.any()):
            cache.query_pseudo_attention_calls += 1
    else:
        attended = ragged_attention(q, k, v, layer, self.scaling)
    cache.append_layer(self.layer_idx, k, v, raw_k=raw_k, q=q)
    return self.o_proj(attended.contiguous()), None


def install():
    Qwen2Attention.forward = attention_forward


def uninstall():
    Qwen2Attention.forward = _ORIGINAL
