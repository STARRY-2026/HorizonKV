"""Method-specific attention patch; all decoder/MLP/norm code remains native HF.

The original SparseMM replaces Qwen2-VL FlashAttention2.forward. Transformers
4.57 uses Qwen2_5_VLAttention instead. Only our explicit StreamingCache opts in;
ordinary HF caches continue to use the original forward implementation.
"""

import torch
import torch.nn.functional as F
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLAttention, apply_multimodal_rotary_pos_emb,
)

from sparsemm_baselines.adapter import ragged_attention

_ORIGINAL = Qwen2_5_VLAttention.forward


def biased_attention(q, k, v, layer, bias, scaling):
    """GQA with explicit bottom-right causality and per-KV-head log mass."""
    _, query_heads, q_len, _ = q.shape
    kv_heads = k.shape[1]
    if query_heads % kv_heads:
        raise ValueError("Nonintegral grouped-query attention ratio")
    lengths = layer.head_lens.tolist()
    if len(set(lengths)) != 1 or bias.shape != (kv_heads, lengths[0]):
        raise ValueError("ProtoKV query cache/bias length mismatch")
    keys = torch.stack([torch.cat((layer.get_segment(h)[0], k[0, h])) for h in range(kv_heads)])[None]
    values = torch.stack([torch.cat((layer.get_segment(h)[1], v[0, h])) for h in range(kv_heads)])[None]
    old = lengths[0]
    total = old + q_len
    key_index = torch.arange(total, device=q.device)
    query_index = old + torch.arange(q_len, device=q.device)
    causal = key_index[None] <= query_index[:, None]
    padded_bias = F.pad(bias.float(), (0, q_len))
    mask = padded_bias.repeat_interleave(query_heads // kv_heads, dim=0)[:, None, :].expand(-1, q_len, -1)
    mask = mask.masked_fill(~causal, -torch.inf)[None]
    output = F.scaled_dot_product_attention(
        q, keys, values, attn_mask=mask, dropout_p=0.0,
        is_causal=False, scale=scaling, enable_gqa=True,
    )
    return output.transpose(1, 2).reshape(1, q_len, -1)


def attention_forward(self, hidden_states, attention_mask=None, position_ids=None,
                      past_key_values=None, output_attentions=False, use_cache=False,
                      cache_position=None, position_embeddings=None, **kwargs):
    from sparsemm_protokv.cache import StreamingCache
    if not isinstance(past_key_values, StreamingCache):
        return _ORIGINAL(self, hidden_states, attention_mask=attention_mask,
                         position_ids=position_ids, past_key_values=past_key_values,
                         output_attentions=output_attentions, use_cache=use_cache,
                         cache_position=cache_position, position_embeddings=position_embeddings, **kwargs)
    cache = past_key_values
    if self.training or hidden_states.shape[0] != 1 or not use_cache:
        raise ValueError("SparseMM streaming plugins require inference, batch_size=1 and use_cache=True")
    if attention_mask is not None:
        raise ValueError("Streaming plugins construct their own causal mask; padding masks are unsupported")
    if position_embeddings is None:
        raise ValueError("Native Qwen must supply explicit M-RoPE position embeddings")
    bsz, count, _ = hidden_states.shape
    q = self.q_proj(hidden_states).view(bsz, count, -1, self.head_dim).transpose(1, 2)
    raw_k = self.k_proj(hidden_states).view(bsz, count, -1, self.head_dim).transpose(1, 2)
    v = self.v_proj(hidden_states).view(bsz, count, -1, self.head_dim).transpose(1, 2)
    if not all(torch.isfinite(t).all().item() for t in (q, raw_k, v)):
        raise FloatingPointError("Non-finite native Qwen Q/K/V projection")
    q, k = apply_multimodal_rotary_pos_emb(q, raw_k, *position_embeddings, self.rope_scaling["mrope_section"])
    layer = cache.layers[self.layer_idx]
    cache.native_attention_calls = getattr(cache, "native_attention_calls", 0) + 1
    if cache.phase == "query":
        cache.query_attention_calls = getattr(cache, "query_attention_calls", 0) + 1
    if cache.config.method == "protokv" and cache.phase == "query":
        bias = cache.query_bias[self.layer_idx]
        attended = biased_attention(q, k, v, layer, bias, self.scaling)
        if cache.bank is not None and bool(cache.bank.active.any()):
            cache.query_pseudo_attention_calls = getattr(cache, "query_pseudo_attention_calls", 0) + 1
    else:
        attended = ragged_attention(q, k, v, layer, self.scaling)
    cache.append_layer(self.layer_idx, k, v, raw_k=raw_k, q=q)
    return self.o_proj(attended.contiguous()), None


def install():
    """Idempotent patch; method and memory state live in each cache/request."""
    Qwen2_5_VLAttention.forward = attention_forward


def uninstall():
    Qwen2_5_VLAttention.forward = _ORIGINAL
