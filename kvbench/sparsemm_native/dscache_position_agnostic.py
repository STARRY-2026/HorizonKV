"""Qwen2.5-VL M-RoPE helpers for a position-agnostic KV-cache pilot.

The returned raw key is the key projection *before* M-RoPE.  Reapplying
M-RoPE at the same positions must reproduce the original cached key before
any experiment reassigns positions to a shortened active cache.
"""

import torch
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb,
)


def _mrope_key(key, cos, sin, sections):
    return apply_multimodal_rotary_pos_emb(
        key, key, cos, sin, sections,
    )[1]


def remove_mrope_from_key(model, rotated_key, position_ids, sections):
    """Undo M-RoPE for a dense key tensor shaped [1, KV heads, tokens, dim]."""
    if rotated_key.ndim != 4 or rotated_key.shape[0] != 1:
        raise ValueError('Expected one dense Qwen key batch')
    if position_ids.shape != (3, 1, rotated_key.shape[2]):
        raise ValueError('M-RoPE positions do not match key length')
    key_fp32 = rotated_key.float()
    cos, sin = model.language_model.rotary_emb(key_fp32, position_ids)
    scaling = float(model.language_model.rotary_emb.attention_scaling)
    return _mrope_key(key_fp32, cos, -sin, sections) / (scaling * scaling)


def apply_mrope_to_key(model, raw_key, position_ids, sections):
    """Assign Qwen's three-axis M-RoPE positions to an unrotated key."""
    if raw_key.ndim != 4 or raw_key.shape[0] != 1:
        raise ValueError('Expected one dense Qwen key batch')
    if position_ids.shape != (3, 1, raw_key.shape[2]):
        raise ValueError('M-RoPE positions do not match key length')
    key_fp32 = raw_key.float()
    cos, sin = model.language_model.rotary_emb(key_fp32, position_ids)
    return _mrope_key(key_fp32, cos, sin, sections)
