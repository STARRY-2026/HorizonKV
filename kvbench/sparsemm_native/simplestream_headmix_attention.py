"""Per-KV-head direct frame visibility for the dynamic-resolution SimpleStream run.

Eleven heads cannot directly attend the oldest two of four image frames while
the other heads keep the released FlashAttention path. This does not promise
complete semantic independence: residual streams mix information across heads.
"""

import contextvars
import csv
import hashlib
from pathlib import Path


SOURCE_TABLE = Path(__file__).resolve().parents[2] / (
    'hermes/results/myobservations/two_bank_grid_dp_k500_floor375_recent32_20261020/'
    'per_head_allocation_fixed_capacity.csv')
SOURCE_SHA256 = hashlib.sha256(SOURCE_TABLE.read_bytes()).hexdigest()


def select_heads():
    with SOURCE_TABLE.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    candidates = [r for r in rows if int(r['recent_r']) == 32]
    if len(rows) != 112 or len(candidates) != 19:
        raise ValueError('Old DP allocation table/candidate set changed')
    chosen = sorted(candidates, key=lambda r: (-int(r['history_q']), int(r['layer']), int(r['kv_head'])))[:11]
    return tuple((int(r['layer']), int(r['kv_head'])) for r in chosen)


HEADS = select_heads()
BY_LAYER = {layer: tuple(head for l, head in HEADS if l == layer) for layer in sorted({l for l, _ in HEADS})}
_visibility = contextvars.ContextVar('simplestream_headmix_visibility', default=None)
_installed = False


def install():
    """Patch only language FlashAttention layers with selected KV heads."""
    global _installed
    if _installed:
        return
    import torch
    import torch.nn.functional as F
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as m
    from transformers.modeling_flash_attention_utils import _flash_attention_forward

    attention_class = m.Qwen2_5_VLFlashAttention2
    original_forward = attention_class.forward

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, position_embeddings=None):
        layout = _visibility.get()
        local_heads = BY_LAYER.get(self.layer_idx, ())
        if layout is None or not local_heads:
            return original_forward(self, hidden_states, attention_mask, position_ids,
                                    past_key_value, output_attentions, use_cache,
                                    cache_position, position_embeddings)
        bsz, q_len, _ = hidden_states.size()
        q = self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = m.apply_multimodal_rotary_pos_emb(q, k, cos, sin, self.rope_scaling['mrope_section'])
        if past_key_value is not None:
            cache_kwargs = {'sin': sin, 'cos': cos, 'cache_position': cache_position}
            k, v = past_key_value.update(k, v, self.layer_idx, cache_kwargs)
        k = m.repeat_kv(k, self.num_key_value_groups)
        v = m.repeat_kv(v, self.num_key_value_groups)
        if q.dtype == torch.float32:
            target = (torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else
                      getattr(self.config, '_pre_quantization_dtype', self.q_proj.weight.dtype))
            q, k, v = q.to(target), k.to(target), v.to(target)
        dropout = 0.0 if not self.training else self.attention_dropout
        sliding = (self.config.sliding_window if self.config.use_sliding_window
                   and getattr(self.config, 'sliding_window', None) is not None
                   and self.layer_idx >= self.config.max_window_layers else None)
        output = _flash_attention_forward(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attention_mask, q_len, dropout=dropout, sliding_window=sliding,
            is_causal=self.is_causal, use_top_left_mask=self._flash_attn_uses_top_left_mask)
        key_len = k.shape[2]
        q_index = torch.arange(key_len - q_len, key_len, device=q.device)
        k_index = torch.arange(key_len, device=q.device)
        allow = k_index[None, :] <= q_index[:, None]
        old_start, recent_start, audit = layout
        old_keys = (k_index >= old_start) & (k_index < recent_start)
        allow &= ~((q_index[:, None] >= recent_start) & old_keys[None, :])
        if sliding is not None:
            allow &= k_index[None, :] >= q_index[:, None] - sliding + 1
        if attention_mask is not None:
            if attention_mask.ndim != 2 or bsz != 1:
                raise RuntimeError('Unexpected nontrivial upstream FlashAttention mask')
            allow &= attention_mask[0, :key_len].bool()[None, :]
        mask = allow[None, None]
        for kv_head in local_heads:
            lo = kv_head * self.num_key_value_groups
            hi = lo + self.num_key_value_groups
            replacement = F.scaled_dot_product_attention(
                q[:, lo:hi], k[:, lo:hi], v[:, lo:hi], attn_mask=mask,
                dropout_p=dropout, is_causal=False, scale=self.head_dim ** -0.5)
            output[:, :, lo:hi] = replacement.transpose(1, 2)
        output = self.o_proj(output.reshape(bsz, q_len, -1).contiguous())
        audit['calls'] += len(local_heads)
        audit['layers'].add(self.layer_idx)
        return output, None, past_key_value

    # A context is set only for the current question. Other code paths,
    # including image encoding, execute the original method.
    attention_class.forward = forward
    _installed = True


def install_question_wrapper(output):
    """Set absolute visual token spans for each released Qwen generation."""
    import json
    from lib import recent_window_eval_qwen25 as upstream
    from sparsemm_native.supervise_simplestream_upstream import append

    original = upstream.RecentWindowQAModel.generate_with_cached_vision

    def generate(self, cached_embeds, cached_grid_thw, question):
        counts = (cached_grid_thw.prod(dim=-1) // self.merge_size ** 2).tolist()
        if len(counts) > 4 or sum(counts) != int(cached_embeds.shape[0]):
            raise ValueError('Unexpected dynamic-resolution per-frame grid')
        prefix = 1 + len(self.processor.tokenizer.encode('user\n', add_special_tokens=False)) + 1
        old_start = prefix
        recent_start = prefix + sum(counts[:-2])
        audit = {'calls': 0, 'layers': set()}
        token = _visibility.set((old_start, recent_start, audit) if len(counts) > 2 else None)
        try:
            answer = original(self, cached_embeds, cached_grid_thw, question)
        finally:
            _visibility.reset(token)
        if len(counts) > 2 and (audit['calls'] < 11 or audit['layers'] != set(BY_LAYER)):
            raise RuntimeError('Eleven-head visibility mask did not cover every selected layer')
        append(output / 'headmix_query_audit.jsonl', dict(
            prompt_sha256=hashlib.sha256(question.encode()).hexdigest(),
            frame_visual_tokens=counts, masked_old_token_span=[old_start, recent_start],
            selected_heads=[list(item) for item in HEADS],
            masked_attention_calls=audit['calls'], layers=sorted(audit['layers']),
            direct_old_frame_visibility_blocked=len(counts) > 2))
        return answer

    upstream.RecentWindowQAModel.generate_with_cached_vision = generate
