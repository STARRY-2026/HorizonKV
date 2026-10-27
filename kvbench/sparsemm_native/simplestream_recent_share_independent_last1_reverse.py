"""Reverse the recent-share split for independent one/four-frame prefixes.

The highest 50% recent-share heads use K/V from a prefix containing only the
last available image; the others use K/V from the full four-image prefix.
The question is prefetched once against this combined ragged cache. The
single-frame branch never receives older image embeddings during its prefill.
"""

import contextvars
import hashlib
import sys
import time
import weakref
from pathlib import Path

import torch
from transformers import DynamicCache
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb

from sparsemm_native.simplestream_recent_share_attention import (
    HEADS as LOW_SHARE_HEADS, SOURCE_SHA256, SOURCE_TABLE,
)

HEADS = tuple((layer, head) for layer in range(28) for head in range(4)
              if (layer, head) not in set(LOW_SHARE_HEADS))
if len(HEADS) != 56 or set(HEADS) & set(LOW_SHARE_HEADS):
    raise RuntimeError('Expected complementary 56 high-share one-frame heads')


HERMES_ROOT = Path(__file__).resolve().parents[2] / 'hermes'
RAGGED_SOURCE = HERMES_ROOT / 'head_analysis/hermes_kv_head_ragged.py'
RAGGED_SHA256 = hashlib.sha256(RAGGED_SOURCE.read_bytes()).hexdigest()
_question = contextvars.ContextVar('physical_recent_share_question', default=None)


def single_frame_position_indices(visual_end, visual_start, last_frame_start):
    """Indices for text prefix plus the final image only."""
    if not 0 <= visual_start <= last_frame_start < visual_end:
        raise ValueError('Invalid image token boundaries')
    return list(range(visual_start)) + list(range(last_frame_start, visual_end))


def _ragged_helpers():
    if str(HERMES_ROOT) not in sys.path:
        sys.path.insert(0, str(HERMES_ROOT))
    from head_analysis.hermes_kv_head_ragged import (
        HeadRaggedLayerCache, HeadRaggedCache, _ragged_language_decode_step,
    )
    return HeadRaggedLayerCache, HeadRaggedCache, _ragged_language_decode_step


def _combine_prefix_caches(full_dense, one_dense, one_frame_heads, Layer):
    """Copy KV from genuinely separate prefix passes into a per-head cache."""
    layers = []
    for layer_idx, ((full_k, full_v), (one_k, one_v)) in enumerate(zip(full_dense, one_dense)):
        if full_k.shape[0:2] != (1, 4) or one_k.shape[0:2] != (1, 4):
            raise RuntimeError('Unexpected dense K/V dimensions')
        k_parts = []
        v_parts = []
        lengths = []
        for head in range(4):
            use_one = (layer_idx, head) in one_frame_heads
            chosen_k, chosen_v = (one_k, one_v) if use_one else (full_k, full_v)
            k_parts.append(chosen_k[0, head])
            v_parts.append(chosen_v[0, head])
            lengths.append(chosen_k.shape[2])
        layers.append(Layer(torch.cat(k_parts, dim=0), torch.cat(v_parts, dim=0),
                            torch.tensor(lengths, device=full_k.device, dtype=torch.int32)))
    return layers


def _prompt_ids(qa, visual_tokens, question):
    tokenizer = qa.processor.tokenizer
    return ([qa._im_start_id] + tokenizer.encode('user\n', add_special_tokens=False)
            + [qa._vision_start_id] + [qa.image_token_id] * visual_tokens
            + [qa._vision_end_id] + tokenizer.encode('\n', add_special_tokens=False)
            + tokenizer.encode(question, add_special_tokens=False)
            + [qa._im_end_id] + tokenizer.encode('\n', add_special_tokens=False)
            + [qa._im_start_id] + tokenizer.encode('assistant\n', add_special_tokens=False))


def _apply_repetition_penalty(logits, ids, factor):
    if factor == 1.0:
        return logits
    indices = torch.as_tensor(sorted(set(ids)), device=logits.device, dtype=torch.long)
    selected = logits.index_select(0, indices)
    selected = torch.where(selected < 0, selected * factor, selected / factor)
    return logits.scatter(0, indices, selected)


def _prefill_question_suffix(model, embeds, position_ids, ragged):
    """Prefill question tokens against physically pruned per-head visual K/V."""
    from flash_attn import flash_attn_varlen_func

    hidden = embeds
    q_len = hidden.shape[1]
    if hidden.shape[0] != 1 or q_len < 1:
        raise ValueError('Expected a nonempty single-question suffix')
    for layer_idx, decoder_layer in enumerate(model.language_model.layers):
        residual = hidden
        normed = decoder_layer.input_layernorm(hidden)
        attn = decoder_layer.self_attn
        n_kv = attn.num_key_value_heads
        group = attn.num_key_value_groups
        dim = attn.head_dim
        q = attn.q_proj(normed).view(1, q_len, n_kv * group, dim).transpose(1, 2)
        k = attn.k_proj(normed).view(1, q_len, n_kv, dim).transpose(1, 2)
        v = attn.v_proj(normed).view(1, q_len, n_kv, dim).transpose(1, 2)
        cos, sin = model.language_model.rotary_emb(normed, position_ids)
        q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, attn.rope_scaling['mrope_section'])
        layer_cache = ragged.layers[layer_idx]
        layer_cache.append_chunk(k, v)
        query = q.view(1, n_kv, group, q_len, dim)[0].permute(0, 2, 1, 3).reshape(
            n_kv * q_len, group, dim).contiguous()
        cu_q = torch.arange(0, (n_kv + 1) * q_len, q_len, dtype=torch.int32, device=q.device)
        attended = flash_attn_varlen_func(
            query, layer_cache.flat_k.view(-1, 1, dim).contiguous(),
            layer_cache.flat_v.view(-1, 1, dim).contiguous(),
            cu_q, layer_cache.cu_klen, max_seqlen_q=q_len,
            max_seqlen_k=layer_cache.max_seqlen_k, causal=True)
        attended = attended.view(n_kv, q_len, group, dim).permute(1, 0, 2, 3).reshape(
            1, q_len, n_kv * group * dim)
        hidden = residual + attn.o_proj(attended.contiguous())
        residual = hidden
        hidden = decoder_layer.post_attention_layernorm(hidden)
        hidden = residual + decoder_layer.mlp(hidden)
    return model.language_model.norm(hidden)


def install(output, *, validation_keep_all=False):
    """Replace only the question decoder; preserve upstream image/prompt work."""
    from lib import recent_window_eval_qwen25 as upstream
    from sparsemm_native.supervise_simplestream_upstream import append

    original_question = upstream.RecentWindowQAModel.generate_with_cached_vision
    original_generate = upstream.RecentWindowQAModel._generate_from_model_inputs
    one_frame_heads = set() if validation_keep_all else set(HEADS)
    Layer, Cache, decode_step = _ragged_helpers()

    def question(self, cached_embeds, cached_grid_thw, prompt):
        counts = (cached_grid_thw.prod(dim=-1) // self.merge_size ** 2).tolist()
        if len(counts) > 4 or sum(counts) != int(cached_embeds.shape[0]):
            raise ValueError('Unexpected image-token layout')
        old_start = 1 + len(self.processor.tokenizer.encode('user\n', add_special_tokens=False)) + 1
        last_frame_start = old_start + sum(counts[:-1])
        ids = _prompt_ids(self, sum(counts), prompt)
        token = _question.set(dict(counts=counts, old_start=old_start,
                                   last_frame_start=last_frame_start,
                                   visual_end=old_start + sum(counts), ids=ids,
                                   prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest()))
        try:
            return original_question(self, cached_embeds, cached_grid_thw, prompt)
        finally:
            _question.reset(token)

    @torch.inference_mode()
    def generate(self, prompt_length, **kwargs):
        layout = _question.get()
        if layout is None:
            return original_generate(self, prompt_length, **kwargs)
        if len(layout['ids']) != prompt_length:
            raise RuntimeError('Prompt token construction differs from upstream')
        model = self.model
        full_dense = DynamicCache()
        started = time.perf_counter()
        visual_end = layout['visual_end']
        full_outputs = model(inputs_embeds=kwargs['inputs_embeds'][:, :visual_end],
                             attention_mask=kwargs['attention_mask'][:, :visual_end],
                             position_ids=kwargs['position_ids'][:, :, :visual_end],
                             past_key_values=full_dense, use_cache=True, return_dict=True)
        del full_outputs
        if len(full_dense) != len(model.language_model.layers):
            raise RuntimeError('Incomplete four-frame prefill cache')
        one_indices = single_frame_position_indices(
            visual_end, layout['old_start'], layout['last_frame_start'])
        independent_pass = bool(one_frame_heads and len(layout['counts']) > 1)
        if independent_pass:
            index = torch.as_tensor(one_indices, device=kwargs['inputs_embeds'].device,
                                    dtype=torch.long)
            one_dense = DynamicCache()
            one_outputs = model(
                inputs_embeds=kwargs['inputs_embeds'].index_select(1, index),
                attention_mask=kwargs['attention_mask'].index_select(1, index),
                position_ids=kwargs['position_ids'].index_select(2, index),
                past_key_values=one_dense, use_cache=True, return_dict=True)
            del one_outputs
            if len(one_dense) != len(full_dense):
                raise RuntimeError('Incomplete independent one-frame prefill cache')
        else:
            one_dense = full_dense
        if any(full_dense[i][0].shape[2] != visual_end or
               one_dense[i][0].shape[2] != len(one_indices)
               for i in range(len(full_dense))):
            raise RuntimeError('Prefix K/V lengths differ from the requested frames')
        full_key_ref = weakref.ref(full_dense[-1][0])
        one_key_ref = weakref.ref(one_dense[-1][0])
        layers = _combine_prefix_caches(full_dense, one_dense, one_frame_heads, Layer)
        ragged = Cache(layers)
        del full_dense, one_dense
        if full_key_ref() is not None or one_key_ref() is not None:
            raise RuntimeError('Dense visual K/V still live after branch combination')
        hidden = _prefill_question_suffix(
            model, kwargs['inputs_embeds'][:, visual_end:],
            kwargs['position_ids'][:, :, visual_end:], ragged)
        logits = model.lm_head(hidden)[0, -1]
        expected_low = prompt_length - (layout['last_frame_start'] - layout['old_start'])
        lens = [layer.head_lens.tolist() for layer in ragged.layers]
        if any(length != (expected_low if (layer_idx, head) in one_frame_heads else prompt_length)
               for layer_idx, row in enumerate(lens) for head, length in enumerate(row)):
            raise RuntimeError('Physical head K/V lengths differ from requested windows')
        physical_slots = sum(sum(row) for row in lens)
        dense_slots = len(lens) * 4 * prompt_length
        physical_bytes = sum(layer.flat_k.numel() * layer.flat_k.element_size()
                             + layer.flat_v.numel() * layer.flat_v.element_size()
                             for layer in ragged.layers)

        config = model.generation_config
        penalty = float(config.repetition_penalty or 1.0)
        eos = config.eos_token_id
        stop_ids = set(eos if isinstance(eos, list) else [eos])
        generated = []
        position = int(kwargs['position_ids'].max().item()) + 1
        for step in range(self.max_new_tokens):
            current = _apply_repetition_penalty(logits, layout['ids'] + generated, penalty)
            next_token = int(current.argmax().item())
            if step == 0:
                self._last_ttft_seconds = time.perf_counter() - started
            generated.append(next_token)
            if next_token in stop_ids or step + 1 == self.max_new_tokens:
                break
            positions = torch.full((3, 1, 1), position, dtype=torch.long,
                                   device=kwargs['inputs_embeds'].device)
            hidden = decode_step(model, torch.tensor([[next_token]],
                                 device=positions.device), ragged,
                                 [positions] * len(lens))
            logits = model.lm_head(hidden)[0, -1]
            position += 1
        append(output / 'ragged_query_audit.jsonl', dict(
            prompt_sha256=layout['prompt_sha256'], frame_visual_tokens=layout['counts'],
            pruned_before_question=True, question_suffix_tokens=prompt_length - visual_end,
            dense_visual_kv_released=True,
            independent_single_frame_prefill=independent_pass,
            one_frame_visual_tokens=layout['counts'][-1],
            one_frame_head_count=len(one_frame_heads), dense_kv_slots=dense_slots,
            physical_kv_slots_at_answer_start=physical_slots,
            physical_kv_bytes_at_answer_start=physical_bytes,
            one_frame_head_tokens=expected_low, four_frame_head_tokens=prompt_length,
            generated_tokens=len(generated)))
        return self.processor.batch_decode([generated], skip_special_tokens=True,
                    clean_up_tokenization_spaces=False)[0].strip()

    upstream.RecentWindowQAModel.generate_with_cached_vision = question
    upstream.RecentWindowQAModel._generate_from_model_inputs = generate
