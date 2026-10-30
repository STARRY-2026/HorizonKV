"""Bounded-position historical KV pilot for Qwen2.5-VL.

The history bank stores raw, pre-RoPE keys.  Every update assigns positions
only to the currently active history union and incoming frame.  At answer
time, the same compact frame order is shared by all layer/head selections.
"""

import torch
from flash_attn import flash_attn_func
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb,
)

from sparsemm_native.dscache_position_agnostic import (
    apply_mrope_to_key,
    remove_mrope_from_key,
)


TOKENS_PER_FRAME = 1024
PATCH_SIDE = 32


def compact_positions(uids, prefix_length, *, device, frame_order=None):
    """Shared 3-axis position map for a chronological active token union.

    Prefix UIDs are -prefix_length..-1.  Visual UIDs are frame*1024+patch.
    Each retained frame receives one 32-position interval, preserving its
    32x32 within-frame coordinates while bounding the timeline by the number
    of active frames, not by the full video duration.
    """
    values = [int(uid) for uid in uids]
    frames = (sorted({uid // TOKENS_PER_FRAME for uid in values if uid >= 0})
              if frame_order is None else list(frame_order))
    rank = {frame: index for index, frame in enumerate(frames)}
    result = torch.empty((3, 1, len(values)), dtype=torch.long, device=device)
    for index, uid in enumerate(values):
        if uid < 0:
            position = uid + prefix_length
            if not 0 <= position < prefix_length:
                raise ValueError('Invalid prefix token UID')
            result[:, 0, index] = position
        else:
            base = prefix_length + PATCH_SIDE * rank[uid // TOKENS_PER_FRAME]
            patch = uid % TOKENS_PER_FRAME
            result[0, 0, index] = base
            result[1, 0, index] = base + patch // PATCH_SIDE
            result[2, 0, index] = base + patch % PATCH_SIDE
    return result


def _prefill_history_frame(model, raw_keys, values, old_uids, embeddings,
                           incoming_uids, prefix_length):
    """Encode one old frame against only the bounded active historical cache."""
    from_uids = [*old_uids, *incoming_uids]
    positions = compact_positions(from_uids, prefix_length, device=embeddings.device)
    old_length = len(old_uids)
    new_positions = positions[:, :, old_length:]
    hidden = embeddings.unsqueeze(0)
    for layer_idx, decoder_layer in enumerate(model.language_model.layers):
        residual = hidden
        normed = decoder_layer.input_layernorm(hidden)
        attn = decoder_layer.self_attn
        q_len = hidden.shape[1]
        q = attn.q_proj(normed).view(1, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
        k = attn.k_proj(normed).view(1, q_len, attn.num_key_value_heads,
                                    attn.head_dim).transpose(1, 2)
        v = attn.v_proj(normed).view(1, q_len, attn.num_key_value_heads,
                                    attn.head_dim).transpose(1, 2)
        cos, sin = model.language_model.rotary_emb(normed, new_positions)
        q, _ = apply_multimodal_rotary_pos_emb(
            q, k, cos, sin, attn.rope_scaling['mrope_section'])
        raw_keys[layer_idx] = torch.cat((raw_keys[layer_idx], k), dim=2)
        values[layer_idx] = torch.cat((values[layer_idx], v), dim=2)
        all_k = apply_mrope_to_key(model, raw_keys[layer_idx], positions,
                                   attn.rope_scaling['mrope_section']).to(k.dtype)
        attended = flash_attn_func(
            q.transpose(1, 2).contiguous(), all_k.transpose(1, 2).contiguous(),
            values[layer_idx].transpose(1, 2).contiguous(), causal=True)
        attended = attended.reshape(1, q_len, attn.num_heads * attn.head_dim)
        hidden = residual + attn.o_proj(attended)
        residual = hidden
        hidden = decoder_layer.post_attention_layernorm(hidden)
        hidden = residual + decoder_layer.mlp(hidden)
    return from_uids, positions


def _select_head_history(model, raw_keys, uids, prefix_length, allocation,
                         probe_fn, selector_fn):
    """Return selected global active indices per layer/head and their union."""
    positions = compact_positions(uids, prefix_length, device=raw_keys[0].device)
    probe_position = int(positions.max().item()) + 1
    probes = probe_fn(model, probe_position)
    selected_by_head = {}
    union = set(range(prefix_length))
    for layer_idx, attn_layer in enumerate(model.language_model.layers):
        sections = attn_layer.self_attn.rope_scaling['mrope_section']
        rotated = apply_mrope_to_key(model, raw_keys[layer_idx], positions,
                                     sections).to(raw_keys[layer_idx].dtype)
        queries, pseudo_keys = probes[layer_idx]
        group = queries.shape[0] // 4
        for head in range(4):
            quota = int(allocation[layer_idx, head]['history_quota'])
            if quota == 0:
                indices = torch.empty(0, dtype=torch.long, device=rotated.device)
            else:
                indices = selector_fn(rotated[0, head],
                                      queries[head * group:(head + 1) * group],
                                      pseudo_keys[head], prefix_length, quota, protect=0)
                union.update(int(index) for index in indices.tolist())
            selected_by_head[layer_idx, head] = indices
    return selected_by_head, sorted(union), probe_position


def build_history_bank(model, full_dense, recent_positions, history_units,
                       prefix_length, allocation, probe_fn, selector_fn,
                       tokenizer, position_limit):
    """Build a cumulative, globally pruned raw-Key bank from old frames."""
    if not history_units:
        return None
    raw_keys = []
    values = []
    prefix_positions = recent_positions[:, :, :prefix_length]
    for layer_idx, decoder_layer in enumerate(model.language_model.layers):
        key, value = full_dense[layer_idx]
        prefix_key = key[:, :, :prefix_length]
        sections = decoder_layer.self_attn.rope_scaling['mrope_section']
        raw_keys.append(remove_mrope_from_key(model, prefix_key, prefix_positions,
                                               sections).to(prefix_key.dtype))
        values.append(value[:, :, :prefix_length].contiguous())
    uids = list(range(-prefix_length, 0))
    peak_position = prefix_length - 1
    peak_active_tokens = prefix_length
    for frame_index, (embeddings, grid) in enumerate(history_units):
        count = int(grid.prod().item()) // 4
        if count != TOKENS_PER_FRAME or int(embeddings.shape[0]) != TOKENS_PER_FRAME:
            raise ValueError('Position pilot requires 1024 tokens per old frame')
        incoming = list(range(frame_index * TOKENS_PER_FRAME,
                              (frame_index + 1) * TOKENS_PER_FRAME))
        candidate_uids, positions = _prefill_history_frame(
            model, raw_keys, values, uids,
            embeddings.to(device=model.device, dtype=raw_keys[0].dtype),
            incoming, prefix_length)
        peak_position = max(peak_position, int(positions.max().item()))
        peak_active_tokens = max(peak_active_tokens, len(candidate_uids))
        selected, union, _ = _select_head_history(
            model, raw_keys, candidate_uids, prefix_length, allocation,
            lambda model_arg, position: probe_fn(model_arg, tokenizer, position),
            selector_fn)
        index = torch.as_tensor(union, device=model.device, dtype=torch.long)
        raw_keys = [key.index_select(2, index).contiguous() for key in raw_keys]
        values = [value.index_select(2, index).contiguous() for value in values]
        uids = [candidate_uids[i] for i in union]
        if peak_position >= position_limit:
            raise RuntimeError('Historical active position reached model limit')
    selected, _, probe_position = _select_head_history(
        model, raw_keys, uids, prefix_length, allocation,
        lambda model_arg, position: probe_fn(model_arg, tokenizer, position),
        selector_fn)
    return dict(raw_keys=raw_keys, values=values, uids=uids,
                selected=selected, peak_position=peak_position,
                peak_active_tokens=peak_active_tokens,
                final_probe_position=probe_position,
                seen_frames=len(history_units))


def combine_answer_cache(model, bank, full_dense, one_dense, recent_positions,
                         one_indices, recent_counts, prefix_length, allocation,
                         recent_probes, selector_fn, Layer, *, include_history,
                         position_limit):
    """Reapply M-RoPE to selected per-head raw keys on one shared timeline."""
    if len(recent_counts) != 4 or any(count != TOKENS_PER_FRAME for count in recent_counts):
        raise ValueError('Pilot needs four 1024-token recent frames')
    history_frames = bank['seen_frames'] if bank else 0
    prefix_uids = list(range(-prefix_length, 0))
    reserved_uids = set(bank['uids'][prefix_length:]) if bank else set()
    recent_full_uids = [frame * TOKENS_PER_FRAME + patch
                        for frame in range(history_frames, history_frames + 4)
                        for patch in range(TOKENS_PER_FRAME)]
    recent_one_uids = recent_full_uids[-TOKENS_PER_FRAME:]
    recent_full_positions = recent_positions[:, :, :prefix_length + 4 * TOKENS_PER_FRAME]
    recent_one_positions = recent_positions.index_select(
        2, torch.as_tensor(one_indices, device=recent_positions.device))
    per_head = []
    history_slots = 0
    recent_slots = 0
    for layer_idx, decoder_layer in enumerate(model.language_model.layers):
        sections = decoder_layer.self_attn.rope_scaling['mrope_section']
        full_rotated, full_value = full_dense[layer_idx]
        one_rotated, one_value = one_dense[layer_idx]
        full_raw = remove_mrope_from_key(model, full_rotated, recent_full_positions,
                                         sections).to(full_rotated.dtype)
        if one_dense is full_dense:
            one_raw = full_raw
        else:
            one_raw = remove_mrope_from_key(model, one_rotated,
                                            recent_one_positions, sections).to(one_rotated.dtype)
        queries, pseudo_keys = recent_probes[layer_idx]
        group = queries.shape[0] // 4
        for head in range(4):
            row = allocation[layer_idx, head]
            one = row['recent_group'] == 'one_frame'
            chosen_rotated, chosen_raw, chosen_value, chosen_uids = (
                (one_rotated, one_raw, one_value, recent_one_uids) if one else
                (full_rotated, full_raw, full_value, recent_full_uids))
            history_idx = bank['selected'][layer_idx, head] if bank else torch.empty(
                0, dtype=torch.long, device=full_rotated.device)
            historical_count = int(history_idx.numel()) if include_history else 0
            history_quota = int(row['history_quota'])
            if historical_count > history_quota:
                raise RuntimeError('Historical selection exceeds per-head quota')
            recent_budget = int(row['recent_quota']) + history_quota - historical_count
            visual_idx = selector_fn(
                chosen_rotated[0, head],
                queries[head * group:(head + 1) * group],
                pseudo_keys[head], prefix_length, recent_budget)
            prefix_idx = torch.arange(prefix_length, device=visual_idx.device)
            recent_idx = torch.cat((prefix_idx, visual_idx))
            raw = chosen_raw[0, head].index_select(0, recent_idx)
            value = chosen_value[0, head].index_select(0, recent_idx)
            uid = [*prefix_uids, *(chosen_uids[int(index) - prefix_length]
                                  for index in visual_idx.tolist())]
            if historical_count:
                hist_raw = bank['raw_keys'][layer_idx][0, head].index_select(0, history_idx)
                hist_value = bank['values'][layer_idx][0, head].index_select(0, history_idx)
                raw = torch.cat((raw[:prefix_length], hist_raw, raw[prefix_length:]), dim=0)
                value = torch.cat((value[:prefix_length], hist_value,
                                   value[prefix_length:]), dim=0)
                uid = [*prefix_uids,
                       *(bank['uids'][int(index)] for index in history_idx.tolist()),
                       *uid[prefix_length:]]
            per_head.append((layer_idx, head, raw, value, uid, historical_count,
                             int(visual_idx.numel())))
            reserved_uids.update(uid[prefix_length:])
            history_slots += historical_count
            recent_slots += int(visual_idx.numel())
    global_positions = compact_positions(sorted(reserved_uids), prefix_length,
                                         device=model.device)
    global_frame_order = sorted({uid // TOKENS_PER_FRAME for uid in reserved_uids})
    max_position = max(prefix_length - 1, int(global_positions.max().item()))
    suffix_start = max_position + 1
    if suffix_start + 256 >= position_limit:
        raise RuntimeError('Answer positions reached model limit')
    layers = []
    for layer_idx, decoder_layer in enumerate(model.language_model.layers):
        keys = []
        values = []
        lengths = []
        sections = decoder_layer.self_attn.rope_scaling['mrope_section']
        for item in per_head[layer_idx * 4:(layer_idx + 1) * 4]:
            _, _, raw, value, uid, _, _ = item
            positions = compact_positions(uid, prefix_length, device=model.device,
                                          frame_order=global_frame_order)
            key = apply_mrope_to_key(model, raw[None, None], positions,
                                     sections).to(raw.dtype)[0, 0]
            keys.append(key)
            values.append(value)
            lengths.append(int(key.shape[0]))
        layers.append(Layer(torch.cat(keys), torch.cat(values),
                            torch.tensor(lengths, dtype=torch.int32,
                                         device=model.device)))
    return layers, history_slots, recent_slots, suffix_start, max_position
