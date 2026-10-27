"""Per-KV-head HERMES pseudo-query ranking for Qwen3 LU calibration/runtime."""

import torch
import torch.nn.functional as F
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

from sparsemm_baselines.qwen3_hermes import predicted_queries, layer_groups


@torch.inference_mode()
def pseudo_head_scores(bridge, query, start_position=None):
    ids = bridge.processor.tokenizer(query, return_tensors="pt").input_ids.to(bridge.device)
    q_len = ids.shape[1]
    start = bridge._next_position if start_position is None else int(start_position)
    position = torch.arange(start, start + q_len, device=bridge.device).view(1, 1, -1).expand(3, 1, -1)
    embeds = bridge.model.get_input_embeddings()(ids)
    cos, sin = bridge.language_model.rotary_emb(embeds, position)
    groups = bridge.num_query_heads // bridge.num_kv_heads
    result = []
    for layer_idx, decoder in enumerate(bridge.language_model.layers):
        norm = decoder.input_layernorm(embeds)
        attn = decoder.self_attn
        query_state = attn.q_norm(attn.q_proj(norm).view(
            1, q_len, bridge.num_query_heads, bridge.head_dim)).transpose(1, 2)
        key_state = attn.k_norm(attn.k_proj(norm).view(
            1, q_len, bridge.num_kv_heads, bridge.head_dim)).transpose(1, 2)
        query_state, key_state = apply_rotary_pos_emb(query_state, key_state, cos, sin)
        heads = []
        for head in range(bridge.num_kv_heads):
            old_key = bridge.kv_cache.layers[layer_idx].get_segment(head)[0]
            if old_key.shape[0] <= bridge.visual_start_idx:
                raise RuntimeError("HERMES pseudo-query requires visual cache")
            keys = torch.cat((old_key, key_state[0, head]), dim=0)
            q = query_state[0, head * groups:(head + 1) * groups]
            logits = q.float() @ keys.float().T * attn.scaling
            attention = logits.softmax(dim=-1)[:, :, bridge.visual_start_idx:old_key.shape[0]]
            heads.append(attention.mean(dim=(0, 1)))
        result.append(heads)
    return result


def _mixed_score(values, alpha, decay):
    values = values.float()
    if values.ndim != 1 or values.numel() < 1 or not bool(torch.isfinite(values).all()):
        raise ValueError("Invalid HERMES pseudo-query attention")
    count = values.numel()
    positions = torch.arange(count, device=values.device, dtype=torch.float32)
    recency = torch.exp(-float(decay) * (count - 1 - positions) / max(count - 1, 1))
    attn = (values - values.min()) / (values.max() - values.min() + 1e-6)
    recent = (recency - recency.min()) / (recency.max() - recency.min() + 1e-6)
    return attn * (1 - float(alpha)) + recent * float(alpha)


def hermes_head_scores(local, global_, mixed):
    """HERMES layer policy and backward refinement, separately per KV head."""
    if not (len(local) == len(global_) == len(mixed)):
        raise ValueError("Pseudo-query layers disagree")
    layers = len(local)
    shallow, deep = layer_groups(layers)
    raw, kinds = [], []
    for index in range(layers):
        if index < shallow:
            source, alpha, decay, kind = local[index], 1., 20., "short-term"
        elif index >= deep:
            source, alpha, decay, kind = global_[index], 0., 0., "long-term"
        else:
            progress = (index - shallow) / (deep - shallow)
            source, alpha, decay, kind = mixed[index], .75 - .6 * progress, 20. - 12. * progress, "mid-term"
        if not (len(local[index]) == len(global_[index]) == len(mixed[index])):
            raise ValueError("Pseudo-query heads disagree")
        raw.append([_mixed_score(row, alpha, decay) for row in source])
        kinds.append(kind)
    refined = [[row.clone() for row in layer] for layer in raw]
    for index in range(layers - 2, -1, -1):
        gamma = {"short-term": .1, "mid-term": .3, "long-term": .4}[kinds[index]]
        for head, row in enumerate(refined[index]):
            following = refined[index + 1][head]
            if following.numel() != row.numel():
                following = F.interpolate(following.reshape(1, 1, -1), size=row.numel(),
                                          mode="linear", align_corners=False).flatten()
            refined[index][head] = (1 - gamma) * row + gamma * following
    return refined, kinds


@torch.inference_mode()
def collect_hermes_head_scores(bridge, start_position=None):
    local_query, global_query = predicted_queries(bridge.conv_history)
    local = pseudo_head_scores(bridge, local_query, start_position)
    global_ = pseudo_head_scores(bridge, global_query, start_position)
    mixed = pseudo_head_scores(bridge, local_query + "; " + global_query, start_position)
    scores, kinds = hermes_head_scores(local, global_, mixed)
    return scores, {"local_query": local_query, "global_query": global_query,
                    "layer_types": kinds, "pseudo_query_forwards": 3}
