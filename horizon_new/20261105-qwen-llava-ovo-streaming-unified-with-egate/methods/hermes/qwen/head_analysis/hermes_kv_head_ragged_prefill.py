"""
Experimental physical per-KV-head ragged prefill/chunk path for Qwen2.5-VL.

This module extends the decode-only HeadRaggedCache prototype to video/text
prefill. Once installed, HERMES keeps the main KV cache in a flat ragged layout:

    flat_k / flat_v / head_lens / cu_klen

For q_len > 1, every KV head attends to its own historical ragged segment plus
the current dense chunk K/V. Compression prunes each KV head independently and
does not form a dense union cache.
"""

import json
import os
import re
import time
from types import MethodType
import csv

import numpy as np
import torch
import torch.nn.functional as F
from logzero import logger
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb

from head_analysis.hermes_kv_head_budget import build_kv_head_budget_table
from head_analysis.hermes_kv_head_budget import build_kv_head_split_budget_table
from head_analysis.hermes_kv_head_budget import compute_memory_current_split_ratios
from head_analysis.hermes_kv_head_budget import load_kv_head_memory_scores
from head_analysis.hermes_kv_head_budget import _select_split_budget_tokens
from head_analysis.hermes_kv_head_ragged import (
    HeadRaggedCache,
    _ragged_language_decode_step,
    flash_attn_varlen_func,
)


def _is_ragged_cache(cache):
    return isinstance(cache, HeadRaggedCache)


def _temporarily_limit_answer_history(model, max_age_seconds):
    """Prune old visual KV for one answer while retaining a restorable snapshot."""
    max_age_seconds = float(max_age_seconds)
    if max_age_seconds <= 0.0 or not _is_ragged_cache(model.kv_cache):
        return None
    frame_span = max(0, int(np.ceil(max_age_seconds * float(model.sample_fps))))
    snapshot = []
    keep_indices = []
    kept_visual = 0
    total_visual = 0
    for layer in model.kv_cache.layers:
        snapshot.append((
            layer.flat_k, layer.flat_v, layer.head_lens, layer.flat_pos,
            layer.flat_source_chunk, layer.flat_attention_ema,
        ))
        layer_keep = []
        for kv_head in range(int(layer.head_lens.numel())):
            frame_ids, _ = layer.get_horizon_aux_segment(kv_head)
            visual = frame_ids >= 0
            total_visual += int(visual.sum().item())
            if bool(visual.any()):
                latest = int(frame_ids[visual].max().item())
                cutoff = latest - frame_span
                keep = torch.nonzero(
                    (frame_ids < 0) | (frame_ids >= cutoff), as_tuple=False
                ).flatten()
                kept_visual += int((frame_ids.index_select(0, keep) >= 0).sum().item())
            else:
                keep = torch.arange(
                    frame_ids.numel(), device=frame_ids.device, dtype=torch.long
                )
            layer_keep.append(keep)
        keep_indices.append(layer_keep)
    position_snapshot = [
        None if value is None else value.clone()
        for value in model._position_ids_cache
    ]
    model.kv_cache.prune_per_head(keep_indices)
    _sync_position_cache_from_ragged(model)
    print(
        "[AnswerHistoryAblation] Temporarily retained "
        f"{kept_visual}/{total_visual} visual KV tokens "
        f"within {max_age_seconds:g}s"
    )
    return snapshot, position_snapshot


def _restore_answer_history(model, state):
    if state is None:
        return
    snapshot, position_snapshot = state
    for layer, values in zip(model.kv_cache.layers, snapshot):
        (layer.flat_k, layer.flat_v, layer.head_lens, layer.flat_pos,
         layer.flat_source_chunk, layer.flat_attention_ema) = values
        layer._refresh_cu_klen()
    model._position_ids_cache = position_snapshot


def _position_ids_2d(position_ids):
    if position_ids.dim() == 3:
        return position_ids[:, 0, :]
    return position_ids


def _attention_shape(model, attn=None):
    config = getattr(model.language_model, "config", None)
    num_query_heads = getattr(config, "num_attention_heads", None)
    num_kv_heads = getattr(config, "num_key_value_heads", None)
    if attn is not None:
        num_query_heads = getattr(attn.config, "num_attention_heads", num_query_heads)
        num_kv_heads = getattr(attn.config, "num_key_value_heads", num_kv_heads)
    num_query_heads = int(num_query_heads or 28)
    num_kv_heads = int(num_kv_heads or 4)
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_query_heads={num_query_heads} must be divisible by num_kv_heads={num_kv_heads}"
        )
    return num_query_heads, num_kv_heads, num_query_heads // num_kv_heads


def _build_total_kv(layer_cache, key_states, value_states, num_kv_heads):
    k_parts = []
    v_parts = []
    total_lens = []
    q_len = key_states.shape[2]

    for kv_head in range(num_kv_heads):
        hist_k, hist_v = layer_cache.get_segment(kv_head)
        curr_k = key_states[0, kv_head]
        curr_v = value_states[0, kv_head]
        k_parts.append(torch.cat([hist_k, curr_k], dim=0))
        v_parts.append(torch.cat([hist_v, curr_v], dim=0))
        total_lens.append(hist_k.shape[0] + q_len)

    flat_k = torch.cat(k_parts, dim=0).view(-1, 1, key_states.shape[-1]).contiguous()
    flat_v = torch.cat(v_parts, dim=0).view(-1, 1, value_states.shape[-1]).contiguous()
    total_lens = torch.tensor(total_lens, device=key_states.device, dtype=torch.int32)
    cu_k = torch.cat([
        torch.zeros(1, device=key_states.device, dtype=torch.int32),
        torch.cumsum(total_lens, dim=0, dtype=torch.int32),
    ]).to(dtype=torch.int32)
    return flat_k, flat_v, total_lens, cu_k


def _ragged_attention_flash_chunk(query_states, key_states, value_states,
                                  layer_cache, num_kv_heads, group_size):
    if flash_attn_varlen_func is None:
        return None
    if query_states.device.type != "cuda":
        return None
    if query_states.dtype not in (torch.float16, torch.bfloat16):
        return None

    _, _, q_len, head_dim = query_states.shape
    q_flat = (
        query_states[0]
        .view(num_kv_heads, group_size, q_len, head_dim)
        .permute(0, 2, 1, 3)
        .reshape(num_kv_heads * q_len, group_size, head_dim)
        .contiguous()
    )
    k_flat, v_flat, total_lens, cu_k = _build_total_kv(
        layer_cache, key_states, value_states, num_kv_heads
    )
    cu_q = torch.arange(
        0,
        (num_kv_heads + 1) * q_len,
        step=q_len,
        device=query_states.device,
        dtype=torch.int32,
    )

    out = flash_attn_varlen_func(
        q_flat,
        k_flat,
        v_flat,
        cu_q,
        cu_k,
        max_seqlen_q=int(q_len),
        max_seqlen_k=int(total_lens.max().item()),
        causal=True,
    )
    out = (
        out.view(num_kv_heads, q_len, group_size, head_dim)
        .permute(0, 2, 1, 3)
        .reshape(num_kv_heads * group_size, q_len, head_dim)
        .transpose(0, 1)
        .reshape(1, q_len, num_kv_heads * group_size * head_dim)
    )
    return out


def _ragged_attention_manual_chunk(query_states, key_states, value_states,
                                   layer_cache, num_kv_heads, group_size,
                                   scaling, collect_scores=False,
                                   score_last_n=None):
    _, _, q_len, head_dim = query_states.shape
    out_groups = []
    score_vectors = []

    for kv_head in range(num_kv_heads):
        hist_k, hist_v = layer_cache.get_segment(kv_head)
        curr_k = key_states[0, kv_head]
        curr_v = value_states[0, kv_head]
        total_k = torch.cat([hist_k, curr_k], dim=0)
        total_v = torch.cat([hist_v, curr_v], dim=0)
        hist_len = hist_k.shape[0]
        total_len = total_k.shape[0]

        q_group = query_states[0, kv_head * group_size:(kv_head + 1) * group_size]
        scores = torch.einsum("gqd,kd->gqk", q_group.float(), total_k.float()) * scaling

        key_idx = torch.arange(total_len, device=query_states.device)
        query_idx = torch.arange(q_len, device=query_states.device)
        causal = key_idx.view(1, -1) <= (hist_len + query_idx).view(-1, 1)
        scores = scores.masked_fill(~causal.view(1, q_len, total_len), -torch.finfo(scores.dtype).max)

        probs = F.softmax(scores, dim=-1, dtype=torch.float32)
        if collect_scores:
            score_probs = probs[:, :, :hist_len]
            if score_last_n is not None and int(score_last_n) > 0:
                score_probs = score_probs[:, -int(score_last_n):, :]
            score_vectors.append(score_probs.mean(dim=(0, 1)).detach())
        probs = probs.to(q_group.dtype)
        out_groups.append(torch.einsum("gqk,kd->gqd", probs, total_v))

    out = (
        torch.cat(out_groups, dim=0)
        .transpose(0, 1)
        .reshape(1, q_len, num_kv_heads * group_size * head_dim)
    )
    if collect_scores:
        return out, score_vectors
    return out, None


def _sampled_visual_read_scores(query_states, key_states, layer_cache,
                                num_kv_heads, group_size, scaling,
                                num_samples=16, top_readers=16):
    """Measure how a sparse set of new visual queries reads every cached key."""
    q_len = int(query_states.shape[2])
    if q_len <= 0 or int(num_samples) <= 0:
        return None
    sample_count = min(q_len, int(num_samples))
    positions = torch.linspace(
        0, q_len - 1, steps=sample_count, device=query_states.device
    ).round().long().unique(sorted=True)
    score_vectors = []
    for kv_head in range(int(num_kv_heads)):
        hist_k, _ = layer_cache.get_segment(kv_head)
        curr_k = key_states[0, kv_head]
        total_k = torch.cat([hist_k, curr_k], dim=0)
        hist_len = int(hist_k.shape[0])
        queries = query_states[
            0, kv_head * int(group_size):(kv_head + 1) * int(group_size)
        ].index_select(1, positions)
        logits = torch.einsum("gqd,kd->gqk", queries.float(), total_k.float())
        logits.mul_(float(scaling))

        key_idx = torch.arange(total_k.shape[0], device=query_states.device)
        causal = key_idx.view(1, -1) <= (hist_len + positions).view(-1, 1)
        logits.masked_fill_(
            ~causal.unsqueeze(0), -torch.finfo(logits.dtype).max
        )
        probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
        readers = probabilities.flatten(0, 1)
        k = min(int(top_readers), int(readers.shape[0]))
        scores = torch.topk(readers, k=k, dim=0, sorted=False).values.mean(0)
        score_vectors.append(scores.detach())
    return score_vectors


def _ragged_chunk_self_attn_forward(model, layer_idx, attn, hidden_states,
                                    ragged_cache, position_embeddings,
                                    position_ids, append=True,
                                    collect_scores=False,
                                    source_chunk_id=None,
                                    source_token_uids=None,
                                    score_last_n=None):
    bsz, q_len, _ = hidden_states.shape
    if bsz != 1:
        raise ValueError("Ragged prefill currently supports batch_size=1 only")

    num_query_heads, num_kv_heads, group_size = _attention_shape(model, attn)
    head_dim = attn.head_dim

    query_states = attn.q_proj(hidden_states)
    key_states = attn.k_proj(hidden_states)
    value_states = attn.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, num_query_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

    target_dtype = attn.q_proj.weight.dtype
    if query_states.dtype != target_dtype:
        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    cos, sin = position_embeddings
    mrope_section = getattr(model, "_mrope_section", attn.rope_scaling["mrope_section"])
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
        mrope_section,
    )

    admission_scores = None
    timestamp_config = getattr(model, "_timestamp_v8_config", None)
    prospective_config = getattr(model, "_prospective_lifecycle_trace_config", None)
    if (
        append
        and isinstance(source_chunk_id, torch.Tensor)
        and (
            prospective_config
            or str((timestamp_config or {}).get("profile_mode", "")).startswith(
                ("survival_", "relative_")
            )
        )
    ):
        admission_scores = _frame_admission_scores(
            query_states,
            key_states,
            source_chunk_id,
            num_kv_heads,
            group_size,
            getattr(attn, "scaling", head_dim ** -0.5),
        )

    layer_cache = ragged_cache.layers[layer_idx]
    visual_read_scores = None
    if append and prospective_config:
        visual_read_scores = _sampled_visual_read_scores(
            query_states,
            key_states,
            layer_cache,
            num_kv_heads,
            group_size,
            getattr(attn, "scaling", head_dim ** -0.5),
            num_samples=int(prospective_config.get("visual_query_samples", 0)),
            top_readers=int(prospective_config.get("visual_top_readers", 16)),
        )
    if collect_scores:
        scaling = getattr(attn, "scaling", head_dim ** -0.5)
        attn_output, layer_scores = _ragged_attention_manual_chunk(
            query_states,
            key_states,
            value_states,
            layer_cache,
            num_kv_heads,
            group_size,
            scaling,
            collect_scores=True,
            score_last_n=score_last_n,
        )
    else:
        attn_output = _ragged_attention_flash_chunk(
            query_states, key_states, value_states, layer_cache, num_kv_heads, group_size
        )
        if attn_output is None:
            scaling = getattr(attn, "scaling", head_dim ** -0.5)
            attn_output, _ = _ragged_attention_manual_chunk(
                query_states,
                key_states,
                value_states,
                layer_cache,
                num_kv_heads,
                group_size,
                scaling,
                collect_scores=False,
            )
        layer_scores = None

    if append:
        ragged_cache.append_chunk(
            layer_idx,
            key_states,
            value_states,
            pos=_position_ids_2d(position_ids),
            source_chunk_id=source_chunk_id,
            attention_scores=admission_scores,
            token_uids=source_token_uids,
        )

    if visual_read_scores is not None:
        layer_scores = visual_read_scores
    answer_gates = getattr(model, "_answer_head_gates", None)
    if answer_gates is not None:
        gates = answer_gates[layer_idx].to(device=attn_output.device, dtype=attn_output.dtype)
        attn_output = attn_output.view(bsz, q_len, num_query_heads, head_dim)
        attn_output = attn_output * gates.view(1, 1, num_query_heads, 1)
        attn_output = attn_output.reshape(bsz, q_len, num_query_heads * head_dim)
    return attn.o_proj(attn_output.contiguous()), layer_scores


def _frame_admission_scores(query_states, key_states, frame_ids,
                            num_kv_heads, group_size, scaling, top_queries=16):
    """Rank same-frame keys by their strongest frame-local Q-K readout."""
    q_len = int(key_states.shape[2])
    result = torch.zeros(
        (int(num_kv_heads), q_len), device=key_states.device, dtype=torch.float32
    )
    frame_ids = frame_ids.to(device=key_states.device).flatten()
    for frame_id in torch.unique(frame_ids, sorted=True):
        if int(frame_id) < 0:
            continue
        positions = torch.nonzero(frame_ids == frame_id, as_tuple=False).flatten()
        count = int(positions.numel())
        if count <= 1:
            result[:, positions] = 0.5
            continue
        for kv_head in range(int(num_kv_heads)):
            queries = query_states[
                0, kv_head * int(group_size):(kv_head + 1) * int(group_size)
            ].index_select(1, positions)
            keys = key_states[0, kv_head].index_select(0, positions)
            logits = torch.einsum("gqd,kd->gqk", queries.float(), keys.float())
            logits.mul_(float(scaling))
            diagonal = torch.eye(
                count, device=logits.device, dtype=torch.bool
            ).unsqueeze(0)
            logits.masked_fill_(diagonal, -torch.finfo(logits.dtype).max)
            probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
            readers = probabilities.flatten(0, 1)
            top_k = min(int(top_queries), int(readers.shape[0]))
            strength = torch.topk(readers, top_k, dim=0, sorted=False).values.mean(0)
            order = torch.argsort(strength, stable=True)
            ranks = torch.empty_like(strength)
            ranks[order] = torch.linspace(
                0.0, 1.0, count, device=strength.device, dtype=strength.dtype
            )
            result[kv_head].index_copy_(0, positions, ranks)
    return result.detach()


def _ragged_language_chunk_forward(model, inputs_embeds, ragged_cache,
                                   position_ids_by_layer, append=True,
                                   collect_scores=False,
                                   source_chunk_id=None,
                                   source_token_uids=None,
                                   score_last_n=None):
    hidden_states = inputs_embeds
    all_scores = []

    for layer_idx, decoder_layer in enumerate(model.language_model.layers):
        residual = hidden_states
        hidden_states_norm = decoder_layer.input_layernorm(hidden_states)
        position_ids = position_ids_by_layer[layer_idx]
        position_embeddings = model.language_model.rotary_emb(hidden_states_norm, position_ids)
        attn_output, layer_scores = _ragged_chunk_self_attn_forward(
            model,
            layer_idx,
            decoder_layer.self_attn,
            hidden_states_norm,
            ragged_cache,
            position_embeddings,
            position_ids,
            append=append,
            collect_scores=collect_scores,
            source_chunk_id=source_chunk_id,
            source_token_uids=source_token_uids,
            score_last_n=score_last_n,
        )
        if collect_scores or layer_scores is not None:
            all_scores.append(layer_scores)
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
        hidden_states = decoder_layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

    hidden_states = model.language_model.norm(hidden_states)
    return hidden_states, all_scores


def _build_text_position_ids(model, offsets, q_len, batch=1):
    return [
        model._build_position_ids_3d_for_text(offsets[layer_idx], q_len, batch)
        for layer_idx in range(model.num_layers)
    ]


def _ragged_query_scores(model, input_ids, offsets, score_last_n=None):
    inputs_embeds = model.get_input_embeddings()(input_ids)
    position_ids_by_layer = _build_text_position_ids(
        model, offsets, inputs_embeds.shape[1], inputs_embeds.shape[0]
    )
    _, scores = _ragged_language_chunk_forward(
        model,
        inputs_embeds,
        model.kv_cache,
        position_ids_by_layer,
        append=False,
        collect_scores=True,
        score_last_n=score_last_n,
    )
    return scores


def install_ragged_readout_trace(model, trace_path, score_last_n=1):
    """Enable a non-mutating true-question attention trace over cached frames."""
    trace_path = os.path.abspath(os.path.expanduser(str(trace_path)))
    os.makedirs(os.path.dirname(trace_path) or ".", exist_ok=True)
    with open(trace_path, "w"):
        pass
    model._ragged_readout_trace_config = {
        "trace_path": trace_path,
        # Zero reproduces the original ourtop query pooling: average all tokens
        # in the real multiple-choice question prompt.
        "score_last_n": max(0, int(score_last_n)),
        "current_frame_id": -1,
        "next_frame_id": 0,
    }
    print(
        "[RaggedReadoutTrace] Installed true-question trace: "
        f"{trace_path} (last_n={max(0, int(score_last_n)) or 'all'})"
    )
    return model


def install_ragged_token_trace(model, trace_path, score_last_n=0):
    """Trace immutable visual-token IDs and true-question attention per head."""
    trace_path = os.path.abspath(os.path.expanduser(str(trace_path)))
    os.makedirs(os.path.dirname(trace_path) or ".", exist_ok=True)
    with open(trace_path, "w"):
        pass
    model._ragged_token_trace_config = {
        "trace_path": trace_path,
        "score_last_n": max(0, int(score_last_n)),
        "current_frame_id": -1,
        "next_frame_id": 0,
        "next_visual_token_uid": 0,
    }
    print(
        "[RaggedTokenTrace] Installed exact token/readout trace: "
        f"{trace_path} (last_n={max(0, int(score_last_n)) or 'all'})"
    )
    return model


def install_ragged_redundancy_trace(model, trace_path, max_tokens=128):
    """Measure content redundancy without changing token selection."""
    trace_path = os.path.abspath(os.path.expanduser(str(trace_path)))
    os.makedirs(os.path.dirname(trace_path) or ".", exist_ok=True)
    with open(trace_path, "w"):
        pass
    model._ragged_redundancy_trace_config = {
        "trace_path": trace_path,
        "max_tokens": max(8, int(max_tokens)),
    }
    print(
        "[RaggedRedundancyTrace] Installed selected-history trace: "
        f"{trace_path} (max_tokens={max(8, int(max_tokens))})"
    )
    return model


def _write_ragged_redundancy_trace(model, keep_indices, budget_table, selector):
    config = getattr(model, "_ragged_redundancy_trace_config", None)
    if not config or not getattr(model, "_ragged_frame_trace_is_last_chunk", False):
        return
    heads = []
    max_tokens = int(config["max_tokens"])
    for layer_idx, layer in enumerate(model.kv_cache.layers):
        for kv_head, keep in enumerate(keep_indices[layer_idx]):
            frame_ids, _ = layer.get_horizon_aux_segment(kv_head)
            _, values = layer.get_segment(kv_head)
            keep = keep.to(device=frame_ids.device, dtype=torch.long)
            selected_frames = frame_ids.index_select(0, keep)
            visual = selected_frames >= 0
            selected_visual = keep[visual]
            visual_frames = selected_frames[visual]
            latest = int(visual_frames.max().item()) if visual_frames.numel() else -1
            historical = visual_frames < latest
            history_indices = selected_visual[historical]
            history_frames = visual_frames[historical]
            original_count = int(history_indices.numel())
            original_distinct_frames = int(torch.unique(history_frames).numel())
            if original_count > max_tokens:
                sample_positions = torch.linspace(
                    0, original_count - 1, steps=max_tokens,
                    device=history_indices.device,
                ).round().long().unique(sorted=True)
                history_indices = history_indices.index_select(0, sample_positions)
                history_frames = history_frames.index_select(0, sample_positions)
            sample_count = int(history_indices.numel())
            metrics = {
                "mean_nearest_cross_frame_cosine": None,
                "p90_nearest_cross_frame_cosine": None,
                "duplicate_fraction_ge_090": None,
                "duplicate_fraction_ge_095": None,
                "mean_content_novelty": None,
                "mean_value_norm": None,
            }
            if sample_count > 0:
                vectors = values.index_select(0, history_indices).float()
                metrics["mean_value_norm"] = float(vectors.norm(dim=-1).mean().item())
                if sample_count > 1 and int(torch.unique(history_frames).numel()) > 1:
                    vectors = torch.nn.functional.normalize(vectors, dim=-1, eps=1e-8)
                    similarity = vectors @ vectors.transpose(0, 1)
                    cross_frame = history_frames[:, None] != history_frames[None, :]
                    similarity = similarity.masked_fill(~cross_frame, -1.0)
                    nearest = similarity.max(dim=1).values
                    metrics.update({
                        "mean_nearest_cross_frame_cosine": float(nearest.mean().item()),
                        "p90_nearest_cross_frame_cosine": float(
                            torch.quantile(nearest, 0.9).item()
                        ),
                        "duplicate_fraction_ge_090": float((nearest >= 0.90).float().mean().item()),
                        "duplicate_fraction_ge_095": float((nearest >= 0.95).float().mean().item()),
                        "mean_content_novelty": float(
                            (1.0 - nearest.clamp(0.0, 1.0)).mean().item()
                        ),
                    })
            heads.append({
                "layer": int(layer_idx),
                "kv_head": int(kv_head),
                "budget": int(budget_table[layer_idx, kv_head]),
                "selected_visual_tokens": int(selected_visual.numel()),
                "selected_history_tokens": original_count,
                "sampled_history_tokens": sample_count,
                "distinct_history_frames": original_distinct_frames,
                **metrics,
            })
    record = {
        "schema_version": 1,
        "selector": str(selector),
        "video_id": getattr(model, "_active_video_id", None),
        "question_index": getattr(model, "_active_question_index", None),
        "question": getattr(model, "_active_question", None),
        "end_time": getattr(model, "_active_question_end_time", None),
        "max_tokens_per_head": max_tokens,
        "heads": heads,
    }
    with open(config["trace_path"], "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def install_prospective_lifecycle_trace(model, trace_path, score_last_n=1,
                                        top_fraction=0.2,
                                        visual_query_samples=16,
                                        visual_top_readers=16,
                                        content_bins=4):
    """Trace each source frame's admission signal and later effective reads."""
    trace_path = os.path.abspath(os.path.expanduser(str(trace_path)))
    os.makedirs(os.path.dirname(trace_path) or ".", exist_ok=True)
    with open(trace_path, "w"):
        pass
    if not 0.0 < float(top_fraction) <= 1.0:
        raise ValueError("top_fraction must be in (0, 1]")
    if int(visual_query_samples) < 0:
        raise ValueError("visual_query_samples must be non-negative")
    if int(visual_top_readers) <= 0:
        raise ValueError("visual_top_readers must be positive")
    if int(content_bins) <= 0:
        raise ValueError("content_bins must be positive")
    model._prospective_lifecycle_trace_config = {
        "trace_path": trace_path,
        "score_last_n": max(0, int(score_last_n)),
        "top_fraction": float(top_fraction),
        "visual_query_samples": int(visual_query_samples),
        "visual_top_readers": int(visual_top_readers),
        "content_bins": int(content_bins),
        "value_proj_dim": 32,
        "current_frame_id": -1,
        "next_frame_id": 0,
        "cache_compressed": False,
    }
    print(
        "[ProspectiveLifecycleTrace] Installed frame trajectory trace: "
        f"{trace_path} (top_fraction={float(top_fraction):g}, "
        f"visual_queries={int(visual_query_samples)}, content_bins={int(content_bins)})"
    )
    return model


def _aggregate_prospective_frames(frame_ids, admission, value_l1, score,
                                  observation_frame_id, sample_fps,
                                  top_fraction, num_content_bins,
                                  value_proj=None):
    """Aggregate one Head on CPU, avoiding synchronization inside frame loops."""
    frames = []
    valid_sources = np.unique(frame_ids[frame_ids >= 0])
    for source_value in valid_sources:
        source_frame_id = int(source_value)
        if source_frame_id > int(observation_frame_id):
            continue
        mask = frame_ids == source_frame_id
        frame_score = score[mask]
        frame_admission = admission[mask]
        frame_value_l1 = value_l1[mask]
        frame_value_proj = (
            value_proj[mask] if value_proj is not None else None
        )
        token_count = int(mask.sum())
        top_k = max(1, int(np.ceil(token_count * float(top_fraction))))
        top_start = max(0, token_count - top_k)
        top_salience = np.partition(frame_admission, top_start)[top_start:]
        frame_entry = {
            "source_frame_id": source_frame_id,
            "age_sec": (
                (int(observation_frame_id) - source_frame_id)
                / max(float(sample_fps), 1e-12)
            ),
            "token_count": token_count,
            "arrival_salience_mean": float(frame_admission.mean()),
            "arrival_salience_top_p": float(top_salience.mean()),
            "value_l1_mean": float(frame_value_l1.mean()),
            "attention_mass": float(frame_score.sum()),
            "attention_density": float(frame_score.mean()),
            "effective_readout_mass": float((frame_score * frame_value_l1).sum()),
        }
        if frame_value_proj is not None:
            frame_vec = frame_value_proj.mean(axis=0)
            frame_entry["value_vec_mean"] = [
                float(x) for x in frame_vec
            ]
        if int(num_content_bins) > 0:
            bin_ids = np.clip(
                (frame_admission * int(num_content_bins)).astype(np.int64),
                0,
                int(num_content_bins) - 1,
            )
            content_rows = []
            for content_bin in range(int(num_content_bins)):
                bin_mask = bin_ids == content_bin
                if not np.any(bin_mask):
                    continue
                bin_score = frame_score[bin_mask]
                bin_admission = frame_admission[bin_mask]
                bin_value_l1 = frame_value_l1[bin_mask]
                content_rows.append({
                    "content_bin": content_bin,
                    "token_count": int(bin_mask.sum()),
                    "arrival_salience_mean": float(bin_admission.mean()),
                    "value_l1_mean": float(bin_value_l1.mean()),
                    "attention_mass": float(bin_score.sum()),
                    "effective_readout_mass": float(
                        (bin_score * bin_value_l1).sum()
                    ),
                })
                if frame_value_proj is not None:
                    bin_vec = frame_value_proj[bin_mask].mean(axis=0)
                    content_rows[-1]["value_vec_mean"] = [
                        float(x) for x in bin_vec
                    ]
            frame_entry["content_bins"] = content_rows
        frames.append(frame_entry)
    return frames


def _write_prospective_lifecycle_trace(model, layer_scores, query_source):
    config = getattr(model, "_prospective_lifecycle_trace_config", None)
    if not config:
        return
    observation_frame_id = int(config["current_frame_id"])
    if observation_frame_id < 0:
        return

    heads = []
    for layer_idx, scores_by_head in enumerate(layer_scores):
        layer_cache = model.kv_cache.layers[layer_idx]
        for kv_head, score in enumerate(scores_by_head):
            frame_ids, admission = layer_cache.get_horizon_aux_segment(kv_head)
            _, values = layer_cache.get_segment(kv_head)
            score = score.float().flatten()
            limit = min(int(score.numel()), int(frame_ids.numel()), int(values.shape[0]))
            frame_ids_np = frame_ids[:limit].detach().cpu().numpy()
            admission_np = admission[:limit].float().detach().cpu().numpy()
            value_l1_np = (
                values[:limit].float().abs().sum(dim=-1).detach().cpu().numpy()
            )
            score_np = score[:limit].detach().cpu().numpy()
            value_proj_dim = int(config.get("value_proj_dim", 0))
            value_proj_np = None
            if value_proj_dim > 0:
                value_proj_np = (
                    values[:limit, :value_proj_dim]
                    .float()
                    .detach()
                    .cpu()
                    .numpy()
                )
            frames = _aggregate_prospective_frames(
                frame_ids_np,
                admission_np,
                value_l1_np,
                score_np,
                observation_frame_id,
                float(getattr(model, "sample_fps", 0.0)),
                config["top_fraction"],
                int(config.get("content_bins", 0)),
                value_proj=value_proj_np,
            )
            heads.append({
                "layer": int(layer_idx),
                "kv_head": int(kv_head),
                "frames": frames,
            })

    fps = float(getattr(model, "sample_fps", 0.0))
    record = {
        "schema_version": 1,
        "video_id": getattr(model, "_active_video_id", None),
        "question_index": getattr(model, "_active_question_index", None),
        "query_source": str(query_source),
        "observation_frame_id": observation_frame_id,
        "observation_time_sec": observation_frame_id / max(fps, 1e-12),
        "sample_fps": fps,
        "full_cache": not bool(config["cache_compressed"]),
        "score_last_n": int(config["score_last_n"]),
        "heads": heads,
    }
    with open(config["trace_path"], "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def _default_hermes_scores_for_trace(model, local_scores, global_scores,
                                     mixed_scores):
    """Reproduce default HERMES ranking scores without selecting or pruning."""
    traced_scores = []
    visual_start = int(model.visual_start_idx)
    for layer_idx, layer_cache in enumerate(model.kv_cache.layers):
        layer_scores, recency_alpha, k_decay = _layer_default_policy(
            model, layer_idx, local_scores, global_scores, mixed_scores
        )
        traced_heads = []
        for kv_head, raw_score in enumerate(layer_scores):
            head_len = int(layer_cache.head_lens[kv_head].item())
            score = raw_score.float().flatten()
            if score.numel() < head_len:
                score = torch.cat([
                    score,
                    torch.zeros(
                        head_len - score.numel(), device=score.device,
                        dtype=score.dtype,
                    ),
                ])
            score = score[:head_len]
            text_keep = min(visual_start, head_len)
            visual_score = score[text_keep:]
            combined = torch.zeros_like(score)
            if visual_score.numel() > 0:
                positions = torch.arange(
                    visual_score.numel(), device=score.device,
                    dtype=torch.float32,
                )
                distances = (
                    visual_score.numel() - 1 - positions
                ) / max(visual_score.numel() - 1, 1)
                recency = torch.exp(-float(k_decay) * distances)
                recency = (recency - recency.min()) / (
                    recency.max() - recency.min() + 1e-6
                )
                attention = (visual_score - visual_score.min()) / (
                    visual_score.max() - visual_score.min() + 1e-6
                )
                combined[text_keep:] = (
                    attention * (1.0 - float(recency_alpha))
                    + recency * float(recency_alpha)
                )
            traced_heads.append(combined.detach())
        traced_scores.append(traced_heads)
    return traced_scores


def _mark_prospective_trace_compressed(model):
    config = getattr(model, "_prospective_lifecycle_trace_config", None)
    if config:
        config["cache_compressed"] = True


def _write_ragged_readout_trace(model, layer_scores):
    config = getattr(model, "_ragged_readout_trace_config", None)
    if not config:
        return

    timestamp_config = getattr(model, "_timestamp_v8_config", None)
    roles = (timestamp_config or {}).get("roles")
    ratios = (timestamp_config or {}).get("attention_ratios")
    heads = []
    for layer_idx, scores_by_head in enumerate(layer_scores):
        layer_cache = model.kv_cache.layers[layer_idx]
        for kv_head, score in enumerate(scores_by_head):
            frame_ids, _ = layer_cache.get_horizon_aux_segment(kv_head)
            score = score.float().flatten()
            frame_ids = frame_ids[:score.numel()]
            visual = frame_ids >= 0
            visual_scores = score[visual]
            visual_frames = frame_ids[visual]
            visual_mass = float(visual_scores.sum().item())

            # Current-config counterpart of ourtop's internal-memory Top-100
            # mean: the newest raw frame is current; all older retained frame
            # tokens form the historical region.
            latest_frame_id = (
                int(visual_frames.max().item()) if visual_frames.numel() > 0 else -1
            )
            historical = visual & (frame_ids < latest_frame_id)
            historical_scores = score[historical]
            historical_top100_mean = None
            if historical_scores.numel() > 0:
                top_k = min(100, int(historical_scores.numel()))
                historical_top100_mean = float(
                    torch.topk(historical_scores, k=top_k, sorted=False)
                    .values.mean().item()
                )

            frame_mass = {}
            frame_tokens = {}
            if visual_scores.numel() > 0:
                unique_frames, inverse, counts = torch.unique(
                    visual_frames, sorted=True, return_inverse=True, return_counts=True
                )
                masses = torch.zeros(
                    unique_frames.numel(), device=score.device, dtype=torch.float32
                )
                masses.scatter_add_(0, inverse, visual_scores)
                frame_mass = {
                    str(int(frame_id)): float(mass)
                    for frame_id, mass in zip(
                        unique_frames.cpu().tolist(), masses.cpu().tolist()
                    )
                }
                frame_tokens = {
                    str(int(frame_id)): int(count)
                    for frame_id, count in zip(
                        unique_frames.cpu().tolist(), counts.cpu().tolist()
                    )
                }

            entry = {
                "layer": int(layer_idx),
                "kv_head": int(kv_head),
                "head_len": int(score.numel()),
                "visual_tokens": int(visual.sum().item()),
                "visual_mass": visual_mass,
                "latest_frame_id": latest_frame_id,
                "historical_tokens": int(historical.sum().item()),
                "historical_top100_mean_attention": historical_top100_mean,
                "frame_mass": frame_mass,
                "frame_tokens": frame_tokens,
            }
            if roles is not None:
                entry["role"] = str(roles[layer_idx, kv_head])
                entry["attention_ratio"] = float(ratios[layer_idx, kv_head])
            heads.append(entry)

    record = {
        "video_id": getattr(model, "_active_video_id", None),
        "question_index": getattr(model, "_active_question_index", None),
        "question": getattr(model, "_active_question", None),
        "end_time": getattr(model, "_active_question_end_time", None),
        "sample_fps": float(getattr(model, "sample_fps", 0.0)),
        "score_last_n": int(config["score_last_n"]),
        "heads": heads,
    }
    with open(config["trace_path"], "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    token_config = getattr(model, "_ragged_token_trace_config", None)
    if token_config:
        token_heads = []
        for layer_idx, scores_by_head in enumerate(layer_scores):
            layer_cache = model.kv_cache.layers[layer_idx]
            for kv_head, score in enumerate(scores_by_head):
                frame_ids, _ = layer_cache.get_horizon_aux_segment(kv_head)
                token_uids = layer_cache.get_token_uid_segment(kv_head)
                limit = min(int(score.numel()), int(frame_ids.numel()),
                            int(token_uids.numel()))
                score = score[:limit].float()
                frame_ids = frame_ids[:limit]
                token_uids = token_uids[:limit]
                visual = (frame_ids >= 0) & (token_uids >= 0)
                token_heads.append({
                    "layer": int(layer_idx),
                    "kv_head": int(kv_head),
                    "visual_token_uids": token_uids[visual].cpu().tolist(),
                    "visual_frame_ids": frame_ids[visual].cpu().tolist(),
                    "visual_attention": score[visual].cpu().tolist(),
                })
        token_record = {
            "schema_version": 1,
            "video_id": getattr(model, "_active_video_id", None),
            "question_index": getattr(model, "_active_question_index", None),
            "question": getattr(model, "_active_question", None),
            "end_time": getattr(model, "_active_question_end_time", None),
            "sample_fps": float(getattr(model, "sample_fps", 0.0)),
            "score_last_n": int(token_config["score_last_n"]),
            "heads": token_heads,
        }
        with open(token_config["trace_path"], "a", encoding="utf-8") as handle:
            handle.write(json.dumps(token_record, ensure_ascii=True) + "\n")


def _ragged_budget_table(model, num_keep):
    override = getattr(model, "_ragged_budget_table_override", None)
    if override is not None:
        override = np.asarray(override, dtype=np.int64)
        expected = (len(model.kv_cache.layers),
                    int(model.kv_cache.layers[0].head_lens.numel()))
        if override.shape != expected:
            raise ValueError(
                f"Ragged budget override must have shape {expected}, "
                f"got {override.shape}"
            )
        return override

    timestamp_config = getattr(model, "_timestamp_v8_config", None)
    if timestamp_config is not None:
        timestamp_budget = timestamp_config.get("kv_budget_table")
        if timestamp_budget is not None:
            return np.asarray(timestamp_budget, dtype=np.int64)

    config = getattr(model, "_kv_head_budget_config", {})
    num_query_heads, num_kv_heads, _ = _attention_shape(model)
    num_layers = int(getattr(model, "num_layers", len(model.kv_cache.layers)))
    scores = getattr(model, "_kv_head_budget_scores", None)
    if scores is None:
        scores = np.zeros((num_layers, num_query_heads), dtype=np.float64)

    return build_kv_head_budget_table(
        scores,
        num_keep,
        num_layers=num_layers,
        num_query_heads=int(config.get("num_query_heads", num_query_heads)),
        num_kv_heads=int(config.get("num_kv_heads", num_kv_heads)),
        strength=float(config.get("strength", 0.75)),
        min_ratio=float(config.get("min_ratio", 0.75)),
        max_ratio=float(config.get("max_ratio", 1.25)),
        scheme=config.get("budget_scheme", "relative"),
        sparsemm_ratio=float(config.get("sparsemm_ratio", 0.1)),
        sparsemm_window_size=int(config.get("sparsemm_window_size", 32)),
        floor_ratio=float(config.get("floor_ratio", 0.0)),
    )


def _ragged_split_budget_table(model, num_keep):
    """Build a split budget table [num_layers, num_kv_heads, 2] for ragged path.

    Reads the config and memory_ratios stored on the model by
    apply_kv_head_budget and combines them with the standard total-budget table
    produced by _ragged_budget_table.
    """
    config = getattr(model, "_kv_head_budget_config", {})
    total_table = _ragged_budget_table(model, num_keep)
    memory_ratios = config.get("memory_ratios", None)
    if memory_ratios is None:
        num_layers, num_kv_heads = total_table.shape
        memory_ratios = np.full((num_layers, num_kv_heads),
                                 float(config.get("base_memory_ratio", 0.7)),
                                 dtype=np.float64)
    return build_kv_head_split_budget_table(total_table, memory_ratios)


def _snapshot_head_lens_as_ints(ragged_cache):
    return [
        [int(x) for x in layer_lens.detach().cpu().tolist()]
        for layer_lens in ragged_cache.snapshot_head_lens()
    ]


def _build_kv_head_policy_roles(csv_path, memory_quantile=0.75,
                                current_quantile=0.75):
    """Build a conservative role table from v8_budget-style scores."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        required = {"layer", "kv_head", "long_score", "short_score"}
        missing = required - fieldnames
        if missing:
            raise ValueError(
                f"{csv_path} missing columns for KV-head policy roles: {sorted(missing)}"
            )
        for row in reader:
            rows.append({
                "layer": int(row["layer"]),
                "kv_head": int(row["kv_head"]),
                "long_score": float(row["long_score"]),
                "short_score": float(row["short_score"]),
            })

    if not rows:
        raise ValueError(f"{csv_path} has no role rows")

    long_values = np.asarray([r["long_score"] for r in rows], dtype=np.float64)
    short_values = np.asarray([r["short_score"] for r in rows], dtype=np.float64)
    long_thr = float(np.quantile(long_values, float(memory_quantile)))
    short_thr = float(np.quantile(short_values, float(current_quantile)))

    roles = {}
    counts = {"memory": 0, "current": 0, "mixed": 0}
    for row in rows:
        long_high = row["long_score"] >= long_thr
        short_high = row["short_score"] >= short_thr
        if long_high and not short_high:
            role = "memory"
        elif short_high and not long_high:
            role = "current"
        else:
            role = "mixed"
        roles[(row["layer"], row["kv_head"])] = role
        counts[role] += 1

    return {
        "roles": roles,
        "counts": counts,
        "long_threshold": long_thr,
        "short_threshold": short_thr,
        "memory_quantile": float(memory_quantile),
        "current_quantile": float(current_quantile),
        "path": csv_path,
    }


def install_kv_head_policy_roles(model, csv_path, memory_quantile=0.75,
                                 current_quantile=0.75,
                                 memory_bonus=0.1,
                                 current_bonus=0.1,
                                 memory_recency_alpha=0.05,
                                 current_recency_alpha=0.75,
                                 current_k_decay=20.0):
    role_info = _build_kv_head_policy_roles(
        csv_path,
        memory_quantile=memory_quantile,
        current_quantile=current_quantile,
    )
    role_info.update({
        "memory_bonus": float(memory_bonus),
        "current_bonus": float(current_bonus),
        "memory_recency_alpha": float(memory_recency_alpha),
        "current_recency_alpha": float(current_recency_alpha),
        "current_k_decay": float(current_k_decay),
    })
    model._kv_head_policy_roles = role_info
    print(
        "[kv_head_policy] Installed role-specific ragged selection: "
        f"roles={role_info['counts']}, "
        f"long_thr={role_info['long_threshold']:.6g}, "
        f"short_thr={role_info['short_threshold']:.6g}, "
        f"memory_bonus={float(memory_bonus):g}, current_bonus={float(current_bonus):g}"
    )
    return model


def install_kv_head_recency_profile(
    model,
    csv_path,
    metric="b_log_per_token_ratio",
    delta=0.2,
    tau=1.0,
    alpha_min=0.0,
    alpha_max=1.0,
    k_delta=0.0,
    k_min=0.0,
    k_max=20.0,
):
    """Install continuous per-KV-head residuals on HERMES recency policy."""
    delta = float(delta)
    tau = float(tau)
    alpha_min = float(alpha_min)
    alpha_max = float(alpha_max)
    k_delta = float(k_delta)
    k_min = float(k_min)
    k_max = float(k_max)
    if delta < 0.0:
        raise ValueError("KV-head recency delta must be non-negative")
    if tau <= 0.0:
        raise ValueError("KV-head recency tau must be positive")
    if not 0.0 <= alpha_min < alpha_max <= 1.0:
        raise ValueError("KV-head recency alpha bounds must satisfy 0 <= min < max <= 1")
    if k_delta < 0.0:
        raise ValueError("KV-head recency k delta must be non-negative")
    if not 0.0 <= k_min < k_max:
        raise ValueError("KV-head recency k bounds must satisfy 0 <= min < max")

    num_layers = getattr(model, "num_layers", None)
    if num_layers is None:
        kv_cache = getattr(model, "kv_cache", None)
        if kv_cache is None or not hasattr(kv_cache, "layers"):
            raise ValueError(
                "Cannot infer the number of layers for KV-head recency profiling"
            )
        num_layers = len(kv_cache.layers)
    num_layers = int(num_layers)
    _, num_kv_heads, _ = _attention_shape(model)
    raw_scores = np.full((num_layers, num_kv_heads), np.nan, dtype=np.float64)
    seen = set()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        required = {"layer", "kv_head", metric}
        missing = required - fieldnames
        if missing:
            raise ValueError(
                f"{csv_path} missing columns for KV-head recency profile: {sorted(missing)}"
            )
        for row in reader:
            key = (int(row["layer"]), int(row["kv_head"]))
            if key in seen:
                raise ValueError(f"{csv_path} contains duplicate head {key}")
            if not (0 <= key[0] < num_layers and 0 <= key[1] < num_kv_heads):
                raise ValueError(f"{csv_path} contains out-of-range head {key}")
            value = float(row[metric])
            if not np.isfinite(value):
                raise ValueError(f"{csv_path} has non-finite {metric} for head {key}")
            seen.add(key)
            raw_scores[key] = value

    expected = num_layers * num_kv_heads
    if len(seen) != expected:
        missing_heads = [
            (layer_idx, kv_head)
            for layer_idx in range(num_layers)
            for kv_head in range(num_kv_heads)
            if (layer_idx, kv_head) not in seen
        ]
        raise ValueError(
            f"{csv_path} covers {len(seen)}/{expected} KV heads; "
            f"missing {missing_heads[:8]}"
        )

    # Center within each layer so the profile captures head heterogeneity rather
    # than re-encoding the layer-depth prior already present in HERMES.
    centered_scores = np.zeros_like(raw_scores)
    for layer_idx in range(num_layers):
        values = raw_scores[layer_idx]
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        scale = 1.4826 * mad
        if scale < 1e-12:
            centered_scores[layer_idx] = 0.0
        else:
            centered_scores[layer_idx] = (values - median) / scale

    model._kv_head_recency_profile = {
        "path": csv_path,
        "metric": str(metric),
        "raw_scores": raw_scores,
        "centered_scores": centered_scores,
        "delta": delta,
        "tau": tau,
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        "k_delta": k_delta,
        "k_min": k_min,
        "k_max": k_max,
    }
    print(
        "[kv_head_recency] Installed continuous head-wise recency policy: "
        f"metric={metric}, delta={delta:g}, tau={tau:g}, "
        f"alpha_bounds=[{alpha_min:g},{alpha_max:g}], "
        f"k_delta={k_delta:g}, k_bounds=[{k_min:g},{k_max:g}]"
    )
    return model


def _apply_kv_head_recency_alpha(model, layer_idx, kv_head, base_alpha):
    profile = getattr(model, "_kv_head_recency_profile", None)
    if not profile:
        return float(base_alpha)
    z_score = float(profile["centered_scores"][layer_idx, kv_head])
    residual = float(profile["delta"]) * np.tanh(z_score / float(profile["tau"]))
    return float(np.clip(
        float(base_alpha) + residual,
        float(profile["alpha_min"]),
        float(profile["alpha_max"]),
    ))


def _apply_kv_head_recency_k_decay(model, layer_idx, kv_head, base_k_decay):
    profile = getattr(model, "_kv_head_recency_profile", None)
    if not profile:
        return float(base_k_decay)
    z_score = float(profile["centered_scores"][layer_idx, kv_head])
    residual = float(profile["k_delta"]) * np.tanh(
        z_score / float(profile["tau"])
    )
    return float(np.clip(
        float(base_k_decay) + residual,
        float(profile["k_min"]),
        float(profile["k_max"]),
    ))


def _layer_default_policy(model, layer_idx, local_scores, global_scores, mixed_scores):
    selector_policy = str(
        getattr(model, "_kv_head_selector_policy", "deployed") or "deployed"
    ).lower()
    if selector_policy == "uniform_joint":
        # The third probe is the concatenated Local+Global pseudo query.
        # Rank history by its attention at every layer; recent-window
        # protection is applied separately by the selector.
        return mixed_scores[layer_idx], 0.0, 0.0
    if selector_policy != "deployed":
        raise ValueError(f"Unsupported KV-head selector policy: {selector_policy}")
    if layer_idx < model.short_term_threshold:
        return local_scores[layer_idx], 1.0, 20.0
    if layer_idx >= model.long_term_threshold:
        return global_scores[layer_idx], 0.0, 0.0

    progress = (
        (layer_idx - model.short_term_threshold)
        / (model.long_term_threshold - model.short_term_threshold)
    )
    return mixed_scores[layer_idx], 0.75 - 0.6 * progress, 20.0 - 12.0 * progress


def _head_policy_score(model, layer_idx, kv_head, local_scores, global_scores,
                       mixed_scores, default_layer_scores,
                       default_recency_alpha, default_k_decay):
    policy = getattr(model, "_kv_head_policy_roles", None)
    if not policy:
        return (
            default_layer_scores[kv_head],
            _apply_kv_head_recency_alpha(
                model, layer_idx, kv_head, default_recency_alpha
            ),
            _apply_kv_head_recency_k_decay(
                model, layer_idx, kv_head, default_k_decay
            ),
            "default",
        )

    role = policy.get("roles", {}).get((int(layer_idx), int(kv_head)), "mixed")
    if role == "memory":
        alpha = _apply_kv_head_recency_alpha(
            model, layer_idx, kv_head,
            float(policy.get("memory_recency_alpha", 0.05)),
        )
        return (
            global_scores[layer_idx][kv_head],
            alpha,
            _apply_kv_head_recency_k_decay(
                model, layer_idx, kv_head, 0.0
            ),
            role,
        )
    if role == "current":
        alpha = _apply_kv_head_recency_alpha(
            model, layer_idx, kv_head,
            float(policy.get("current_recency_alpha", 0.75)),
        )
        return (
            local_scores[layer_idx][kv_head],
            alpha,
            _apply_kv_head_recency_k_decay(
                model, layer_idx, kv_head,
                float(policy.get("current_k_decay", 20.0)),
            ),
            role,
        )
    return (
        default_layer_scores[kv_head],
        _apply_kv_head_recency_alpha(
            model, layer_idx, kv_head, default_recency_alpha
        ),
        _apply_kv_head_recency_k_decay(
            model, layer_idx, kv_head, default_k_decay
        ),
        role,
    )


def _clamp01(value):
    return max(0.0, min(float(value), 1.0))


def _layerwise_last_frame_ratio(config, layer_idx):
    if not bool(config.get("layerwise_last_frame", False)):
        return 0.0

    shallow_end = int(config.get("layerwise_shallow_end", 10))
    mid_end = int(config.get("layerwise_mid_end", 20))
    if int(layer_idx) < shallow_end:
        return _clamp01(config.get("layerwise_shallow_ratio", 1.0))
    if int(layer_idx) < mid_end:
        return _clamp01(config.get("layerwise_mid_ratio", 0.5))
    return _clamp01(config.get("layerwise_deep_ratio", 0.0))


def _select_layerwise_last_frame_tokens(model, layer_idx, kv_head, text_keep,
                                        num_visual_tokens, combined_score):
    config = getattr(model, "_kv_head_budget_config", {})
    role_ratio = _role_last_frame_ratio(model, layer_idx, kv_head)
    if role_ratio is not None:
        ratio = role_ratio
    else:
        ratio = _layerwise_last_frame_ratio(config, layer_idx)
    device = combined_score.device
    if ratio <= 0.0 or num_visual_tokens <= 0:
        return torch.empty(0, device=device, dtype=torch.long)

    layer_cache = model.kv_cache.layers[layer_idx]
    pos = layer_cache.get_position_segment(kv_head)
    if pos is not None and pos.shape[1] >= text_keep + num_visual_tokens:
        video_pos = pos[:, text_keep:text_keep + num_visual_tokens]
        if video_pos.numel() > 0:
            temporal = video_pos[0]
            max_temporal = temporal.max()
            last_frame_positions = torch.nonzero(
                temporal == max_temporal, as_tuple=False
            ).flatten().to(device=device, dtype=torch.long)
        else:
            last_frame_positions = torch.empty(0, device=device, dtype=torch.long)
    else:
        total_frames = max(1, int(getattr(model, "total_processed_frames", 1) or 1))
        est_tokens = max(1, num_visual_tokens // total_frames)
        start = max(0, num_visual_tokens - est_tokens)
        last_frame_positions = torch.arange(
            start, num_visual_tokens, device=device, dtype=torch.long
        )

    if last_frame_positions.numel() == 0:
        return last_frame_positions

    keep_k = int(np.ceil(float(last_frame_positions.numel()) * ratio))
    keep_k = max(0, min(keep_k, int(last_frame_positions.numel())))
    if keep_k <= 0:
        return torch.empty(0, device=device, dtype=torch.long)
    if keep_k >= int(last_frame_positions.numel()):
        return last_frame_positions

    local_scores = combined_score[last_frame_positions]
    top_local = torch.topk(local_scores, keep_k, sorted=False)[1]
    return last_frame_positions[top_local]


def _unique_visual_parts(parts):
    valid = [part for part in parts if part is not None and part.numel() > 0]
    if not valid:
        return None
    return torch.unique(torch.cat(valid), sorted=True)


def _pick_score_column(fieldnames):
    preferred = [
        "score",
        "internal_top100_mean_attention",
        "observed_memory_ratio",
        "memory_readout_share",
        "readout_shape_score",
    ]
    for name in preferred:
        if name in fieldnames:
            return name
    for name in fieldnames:
        if name not in ("layer", "kv_head", "head", "q_heads"):
            return name
    raise ValueError("No score column found for role-aware recent retention")


def _build_kv_head_score_roles(csv_path, memory_quantile=0.75, current_quantile=0.25,
                               memory_count=-1, current_count=-1):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "layer" not in fieldnames or "kv_head" not in fieldnames:
            raise ValueError(f"{csv_path} must contain layer and kv_head columns")
        score_col = _pick_score_column(fieldnames)
        for row in reader:
            rows.append({
                "layer": int(row["layer"]),
                "kv_head": int(row["kv_head"]),
                "score": float(row[score_col]),
            })

    if not rows:
        raise ValueError(f"{csv_path} has no score rows")

    requested_memory_count = int(memory_count)
    requested_current_count = int(current_count)
    ordered = sorted(rows, key=lambda item: (item["score"], item["layer"], item["kv_head"]))
    total = len(ordered)
    use_count = requested_memory_count >= 0 or requested_current_count >= 0
    if use_count:
        resolved_current_count = requested_current_count if requested_current_count >= 0 else int(round(total * float(current_quantile)))
        resolved_memory_count = requested_memory_count if requested_memory_count >= 0 else int(round(total * (1.0 - float(memory_quantile))))
    else:
        resolved_current_count = int(round(total * float(current_quantile)))
        resolved_memory_count = int(round(total * (1.0 - float(memory_quantile))))
    current_count = max(0, min(resolved_current_count, total))
    memory_count = max(0, min(resolved_memory_count, total - current_count))

    current_keys = {
        (int(row["layer"]), int(row["kv_head"]))
        for row in ordered[:current_count]
    }
    memory_keys = {
        (int(row["layer"]), int(row["kv_head"]))
        for row in ordered[total - memory_count:]
    }
    current_thr = float(ordered[current_count - 1]["score"]) if current_count > 0 else float("nan")
    memory_thr = float(ordered[total - memory_count]["score"]) if memory_count > 0 else float("nan")

    roles = {}
    counts = {"memory": 0, "mixed": 0, "current": 0}
    for row in rows:
        key = (int(row["layer"]), int(row["kv_head"]))
        if key in memory_keys:
            role = "memory"
        elif key in current_keys:
            role = "current"
        else:
            role = "mixed"
        roles[key] = role
        counts[role] += 1

    return {
        "roles": roles,
        "counts": counts,
        "score_column": score_col,
        "memory_threshold": memory_thr,
        "current_threshold": current_thr,
        "memory_quantile": float(memory_quantile),
        "current_quantile": float(current_quantile),
        "memory_count_arg": requested_memory_count if use_count else -1,
        "current_count_arg": requested_current_count if use_count else -1,
        "resolved_memory_count": int(len(memory_keys)),
        "resolved_current_count": int(len(current_keys)),
        "use_count": bool(use_count),
        "path": csv_path,
    }


def _pick_dual_score_column(fieldnames, role):
    preferred = {
        "memory": [
            "memory_score",
            "long_score",
            "memory_attention_ratio",
            "old_attention_ratio",
            "score",
        ],
        "current": [
            "current_score",
            "short_score",
            "current_attention_ratio",
            "recent_attention_ratio",
            "score",
        ],
    }
    for name in preferred[role]:
        if name in fieldnames:
            return name
    raise ValueError(
        f"Dual role score CSV has no supported {role} score column. "
        f"Expected one of {preferred[role]}."
    )


def _build_kv_head_dual_score_roles(
    csv_path,
    memory_quantile=0.75,
    current_quantile=0.25,
    memory_count=-1,
    current_count=-1,
):
    """Assign exact, disjoint roles from independent memory/current scores."""
    rows = []
    seen = set()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        if "layer" not in fieldnames or "kv_head" not in fieldnames:
            raise ValueError(f"{csv_path} must contain layer and kv_head columns")
        memory_col = _pick_dual_score_column(fieldnames, "memory")
        current_col = _pick_dual_score_column(fieldnames, "current")
        if memory_col == current_col:
            raise ValueError(
                "Dual role classification requires different memory and current "
                f"score columns, but both resolved to {memory_col!r}."
            )
        for row in reader:
            key = (int(row["layer"]), int(row["kv_head"]))
            if key in seen:
                raise ValueError(f"{csv_path} contains duplicate head {key}")
            seen.add(key)
            rows.append({
                "layer": key[0],
                "kv_head": key[1],
                "memory_score": float(row[memory_col]),
                "current_score": float(row[current_col]),
            })

    if not rows:
        raise ValueError(f"{csv_path} has no score rows")

    total = len(rows)
    requested_memory_count = int(memory_count)
    requested_current_count = int(current_count)
    if requested_memory_count < -1 or requested_current_count < -1:
        raise ValueError("memory_count and current_count must be -1 or non-negative")
    use_count = requested_memory_count >= 0 or requested_current_count >= 0
    if use_count:
        resolved_memory_count = (
            requested_memory_count
            if requested_memory_count >= 0
            else int(round(total * (1.0 - float(memory_quantile))))
        )
        resolved_current_count = (
            requested_current_count
            if requested_current_count >= 0
            else int(round(total * float(current_quantile)))
        )
    else:
        resolved_memory_count = int(round(total * (1.0 - float(memory_quantile))))
        resolved_current_count = int(round(total * float(current_quantile)))

    if resolved_memory_count + resolved_current_count > total:
        raise ValueError(
            "Requested memory/current head counts exceed available heads: "
            f"memory={resolved_memory_count}, current={resolved_current_count}, "
            f"total={total}"
        )
    resolved_memory_count = max(0, min(resolved_memory_count, total))
    resolved_current_count = max(0, min(resolved_current_count, total))

    def ordered_keys(score_name):
        ordered_rows = sorted(
            rows,
            key=lambda item: (
                -item[score_name], item["layer"], item["kv_head"]
            ),
        )
        return [
            (int(item["layer"]), int(item["kv_head"]))
            for item in ordered_rows
        ]

    memory_order = ordered_keys("memory_score")
    current_order = ordered_keys("current_score")
    denom = max(total - 1, 1)
    memory_strength = {
        key: 1.0 - rank / denom for rank, key in enumerate(memory_order)
    }
    current_strength = {
        key: 1.0 - rank / denom for rank, key in enumerate(current_order)
    }

    # Resolve overlap by assigning the strongest and most role-specific pairs
    # first. The remaining candidates then fill both exact role capacities.
    candidates = []
    for key in seen:
        mem_strength = memory_strength[key]
        cur_strength = current_strength[key]
        candidates.append((mem_strength, mem_strength - cur_strength, "memory", key))
        candidates.append((cur_strength, cur_strength - mem_strength, "current", key))
    candidates.sort(
        key=lambda item: (
            -item[0], -item[1], item[2], item[3][0], item[3][1]
        )
    )

    limits = {
        "memory": resolved_memory_count,
        "current": resolved_current_count,
    }
    selected = {"memory": set(), "current": set()}
    assigned = set()
    for _, _, role, key in candidates:
        if len(selected[role]) >= limits[role] or key in assigned:
            continue
        selected[role].add(key)
        assigned.add(key)
        if all(len(selected[name]) >= limits[name] for name in limits):
            break

    if any(len(selected[name]) != limits[name] for name in limits):
        selected_counts = {name: len(keys) for name, keys in selected.items()}
        raise RuntimeError(
            "Failed to resolve exact disjoint memory/current head counts: "
            f"wanted={limits}, got={selected_counts}"
        )

    row_by_key = {
        (int(row["layer"]), int(row["kv_head"])): row for row in rows
    }
    roles = {}
    counts = {"memory": 0, "mixed": 0, "current": 0}
    for key in sorted(seen):
        if key in selected["memory"]:
            role = "memory"
        elif key in selected["current"]:
            role = "current"
        else:
            role = "mixed"
        roles[key] = role
        counts[role] += 1

    memory_thr = (
        min(row_by_key[key]["memory_score"] for key in selected["memory"])
        if selected["memory"] else float("nan")
    )
    current_thr = (
        min(row_by_key[key]["current_score"] for key in selected["current"])
        if selected["current"] else float("nan")
    )
    return {
        "roles": roles,
        "counts": counts,
        "score_mode": "dual",
        "memory_score_column": memory_col,
        "current_score_column": current_col,
        "memory_threshold": float(memory_thr),
        "current_threshold": float(current_thr),
        "memory_quantile": float(memory_quantile),
        "current_quantile": float(current_quantile),
        "memory_count_arg": requested_memory_count if use_count else -1,
        "current_count_arg": requested_current_count if use_count else -1,
        "resolved_memory_count": int(len(selected["memory"])),
        "resolved_current_count": int(len(selected["current"])),
        "use_count": bool(use_count),
        "path": csv_path,
    }


def install_kv_head_role_recent_retention(
    model,
    csv_path=None,
    dual_scores_path=None,
    memory_quantile=0.75,
    current_quantile=0.25,
    memory_count=-1,
    current_count=-1,
    memory_recent_tokens=0,
    mixed_recent_tokens=2000,
    current_recent_tokens=4000,
    window_mode="override",
    uniform_budget=False,
    role_budget_enable=False,
    memory_budget=8000,
    mixed_budget=6000,
    current_budget=4000,
    budget_auto_mixed=False,
    history_uniform_memory_ratio=0.0,
    history_uniform_mixed_ratio=0.0,
    history_uniform_bins=32,
    last_frame_by_role=False,
    last_frame_ratio_current=1.0,
    last_frame_ratio_memory=0.0,
    last_frame_ratio_mixed=0.0,
):
    if dual_scores_path:
        role_info = _build_kv_head_dual_score_roles(
            dual_scores_path,
            memory_quantile=memory_quantile,
            current_quantile=current_quantile,
            memory_count=memory_count,
            current_count=current_count,
        )
    elif csv_path:
        role_info = _build_kv_head_score_roles(
            csv_path,
            memory_quantile=memory_quantile,
            current_quantile=current_quantile,
            memory_count=memory_count,
            current_count=current_count,
        )
        role_info["score_mode"] = "single"
    else:
        raise ValueError("A single-score or dual-score role CSV is required")
    mode = str(window_mode or "override").lower()
    if mode not in ("override", "max"):
        raise ValueError("window_mode must be 'override' or 'max'")
    memory_uniform_ratio = float(history_uniform_memory_ratio)
    mixed_uniform_ratio = float(history_uniform_mixed_ratio)
    if not 0.0 <= memory_uniform_ratio <= 1.0:
        raise ValueError("history_uniform_memory_ratio must be in [0, 1]")
    if not 0.0 <= mixed_uniform_ratio <= 1.0:
        raise ValueError("history_uniform_mixed_ratio must be in [0, 1]")
    uniform_bins = int(history_uniform_bins)
    if uniform_bins <= 0:
        raise ValueError("history_uniform_bins must be positive")
    role_info.update({
        "memory_recent_tokens": int(memory_recent_tokens),
        "mixed_recent_tokens": int(mixed_recent_tokens),
        "current_recent_tokens": int(current_recent_tokens),
        "window_mode": mode,
        "last_frame_by_role": bool(last_frame_by_role),
        "last_frame_ratio_by_role": {
            "current": float(last_frame_ratio_current),
            "memory": float(last_frame_ratio_memory),
            "mixed": float(last_frame_ratio_mixed),
        },
        "uniform_budget": bool(uniform_budget),
        "role_budget_enable": bool(role_budget_enable),
        "memory_budget": int(memory_budget),
        "mixed_budget": int(mixed_budget),
        "current_budget": int(current_budget),
        "budget_auto_mixed": bool(budget_auto_mixed),
        "history_uniform_memory_ratio": memory_uniform_ratio,
        "history_uniform_mixed_ratio": mixed_uniform_ratio,
        "history_uniform_bins": uniform_bins,
    })
    model._kv_head_role_recent_retention = role_info
    budget_msg = "disabled"
    if role_budget_enable:
        budget_msg = (
            f"memory:{int(memory_budget)}, mixed:{int(mixed_budget)}, "
            f"current:{int(current_budget)}, auto_mixed={bool(budget_auto_mixed)}"
        )
    print(
        "[kv_head_role_recent] Installed role-aware recent retention: "
        f"roles={role_info['counts']}, score_mode={role_info['score_mode']}, "
        f"split={'count' if role_info.get('use_count') else 'quantile'}, "
        f"current={role_info['counts']['current']} (thr {role_info['current_threshold']:.6g}), "
        f"memory={role_info['counts']['memory']} (thr {role_info['memory_threshold']:.6g}), "
        f"recent_tokens=(memory:{int(memory_recent_tokens)}, mixed:{int(mixed_recent_tokens)}, "
        f"current:{int(current_recent_tokens)}), window_mode={mode}, "
        f"history_uniform=(memory:{memory_uniform_ratio:g}, "
        f"mixed:{mixed_uniform_ratio:g}, bins:{uniform_bins}), "
        f"uniform_budget={bool(uniform_budget)}, role_budget={budget_msg}"
    )
    return model


def _apply_role_budget_override(model, budget_table, num_keep):
    cfg = getattr(model, "_kv_head_role_recent_retention", None)
    if not cfg or not bool(cfg.get("role_budget_enable", False)):
        return budget_table

    roles = cfg.get("roles", {})
    actual_counts = {"memory": 0, "mixed": 0, "current": 0}
    for layer_idx in range(budget_table.shape[0]):
        for kv_head in range(budget_table.shape[1]):
            role = roles.get((int(layer_idx), int(kv_head)), "mixed")
            if role not in actual_counts:
                role = "mixed"
            actual_counts[role] += 1
    cfg["resolved_budget_role_counts"] = dict(actual_counts)

    memory_budget = int(cfg.get("memory_budget", int(num_keep)))
    current_budget = int(cfg.get("current_budget", int(num_keep)))
    mixed_budget = int(cfg.get("mixed_budget", int(num_keep)))

    if bool(cfg.get("budget_auto_mixed", False)):
        mixed_count = int(actual_counts.get("mixed", 0))
        if mixed_count <= 0:
            raise ValueError("Cannot auto-compute mixed role budget when no mixed KV heads exist.")
        total_heads = int(budget_table.size)
        target_total = int(total_heads * int(num_keep))
        fixed_total = (
            int(actual_counts.get("memory", 0)) * memory_budget
            + int(actual_counts.get("current", 0)) * current_budget
        )
        mixed_budget = int(round((target_total - fixed_total) / mixed_count))
        if mixed_budget < 0:
            raise ValueError(
                "Auto-computed mixed role budget is negative. Reduce memory/current "
                "budgets or selected head counts."
            )
        cfg["resolved_mixed_budget"] = mixed_budget

    role_budgets = {
        "memory": memory_budget,
        "mixed": mixed_budget,
        "current": current_budget,
    }
    override_table = np.empty_like(budget_table)
    for layer_idx in range(budget_table.shape[0]):
        for kv_head in range(budget_table.shape[1]):
            role = roles.get((int(layer_idx), int(kv_head)), "mixed")
            override_table[layer_idx, kv_head] = int(role_budgets.get(role, mixed_budget))
    return override_table


def _role_recent_window_tokens(model, layer_idx, kv_head, default_window):
    cfg = getattr(model, "_kv_head_role_recent_retention", None)
    if not cfg:
        return int(default_window)
    role = cfg.get("roles", {}).get((int(layer_idx), int(kv_head)), "mixed")
    role_window = int(cfg.get(f"{role}_recent_tokens", int(default_window)))
    if cfg.get("window_mode", "override") == "max":
        return max(int(default_window), role_window)
    return max(0, role_window)


def _role_last_frame_ratio(model, layer_idx, kv_head):
    cfg = getattr(model, "_kv_head_role_recent_retention", None)
    if not cfg or not bool(cfg.get("last_frame_by_role", False)):
        return None
    role = _role_recent_role(model, layer_idx, kv_head)
    ratio_map = cfg.get("last_frame_ratio_by_role", {})
    return _clamp01(ratio_map.get(role, 0.0))


def _role_recent_role(model, layer_idx, kv_head):
    cfg = getattr(model, "_kv_head_role_recent_retention", None)
    if not cfg:
        return "mixed"
    role = cfg.get("roles", {}).get((int(layer_idx), int(kv_head)), "mixed")
    return role if role in ("memory", "mixed", "current") else "mixed"


def _role_history_uniform_ratio(model, layer_idx, kv_head):
    cfg = getattr(model, "_kv_head_role_recent_retention", None)
    if not cfg:
        return 0.0
    role = _role_recent_role(model, layer_idx, kv_head)
    if role == "memory":
        return float(cfg.get("history_uniform_memory_ratio", 0.0))
    if role == "mixed":
        return float(cfg.get("history_uniform_mixed_ratio", 0.0))
    return 0.0


def _select_temporally_uniform_history_tokens(
    model,
    layer_idx,
    kv_head,
    text_keep,
    history_positions,
    combined_score,
    target_k,
):
    """Spread a fixed history sub-budget uniformly across temporal bins."""
    target_k = min(max(0, int(target_k)), int(history_positions.numel()))
    if target_k <= 0:
        return torch.empty(
            0, device=combined_score.device, dtype=torch.long
        )
    if target_k >= int(history_positions.numel()):
        return history_positions

    cfg = getattr(model, "_kv_head_role_recent_retention", {})
    max_bins = max(1, int(cfg.get("history_uniform_bins", 32)))
    layer_cache = model.kv_cache.layers[layer_idx]
    pos = layer_cache.get_position_segment(kv_head)
    num_visual_tokens = int(combined_score.numel())
    if pos is not None and pos.shape[1] >= text_keep + num_visual_tokens:
        temporal = pos[0, text_keep:text_keep + num_visual_tokens]
        temporal = temporal.index_select(0, history_positions)
        _, inverse = torch.unique(
            temporal, sorted=True, return_inverse=True
        )
        unique_count = int(inverse.max().item()) + 1
        num_bins = min(max_bins, unique_count)
        bin_ids = torch.div(
            inverse * num_bins,
            max(unique_count, 1),
            rounding_mode="floor",
        ).clamp(max=num_bins - 1)
    else:
        num_bins = min(max_bins, int(history_positions.numel()))
        sequence_rank = torch.arange(
            history_positions.numel(),
            device=history_positions.device,
            dtype=torch.long,
        )
        bin_ids = torch.div(
            sequence_rank * num_bins,
            max(int(history_positions.numel()), 1),
            rounding_mode="floor",
        ).clamp(max=num_bins - 1)

    occupied_bins = torch.unique(bin_ids, sorted=True)
    num_occupied = int(occupied_bins.numel())
    quotas = torch.full(
        (num_occupied,),
        target_k // max(num_occupied, 1),
        device=history_positions.device,
        dtype=torch.long,
    )
    remainder = target_k - int(quotas.sum().item())
    if remainder > 0:
        # Place remainder quotas evenly over the timeline instead of favoring
        # the oldest or newest bins.
        extra_indices = torch.div(
            (torch.arange(remainder, device=history_positions.device) * 2 + 1)
            * num_occupied,
            2 * remainder,
            rounding_mode="floor",
        ).clamp(max=num_occupied - 1)
        quotas[extra_indices] += 1

    selected_parts = []
    selected_mask = torch.zeros(
        num_visual_tokens, device=combined_score.device, dtype=torch.bool
    )
    for bin_offset, bin_id in enumerate(occupied_bins):
        candidates = history_positions[bin_ids == bin_id]
        keep_k = min(int(quotas[bin_offset].item()), int(candidates.numel()))
        if keep_k <= 0:
            continue
        local_top = torch.topk(
            combined_score[candidates], keep_k, sorted=False
        )[1]
        chosen = candidates[local_top]
        selected_parts.append(chosen)
        selected_mask[chosen] = True

    selected_count = sum(int(part.numel()) for part in selected_parts)
    remaining_k = target_k - selected_count
    if remaining_k > 0:
        remaining_positions = history_positions[
            ~selected_mask[history_positions]
        ]
        fill_k = min(remaining_k, int(remaining_positions.numel()))
        if fill_k > 0:
            fill_top = torch.topk(
                combined_score[remaining_positions], fill_k, sorted=False
            )[1]
            selected_parts.append(remaining_positions[fill_top])

    if not selected_parts:
        return torch.empty(
            0, device=combined_score.device, dtype=torch.long
        )
    return torch.unique(torch.cat(selected_parts), sorted=True)


def _select_ragged_keep_indices(model, local_scores, global_scores, mixed_scores, num_keep):
    device = model.device
    visual_start_idx = int(model.visual_start_idx)
    num_layers = len(model.kv_cache.layers)
    num_kv_heads = int(model.kv_cache.layers[0].head_lens.numel())
    config = getattr(model, "_kv_head_budget_config", {})
    scheme = str(config.get("budget_scheme", "relative") or "relative").lower()
    protected_recent_window = (
        max(0, int(config.get("sparsemm_window_size", 0)))
        if scheme.startswith("sparsemm")
        else 0
    )
    memory_current_split = bool(config.get("memory_current_split", False))

    if memory_current_split:
        budget_table = _ragged_budget_table(model, num_keep)
        split_budget_table = _ragged_split_budget_table(model, num_keep)
        pre_head_lens = getattr(
            model, "_kv_head_budget_last_pre_lens_ragged", None
        )
    else:
        budget_table = _ragged_budget_table(model, num_keep)
        split_budget_table = None
        pre_head_lens = None

    role_recent_cfg = getattr(model, "_kv_head_role_recent_retention", None)
    if (
        split_budget_table is not None
        and role_recent_cfg
        and (
            float(role_recent_cfg.get("history_uniform_memory_ratio", 0.0)) > 0.0
            or float(role_recent_cfg.get("history_uniform_mixed_ratio", 0.0)) > 0.0
        )
    ):
        raise ValueError(
            "Role history-uniform retention is not supported with "
            "--kv_head_budget_memory_current_split."
        )
    if role_recent_cfg and bool(role_recent_cfg.get("role_budget_enable", False)):
        if split_budget_table is not None:
            raise ValueError(
                "--kv_head_role_budget_enable is not supported with "
                "--kv_head_budget_memory_current_split in the current implementation."
            )
        budget_table = _apply_role_budget_override(model, budget_table, num_keep)
    elif role_recent_cfg and bool(role_recent_cfg.get("uniform_budget", False)):
        budget_table = np.full_like(budget_table, int(num_keep))
        if split_budget_table is not None:
            split_budget_table = _ragged_split_budget_table(model, num_keep)

    keep_indices = []
    visible_lengths = []

    for layer_idx in range(num_layers):
        layer_scores, recency_alpha, k_decay = _layer_default_policy(
            model, layer_idx, local_scores, global_scores, mixed_scores
        )

        layer_keep = []
        for kv_head in range(num_kv_heads):
            head_len = int(model.kv_cache.layers[layer_idx].head_lens[kv_head].item())
            text_keep = min(visual_start_idx, head_len)
            score, head_recency_alpha, head_k_decay, head_role = _head_policy_score(
                model,
                layer_idx,
                kv_head,
                local_scores,
                global_scores,
                mixed_scores,
                layer_scores,
                recency_alpha,
                k_decay,
            )
            score = score.to(device=device)
            if score.numel() < head_len:
                pad = torch.zeros(head_len - score.numel(), device=device, dtype=score.dtype)
                score = torch.cat([score, pad], dim=0)
            score = score[:head_len]

            num_visual_tokens = max(0, head_len - text_keep)
            if num_visual_tokens <= 0:
                keep = torch.arange(head_len, device=device, dtype=torch.long)
                layer_keep.append(keep)
                visible_lengths.append(int(keep.numel()))
                continue

            visual_score = score[text_keep:]
            positions = torch.arange(num_visual_tokens, device=device, dtype=torch.float32)
            time_distances = (num_visual_tokens - 1 - positions) / max(num_visual_tokens - 1, 1)
            recency = torch.exp(-head_k_decay * time_distances)
            recency = (recency - recency.min()) / (recency.max() - recency.min() + 1e-6)
            attn_norm = (visual_score - visual_score.min()) / (
                visual_score.max() - visual_score.min() + 1e-6
            )
            combined = attn_norm * (1.0 - head_recency_alpha) + recency * head_recency_alpha

            policy = getattr(model, "_kv_head_policy_roles", None)
            if policy and head_role in ("memory", "current"):
                pre_head_lens_policy = getattr(
                    model, "_kv_head_budget_last_pre_lens_ragged", None
                )
                if pre_head_lens_policy is not None:
                    try:
                        pre_len_policy = int(pre_head_lens_policy[layer_idx][kv_head])
                    except (IndexError, TypeError):
                        pre_len_policy = text_keep
                    pre_vis_len_policy = max(0, pre_len_policy - text_keep)
                    pre_vis_len_policy = min(pre_vis_len_policy, num_visual_tokens)
                    if head_role == "memory" and pre_vis_len_policy > 0:
                        combined[:pre_vis_len_policy] += float(policy.get("memory_bonus", 0.1))
                    elif head_role == "current" and pre_vis_len_policy < num_visual_tokens:
                        combined[pre_vis_len_policy:] += float(policy.get("current_bonus", 0.1))

            if split_budget_table is not None:
                # ---- memory/current split selection (ragged) ----
                mem_budget = int(split_budget_table[layer_idx, kv_head, 0])
                cur_budget = int(split_budget_table[layer_idx, kv_head, 1])

                # Per-KV-head region boundary: the pre-chunk head length
                # from _kv_head_budget_last_pre_lens_ragged.
                if pre_head_lens is not None:
                    try:
                        pre_len = int(pre_head_lens[layer_idx][kv_head])
                    except (IndexError, TypeError):
                        pre_len = text_keep
                    pre_vis_len = max(0, pre_len - text_keep)
                    pre_vis_len = min(pre_vis_len, num_visual_tokens)
                else:
                    pre_vis_len = 0

                recent_k = min(_role_recent_window_tokens(model, layer_idx, kv_head, protected_recent_window), num_visual_tokens)
                if recent_k > 0:
                    protected_recent = torch.arange(
                        num_visual_tokens - recent_k,
                        num_visual_tokens,
                        device=device,
                        dtype=torch.long,
                    )
                else:
                    protected_recent = torch.empty(0, device=device, dtype=torch.long)
                protected_last_frame = _select_layerwise_last_frame_tokens(
                    model, layer_idx, kv_head, text_keep, num_visual_tokens, combined
                )
                protected_visual = _unique_visual_parts([protected_recent, protected_last_frame])
                if protected_visual is None:
                    protected_visual = torch.empty(0, device=device, dtype=torch.long)

                protected_mask = torch.zeros(num_visual_tokens, device=device, dtype=torch.bool)
                if protected_visual.numel() > 0:
                    protected_mask[protected_visual] = True
                history_positions = torch.nonzero(~protected_mask, as_tuple=False).flatten()
                split_point = int((history_positions < pre_vis_len).sum().item())
                prev_positions = history_positions[:split_point]
                curr_positions = history_positions[split_point:]

                total_avail = mem_budget + cur_budget
                history_budget = max(0, total_avail - int(protected_visual.numel()))
                if history_budget > 0:
                    mem_frac = mem_budget / max(total_avail, 1)
                    hist_mem_budget = max(0, int(round(history_budget * mem_frac)))
                    hist_cur_budget = history_budget - hist_mem_budget
                else:
                    hist_mem_budget = 0
                    hist_cur_budget = 0

                history_selected = _select_split_budget_tokens(
                    combined_score=combined,
                    previous_positions=prev_positions,
                    current_positions=curr_positions,
                    visual_positions=history_positions,
                    memory_budget=hist_mem_budget,
                    current_budget=hist_cur_budget,
                )
                selected_visual = torch.cat([history_selected, protected_visual])
                selected_visual = torch.unique(selected_visual, sorted=True) + text_keep
            else:
                # ---- original top-k selection + Plan3 frame protection ----
                budget = min(int(budget_table[layer_idx, kv_head]), num_visual_tokens)
                if budget <= 0:
                    selected_visual = torch.empty(0, device=device, dtype=torch.long)
                else:
                    # --- Plan3: Frame-level short-term protection ---
                    stf = int(getattr(model, 'short_term_frames', 0) or 0)
                    frame_protect_tokens = 0
                    if stf > 0:
                        est_tokens_per_frame = max(
                            1,
                            num_visual_tokens // max(model.total_processed_frames, 1),
                        )
                        frame_protect_tokens = min(
                            int(stf * est_tokens_per_frame),
                            num_visual_tokens,
                        )
                    role_recent_window = _role_recent_window_tokens(model, layer_idx, kv_head, protected_recent_window)
                    recent_k = max(int(role_recent_window), frame_protect_tokens)
                    recent_k = min(recent_k, num_visual_tokens)
                    if stf > 0 and layer_idx == 0 and kv_head == 0:
                        print(f"[Plan3] short_term_frames={stf}, "
                              f"est_tok/frame={est_tokens_per_frame:.0f}, "
                              f"protect={frame_protect_tokens}, "
                              f"recent_k={recent_k}, num_vis={num_visual_tokens}")
                    # --------------------------------------------------
                    if recent_k > 0:
                        protected_recent = torch.arange(
                            num_visual_tokens - recent_k,
                            num_visual_tokens,
                            device=device,
                            dtype=torch.long,
                        )
                    else:
                        protected_recent = torch.empty(0, device=device, dtype=torch.long)
                    protected_last_frame = _select_layerwise_last_frame_tokens(
                        model, layer_idx, kv_head, text_keep, num_visual_tokens, combined
                    )
                    protected_visual = _unique_visual_parts([protected_recent, protected_last_frame])
                    if protected_visual is None:
                        protected_visual = torch.empty(0, device=device, dtype=torch.long)

                    protected_mask = torch.zeros(num_visual_tokens, device=device, dtype=torch.bool)
                    if protected_visual.numel() > 0:
                        protected_mask[protected_visual] = True
                    history_positions = torch.nonzero(~protected_mask, as_tuple=False).flatten()
                    history_k = min(max(budget - int(protected_visual.numel()), 0), int(history_positions.numel()))

                    pieces = []
                    if history_k > 0:
                        uniform_ratio = _role_history_uniform_ratio(
                            model, layer_idx, kv_head
                        )
                        uniform_k = min(
                            history_k,
                            max(0, int(round(history_k * uniform_ratio))),
                        )
                        uniform_selected = _select_temporally_uniform_history_tokens(
                            model=model,
                            layer_idx=layer_idx,
                            kv_head=kv_head,
                            text_keep=text_keep,
                            history_positions=history_positions,
                            combined_score=combined,
                            target_k=uniform_k,
                        )
                        if uniform_selected.numel() > 0:
                            pieces.append(uniform_selected)

                        score_k = history_k - int(uniform_selected.numel())
                        if score_k > 0:
                            score_mask = torch.ones(
                                num_visual_tokens,
                                device=device,
                                dtype=torch.bool,
                            )
                            if uniform_selected.numel() > 0:
                                score_mask[uniform_selected] = False
                            score_positions = history_positions[
                                score_mask[history_positions]
                            ]
                            score_k = min(score_k, int(score_positions.numel()))
                            if score_k > 0:
                                score_top = torch.topk(
                                    combined[score_positions],
                                    score_k,
                                    sorted=False,
                                )[1]
                                pieces.append(score_positions[score_top])
                    if protected_visual.numel() > 0:
                        pieces.append(protected_visual)
                    if pieces:
                        selected_visual = torch.unique(torch.cat(pieces), sorted=True) + text_keep
                    else:
                        selected_visual = torch.empty(0, device=device, dtype=torch.long)

            keep = torch.cat([
                torch.arange(text_keep, device=device, dtype=torch.long),
                selected_visual,
            ])
            keep = torch.unique(keep, sorted=True)
            layer_keep.append(keep)
            visible_lengths.append(int(keep.numel()))
        keep_indices.append(layer_keep)

    return keep_indices, budget_table, visible_lengths


def _sync_position_cache_from_ragged(model):
    if not _is_ragged_cache(model.kv_cache):
        return
    for layer_idx, layer in enumerate(model.kv_cache.layers):
        if layer.flat_pos is None:
            model._position_ids_cache[layer_idx] = None
        else:
            model._position_ids_cache[layer_idx] = layer.flat_pos


def _apply_plan2_partition_ragged(model):
    """Plan2: partition position IDs per-head in ragged cache.

    After pruning, each head may have fragmented position IDs.  This function
    compacts video-token positions, assigning old-frame tokens to a compact
    low range and recent-frame tokens to a higher range (temporal dimension
    only; spatial dims stay uniformly compacted).  Keys are re-rotated to
    match the new position assignments.
    """
    pframes = getattr(model, 'partition_position_frames', 0) or 0
    if pframes <= 0:
        return

    from inference.reindex_3d import (
        compute_cos_sin_for_positions, rotary_delta,
        apply_rotary_delta_to_keys_only,
    )

    dtype = model.kv_cache.layers[0].flat_k.dtype
    device = model.kv_cache.layers[0].flat_k.device
    mrope_section = model._mrope_section
    text_offset = int(model.visual_start_idx)
    did_any = False

    for layer_idx, layer in enumerate(model.kv_cache.layers):
        if layer.flat_pos is None:
            continue

        head_start = 0
        new_pos_parts = []

        for kv_head in range(int(layer.head_lens.numel())):
            head_len = int(layer.head_lens[kv_head].item())
            head_end = head_start + head_len

            if head_len == 0:
                new_pos_parts.append(torch.empty((3, 0), device=device, dtype=torch.float32))
                head_start = head_end
                continue

            old_pos = layer.flat_pos[:, head_start:head_end]   # [3, head_len]

            num_text = int((old_pos[0] < text_offset).sum().item())
            num_video = head_len - num_text
            if num_video <= 1:
                # Not enough video tokens to partition; keep original
                new_pos_parts.append(old_pos)
                head_start = head_end
                continue

            video_indices = torch.arange(num_text, head_len, device=device, dtype=torch.long)
            temporal_vals = old_pos[0, video_indices]
            unique_temporal = torch.unique(temporal_vals, sorted=True)

            if len(unique_temporal) <= pframes:
                # Not enough unique temporal positions; keep original
                new_pos_parts.append(old_pos)
                head_start = head_end
                continue

            # --- Partition: oldest temporal → low, newest → high ---
            split_point = len(unique_temporal) - pframes
            recent_threshold = unique_temporal[split_point].item()
            recent_mask = temporal_vals >= recent_threshold
            old_mask = ~recent_mask

            old_video_idx = video_indices[old_mask]
            recent_video_idx = video_indices[recent_mask]

            new_pos = old_pos.clone()

            for dim in range(3):
                if dim == 0:
                    # Temporal: old tokens → low, recent tokens → high
                    old_vals = old_pos[dim, old_video_idx]
                    uniq_old, inv_old = torch.unique(old_vals, sorted=True,
                                                     return_inverse=True)
                    compact_old = torch.arange(len(uniq_old), device=device) + text_offset
                    new_pos[dim, old_video_idx] = compact_old[inv_old].to(new_pos.dtype)

                    recent_vals = old_pos[dim, recent_video_idx]
                    uniq_recent, inv_recent = torch.unique(recent_vals, sorted=True,
                                                            return_inverse=True)
                    recent_start = text_offset + len(uniq_old)
                    compact_recent = torch.arange(len(uniq_recent), device=device) + recent_start
                    new_pos[dim, recent_video_idx] = compact_recent[inv_recent].to(new_pos.dtype)
                else:
                    # Spatial (h, w): uniform compaction for all video tokens
                    all_vals = old_pos[dim]
                    uniq_all, inv_all = torch.unique(all_vals, sorted=True,
                                                     return_inverse=True)
                    compact_all = torch.arange(len(uniq_all), device=device) + text_offset
                    new_pos[dim, :] = compact_all[inv_all].to(new_pos.dtype)

            # Re-rotate keys to match new positions
            cos_old, sin_old = compute_cos_sin_for_positions(
                model.language_model, head_len, old_pos.unsqueeze(1), dtype, device)
            cos_new, sin_new = compute_cos_sin_for_positions(
                model.language_model, head_len, new_pos.unsqueeze(1), dtype, device)
            cos_delta, sin_delta = rotary_delta(cos_old, sin_old, cos_new, sin_new)

            k_head = layer.flat_k[head_start:head_end].unsqueeze(0).unsqueeze(1)
            k_new = apply_rotary_delta_to_keys_only(k_head, cos_delta, sin_delta, mrope_section)
            layer.flat_k[head_start:head_end] = k_new[0, 0]

            new_pos_parts.append(new_pos)
            head_start = head_end
            did_any = True

        if did_any and all(p is not None for p in new_pos_parts):
            layer.flat_pos = torch.cat(new_pos_parts, dim=1).contiguous()

    if did_any:
        _sync_position_cache_from_ragged(model)
        print(f"[Plan2] Partitioned position IDs in ragged cache "
              f"(protect_frames={pframes})")


def _ragged_pseudo_forward(model, local_question=None, global_question=None):
    device = model.device
    horizon_config = getattr(model, "_horizonkv_config", None)
    if horizon_config:
        from head_analysis.horizonkv_policy import (
            horizonkv_needs_compression,
            select_horizonkv_keep_indices,
        )

        offsets = model._get_next_global_offset_per_layer()
        probe_scores = []
        for probe in horizon_config["pseudo_probes"]:
            probe_prompt = model.get_prompt(probe)
            input_ids = model.processor.tokenizer(probe_prompt).input_ids
            input_ids = torch.as_tensor(
                [input_ids], device=device, dtype=torch.long
            )
            probe_scores.append(
                _ragged_query_scores(
                    model,
                    input_ids,
                    offsets,
                    score_last_n=int(horizon_config["query_pool_last_n"]),
                )
            )

        keep_indices, budget_table, visible_lengths = (
            select_horizonkv_keep_indices(model, probe_scores)
        )
        if not horizonkv_needs_compression(model):
            return

        print(
            "Applying ragged KV-Cache compression via HorizonKV "
            "because at least one Head exceeds its fixed total budget"
        )
        model.kv_cache.prune_per_head(keep_indices)
        _mark_prospective_trace_compressed(model)
        _sync_position_cache_from_ragged(model)
        _apply_plan2_partition_ragged(model)
        rmin, rmax, rmean = model.kv_cache.stats()
        stats = model._horizonkv_last_stats
        print(
            "[HorizonKV] prune stats: "
            f"kv_budget=[{budget_table.min()}, {budget_table.max()}], "
            f"recent=[{stats['recent_min']}, {stats['recent_max']}], "
            f"memory=[{stats['memory_min']}, {stats['memory_max']}], "
            f"visible=[{min(visible_lengths)}, {max(visible_lengths)}], "
            f"head_lens=[{rmin}, {rmax}], mean={rmean:.1f}"
        )
        return

    if local_question is None:
        local_question = "What is happening in the video?"
    if global_question is None:
        global_question = "What is the main topic of the video?"

    offsets = model._get_next_global_offset_per_layer()

    local_input_ids = model.processor.tokenizer(local_question).input_ids
    local_input_ids = torch.as_tensor([local_input_ids], device=device, dtype=torch.long)
    local_scores = _ragged_query_scores(model, local_input_ids, offsets)

    global_input_ids = model.processor.tokenizer(global_question).input_ids
    global_input_ids = torch.as_tensor([global_input_ids], device=device, dtype=torch.long)
    global_scores = _ragged_query_scores(model, global_input_ids, offsets)

    mixed_question = local_question + "; " + global_question
    mixed_input_ids = model.processor.tokenizer(mixed_question).input_ids
    mixed_input_ids = torch.as_tensor([mixed_input_ids], device=device, dtype=torch.long)
    mixed_scores = _ragged_query_scores(model, mixed_input_ids, offsets)
    prospective_config = getattr(model, "_prospective_lifecycle_trace_config", None)
    if prospective_config:
        hermes_scores = _default_hermes_scores_for_trace(
            model, local_scores, global_scores, mixed_scores
        )
        _write_prospective_lifecycle_trace(model, hermes_scores, "hermes")
        prospective_scores = mixed_scores
        if int(prospective_config["score_last_n"]) > 0:
            prospective_scores = _ragged_query_scores(
                model,
                mixed_input_ids,
                offsets,
                score_last_n=prospective_config["score_last_n"],
            )
        _write_prospective_lifecycle_trace(
            model, prospective_scores, "pseudo_mixed"
        )

    print(f"GPU memory usage: {model.get_gpu_memory_usage_gb()} GB")
    timestamp_config = getattr(model, "_timestamp_v8_config", None)
    if timestamp_config:
        from head_analysis.timestamp_v8_policy import (
            select_timestamp_v8_keep_indices,
            timestamp_v8_needs_compression,
        )

        budget_table = _ragged_budget_table(model, model.kv_size)
        if not timestamp_v8_needs_compression(model, budget_table):
            return
        keep_indices, visible_lengths = select_timestamp_v8_keep_indices(
            model,
            local_scores,
            global_scores,
            mixed_scores,
            budget_table,
        )
        from head_analysis.timestamp_v8_policy import write_ragged_frame_trace
        write_ragged_frame_trace(
            model, keep_indices, budget_table, "timestamp-v8",
            local_question=local_question, global_question=global_question,
        )
        model.kv_cache.prune_per_head(keep_indices)
        _mark_prospective_trace_compressed(model)
        _sync_position_cache_from_ragged(model)
        _apply_plan2_partition_ragged(model)
        rmin, rmax, rmean = model.kv_cache.stats()
        stats = model._timestamp_v8_last_stats
        print(
            "[TimestampV8] prune stats: "
            f"kv_budget=[{budget_table.min()}, {budget_table.max()}], "
            f"protected=[{stats['protected_min']}, {stats['protected_max']}], "
            f"attention=[{stats['attention_min']}, {stats['attention_max']}], "
            f"recent=[{stats['recent_min']}, {stats['recent_max']}], "
            f"middle=[{stats.get('middle_min', 0)}, {stats.get('middle_max', 0)}], "
            f"far=[{stats.get('far_min', 0)}, {stats.get('far_max', 0)}], "
            f"visible=[{min(visible_lengths)}, {max(visible_lengths)}], "
            f"head_lens=[{rmin}, {rmax}], mean={rmean:.1f}"
        )
        return

    current_len = max(model.kv_cache.max_lens_per_layer())
    if current_len <= model.kv_size:
        return

    if getattr(model, "_rolekv_ragged_config", None):
        from head_analysis.rolekv_ragged_policy import select_rolekv_ragged_keep_indices

        budget_table = _ragged_budget_table(model, model.kv_size)
        keep_indices, visible_lengths = select_rolekv_ragged_keep_indices(
            model,
            local_scores,
            global_scores,
            mixed_scores,
            num_keep=model.kv_size,
            budget_table=budget_table,
        )
        selector_name = "RoleKV-ragged"
    else:
        keep_indices, budget_table, visible_lengths = _select_ragged_keep_indices(
            model,
            local_scores,
            global_scores,
            mixed_scores,
            num_keep=model.kv_size,
        )
        config = getattr(model, "_kv_head_budget_config", {})
        if config.get("memory_current_split"):
            selector_name = "split-ragged"
        else:
            selector_name = "default-ragged"

    frame_trace_config = getattr(model, "_ragged_frame_trace_config", None)
    if frame_trace_config:
        from head_analysis.timestamp_v8_policy import write_ragged_frame_trace
        write_ragged_frame_trace(
            model, keep_indices, budget_table, selector_name,
            local_question=local_question, global_question=global_question,
        )
    _write_ragged_redundancy_trace(
        model, keep_indices, budget_table, selector_name
    )

    print(
        f"Applying ragged KV-Cache compression via {selector_name} "
        f"due to max head_len > {model.kv_size}"
    )
    model.kv_cache.prune_per_head(keep_indices)
    _mark_prospective_trace_compressed(model)
    _sync_position_cache_from_ragged(model)
    _apply_plan2_partition_ragged(model)  # Plan2: partition position IDs in ragged cache
    rmin, rmax, rmean = model.kv_cache.stats()
    print(
        "[kv_head_ragged_prefill] prune stats: "
        f"kv_budget=[{budget_table.min()}, {budget_table.max()}], "
        f"visible=[{min(visible_lengths)}, {max(visible_lengths)}], "
        f"head_lens=[{rmin}, {rmax}], mean={rmean:.1f}"
    )
    role_stats = getattr(model, "_rolekv_ragged_last_stats", None)
    if role_stats:
        print(
            "[RoleKV-ragged] selection stats: "
            f"roles={role_stats.get('selected_kv_roles')}, "
            f"quota={role_stats.get('quota_stats')}, "
            f"mode={role_stats.get('mode')}, "
            f"quota_ratio={role_stats.get('quota_ratio')}"
        )


def _prune_partial_deferred_last_frame_ragged(model, history_lens, keep_ratio):
    """Prune only newly appended final-frame KV while preserving history."""
    if not _is_ragged_cache(model.kv_cache):
        raise RuntimeError("Partial defer requires a physical ragged KV cache")
    ratio = float(keep_ratio)
    if not 0.0 < ratio < 1.0:
        raise ValueError("Partial defer pruning requires keep_ratio in (0, 1)")
    if len(history_lens) != len(model.kv_cache.layers):
        raise ValueError("Partial defer history snapshot does not match model layers")

    local_question, global_question = model.predict_next_question()
    offsets = model._get_next_global_offset_per_layer()

    def query_scores(question):
        input_ids = model.processor.tokenizer(question).input_ids
        input_ids = torch.as_tensor(
            [input_ids], device=model.device, dtype=torch.long
        )
        return _ragged_query_scores(model, input_ids, offsets)

    local_scores = query_scores(local_question)
    global_scores = query_scores(global_question)
    mixed_scores = query_scores(local_question + "; " + global_question)

    keep_indices = []
    appended_counts = []
    retained_counts = []
    for layer_idx, layer_cache in enumerate(model.kv_cache.layers):
        layer_scores, recency_alpha, k_decay = _layer_default_policy(
            model, layer_idx, local_scores, global_scores, mixed_scores
        )
        layer_keep = []
        for kv_head in range(int(layer_cache.head_lens.numel())):
            head_len = int(layer_cache.head_lens[kv_head].item())
            history_len = int(history_lens[layer_idx][kv_head].item())
            history_len = max(0, min(history_len, head_len))
            appended = head_len - history_len
            keep_k = min(appended, int(np.ceil(appended * ratio)))

            prefix = torch.arange(
                history_len, device=model.device, dtype=torch.long
            )
            if keep_k >= appended:
                frame_keep = torch.arange(
                    history_len, head_len, device=model.device, dtype=torch.long
                )
            elif keep_k > 0:
                score, head_recency_alpha, head_k_decay, _ = _head_policy_score(
                    model,
                    layer_idx,
                    kv_head,
                    local_scores,
                    global_scores,
                    mixed_scores,
                    layer_scores,
                    recency_alpha,
                    k_decay,
                )
                score = score.to(device=model.device)
                if score.numel() < head_len:
                    score = torch.cat([
                        score,
                        torch.zeros(
                            head_len - score.numel(),
                            device=model.device,
                            dtype=score.dtype,
                        ),
                    ])
                score = score[:head_len]
                positions = torch.arange(
                    head_len, device=model.device, dtype=torch.float32
                )
                distances = (head_len - 1 - positions) / max(head_len - 1, 1)
                recency = torch.exp(-head_k_decay * distances)
                recency = (recency - recency.min()) / (
                    recency.max() - recency.min() + 1e-6
                )
                attn_norm = (score - score.min()) / (
                    score.max() - score.min() + 1e-6
                )
                combined = (
                    attn_norm * (1.0 - head_recency_alpha)
                    + recency * head_recency_alpha
                )
                frame_positions = torch.arange(
                    history_len, head_len, device=model.device, dtype=torch.long
                )
                local_top = torch.topk(
                    combined[frame_positions], keep_k, sorted=False
                )[1]
                frame_keep = frame_positions[local_top]
            else:
                frame_keep = torch.empty(
                    0, device=model.device, dtype=torch.long
                )
            layer_keep.append(torch.cat([prefix, frame_keep]))
            appended_counts.append(appended)
            retained_counts.append(keep_k)
        keep_indices.append(layer_keep)

    model.kv_cache.prune_per_head(keep_indices)
    _mark_prospective_trace_compressed(model)
    _sync_position_cache_from_ragged(model)
    rmin, rmax, rmean = model.kv_cache.stats()
    print(
        "[partial_defer] Pruned only final-frame KV: "
        f"ratio={ratio:g}, appended=[{min(appended_counts)}, {max(appended_counts)}], "
        f"retained=[{min(retained_counts)}, {max(retained_counts)}], "
        f"head_lens=[{rmin}, {rmax}], mean={rmean:.1f}"
    )


def _prepare_head_budget_deferred_history_ragged(model, keep_ratio,
                                                  budget_threshold):
    """Reserve a fixed within-head bank only for heads above a budget cutoff."""
    if not _is_ragged_cache(model.kv_cache):
        raise RuntimeError("Head-budget defer requires a physical ragged KV cache")
    ratio = float(keep_ratio)
    threshold = int(budget_threshold)
    if not 0.0 < ratio < 1.0:
        raise ValueError("Head-budget defer ratio must be in (0, 1)")
    base_budget = _ragged_budget_table(model, model.kv_size).copy()
    protected = base_budget > threshold
    reserve = np.zeros_like(base_budget)
    reserve[protected] = np.ceil(base_budget[protected] * ratio).astype(np.int64)
    reserve = np.minimum(reserve, np.maximum(base_budget - 1, 0))
    history_budget = base_budget - reserve

    original_kv_size = model.kv_size
    model._ragged_budget_table_override = history_budget
    try:
        # Force the regular selector to run even when every physical length is
        # below the original nominal K. The exact per-head table is supplied by
        # the override above.
        model.kv_size = 1
        model.predict_and_compress()
    finally:
        model.kv_size = original_kv_size
        del model._ragged_budget_table_override

    print(
        "[head_budget_defer] Reserved current-frame bank: "
        f"heads={int(protected.sum())}/{base_budget.size}, "
        f"threshold={threshold}, ratio={ratio:g}, "
        f"reserve=[{int(reserve[protected].min()) if protected.any() else 0}, "
        f"{int(reserve[protected].max()) if protected.any() else 0}], "
        f"history_budget=[{int(history_budget.min())}, {int(history_budget.max())}]"
    )
    return base_budget, reserve


def _prune_head_budget_deferred_last_frame_ragged(
    model, history_lens, base_budget_table, reserve_table
):
    """Use normal LU selection on other heads and a strict dual bank on large heads."""
    if not _is_ragged_cache(model.kv_cache):
        raise RuntimeError("Head-budget defer requires a physical ragged KV cache")
    base_budget_table = np.asarray(base_budget_table, dtype=np.int64)
    reserve_table = np.asarray(reserve_table, dtype=np.int64)
    if base_budget_table.shape != reserve_table.shape:
        raise ValueError("Base-budget and reserve tables must have identical shapes")

    local_question, global_question = model.predict_next_question()
    offsets = model._get_next_global_offset_per_layer()

    def query_scores(question):
        input_ids = model.processor.tokenizer(question).input_ids
        input_ids = torch.as_tensor(
            [input_ids], device=model.device, dtype=torch.long
        )
        return _ragged_query_scores(model, input_ids, offsets)

    local_scores = query_scores(local_question)
    global_scores = query_scores(global_question)
    mixed_scores = query_scores(local_question + "; " + global_question)

    model._ragged_budget_table_override = base_budget_table
    try:
        normal_keep, _, _ = _select_ragged_keep_indices(
            model, local_scores, global_scores, mixed_scores,
            num_keep=model.kv_size,
        )
    finally:
        del model._ragged_budget_table_override

    keep_indices = []
    appended_counts = []
    retained_counts = []
    protected_heads = 0
    for layer_idx, layer_cache in enumerate(model.kv_cache.layers):
        layer_scores, recency_alpha, k_decay = _layer_default_policy(
            model, layer_idx, local_scores, global_scores, mixed_scores
        )
        layer_keep = []
        for kv_head in range(int(layer_cache.head_lens.numel())):
            reserve_k = int(reserve_table[layer_idx, kv_head])
            if reserve_k <= 0:
                layer_keep.append(normal_keep[layer_idx][kv_head])
                continue

            protected_heads += 1
            head_len = int(layer_cache.head_lens[kv_head].item())
            history_len = int(history_lens[layer_idx][kv_head].item())
            history_len = max(0, min(history_len, head_len))
            appended = head_len - history_len
            keep_k = min(appended, reserve_k)
            prefix = torch.arange(
                history_len, device=model.device, dtype=torch.long
            )
            if keep_k >= appended:
                frame_keep = torch.arange(
                    history_len, head_len, device=model.device, dtype=torch.long
                )
            elif keep_k > 0:
                score, head_recency_alpha, head_k_decay, _ = _head_policy_score(
                    model, layer_idx, kv_head,
                    local_scores, global_scores, mixed_scores,
                    layer_scores, recency_alpha, k_decay,
                )
                score = score.to(device=model.device)
                if score.numel() < head_len:
                    score = torch.cat([
                        score,
                        torch.zeros(
                            head_len - score.numel(), device=model.device,
                            dtype=score.dtype,
                        ),
                    ])
                score = score[:head_len]
                positions = torch.arange(
                    head_len, device=model.device, dtype=torch.float32
                )
                distances = (head_len - 1 - positions) / max(head_len - 1, 1)
                recency = torch.exp(-head_k_decay * distances)
                recency = (recency - recency.min()) / (
                    recency.max() - recency.min() + 1e-6
                )
                attention = (score - score.min()) / (
                    score.max() - score.min() + 1e-6
                )
                combined = (
                    attention * (1.0 - head_recency_alpha)
                    + recency * head_recency_alpha
                )
                frame_positions = torch.arange(
                    history_len, head_len, device=model.device, dtype=torch.long
                )
                local_top = torch.topk(
                    combined.index_select(0, frame_positions),
                    keep_k, sorted=False,
                ).indices
                frame_keep = frame_positions.index_select(0, local_top)
            else:
                frame_keep = torch.empty(
                    0, device=model.device, dtype=torch.long
                )
            layer_keep.append(torch.cat([prefix, frame_keep]))
            appended_counts.append(appended)
            retained_counts.append(keep_k)
        keep_indices.append(layer_keep)

    model.kv_cache.prune_per_head(keep_indices)
    _mark_prospective_trace_compressed(model)
    _sync_position_cache_from_ragged(model)
    rmin, rmax, rmean = model.kv_cache.stats()
    print(
        "[head_budget_defer] Built strict dual-bank answer cache: "
        f"protected_heads={protected_heads}, "
        f"appended=[{min(appended_counts)}, {max(appended_counts)}], "
        f"retained=[{min(retained_counts)}, {max(retained_counts)}], "
        f"head_lens=[{rmin}, {rmax}], mean={rmean:.1f}"
    )


def apply_kv_head_ragged_prefill(model):
    """Install physical per-KV-head ragged cache for prefill/chunk/decode."""
    original_encode_init_prompt = model.encode_init_prompt
    original_encode_video_chunk = model.encode_video_chunk
    original_predict_and_compress = model.predict_and_compress
    original_question_answering = model.question_answering
    original_get_cache_seq_len_per_layer = model._get_cache_seq_len_per_layer
    original_get_next_global_offset_per_layer = model._get_next_global_offset_per_layer
    original_truncate_kv_cache = model._truncate_kv_cache

    @torch.inference_mode()
    def get_cache_seq_len_per_layer_ragged(self):
        if _is_ragged_cache(self.kv_cache):
            return self.kv_cache.max_lens_per_layer()
        return original_get_cache_seq_len_per_layer()

    def get_next_global_offset_per_layer_ragged(self):
        if _is_ragged_cache(self.kv_cache):
            offsets = []
            for layer_idx, layer in enumerate(self.kv_cache.layers):
                max_pos = None
                if layer.flat_pos is not None and layer.flat_pos.numel() > 0:
                    max_pos = layer.flat_pos.max().item()
                pos_cache = self._position_ids_cache[layer_idx]
                if pos_cache is not None and pos_cache.numel() > 0:
                    cache_max = pos_cache.max().item()
                    max_pos = cache_max if max_pos is None else max(max_pos, cache_max)
                offsets.append(0 if max_pos is None else int(max_pos) + 1)
            return offsets
        return original_get_next_global_offset_per_layer()

    def truncate_kv_cache_ragged(self, target_lengths):
        if _is_ragged_cache(self.kv_cache):
            self.kv_cache.truncate(target_lengths)
            _sync_position_cache_from_ragged(self)
            return
        return original_truncate_kv_cache(target_lengths)

    @torch.inference_mode()
    def encode_init_prompt_ragged(self):
        original_encode_init_prompt()
        self.kv_cache = HeadRaggedCache.from_dense_cache(self, self.kv_cache)
        horizon_config = getattr(self, "_horizonkv_config", None)
        if horizon_config:
            horizon_config["current_chunk_id"] = -1
            horizon_config["next_chunk_id"] = 0
        timestamp_config = getattr(self, "_timestamp_v8_config", None)
        if timestamp_config:
            timestamp_config["current_frame_id"] = -1
            timestamp_config["next_frame_id"] = 0
        frame_trace_config = getattr(self, "_ragged_frame_trace_config", None)
        if frame_trace_config:
            frame_trace_config["current_frame_id"] = -1
            frame_trace_config["next_frame_id"] = 0
        readout_trace_config = getattr(self, "_ragged_readout_trace_config", None)
        if readout_trace_config:
            readout_trace_config["current_frame_id"] = -1
            readout_trace_config["next_frame_id"] = 0
        token_trace_config = getattr(self, "_ragged_token_trace_config", None)
        if token_trace_config:
            token_trace_config["current_frame_id"] = -1
            token_trace_config["next_frame_id"] = 0
            token_trace_config["next_visual_token_uid"] = 0
        prospective_config = getattr(self, "_prospective_lifecycle_trace_config", None)
        if prospective_config:
            prospective_config["current_frame_id"] = -1
            prospective_config["next_frame_id"] = 0
            prospective_config["cache_compressed"] = False
        _sync_position_cache_from_ragged(self)
        rmin, rmax, rmean = self.kv_cache.stats()
        print(
            "[kv_head_ragged_prefill] init cache converted to ragged: "
            f"min={rmin}, max={rmax}, mean={rmean:.1f}"
        )

    @torch.inference_mode()
    def encode_video_chunk_ragged(self, video_chunk, is_last_chunk=False, **kwargs):
        if not _is_ragged_cache(self.kv_cache):
            original_encode_video_chunk(video_chunk, is_last_chunk=is_last_chunk, **kwargs)
            return
        if video_chunk is None or (hasattr(video_chunk, "shape") and video_chunk.shape[0] == 0):
            return
        self._ragged_frame_trace_is_last_chunk = bool(is_last_chunk)

        from inference.qwenvl_hermes import get_qwen2_5_vl_position_ids

        if len(video_chunk.shape) == 4 and video_chunk.shape[-1] == 3:
            video_chunk = video_chunk.permute(0, 3, 1, 2)

        # --- Plan1: Last-frame high resolution (ragged path) ---
        use_highres_last = (
            getattr(self, 'last_frame_stride', 1.0) != 1.0
            and is_last_chunk
            and getattr(self, 'processor_last', None) is not None
        )

        multi = getattr(self, 'multi_scale', False)
        layered = getattr(self, 'layered_resolution', False)

        if use_highres_last and video_chunk.shape[0] > 1:
            if layered:
                logger.info(f"[PlanA] Ragged layered-res: {video_chunk.shape[0] - 1} "
                            f"normal + 1 frame layered "
                            f"(stride={self.last_frame_stride})")
                _encode_ragged_frames(self, video_chunk[:-1], self.processor)
                _snapshot_shallow_ragged(self)
                _encode_ragged_frames(self, video_chunk[-1:], self.processor)
                _snapshot_deep_ragged(self)
                _restore_shallow_ragged(self)
                _encode_ragged_frames(self, video_chunk[-1:], self.processor_last,
                                       count_as_frame=False)
                _restore_deep_ragged(self)
            elif multi:
                logger.info(f"[Plan1] Ragged multi-scale: {video_chunk.shape[0] - 1} "
                            f"normal + 1 frame @ normal+high-res "
                            f"(stride={self.last_frame_stride})")
                _encode_ragged_frames(self, video_chunk[:-1], self.processor)
                _encode_ragged_frames(self, video_chunk[-1:], self.processor)
                _encode_ragged_frames(self, video_chunk[-1:], self.processor_last,
                                       count_as_frame=False)
            else:
                logger.info(f"[Plan1] Ragged high-res: {video_chunk.shape[0] - 1} "
                            f"normal + 1 high-res (stride={self.last_frame_stride})")
                _encode_ragged_frames(self, video_chunk[:-1], self.processor)
                if getattr(self, 'transient_highres', False):
                    _snapshot_ragged_for_transient(self, video_chunk[-1:])
                _encode_ragged_frames(self, video_chunk[-1:], self.processor_last)
            self.last_encoded_frames = video_chunk.shape[0]
        elif use_highres_last and video_chunk.shape[0] == 1:
            if layered:
                logger.info(f"[PlanA] Ragged layered-res: single frame "
                            f"(stride={self.last_frame_stride})")
                _snapshot_shallow_ragged(self)
                _encode_ragged_frames(self, video_chunk, self.processor)
                _snapshot_deep_ragged(self)
                _restore_shallow_ragged(self)
                _encode_ragged_frames(self, video_chunk, self.processor_last,
                                       count_as_frame=False)
                _restore_deep_ragged(self)
            elif multi:
                logger.info(f"[Plan1] Ragged multi-scale: single frame "
                            f"@ normal+high-res (stride={self.last_frame_stride})")
                _encode_ragged_frames(self, video_chunk, self.processor)
                _encode_ragged_frames(self, video_chunk, self.processor_last,
                                       count_as_frame=False)
            else:
                logger.info(f"[Plan1] Ragged high-res: single frame "
                            f"(stride={self.last_frame_stride})")
                if getattr(self, 'transient_highres', False):
                    _snapshot_ragged_for_transient(self, video_chunk)
                _encode_ragged_frames(self, video_chunk, self.processor_last)
        else:
            _encode_ragged_frames(self, video_chunk, self.processor)

    def _snapshot_ragged_for_transient(self, last_frame):
        """Snapshot ragged head_lens before transient high-res encoding."""
        self._transient_head_lens = {}
        for layer_idx in range(self.num_layers):
            self._transient_head_lens[layer_idx] = (
                self.kv_cache.layers[layer_idx].head_lens.clone())
        self._transient_last_frame = last_frame.detach().clone()
        self._transient_total_frames = self.total_processed_frames

    def replace_highres_with_normal_ragged(self):
        """Restore ragged cache, re-encode last frame at normal res, re-compress."""
        if not hasattr(self, '_transient_head_lens'):
            return
        logger.info("[PlanE] Ragged: replacing transient high-res with normal-res")

        for layer_idx in range(self.num_layers):
            target = self._transient_head_lens[layer_idx]
            self.kv_cache.layers[layer_idx].truncate(target)
        _sync_position_cache_from_ragged(self)
        self.total_processed_frames = self._transient_total_frames

        last_frame = self._transient_last_frame
        del self._transient_head_lens, self._transient_last_frame
        del self._transient_total_frames

        _encode_ragged_frames(self, last_frame, self.processor,
                               count_as_frame=False)
        self.predict_and_compress()

    def _snapshot_deep_ragged(self):
        """Snapshot non-sensory layer (L8-27) head_lens."""
        start = self.short_term_threshold  # L8
        self._layered_ragged_deep = {}
        for layer_idx in range(start, self.num_layers):
            self._layered_ragged_deep[layer_idx] = (
                self.kv_cache.layers[layer_idx].head_lens.clone())

    def _restore_deep_ragged(self):
        """Truncate L8-27 ragged layers to snapshot (normal-res only)."""
        if not hasattr(self, '_layered_ragged_deep'):
            return
        start = self.short_term_threshold  # L8
        for layer_idx in range(start, self.num_layers):
            self.kv_cache.layers[layer_idx].truncate(
                self._layered_ragged_deep[layer_idx])
        del self._layered_ragged_deep
        _sync_position_cache_from_ragged(self)
        logger.info(f"[PlanA] Ragged layers L{start}-{self.num_layers-1} "
                     "restored to normal-res only")

    def _snapshot_shallow_ragged(self):
        """Snapshot ALL layers' head_lens before normal-res pass."""
        self._layered_ragged_shallow = {}
        for layer_idx in range(self.num_layers):
            self._layered_ragged_shallow[layer_idx] = (
                self.kv_cache.layers[layer_idx].head_lens.clone())

    def _restore_shallow_ragged(self):
        """Truncate ALL ragged layers to pre-normal-res snapshot."""
        if not hasattr(self, '_layered_ragged_shallow'):
            return
        for layer_idx in range(self.num_layers):
            self.kv_cache.layers[layer_idx].truncate(
                self._layered_ragged_shallow[layer_idx])
        del self._layered_ragged_shallow
        _sync_position_cache_from_ragged(self)
        logger.info("[PlanA] Ragged all layers restored to normal-frames-only "
                     "(sensory L0-7 will receive high-res)")

    def _encode_ragged_frames(self, video_chunk, processor,
                                 count_as_frame=True):
        """Encode frames into ragged KV cache.

        When count_as_frame=False, tokens are added but frame counters
        are untouched (multi-scale encoding guard).
        """
        from inference.qwenvl_hermes import get_qwen2_5_vl_position_ids

        if hasattr(self, "_apply_fixed_frame_canvas"):
            video_chunk = self._apply_fixed_frame_canvas(video_chunk)
        video_input = processor(text=[""], videos=video_chunk, return_tensors="pt").to(
            self.device, self.dtype
        )
        pixel_values_videos = video_input["pixel_values_videos"]
        video_grid_thw = video_input["video_grid_thw"]
        video_features = self.get_video_features(pixel_values_videos, video_grid_thw)[0].unsqueeze(0)

        offsets = self._get_next_global_offset_per_layer()
        q_len = video_features.shape[1]
        batch = video_features.shape[0]
        base_offset = offsets[0]
        grid_pos_ids = get_qwen2_5_vl_position_ids(
            video_grid_thw[0].tolist(),
            q_len,
            offset=base_offset,
            vision_config=self.config.vision_config,
            sample_fps=self.sample_fps,
        ).to(self.device)

        position_ids_by_layer = []
        for layer_idx in range(self.num_layers):
            layer_offset = offsets[layer_idx]
            current_layer_pos = grid_pos_ids.clone()
            if layer_offset != base_offset:
                current_layer_pos = current_layer_pos + (layer_offset - base_offset)
            position_ids_by_layer.append(self._build_position_ids_3d_for_vision(current_layer_pos, batch))

        horizon_config = getattr(self, "_horizonkv_config", None)
        timestamp_config = getattr(self, "_timestamp_v8_config", None)
        frame_trace_config = getattr(self, "_ragged_frame_trace_config", None)
        readout_trace_config = getattr(self, "_ragged_readout_trace_config", None)
        token_trace_config = getattr(self, "_ragged_token_trace_config", None)
        prospective_config = getattr(self, "_prospective_lifecycle_trace_config", None)
        source_chunk_id = None
        if horizon_config:
            source_chunk_id = int(horizon_config["next_chunk_id"])
            horizon_config["current_chunk_id"] = source_chunk_id
        elif (timestamp_config or frame_trace_config or readout_trace_config
              or token_trace_config or prospective_config):
            from head_analysis.timestamp_v8_policy import build_source_frame_ids

            source_configs = [
                config for config in (
                    timestamp_config, frame_trace_config, readout_trace_config,
                    token_trace_config,
                    prospective_config,
                ) if config
            ]
            source_config = source_configs[0]
            raw_frame_count = int(video_chunk.shape[0])
            first_frame_id = int(source_config["next_frame_id"])
            source_chunk_id = build_source_frame_ids(
                q_len=q_len,
                temporal_grid=int(video_grid_thw[0][0].item()),
                raw_frame_count=raw_frame_count,
                first_frame_id=first_frame_id,
                device=self.device,
            )
            current_frame_id = first_frame_id + raw_frame_count - 1
            for active_config in source_configs:
                active_config["current_frame_id"] = current_frame_id

        source_token_uids = None
        if token_trace_config:
            first_uid = int(token_trace_config["next_visual_token_uid"])
            source_token_uids = torch.arange(
                first_uid, first_uid + int(q_len),
                device=self.device, dtype=torch.int64,
            )

        pre_head_lens = _snapshot_head_lens_as_ints(self.kv_cache)
        _, visual_read_scores = _ragged_language_chunk_forward(
            self,
            video_features,
            self.kv_cache,
            position_ids_by_layer,
            append=True,
            collect_scores=False,
            source_chunk_id=source_chunk_id,
            source_token_uids=source_token_uids,
        )
        if prospective_config and visual_read_scores:
            _write_prospective_lifecycle_trace(
                self, visual_read_scores, "visual_future"
            )
        if horizon_config:
            horizon_config["next_chunk_id"] = source_chunk_id + 1
        elif (timestamp_config or frame_trace_config or readout_trace_config
              or token_trace_config or prospective_config):
            next_frame_id = int(source_config["current_frame_id"]) + 1
            for active_config in source_configs:
                active_config["next_frame_id"] = next_frame_id
        if token_trace_config:
            token_trace_config["next_visual_token_uid"] += int(q_len)
        post_head_lens = _snapshot_head_lens_as_ints(self.kv_cache)
        self._rolekv_last_pre_lens_ragged = pre_head_lens
        self._rolekv_last_post_lens_ragged = post_head_lens
        self._kv_head_budget_last_pre_lens_ragged = pre_head_lens
        self._kv_head_budget_last_post_lens_ragged = post_head_lens
        _sync_position_cache_from_ragged(self)
        if count_as_frame:
            self.last_encoded_frames = video_chunk.shape[0]
            self.total_processed_frames += video_chunk.shape[0]
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def predict_and_compress_ragged(self):
        if not _is_ragged_cache(self.kv_cache):
            return original_predict_and_compress()
        if self.compress_mode == "streamingvlm":
            raise NotImplementedError("ragged prefill currently supports HERMES compression only")
        prospective_config = getattr(self, "_prospective_lifecycle_trace_config", None)
        if prospective_config and int(prospective_config.get("visual_query_samples", 0)) > 0:
            _, max_len, _ = self.kv_cache.stats()
            if int(max_len) > int(self.kv_size):
                raise RuntimeError(
                    "Prospective visual lifecycle collection requires a full cache; "
                    f"observed head length {int(max_len)} > kv_size {int(self.kv_size)}"
                )
            return
        if getattr(self, "_horizonkv_config", None):
            _ragged_pseudo_forward(self)
            return
        local_question, global_question = self.predict_next_question()
        _ragged_pseudo_forward(self, local_question, global_question)

    @torch.inference_mode()
    def question_answering_ragged_prefill(self, input_text, max_new_tokens=128,
                                          temperature=0, repetition_penalty=1.1,
                                          pseudo_forward=False):
        if pseudo_forward or not _is_ragged_cache(self.kv_cache):
            return original_question_answering(
                input_text,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                pseudo_forward=pseudo_forward,
            )

        from head_analysis.answer_head_router import (
            activate_answer_head_router, deactivate_answer_head_router,
        )
        activate_answer_head_router(self, input_text["question"])
        answer_history_state = _temporarily_limit_answer_history(
            self, getattr(self, "_answer_history_max_age_seconds", 0.0)
        )
        device = self.device
        stop_token_ids = [self.processor.tokenizer.eos_token_id]
        output_ids = []
        start_time = time.perf_counter()

        prompt = input_text["prompt"]
        input_ids = self.processor.tokenizer(prompt).input_ids
        input_ids = torch.as_tensor([input_ids], device=device, dtype=torch.long)
        inputs_embeds = self.get_input_embeddings()(input_ids)
        q_len_prefill = inputs_embeds.shape[1]
        batch = inputs_embeds.shape[0]

        past_lens_prefill = self.kv_cache.snapshot_head_lens()
        offsets = self._get_next_global_offset_per_layer()
        position_ids_by_layer = _build_text_position_ids(self, offsets, q_len_prefill, batch)

        readout_config = getattr(self, "_ragged_readout_trace_config", None)
        layer_scores = None
        if readout_config:
            layer_scores = _ragged_query_scores(
                self,
                input_ids,
                offsets,
                score_last_n=readout_config["score_last_n"],
            )
            _write_ragged_readout_trace(self, layer_scores)
        prospective_config = getattr(self, "_prospective_lifecycle_trace_config", None)
        if prospective_config:
            if layer_scores is None or (
                int(readout_config["score_last_n"])
                != int(prospective_config["score_last_n"])
            ):
                layer_scores = _ragged_query_scores(
                    self,
                    input_ids,
                    offsets,
                    score_last_n=prospective_config["score_last_n"],
                )
            _write_prospective_lifecycle_trace(
                self, layer_scores, "true_question"
            )

        hidden_states, _ = _ragged_language_chunk_forward(
            self,
            inputs_embeds,
            self.kv_cache,
            position_ids_by_layer,
            append=True,
            collect_scores=False,
        )
        logits = self.lm_head(hidden_states)

        for step in range(max_new_tokens):
            last_token_logits = logits[0, -1, :]
            if repetition_penalty != 1.0 and len(output_ids) > 0:
                for token_id in set(output_ids):
                    if last_token_logits[token_id] < 0:
                        last_token_logits[token_id] *= repetition_penalty
                    else:
                        last_token_logits[token_id] /= repetition_penalty

            if temperature == 0.0:
                _, indices = torch.topk(last_token_logits, 1)
                token = int(indices[0])
            else:
                scaled_logits = last_token_logits / temperature
                scaled_logits = torch.nan_to_num(
                    scaled_logits,
                    nan=-float("inf"),
                    posinf=float("inf"),
                    neginf=-float("inf"),
                )
                probs = F.softmax(scaled_logits, dim=-1)
                probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
                probs_sum = probs.sum()
                if probs_sum > 0:
                    probs = probs / probs_sum
                    token = torch.multinomial(probs, num_samples=1).item()
                else:
                    _, indices = torch.topk(last_token_logits, 1)
                    token = int(indices[0])

            output_ids.append(token)
            if step == 0:
                end_time = time.perf_counter()
                print(f"TTFT: {end_time - start_time} seconds")
            if token in stop_token_ids:
                break

            position_ids_step = []
            for layer_idx in range(self.num_layers):
                offset = offsets[layer_idx] + q_len_prefill + step
                position_ids_step.append(self._build_position_ids_3d_for_text(offset, 1, 1))

            hidden_states = _ragged_language_decode_step(
                self,
                torch.as_tensor([[token]], device=device),
                self.kv_cache,
                position_ids_step,
            )
            logits = self.lm_head(hidden_states)

        output = self.processor.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )

        current_question = input_text["question"]
        current_options = None
        formatted_question = input_text.get("formatted_question", None)
        if formatted_question:
            option_matches = re.findall(
                r"\([A-Z]\)\s*(.+?)(?=\n\([A-Z]\)|\nThe best answer|\n*$)",
                formatted_question,
                re.DOTALL,
            )
            if option_matches:
                current_options = [opt.strip() for opt in option_matches]
        self.conv_history.append((current_question, output, current_options))
        logger.info(f"Saved conversation to history. Total conversations: {len(self.conv_history)}")

        self.kv_cache.truncate(past_lens_prefill)
        _sync_position_cache_from_ragged(self)
        new_lens = self._get_cache_seq_len_per_layer()
        print(f"Answering Cache lengths: min={min(new_lens)}, max={max(new_lens)}")
        _restore_answer_history(self, answer_history_state)
        deactivate_answer_head_router(self)
        torch.cuda.empty_cache()
        return output

    model._get_cache_seq_len_per_layer = MethodType(get_cache_seq_len_per_layer_ragged, model)
    model._get_next_global_offset_per_layer = MethodType(get_next_global_offset_per_layer_ragged, model)
    model._truncate_kv_cache = MethodType(truncate_kv_cache_ragged, model)
    model.encode_init_prompt = MethodType(encode_init_prompt_ragged, model)
    model.encode_video_chunk = MethodType(encode_video_chunk_ragged, model)
    model.predict_and_compress = MethodType(predict_and_compress_ragged, model)
    model.prune_partial_deferred_last_frame = MethodType(
        _prune_partial_deferred_last_frame_ragged, model
    )
    model.prepare_head_budget_deferred_history = MethodType(
        _prepare_head_budget_deferred_history_ragged, model
    )
    model.prune_head_budget_deferred_last_frame = MethodType(
        _prune_head_budget_deferred_last_frame_ragged, model
    )
    model.question_answering = MethodType(question_answering_ragged_prefill, model)
    model._kv_head_ragged_prefill_enabled = True
    print("[kv_head_ragged_prefill] Installed physical per-KV-head ragged prefill/chunk/decode path.")
    return model
