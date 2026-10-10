"""Fixed-region HorizonKV policy for physical per-KV-head ragged caches."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch


PSEUDO_PROBES = (
    "Summarize the important visual information observed in the video so far.",
    (
        "What information introduced much earlier in the video is still important "
        "for understanding the video now?"
    ),
    "What new visual information appeared or changed in the most recent segment?",
)

HERMES_LOCAL_PROBE = (
    "Describe the current scene in detail, focusing on specific objects, "
    "fine-grained actions, and spatial relationships."
)
HERMES_GLOBAL_PROBE = (
    "Summarize the video narrative, identifying main characters, key events, "
    "timeline changes, and the overall theme."
)
HERMES_PSEUDO_PROBES = (
    HERMES_LOCAL_PROBE,
    HERMES_LOCAL_PROBE + "; " + HERMES_GLOBAL_PROBE,
    HERMES_GLOBAL_PROBE,
)


def get_horizonkv_pseudo_probes(probe_style):
    style = str(probe_style or "original").lower()
    if style == "original":
        return PSEUDO_PROBES
    if style == "hermes_routed":
        return HERMES_PSEUDO_PROBES
    raise ValueError(f"Unsupported HorizonKV probe style: {probe_style!r}")


def hermes_layer_recency_alpha(
    layer_idx, short_term_threshold=8, long_term_threshold=19
):
    """Return the layer-wise recency mixture used by the HERMES selector."""
    layer_idx = int(layer_idx)
    short_term_threshold = int(short_term_threshold)
    long_term_threshold = int(long_term_threshold)
    if layer_idx < short_term_threshold:
        return 1.0
    if layer_idx >= long_term_threshold:
        return 0.0
    progress = (layer_idx - short_term_threshold) / (
        long_term_threshold - short_term_threshold
    )
    return 0.75 - 0.6 * progress


HEAD_TYPE_CONFIG = {
    "decay_aggressive": {
        "recency_weight": 0.70,
        "attention_weight": 0.30,
        "cold_anchor": 0.25,
        "age_exponent": 1.00,
        "cold_protect": False,
    },
    "normal_sustained": {
        "recency_weight": 0.60,
        "attention_weight": 0.40,
        "cold_anchor": 0.35,
        "age_exponent": 0.50,
        "cold_protect": False,
    },
    "sustained_gentle": {
        "recency_weight": 0.50,
        "attention_weight": 0.50,
        "cold_anchor": 0.40,
        "age_exponent": 0.25,
        "cold_protect": False,
    },
    "rising_far": {
        "recency_weight": 0.50,
        "attention_weight": 0.50,
        "cold_anchor": 0.50,
        "age_exponent": 0.25,
        "cold_protect": True,
    },
}


def validate_horizonkv_args(args):
    """Validate the intentionally narrow first-version integration surface."""
    if not getattr(args, "horizonkv_policy", None):
        return
    if not str(getattr(args, "model", "")).startswith("qwen2.5_vl"):
        raise ValueError("HorizonKV currently supports Qwen2.5-VL models only")
    if getattr(args, "compress_mode", "hermes") != "hermes":
        raise ValueError("HorizonKV requires --compress_mode hermes")
    if int(getattr(args, "horizonkv_recent_capacity", 0)) <= 0:
        raise ValueError("--horizonkv_recent_capacity must be positive")
    gamma = float(getattr(args, "horizonkv_ema_gamma", 0.9))
    if not 0.0 <= gamma < 1.0:
        raise ValueError("--horizonkv_ema_gamma must be in [0, 1)")
    get_horizonkv_pseudo_probes(
        getattr(args, "horizonkv_probe_style", "original")
    )
    score_blend = str(
        getattr(args, "horizonkv_score_blend", "head_type")
    ).lower()
    if score_blend not in {"head_type", "hermes_layer"}:
        raise ValueError(f"Unsupported HorizonKV score blend: {score_blend!r}")

    conflicts = {
        "--head_budget_scores": bool(getattr(args, "head_budget_scores", None)),
        "--layer_budget_scores": bool(getattr(args, "layer_budget_scores", None)),
        "--kv_head_budget_scores": bool(getattr(args, "kv_head_budget_scores", None)),
        "--kv_head_ragged_decode": bool(getattr(args, "kv_head_ragged_decode", False)),
        "--kv_head_policy_roles": bool(getattr(args, "kv_head_policy_roles", None)),
        "--kv_head_role_recent_scores": bool(getattr(args, "kv_head_role_recent_scores", None)),
        "--kv_head_role_dual_scores": bool(getattr(args, "kv_head_role_dual_scores", None)),
        "--rolekv_ragged": bool(getattr(args, "rolekv_ragged", False)),
        "--defer_last_frame_compress": bool(getattr(args, "defer_last_frame_compress", False)),
        "--multi_scale": bool(getattr(args, "multi_scale", False)),
        "--layered_resolution": bool(getattr(args, "layered_resolution", False)),
        "--fixed_base_resolution": bool(getattr(args, "fixed_base_resolution", False)),
        "--fixed_frame_canvas_size": int(getattr(args, "fixed_frame_canvas_size", 0)) > 0,
        "--transient_highres": bool(getattr(args, "transient_highres", False)),
        "--feature_buffer_frames": int(getattr(args, "feature_buffer_frames", 0)) > 0,
        "--last_frame_stride": float(getattr(args, "last_frame_stride", 1.0)) != 1.0,
        "--partition_position_frames": int(getattr(args, "partition_position_frames", 0)) > 0,
        "--short_term_frames": int(getattr(args, "short_term_frames", 0)) > 0,
    }
    enabled = [name for name, active in conflicts.items() if active]
    if enabled:
        raise ValueError(
            "HorizonKV v0.6 cannot be combined with: " + ", ".join(enabled)
        )
    args.kv_head_ragged_prefill = True


def load_horizonkv_policy(
    csv_path,
    num_layers=28,
    num_kv_heads=4,
    recent_capacity=2691,
    expected_average_budget=None,
):
    """Load and strictly validate one frozen policy row per physical KV head."""
    path = Path(csv_path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = set(reader.fieldnames or [])

    required = {
        "layer",
        "kv_head",
        "HeadType",
        "RecentBudget",
        "MemoryBudget",
        "TotalBudget",
    }
    missing = sorted(required - fields)
    if missing:
        raise ValueError(f"HorizonKV policy is missing columns: {missing}")

    expected_keys = {
        (layer, head)
        for layer in range(int(num_layers))
        for head in range(int(num_kv_heads))
    }
    seen = set()
    head_types = np.empty((num_layers, num_kv_heads), dtype=object)
    recent = np.zeros((num_layers, num_kv_heads), dtype=np.int32)
    memory = np.zeros((num_layers, num_kv_heads), dtype=np.int32)
    total = np.zeros((num_layers, num_kv_heads), dtype=np.int32)

    for row in rows:
        key = (int(row["layer"]), int(row["kv_head"]))
        if key in seen:
            raise ValueError(f"Duplicate HorizonKV policy row for head {key}")
        if key not in expected_keys:
            raise ValueError(f"Out-of-range HorizonKV head {key}")
        seen.add(key)

        head_type = row["HeadType"].strip()
        if head_type not in HEAD_TYPE_CONFIG:
            raise ValueError(f"Unknown HorizonKV HeadType {head_type!r} for head {key}")
        recent_budget = int(row["RecentBudget"])
        memory_budget = int(row["MemoryBudget"])
        total_budget = int(row["TotalBudget"])
        if recent_budget != int(recent_capacity):
            raise ValueError(
                f"Head {key} has RecentBudget={recent_budget}, expected {recent_capacity}"
            )
        if memory_budget < 0 or total_budget <= 0:
            raise ValueError(f"Head {key} has a non-positive budget")
        if recent_budget + memory_budget != total_budget:
            raise ValueError(
                f"Head {key}: RecentBudget + MemoryBudget != TotalBudget"
            )

        layer, head = key
        head_types[layer, head] = head_type
        recent[layer, head] = recent_budget
        memory[layer, head] = memory_budget
        total[layer, head] = total_budget

    missing_heads = sorted(expected_keys - seen)
    if missing_heads:
        raise ValueError(
            f"HorizonKV policy is incomplete: missing {len(missing_heads)} heads; "
            f"first={missing_heads[0]}"
        )

    if expected_average_budget is not None:
        expected_total = int(expected_average_budget) * len(expected_keys)
        actual_total = int(total.sum())
        if actual_total != expected_total:
            raise ValueError(
                f"HorizonKV total budget is {actual_total}, expected {expected_total} "
                f"({expected_average_budget} per head)"
            )

    return {
        "path": str(path),
        "head_types": head_types,
        "recent_budgets": recent,
        "memory_budgets": memory,
        "total_budgets": total,
        "num_layers": int(num_layers),
        "num_kv_heads": int(num_kv_heads),
    }


def install_horizonkv_policy(
    model,
    csv_path,
    recent_capacity=2691,
    ema_gamma=0.9,
    apply_age_penalty=True,
    temporal_coverage="cold",
    probe_style="original",
    score_blend="head_type",
):
    if not 0.0 <= float(ema_gamma) < 1.0:
        raise ValueError("HorizonKV ema_gamma must be in [0, 1)")
    if temporal_coverage not in {"cold", "all"}:
        raise ValueError("HorizonKV temporal_coverage must be 'cold' or 'all'")
    pseudo_probes = get_horizonkv_pseudo_probes(probe_style)
    score_blend = str(score_blend).lower()
    if score_blend not in {"head_type", "hermes_layer"}:
        raise ValueError(f"Unsupported HorizonKV score blend: {score_blend!r}")

    language_model = getattr(model, "language_model", None)
    language_config = getattr(language_model, "config", None)
    if language_config is None and hasattr(language_model, "model"):
        language_config = getattr(language_model.model, "config", None)
    num_layers = int(getattr(model, "num_layers", 28))
    num_kv_heads = int(getattr(language_config, "num_key_value_heads", 4))

    policy = load_horizonkv_policy(
        csv_path,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        recent_capacity=recent_capacity,
        expected_average_budget=int(model.kv_size),
    )
    policy.update(
        {
            "recent_capacity": int(recent_capacity),
            "ema_gamma": float(ema_gamma),
            "apply_age_penalty": bool(apply_age_penalty),
            "temporal_coverage": temporal_coverage,
            "probe_style": str(probe_style),
            "score_blend": score_blend,
            "pseudo_probes": pseudo_probes,
            "short_term_threshold": int(
                getattr(model, "short_term_threshold", 8)
            ),
            "long_term_threshold": int(
                getattr(model, "long_term_threshold", 19)
            ),
            "query_pool_last_n": 8,
            "current_chunk_id": -1,
            "next_chunk_id": 0,
        }
    )
    model._horizonkv_config = policy
    print(
        "[HorizonKV] Installed fixed-region policy: "
        f"recent={recent_capacity}, ema_gamma={ema_gamma:g}, "
        f"age_penalty={bool(apply_age_penalty)}, "
        f"temporal_coverage={temporal_coverage}, "
        f"probe_style={probe_style}, "
        f"score_blend={score_blend}, "
        f"memory=[{policy['memory_budgets'].min()}, "
        f"{policy['memory_budgets'].max()}], "
        f"total={int(policy['total_budgets'].sum())}"
    )
    return model


def _normalize(values, constant_value=0.0):
    values = values.float()
    if values.numel() == 0:
        return values
    vmin = values.min()
    vmax = values.max()
    if float((vmax - vmin).abs().item()) < 1e-12:
        return torch.full_like(values, float(constant_value))
    return (values - vmin) / (vmax - vmin)


def aggregate_probe_attention(
    probe_scores,
    layer_idx,
    kv_head,
    visual_indices,
    probe_style="original",
    short_term_threshold=8,
    long_term_threshold=19,
):
    """Normalize probes, then average or apply HERMES-style layer routing."""
    if probe_style == "hermes_routed":
        if layer_idx < int(short_term_threshold):
            probe_scores = probe_scores[:1]
        elif layer_idx >= int(long_term_threshold):
            probe_scores = probe_scores[2:3]
        else:
            probe_scores = probe_scores[1:2]
    parts = []
    for scores in probe_scores:
        values = scores[layer_idx][kv_head].to(device=visual_indices.device).float()
        if values.numel() <= int(visual_indices.max().item()):
            raise ValueError(
                f"Probe score length {values.numel()} does not cover cache index "
                f"{int(visual_indices.max().item())}"
            )
        parts.append(_normalize(values.index_select(0, visual_indices)))
    if not parts:
        return torch.zeros(visual_indices.numel(), device=visual_indices.device)
    return torch.stack(parts, dim=0).mean(dim=0)


def update_attention_ema(old_ema, current_attention, source_chunks,
                         current_chunk_id, gamma):
    updated = float(gamma) * old_ema.float() + (1.0 - float(gamma)) * current_attention.float()
    new_mask = source_chunks == int(current_chunk_id)
    updated[new_mask] = current_attention.float()[new_mask]
    return updated


def _temporal_anchor_indices(candidate_indices, source_chunks, scores, target_k):
    """Choose score-high tokens while distributing quota over occupied chunks."""
    target_k = min(max(0, int(target_k)), int(candidate_indices.numel()))
    if target_k == 0:
        return torch.empty(0, device=candidate_indices.device, dtype=torch.long)

    chunk_values = source_chunks.index_select(0, candidate_indices)
    unique_chunks = torch.unique(chunk_values, sorted=True)
    base = target_k // int(unique_chunks.numel())
    remainder = target_k % int(unique_chunks.numel())
    quotas = torch.full(
        (unique_chunks.numel(),), base, device=candidate_indices.device, dtype=torch.long
    )
    if remainder:
        extra = torch.div(
            (torch.arange(remainder, device=candidate_indices.device) * 2 + 1)
            * unique_chunks.numel(),
            2 * remainder,
            rounding_mode="floor",
        ).clamp(max=unique_chunks.numel() - 1)
        quotas[extra] += 1

    selected = []
    selected_mask = torch.zeros(source_chunks.numel(), device=candidate_indices.device, dtype=torch.bool)
    for offset, chunk_id in enumerate(unique_chunks):
        chunk_candidates = candidate_indices[chunk_values == chunk_id]
        keep_k = min(int(quotas[offset].item()), int(chunk_candidates.numel()))
        if keep_k:
            top = torch.topk(scores.index_select(0, chunk_candidates), keep_k, sorted=False).indices
            chosen = chunk_candidates.index_select(0, top)
            selected.append(chosen)
            selected_mask[chosen] = True

    selected_count = sum(int(part.numel()) for part in selected)
    if selected_count < target_k:
        remaining = candidate_indices[~selected_mask.index_select(0, candidate_indices)]
        fill_k = min(target_k - selected_count, int(remaining.numel()))
        if fill_k:
            top = torch.topk(scores.index_select(0, remaining), fill_k, sorted=False).indices
            selected.append(remaining.index_select(0, top))
    return torch.unique(torch.cat(selected), sorted=True) if selected else torch.empty(
        0, device=candidate_indices.device, dtype=torch.long
    )


def select_memory_tokens(memory_indices, source_chunks, attention_ema,
                         current_chunk_id, memory_budget, head_type,
                         apply_age_penalty=True, temporal_coverage="cold",
                         recency_weight_override=None):
    """Apply temporal coverage and the selected HeadType retention policy."""
    memory_budget = min(max(0, int(memory_budget)), int(memory_indices.numel()))
    if memory_budget == 0:
        return torch.empty(0, device=memory_indices.device, dtype=torch.long)
    if memory_indices.numel() <= memory_budget:
        return torch.sort(memory_indices).values

    cfg = HEAD_TYPE_CONFIG[head_type]
    ages_all = (int(current_chunk_id) - source_chunks).clamp(min=0).float()
    ages = ages_all.index_select(0, memory_indices)
    recency = _normalize(1.0 / (1.0 + ages), constant_value=1.0)
    attention = _normalize(attention_ema.index_select(0, memory_indices))
    if recency_weight_override is None:
        recency_weight = float(cfg["recency_weight"])
        attention_weight = float(cfg["attention_weight"])
    else:
        recency_weight = max(0.0, min(1.0, float(recency_weight_override)))
        attention_weight = 1.0 - recency_weight
    base = recency_weight * recency + attention_weight * attention

    if apply_age_penalty:
        exponent = torch.full_like(ages, float(cfg["age_exponent"]))
        if cfg["cold_protect"]:
            exponent[ages >= 4] *= 0.5
        retention_local = base / torch.pow(1.0 + ages, exponent)
    else:
        retention_local = base
    retention = torch.full_like(attention_ema, -torch.inf, dtype=torch.float32)
    retention[memory_indices] = retention_local

    if temporal_coverage == "cold":
        anchor_candidates = memory_indices[ages >= 3]
    elif temporal_coverage == "all":
        anchor_candidates = memory_indices
    else:
        raise ValueError(f"Unknown temporal coverage mode: {temporal_coverage}")
    anchor_target = min(
        int(round(memory_budget * float(cfg["cold_anchor"]))),
        int(anchor_candidates.numel()),
    )
    anchors = _temporal_anchor_indices(
        anchor_candidates, source_chunks, retention, anchor_target
    )

    selected_mask = torch.zeros(source_chunks.numel(), device=memory_indices.device, dtype=torch.bool)
    if anchors.numel():
        selected_mask[anchors] = True
    remaining = memory_indices[~selected_mask.index_select(0, memory_indices)]
    fill_k = min(memory_budget - int(anchors.numel()), int(remaining.numel()))
    pieces = [anchors] if anchors.numel() else []
    if fill_k:
        top = torch.topk(retention.index_select(0, remaining), fill_k, sorted=False).indices
        pieces.append(remaining.index_select(0, top))
    return torch.unique(torch.cat(pieces), sorted=True)


def horizonkv_needs_compression(model):
    config = model._horizonkv_config
    total_budgets = config["total_budgets"]
    for layer_idx, layer in enumerate(model.kv_cache.layers):
        for kv_head in range(int(layer.head_lens.numel())):
            chunks, _ = layer.get_horizon_aux_segment(kv_head)
            visual_count = int((chunks >= 0).sum().item())
            if visual_count > int(total_budgets[layer_idx, kv_head]):
                return True
    return False


def select_horizonkv_keep_indices(model, probe_scores):
    """Return per-layer/per-head physical keep indices for fixed-region HorizonKV."""
    config = model._horizonkv_config
    current_chunk_id = int(config["current_chunk_id"])
    if current_chunk_id < 0:
        raise RuntimeError("HorizonKV has no encoded video chunk to compress")

    keep_indices = []
    visible_lengths = []
    recent_counts = []
    memory_counts = []

    for layer_idx, layer in enumerate(model.kv_cache.layers):
        layer_keep = []
        for kv_head in range(int(layer.head_lens.numel())):
            chunks, old_ema = layer.get_horizon_aux_segment(kv_head)
            visual_indices = torch.nonzero(chunks >= 0, as_tuple=False).flatten()
            text_indices = torch.nonzero(chunks < 0, as_tuple=False).flatten()
            if visual_indices.numel() == 0:
                layer_keep.append(text_indices)
                visible_lengths.append(int(text_indices.numel()))
                recent_counts.append(0)
                memory_counts.append(0)
                continue

            current_attention = aggregate_probe_attention(
                probe_scores,
                layer_idx,
                kv_head,
                visual_indices,
                probe_style=config.get("probe_style", "original"),
                short_term_threshold=config.get("short_term_threshold", 8),
                long_term_threshold=config.get("long_term_threshold", 19),
            )
            visual_chunks = chunks.index_select(0, visual_indices)
            visual_old_ema = old_ema.index_select(0, visual_indices)
            visual_ema = update_attention_ema(
                visual_old_ema,
                current_attention,
                visual_chunks,
                current_chunk_id,
                config["ema_gamma"],
            )
            layer.set_horizon_attention_ema(kv_head, visual_indices, visual_ema)

            recent_budget = int(config["recent_budgets"][layer_idx, kv_head])
            recent_count = min(recent_budget, int(visual_indices.numel()))
            recent = visual_indices[-recent_count:] if recent_count else visual_indices[:0]
            memory_candidates = visual_indices[:-recent_count] if recent_count else visual_indices
            recency_weight_override = None
            if config.get("score_blend", "head_type") == "hermes_layer":
                recency_weight_override = hermes_layer_recency_alpha(
                    layer_idx,
                    config.get("short_term_threshold", 8),
                    config.get("long_term_threshold", 19),
                )
            memory = select_memory_tokens(
                memory_candidates,
                chunks,
                layer.get_horizon_aux_segment(kv_head)[1],
                current_chunk_id,
                int(config["memory_budgets"][layer_idx, kv_head]),
                config["head_types"][layer_idx, kv_head],
                apply_age_penalty=config.get("apply_age_penalty", True),
                temporal_coverage=config.get("temporal_coverage", "cold"),
                recency_weight_override=recency_weight_override,
            )

            keep = torch.unique(torch.cat([text_indices, memory, recent]), sorted=True)
            layer_keep.append(keep)
            visible_lengths.append(int(keep.numel()))
            recent_counts.append(int(recent.numel()))
            memory_counts.append(int(memory.numel()))
        keep_indices.append(layer_keep)

    model._horizonkv_last_stats = {
        "recent_min": min(recent_counts) if recent_counts else 0,
        "recent_max": max(recent_counts) if recent_counts else 0,
        "memory_min": min(memory_counts) if memory_counts else 0,
        "memory_max": max(memory_counts) if memory_counts else 0,
        "visible_min": min(visible_lengths) if visible_lengths else 0,
        "visible_max": max(visible_lengths) if visible_lengths else 0,
    }
    return keep_indices, config["total_budgets"], visible_lengths
