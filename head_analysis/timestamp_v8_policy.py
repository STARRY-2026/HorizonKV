"""Timestamp-gated, v8 role-aware eviction for physical ragged KV caches."""

from collections import Counter
import csv
import json
import os

import numpy as np
import torch


MEMORY_ROLE = "memory"
MIXED_ROLE = "mixed"
CURRENT_ROLE = "current"


def _percentile_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size <= 1:
        return np.zeros_like(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks / float(values.size - 1)


def load_v8_head_profiles(csv_path, num_layers=28, num_kv_heads=4,
                          role_quantile=0.75):
    identity = {"layer", "kv_head"}
    legacy_columns = {"long_score", "short_score"}
    temporal_columns = {"recent_ratio", "middle_ratio", "far_ratio"}
    annealed_columns = {
        "profile_mode", "alpha_initial", "alpha_floor", "alpha_tau_sec",
        "protect_latest",
    }
    utility_columns = {
        "time_0_4", "time_4_8", "time_8_16", "time_16_32",
        "time_32_64", "time_64_128", "time_128_plus",
        "current_affinity", "temporal_weight", "novelty_weight",
        "novelty_tau_sec", "profile_mode",
    }
    prospective_columns = {
        "profile_mode", "lifecycle_weight",
        *(f"content_prior_{content_bin}" for content_bin in range(4)),
        *(
            f"lifecycle_c{content_bin}_t{age_bin}"
            for content_bin in range(4)
            for age_bin in range(8)
        ),
    }
    rows = []
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = identity - fieldnames
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
        if prospective_columns <= fieldnames:
            policy_mode = "prospective_lifecycle"
            value_columns = (
                *(f"content_prior_{content_bin}" for content_bin in range(4)),
                *(
                    f"lifecycle_c{content_bin}_t{age_bin}"
                    for content_bin in range(4)
                    for age_bin in range(8)
                ),
                "lifecycle_weight",
            )
        elif annealed_columns <= fieldnames:
            policy_mode = "annealed_recency"
            value_columns = (
                "alpha_initial", "alpha_floor", "alpha_tau_sec",
                "protect_latest",
            )
        elif utility_columns <= fieldnames:
            policy_mode = "temporal_utility"
            value_columns = (
                "time_0_4", "time_4_8", "time_8_16", "time_16_32",
                "time_32_64", "time_64_128", "time_128_plus",
                "current_affinity", "temporal_weight", "novelty_weight",
                "novelty_tau_sec",
            )
        elif temporal_columns <= fieldnames:
            policy_mode = "temporal_zones"
            value_columns = ("recent_ratio", "middle_ratio", "far_ratio")
        elif legacy_columns <= fieldnames:
            policy_mode = "legacy_v8"
            value_columns = ("long_score", "short_score")
        else:
            raise ValueError(
                f"{csv_path} must contain either {sorted(legacy_columns)} or "
                f"{sorted(temporal_columns)}"
            )
        for row in reader:
            rows.append(
                (
                    int(row["layer"]),
                    int(row["kv_head"]),
                    *(float(row[column]) for column in value_columns),
                )
            )

    expected = int(num_layers) * int(num_kv_heads)
    valid = [
        row for row in rows
        if 0 <= row[0] < int(num_layers) and 0 <= row[1] < int(num_kv_heads)
    ]
    keys = {(row[0], row[1]) for row in valid}
    if len(keys) != expected or len(valid) != expected:
        raise ValueError(
            f"{csv_path} must contain exactly {expected} unique physical KV heads; "
            f"found {len(keys)}"
        )

    valid.sort(key=lambda row: (row[0], row[1]))
    shape = (int(num_layers), int(num_kv_heads))
    if policy_mode == "prospective_lifecycle":
        values = np.asarray([row[2:] for row in valid], dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(
                f"{csv_path} prospective lifecycle values must be finite and non-negative"
            )
        content_priors = values[:, :4]
        lifecycle_values = values[:, 4:36]
        lifecycle_weights = values[:, 36]
        if not np.allclose(content_priors, content_priors[0]):
            raise ValueError(f"{csv_path} content priors must be global")
        if not np.allclose(lifecycle_weights, lifecycle_weights[0]):
            raise ValueError(f"{csv_path} lifecycle weight must be global")
        modes = set()
        with open(csv_path, newline="") as handle:
            reader = csv.DictReader(handle)
            modes.update(row["profile_mode"] for row in reader)
        if modes != {"prospective_lifecycle"}:
            raise ValueError(
                f"{csv_path} profile_mode must be prospective_lifecycle"
            )
        return {
            "policy_mode": policy_mode,
            "profile_mode": modes.pop(),
            "content_priors": content_priors[0],
            "lifecycle_values": lifecycle_values.reshape(*shape, 4, 8),
            "lifecycle_weight": float(lifecycle_weights[0]),
            "role_counts": {},
        }
    if policy_mode == "annealed_recency":
        values = np.asarray([row[2:] for row in valid], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{csv_path} annealed-recency values must be finite")
        alpha_initial = values[:, 0]
        alpha_floor = values[:, 1]
        alpha_tau_sec = values[:, 2]
        protect_latest = values[:, 3].astype(np.int64)
        if np.any((alpha_initial < 0.0) | (alpha_initial > 1.0)):
            raise ValueError(f"{csv_path} alpha_initial must be within [0, 1]")
        if np.any((alpha_floor < 0.0) | (alpha_floor > 1.0)):
            raise ValueError(f"{csv_path} alpha_floor must be within [0, 1]")
        if np.any(alpha_initial < alpha_floor):
            raise ValueError(f"{csv_path} alpha_initial must be >= alpha_floor")
        if np.any(alpha_tau_sec <= 0.0):
            raise ValueError(f"{csv_path} alpha_tau_sec must be positive")
        if np.any((protect_latest != 0) & (protect_latest != 1)):
            raise ValueError(f"{csv_path} protect_latest must be 0 or 1")
        modes = set()
        alpha_grace_sec = np.zeros(expected, dtype=np.float64)
        recency_k_decay = np.full(expected, np.nan, dtype=np.float64)
        row_by_key = {(row[0], row[1]): idx for idx, row in enumerate(valid)}
        with open(csv_path, newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                idx = row_by_key[(int(row["layer"]), int(row["kv_head"]))]
                modes.add(row["profile_mode"])
                if "alpha_grace_sec" in fieldnames:
                    alpha_grace_sec[idx] = float(row["alpha_grace_sec"])
                if "recency_k_decay" in fieldnames:
                    recency_k_decay[idx] = float(row["recency_k_decay"])
        if len(modes) != 1:
            raise ValueError(f"{csv_path} must contain exactly one profile_mode")
        if np.any(~np.isfinite(alpha_grace_sec)) or np.any(alpha_grace_sec < 0.0):
            raise ValueError(f"{csv_path} alpha_grace_sec must be finite and non-negative")
        finite_k = np.isfinite(recency_k_decay)
        if np.any(recency_k_decay[finite_k] < 0.0):
            raise ValueError(f"{csv_path} recency_k_decay must be non-negative")
        return {
            "policy_mode": policy_mode,
            "profile_mode": modes.pop(),
            "alpha_initial": alpha_initial.reshape(shape),
            "alpha_floor": alpha_floor.reshape(shape),
            "alpha_tau_sec": alpha_tau_sec.reshape(shape),
            "alpha_grace_sec": alpha_grace_sec.reshape(shape),
            "recency_k_decay": recency_k_decay.reshape(shape),
            "protect_latest": protect_latest.astype(bool).reshape(shape),
            "role_counts": {},
        }

    if policy_mode == "temporal_utility":
        values = np.asarray([row[2:] for row in valid], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{csv_path} temporal utility values must be finite")
        temporal_values = values[:, :7]
        current_affinity = values[:, 7]
        temporal_weights = values[:, 8]
        novelty_weights = values[:, 9]
        novelty_taus = values[:, 10]
        if np.any((temporal_values < 0.0) | (temporal_values > 1.0)):
            raise ValueError(f"{csv_path} time values must be within [0, 1]")
        if np.any((current_affinity < 0.0) | (current_affinity > 1.0)):
            raise ValueError(f"{csv_path} current affinity must be within [0, 1]")
        if np.any(temporal_weights < 0.0) or np.any(novelty_weights < 0.0):
            raise ValueError(f"{csv_path} utility weights must be non-negative")
        if np.any(novelty_taus <= 0.0):
            raise ValueError(f"{csv_path} novelty tau must be positive")
        if not (
            np.allclose(temporal_weights, temporal_weights[0])
            and np.allclose(novelty_weights, novelty_weights[0])
            and np.allclose(novelty_taus, novelty_taus[0])
        ):
            raise ValueError(f"{csv_path} utility hyperparameters must be global")
        modes = set()
        with open(csv_path, newline="") as handle:
            reader = csv.DictReader(handle)
            modes.update(row["profile_mode"] for row in reader)
        if len(modes) != 1:
            raise ValueError(f"{csv_path} must contain exactly one profile_mode")
        return {
            "policy_mode": policy_mode,
            "profile_mode": modes.pop(),
            "temporal_values": temporal_values.reshape(*shape, 7),
            "global_temporal_values": temporal_values.reshape(*shape, 7).mean(axis=(0, 1)),
            "current_affinity": current_affinity.reshape(shape),
            "temporal_weight": float(temporal_weights[0]),
            "novelty_weight": float(novelty_weights[0]),
            "novelty_tau_sec": float(novelty_taus[0]),
            "role_counts": {},
        }

    if policy_mode == "temporal_zones":
        ratios = np.asarray([row[2:] for row in valid], dtype=np.float64)
        if not np.all(np.isfinite(ratios)) or np.any(ratios < 0.0):
            raise ValueError(f"{csv_path} temporal ratios must be finite and non-negative")
        totals = ratios.sum(axis=1, keepdims=True)
        if np.any(totals <= 0.0):
            raise ValueError(f"{csv_path} each temporal profile must have positive mass")
        ratios = ratios / totals
        protect_latest = np.ones(expected, dtype=bool)
        current_affinity = np.full(expected, np.nan, dtype=np.float64)
        kv_budget = np.full(expected, -1, dtype=np.int64)
        row_by_key = {(row[0], row[1]): idx for idx, row in enumerate(valid)}
        with open(csv_path, newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                key = (int(row["layer"]), int(row["kv_head"]))
                if key not in row_by_key:
                    continue
                idx = row_by_key[key]
                if "protect_latest" in fieldnames:
                    protect_latest[idx] = bool(int(row["protect_latest"]))
                if "current_affinity" in fieldnames:
                    current_affinity[idx] = float(row["current_affinity"])
                if "kv_budget" in fieldnames:
                    kv_budget[idx] = int(float(row["kv_budget"]))
        if "kv_budget" in fieldnames:
            if np.any(kv_budget <= 0):
                raise ValueError(f"{csv_path} kv_budget must be positive for every head")
            kv_budget_table = kv_budget.reshape(shape)
        else:
            kv_budget_table = None
        return {
            "policy_mode": policy_mode,
            "temporal_ratios": ratios.reshape(*shape, 3),
            "protect_latest": protect_latest.reshape(shape),
            "current_affinity": current_affinity.reshape(shape),
            "kv_budget_table": kv_budget_table,
            "role_counts": {},
        }

    long_flat = np.asarray([row[2] for row in valid], dtype=np.float64)
    short_flat = np.asarray([row[3] for row in valid], dtype=np.float64)
    long_threshold = float(np.quantile(long_flat, float(role_quantile)))
    short_threshold = float(np.quantile(short_flat, float(role_quantile)))
    long_rank = _percentile_ranks(long_flat)
    short_rank = _percentile_ranks(short_flat)

    roles = np.empty(expected, dtype=object)
    attention_ratios = np.empty(expected, dtype=np.float64)
    for idx, (_, _, long_score, short_score) in enumerate(valid):
        long_high = long_score >= long_threshold
        short_high = short_score >= short_threshold
        if long_high and not short_high:
            roles[idx] = MEMORY_ROLE
            attention_ratios[idx] = 1.0
        elif short_high and not long_high:
            roles[idx] = CURRENT_ROLE
            attention_ratios[idx] = 0.0
        else:
            roles[idx] = MIXED_ROLE
            denom = long_rank[idx] + short_rank[idx]
            attention_ratios[idx] = (
                0.5 if denom <= 1e-12 else long_rank[idx] / denom
            )

    return {
        "policy_mode": policy_mode,
        "roles": roles.reshape(shape),
        "attention_ratios": attention_ratios.reshape(shape),
        "long_scores": long_flat.reshape(shape),
        "short_scores": short_flat.reshape(shape),
        "long_threshold": long_threshold,
        "short_threshold": short_threshold,
        "role_counts": dict(Counter(roles.tolist())),
    }


def _initialize_trace_path(trace_path):
    if not trace_path:
        return None
    trace_path = os.path.abspath(os.path.expanduser(str(trace_path)))
    parent = os.path.dirname(trace_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(trace_path, "w"):
        pass
    return trace_path


def install_ragged_frame_trace(model, trace_path):
    """Track source-frame IDs for diagnostic traces without changing eviction."""
    model._ragged_frame_trace_config = {
        "trace_path": _initialize_trace_path(trace_path),
        "current_frame_id": -1,
        "next_frame_id": 0,
    }
    print(f"[RaggedFrameTrace] Installed trace: {model._ragged_frame_trace_config['trace_path']}")
    return model


def install_timestamp_v8_policy(model, csv_path, grace_frames=1,
                                role_quantile=0.75, trace_path=None):
    if getattr(model, "_horizonkv_config", None):
        raise ValueError("Timestamp-v8 and HorizonKV cannot be enabled together")
    if int(grace_frames) < 0:
        raise ValueError("grace_frames must be non-negative")

    layer_config = getattr(model.language_model, "config", None)
    if layer_config is None and hasattr(model.language_model, "model"):
        layer_config = model.language_model.model.config
    num_kv_heads = int(getattr(layer_config, "num_key_value_heads", 4))
    profile = load_v8_head_profiles(
        csv_path,
        num_layers=int(model.num_layers),
        num_kv_heads=num_kv_heads,
        role_quantile=role_quantile,
    )
    profile.update(
        {
            "csv_path": csv_path,
            "grace_frames": int(grace_frames),
            "current_frame_id": -1,
            "next_frame_id": 0,
            "trace_path": _initialize_trace_path(trace_path),
        }
    )
    model._timestamp_v8_config = profile
    if profile["policy_mode"] == "prospective_lifecycle":
        print(
            "[TimestampV8] Installed task-facing prospective lifecycle eviction: "
            f"lifecycle_weight={profile['lifecycle_weight']:g}, "
            f"content_bins={int(profile['content_priors'].size)}, "
            f"age_bins={int(profile['lifecycle_values'].shape[-1])}"
        )
    elif profile["policy_mode"] == "annealed_recency":
        print(
            "[TimestampV8] Installed head-specific annealed-recency eviction: "
            f"alpha_initial_mean={profile['alpha_initial'].mean():.4f}, "
            f"alpha_floor_mean={profile['alpha_floor'].mean():.4f}, "
            f"tau_mean={profile['alpha_tau_sec'].mean():.2f}s, "
            f"alpha_grace_mean={profile['alpha_grace_sec'].mean():.2f}s, "
            f"grace_frames={int(grace_frames)}, "
            f"protected_heads={int(profile['protect_latest'].sum())}"
        )
    elif profile["policy_mode"] == "temporal_utility":
        print(
            "[TimestampV8] Installed temporal-utility eviction: "
            f"mode={profile['profile_mode']}, "
            f"temporal_weight={profile['temporal_weight']:g}, "
            f"novelty_weight={profile['novelty_weight']:g}, "
            f"novelty_tau_sec={profile['novelty_tau_sec']:g}"
        )
    elif profile["policy_mode"] == "temporal_zones":
        means = profile["temporal_ratios"].mean(axis=(0, 1))
        print(
            "[TimestampV8] Installed head-specific temporal-zone eviction: "
            f"grace_frames={int(grace_frames)}, mean_ratios="
            f"({means[0]:.4f}, {means[1]:.4f}, {means[2]:.4f}), "
            f"protected_heads={int(profile['protect_latest'].sum())}"
        )
    else:
        print(
            "[TimestampV8] Installed timestamp-gated v8 eviction: "
            f"roles={profile['role_counts']}, grace_frames={int(grace_frames)}, "
            f"attention_mean={profile['attention_ratios'].mean():.4f}, "
            f"thresholds=({profile['long_threshold']:.12g}, "
            f"{profile['short_threshold']:.12g})"
        )
    return model


def _frame_histogram(frame_ids, indices):
    if indices.numel() == 0:
        return {}
    selected = frame_ids.index_select(0, indices)
    selected = selected[selected >= 0]
    if selected.numel() == 0:
        return {}
    values, counts = torch.unique(selected, sorted=True, return_counts=True)
    return {
        str(int(frame_id)): int(count)
        for frame_id, count in zip(values.cpu().tolist(), counts.cpu().tolist())
    }


def write_ragged_frame_trace(model, keep_indices, budget_table, selector,
                             local_question=None, global_question=None):
    """Write the question-boundary cache allocation without changing selection."""
    timestamp_config = getattr(model, "_timestamp_v8_config", None)
    trace_config = timestamp_config or getattr(model, "_ragged_frame_trace_config", None)
    trace_path = (trace_config or {}).get("trace_path")
    if not trace_path or not getattr(model, "_ragged_frame_trace_is_last_chunk", False):
        return

    roles = (timestamp_config or {}).get("roles")
    ratios = (timestamp_config or {}).get("attention_ratios")
    heads = []
    for layer_idx, layer in enumerate(model.kv_cache.layers):
        for kv_head, keep in enumerate(keep_indices[layer_idx]):
            frame_ids, _ = layer.get_horizon_aux_segment(kv_head)
            visual = torch.nonzero(frame_ids >= 0, as_tuple=False).flatten()
            keep = keep.to(device=frame_ids.device, dtype=torch.long)
            entry = {
                "layer": int(layer_idx),
                "kv_head": int(kv_head),
                "budget": int(budget_table[layer_idx, kv_head]),
                "before_visual_tokens": int(visual.numel()),
                "after_visual_tokens": int((frame_ids.index_select(0, keep) >= 0).sum()),
                "before_frame_tokens": _frame_histogram(frame_ids, visual),
                "after_frame_tokens": _frame_histogram(frame_ids, keep),
            }
            if roles is not None:
                entry["role"] = str(roles[layer_idx, kv_head])
                entry["attention_ratio"] = float(ratios[layer_idx, kv_head])
            heads.append(entry)

    record = {
        "selector": str(selector),
        "video_id": getattr(model, "_active_video_id", None),
        "question_index": getattr(model, "_active_question_index", None),
        "question": getattr(model, "_active_question", None),
        "end_time": getattr(model, "_active_question_end_time", None),
        "sample_fps": float(getattr(model, "sample_fps", 0.0)),
        "current_frame_id": int(trace_config["current_frame_id"]),
        "local_probe": local_question,
        "global_probe": global_question,
        "heads": heads,
    }
    with open(trace_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def build_source_frame_ids(q_len, temporal_grid, raw_frame_count,
                           first_frame_id, device=None):
    """Map Qwen temporal-patch tokens to the latest raw frame they cover."""
    q_len = int(q_len)
    temporal_grid = int(temporal_grid)
    raw_frame_count = int(raw_frame_count)
    first_frame_id = int(first_frame_id)
    if q_len <= 0 or temporal_grid <= 0 or raw_frame_count <= 0:
        raise ValueError("q_len, temporal_grid, and raw_frame_count must be positive")
    if q_len % temporal_grid != 0:
        raise ValueError(
            f"Visual token count {q_len} is not divisible by temporal grid "
            f"{temporal_grid}; cannot assign exact frame timestamps"
        )
    temporal_idx = torch.arange(temporal_grid, device=device, dtype=torch.long)
    latest_raw_offset = torch.div(
        (temporal_idx + 1) * raw_frame_count - 1,
        temporal_grid,
        rounding_mode="floor",
    ).clamp(max=raw_frame_count - 1)
    frame_ids = latest_raw_offset + first_frame_id
    return frame_ids.repeat_interleave(q_len // temporal_grid).to(torch.int32)


def _topk_indices(candidates, utility, target_k):
    target_k = min(max(0, int(target_k)), int(candidates.numel()))
    if target_k == 0:
        return candidates[:0]
    if target_k == int(candidates.numel()):
        return torch.sort(candidates).values
    top = torch.topk(utility.index_select(0, candidates), target_k, sorted=False).indices
    return torch.sort(candidates.index_select(0, top)).values


def _percentile_rank_tensor(values):
    """Return deterministic ranks in [0, 1] without moving scores to CPU."""
    if values.numel() <= 1:
        return torch.zeros_like(values, dtype=torch.float32)
    _, inverse, counts = torch.unique(
        values.float(), sorted=True, return_inverse=True, return_counts=True
    )
    ends = torch.cumsum(counts, dim=0).to(torch.float32) - 1.0
    starts = ends - counts.to(torch.float32) + 1.0
    average_ranks = 0.5 * (starts + ends) / float(values.numel() - 1)
    return average_ranks.index_select(0, inverse)


def select_temporal_utility_tokens(
    visual_indices,
    frame_ids,
    utility,
    current_frame_id,
    sample_fps,
    budget,
    temporal_values,
    current_affinity,
    temporal_weight,
    novelty_weight,
    novelty_tau_sec,
):
    """Select one shared Top-K using content, temporal prior, and novelty."""
    budget = min(max(0, int(budget)), int(visual_indices.numel()))
    if budget == 0:
        return visual_indices[:0], (0, 0, 0)
    if sample_fps <= 0.0:
        raise ValueError("sample_fps must be positive for temporal utility")

    ages = (
        int(current_frame_id)
        - frame_ids.index_select(0, visual_indices).to(torch.float32)
    ).clamp_min(0.0) / float(sample_fps)
    boundaries = torch.tensor(
        [4.0, 8.0, 16.0, 32.0, 64.0, 128.0],
        device=ages.device,
        dtype=ages.dtype,
    )
    age_bins = torch.bucketize(ages, boundaries)
    time_curve = torch.as_tensor(
        temporal_values, device=ages.device, dtype=torch.float32
    )
    temporal_score = time_curve.index_select(0, age_bins)
    novelty_score = float(current_affinity) * torch.exp(
        -ages / float(novelty_tau_sec)
    )
    content_score = _percentile_rank_tensor(
        utility.index_select(0, visual_indices)
    )
    combined = (
        content_score
        + float(temporal_weight) * temporal_score
        + float(novelty_weight) * novelty_score
    )
    if budget == int(visual_indices.numel()):
        chosen = torch.sort(visual_indices).values
    else:
        top = torch.topk(combined, budget, sorted=False).indices
        chosen = torch.sort(visual_indices.index_select(0, top)).values

    chosen_ages = (
        int(current_frame_id)
        - frame_ids.index_select(0, chosen).to(torch.float32)
    ).clamp_min(0.0) / float(sample_fps)
    counts = (
        int((chosen_ages < 16.0).sum()),
        int(((chosen_ages >= 16.0) & (chosen_ages < 64.0)).sum()),
        int((chosen_ages >= 64.0).sum()),
    )
    return chosen, counts


def select_temporal_survival_tokens(
    visual_indices,
    frame_ids,
    admission_scores,
    probe_scores,
    current_frame_id,
    sample_fps,
    budget,
    survival_values,
    temporal_weight,
    global_survival_values=None,
    relative_temporal=False,
    temporal_clip=0.5,
    ema_decay=0.8,
):
    """Evict by observed token content times Head-specific remaining lifetime."""
    budget = min(max(0, int(budget)), int(visual_indices.numel()))
    if budget == 0:
        return visual_indices[:0], (0, 0, 0), admission_scores
    ages = (
        int(current_frame_id)
        - frame_ids.index_select(0, visual_indices).to(torch.float32)
    ).clamp_min(0.0) / float(sample_fps)
    boundaries = torch.tensor(
        [4.0, 8.0, 16.0, 32.0, 64.0, 128.0],
        device=ages.device,
        dtype=ages.dtype,
    )
    age_bins = torch.bucketize(ages, boundaries)
    survival = torch.as_tensor(
        survival_values, device=ages.device, dtype=torch.float32
    ).index_select(0, age_bins).clamp_min(1e-4)

    current_probe = _percentile_rank_tensor(
        probe_scores.index_select(0, visual_indices)
    )
    old_content = admission_scores.index_select(0, visual_indices).float()
    content = (
        float(ema_decay) * old_content
        + (1.0 - float(ema_decay)) * current_probe
    ).clamp(0.0, 1.0)
    updated_scores = admission_scores.clone()
    updated_scores.index_copy_(0, visual_indices, content)
    if relative_temporal:
        if global_survival_values is None:
            raise ValueError("relative temporal selection requires a global curve")
        global_survival = torch.as_tensor(
            global_survival_values, device=ages.device, dtype=torch.float32
        ).index_select(0, age_bins).clamp_min(1e-4)
        temporal_term = torch.log(
            (survival + 1e-3) / (global_survival + 1e-3)
        ).clamp(min=-float(temporal_clip), max=float(temporal_clip))
    else:
        temporal_term = torch.log(survival + 1e-3)
    combined = (
        torch.log(content + 1e-3)
        + float(temporal_weight) * temporal_term
    )
    if budget == int(visual_indices.numel()):
        chosen = torch.sort(visual_indices).values
    else:
        top = torch.topk(combined, budget, sorted=False).indices
        chosen = torch.sort(visual_indices.index_select(0, top)).values
    chosen_ages = (
        int(current_frame_id)
        - frame_ids.index_select(0, chosen).to(torch.float32)
    ).clamp_min(0.0) / float(sample_fps)
    return chosen, (
        int((chosen_ages < 16.0).sum()),
        int(((chosen_ages >= 16.0) & (chosen_ages < 64.0)).sum()),
        int((chosen_ages >= 64.0).sum()),
    ), updated_scores


def select_prospective_lifecycle_tokens(
    visual_indices,
    frame_ids,
    admission_scores,
    current_frame_id,
    sample_fps,
    budget,
    content_priors,
    lifecycle_values,
    lifecycle_weight,
):
    """Apply the frozen task-facing content-bin and Head lifecycle score."""
    budget = min(max(0, int(budget)), int(visual_indices.numel()))
    if budget == 0:
        return visual_indices[:0], (0, 0, 0)
    if sample_fps <= 0.0:
        raise ValueError("sample_fps must be positive for prospective lifecycle")

    ages = (
        int(current_frame_id)
        - frame_ids.index_select(0, visual_indices).to(torch.float32)
    ).clamp_min(0.0) / float(sample_fps)
    boundaries = torch.tensor(
        [4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0],
        device=ages.device,
        dtype=ages.dtype,
    )
    age_bins = torch.bucketize(ages, boundaries)
    admission = admission_scores.index_select(0, visual_indices).float()
    content_bins = torch.clamp(
        (admission * 4).to(torch.long), min=0, max=3
    )
    priors = torch.as_tensor(
        content_priors, device=ages.device, dtype=torch.float32
    ).index_select(0, content_bins)
    lifecycle = torch.as_tensor(
        lifecycle_values, device=ages.device, dtype=torch.float32
    )[content_bins, age_bins]
    combined = (
        torch.log(priors + 1e-12)
        + float(lifecycle_weight) * torch.log(lifecycle + 1e-12)
    )
    if budget == int(visual_indices.numel()):
        chosen = torch.sort(visual_indices).values
    else:
        top = torch.topk(combined, budget, sorted=False).indices
        chosen = torch.sort(visual_indices.index_select(0, top)).values
    chosen_ages = (
        int(current_frame_id)
        - frame_ids.index_select(0, chosen).to(torch.float32)
    ).clamp_min(0.0) / float(sample_fps)
    return chosen, (
        int((chosen_ages < 16.0).sum()),
        int(((chosen_ages >= 16.0) & (chosen_ages < 64.0)).sum()),
        int((chosen_ages >= 64.0).sum()),
    )


def _hermes_utility_components(
    model, local_scores, global_scores, mixed_scores, layer_idx, kv_head,
    frame_ids, recency_k_decay_override=None,
):
    """Return normalized HERMES attention and recency for one physical KV head."""
    from head_analysis.hermes_kv_head_ragged_prefill import (
        _head_policy_score,
        _layer_default_policy,
    )

    layer_scores, recency_alpha, k_decay = _layer_default_policy(
        model, layer_idx, local_scores, global_scores, mixed_scores
    )
    score, head_alpha, head_k_decay, _ = _head_policy_score(
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
    score = score.to(device=frame_ids.device).float()
    head_len = int(frame_ids.numel())
    if score.numel() < head_len:
        score = torch.cat([
            score,
            torch.zeros(head_len - score.numel(), device=score.device, dtype=score.dtype),
        ])
    score = score[:head_len]
    if recency_k_decay_override is not None:
        head_k_decay = float(recency_k_decay_override)
    text_keep = int((frame_ids < 0).sum())
    num_visual = max(0, head_len - text_keep)
    attention_full = torch.zeros(head_len, device=score.device, dtype=torch.float32)
    recency_full = torch.zeros(head_len, device=score.device, dtype=torch.float32)
    if num_visual == 0:
        return attention_full, recency_full, float(head_alpha)
    visual_score = score[text_keep:]
    positions = torch.arange(num_visual, device=score.device, dtype=torch.float32)
    distances = (num_visual - 1 - positions) / max(num_visual - 1, 1)
    recency = torch.exp(-float(head_k_decay) * distances)
    recency = (recency - recency.min()) / (recency.max() - recency.min() + 1e-6)
    attention = (visual_score - visual_score.min()) / (
        visual_score.max() - visual_score.min() + 1e-6
    )
    attention_full[text_keep:] = attention
    recency_full[text_keep:] = recency
    return attention_full, recency_full, float(head_alpha)


def _hermes_combined_utility(
    model, local_scores, global_scores, mixed_scores, layer_idx, kv_head,
    frame_ids,
):
    """Reproduce the unmodified HERMES attention/recency score for one KV head."""
    attention, recency, head_alpha = _hermes_utility_components(
        model,
        local_scores,
        global_scores,
        mixed_scores,
        layer_idx,
        kv_head,
        frame_ids,
    )
    combined = attention * (1.0 - head_alpha) + recency * head_alpha
    return combined


def select_annealed_recency_tokens(
    visual_indices,
    frame_ids,
    attention,
    recency,
    current_frame_id,
    sample_fps,
    budget,
    alpha_initial,
    alpha_floor,
    alpha_tau_sec,
    protected,
    alpha_grace_sec=0.0,
):
    """Let recency yield smoothly to content attention as each token ages."""
    budget = min(max(0, int(budget)), int(visual_indices.numel()))
    if budget == 0:
        return visual_indices[:0], (0, 0, 0)
    if sample_fps <= 0.0:
        raise ValueError("sample_fps must be positive for annealed-recency eviction")
    ages = (
        int(current_frame_id)
        - frame_ids.index_select(0, visual_indices).to(torch.float32)
    ).clamp_min(0.0) / float(sample_fps)
    effective_ages = (ages - float(alpha_grace_sec)).clamp_min(0.0)
    alpha = float(alpha_floor) + (
        float(alpha_initial) - float(alpha_floor)
    ) * torch.exp(-effective_ages / float(alpha_tau_sec))
    content = attention.index_select(0, visual_indices)
    recent = recency.index_select(0, visual_indices)
    score = content * (1.0 - alpha) + recent * alpha

    protected = _topk_indices(protected, score.new_zeros(frame_ids.numel()).index_copy(
        0, visual_indices, score
    ), budget)
    protected_mask = torch.zeros(
        frame_ids.numel(), device=frame_ids.device, dtype=torch.bool
    )
    if protected.numel():
        protected_mask[protected] = True
    candidates = visual_indices[
        ~protected_mask.index_select(0, visual_indices)
    ]
    candidate_scores = score[
        ~protected_mask.index_select(0, visual_indices)
    ]
    remaining = min(budget - int(protected.numel()), int(candidates.numel()))
    if remaining > 0:
        top = torch.topk(candidate_scores, remaining, sorted=False).indices
        chosen = torch.unique(
            torch.cat([protected, candidates.index_select(0, top)]), sorted=True
        )
    else:
        chosen = torch.sort(protected).values

    chosen_ages = (
        int(current_frame_id)
        - frame_ids.index_select(0, chosen).to(torch.float32)
    ).clamp_min(0.0) / float(sample_fps)
    return chosen, (
        int((chosen_ages < 16.0).sum()),
        int(((chosen_ages >= 16.0) & (chosen_ages < 64.0)).sum()),
        int((chosen_ages >= 64.0).sum()),
    )


def select_temporal_residual_tokens(
    visual_indices,
    frame_ids,
    hermes_utility,
    current_frame_id,
    sample_fps,
    budget,
    temporal_values,
    global_temporal_values,
    residual_weight,
):
    """Apply a bounded head-specific temporal residual to HERMES utility."""
    budget = min(max(0, int(budget)), int(visual_indices.numel()))
    if budget == 0:
        return visual_indices[:0], (0, 0, 0)
    ages = (
        int(current_frame_id)
        - frame_ids.index_select(0, visual_indices).to(torch.float32)
    ).clamp_min(0.0) / float(sample_fps)
    boundaries = torch.tensor(
        [4.0, 8.0, 16.0, 32.0, 64.0, 128.0],
        device=ages.device,
        dtype=ages.dtype,
    )
    age_bins = torch.bucketize(ages, boundaries)
    head_curve = torch.as_tensor(
        temporal_values, device=ages.device, dtype=torch.float32
    ).index_select(0, age_bins)
    global_curve = torch.as_tensor(
        global_temporal_values, device=ages.device, dtype=torch.float32
    ).index_select(0, age_bins)
    log_residual = torch.log((head_curve + 1e-6) / (global_curve + 1e-6))
    log_residual = torch.clamp(log_residual, min=-2.0, max=2.0)
    base = hermes_utility.index_select(0, visual_indices).clamp_min(0.0)
    adjusted = base * torch.exp(float(residual_weight) * log_residual)
    if budget == int(visual_indices.numel()):
        chosen = torch.sort(visual_indices).values
    else:
        top = torch.topk(adjusted, budget, sorted=False).indices
        chosen = torch.sort(visual_indices.index_select(0, top)).values
    chosen_ages = (
        int(current_frame_id)
        - frame_ids.index_select(0, chosen).to(torch.float32)
    ).clamp_min(0.0) / float(sample_fps)
    return chosen, (
        int((chosen_ages < 16.0).sum()),
        int(((chosen_ages >= 16.0) & (chosen_ages < 64.0)).sum()),
        int((chosen_ages >= 64.0).sum()),
    )


def _integer_zone_quotas(ratios, budget):
    raw = np.asarray(ratios, dtype=np.float64) * int(budget)
    quotas = np.floor(raw).astype(np.int64)
    remainder = int(budget) - int(quotas.sum())
    if remainder > 0:
        order = np.argsort(-(raw - quotas), kind="stable")
        quotas[order[:remainder]] += 1
    return quotas


def select_temporal_zone_tokens(visual_indices, frame_ids, utility,
                                current_frame_id, sample_fps, budget,
                                ratios, protected):
    """Select fixed-budget tokens with head-specific Recent/Middle/Far quotas."""
    budget = min(max(0, int(budget)), int(visual_indices.numel()))
    if budget == 0:
        return visual_indices[:0], (0, 0, 0)
    if sample_fps <= 0.0:
        raise ValueError("sample_fps must be positive for temporal-zone eviction")
    if protected.numel() > budget:
        raise RuntimeError(
            f"Protected visual tokens ({protected.numel()}) exceed budget ({budget})"
        )

    ages = (
        int(current_frame_id)
        - frame_ids.index_select(0, visual_indices).to(torch.float32)
    ) / float(sample_fps)
    zone_ids = torch.where(
        ages < 16.0,
        torch.zeros_like(ages, dtype=torch.long),
        torch.where(
            ages < 64.0,
            torch.ones_like(ages, dtype=torch.long),
            torch.full_like(ages, 2, dtype=torch.long),
        ),
    )
    zones = [visual_indices[zone_ids == zone] for zone in range(3)]
    quotas = _integer_zone_quotas(ratios, budget)

    # Protection is paid entirely from Recent so Middle/Far capacity is intact.
    protected = _topk_indices(protected, utility, int(quotas[0]))
    protected_count = int(protected.numel())

    protected_mask = torch.zeros(
        frame_ids.numel(), device=frame_ids.device, dtype=torch.bool
    )
    protected_mask[protected] = True
    selected = [protected]
    counts = [protected_count, 0, 0]
    for zone, candidates in enumerate(zones):
        candidates = candidates[~protected_mask.index_select(0, candidates)]
        target = max(0, int(quotas[zone]) - counts[zone])
        chosen = _topk_indices(candidates, utility, target)
        selected.append(chosen)
        counts[zone] += int(chosen.numel())

    chosen = torch.unique(torch.cat(selected), sorted=True)
    deficit = budget - int(chosen.numel())
    if deficit > 0:
        available = torch.ones(
            frame_ids.numel(), device=frame_ids.device, dtype=torch.bool
        )
        available[chosen] = False
        spill_candidates = visual_indices[
            available.index_select(0, visual_indices)
        ]
        spill = _topk_indices(spill_candidates, utility, deficit)
        if spill.numel():
            spill_ages = (
                int(current_frame_id)
                - frame_ids.index_select(0, spill).to(torch.float32)
            ) / float(sample_fps)
            counts[0] += int((spill_ages < 16.0).sum())
            counts[1] += int(((spill_ages >= 16.0) & (spill_ages < 64.0)).sum())
            counts[2] += int((spill_ages >= 64.0).sum())
            chosen = torch.unique(torch.cat([chosen, spill]), sorted=True)
    return chosen, tuple(counts)


def select_recent_frames(history_indices, frame_ids, utility, target_k):
    """Keep whole recent frames; use attention only inside the boundary frame."""
    target_k = min(max(0, int(target_k)), int(history_indices.numel()))
    if target_k == 0:
        return history_indices[:0]
    candidate_frames = frame_ids.index_select(0, history_indices)
    selected = []
    remaining = target_k
    for frame_id in torch.unique(candidate_frames, sorted=True).flip(0):
        frame_candidates = history_indices[candidate_frames == frame_id]
        if frame_candidates.numel() <= remaining:
            selected.append(frame_candidates)
            remaining -= int(frame_candidates.numel())
        else:
            selected.append(_topk_indices(frame_candidates, utility, remaining))
            remaining = 0
        if remaining == 0:
            break
    if not selected:
        return history_indices[:0]
    return torch.unique(torch.cat(selected), sorted=True)


def _hermes_utility(local_scores, global_scores, mixed_scores, layer_idx,
                    kv_head, short_term_threshold, long_term_threshold):
    if layer_idx < int(short_term_threshold):
        scores = local_scores
    elif layer_idx >= int(long_term_threshold):
        scores = global_scores
    else:
        scores = mixed_scores
    return scores[layer_idx][kv_head].float()


def timestamp_v8_needs_compression(model, budget_table):
    for layer_idx, layer in enumerate(model.kv_cache.layers):
        for kv_head in range(int(layer.head_lens.numel())):
            frame_ids, _ = layer.get_horizon_aux_segment(kv_head)
            visual_count = int((frame_ids >= 0).sum().item())
            if visual_count > int(budget_table[layer_idx, kv_head]):
                return True
    return False


def select_timestamp_v8_keep_indices(model, local_scores, global_scores,
                                     mixed_scores, budget_table):
    config = model._timestamp_v8_config
    current_frame_id = int(config["current_frame_id"])
    if current_frame_id < 0:
        raise RuntimeError("Timestamp-v8 has no encoded visual frame")

    keep_indices = []
    visible_lengths = []
    protected_counts = []
    attention_counts = []
    recent_counts = []
    middle_counts = []
    far_counts = []
    grace_frames = int(config["grace_frames"])
    protected_floor = current_frame_id - grace_frames + 1

    for layer_idx, layer in enumerate(model.kv_cache.layers):
        layer_keep = []
        for kv_head in range(int(layer.head_lens.numel())):
            frame_ids, _ = layer.get_horizon_aux_segment(kv_head)
            text_indices = torch.nonzero(frame_ids < 0, as_tuple=False).flatten()
            visual_indices = torch.nonzero(frame_ids >= 0, as_tuple=False).flatten()
            utility = _hermes_utility(
                local_scores,
                global_scores,
                mixed_scores,
                layer_idx,
                kv_head,
                getattr(model, "short_term_threshold", 8),
                getattr(model, "long_term_threshold", 19),
            ).to(device=frame_ids.device)
            if utility.numel() < frame_ids.numel():
                utility = torch.cat([
                    utility,
                    torch.zeros(
                        frame_ids.numel() - utility.numel(),
                        device=frame_ids.device,
                        dtype=utility.dtype,
                    ),
                ])
            utility = utility[:frame_ids.numel()]

            annealed_mode = config.get("policy_mode") == "annealed_recency"
            residual_mode = str(config.get("profile_mode", "")).startswith("residual_")
            if annealed_mode:
                k_override = config["recency_k_decay"][layer_idx, kv_head]
                attention_utility, recency_utility, _ = _hermes_utility_components(
                    model,
                    local_scores,
                    global_scores,
                    mixed_scores,
                    layer_idx,
                    kv_head,
                    frame_ids,
                    recency_k_decay_override=(
                        float(k_override) if np.isfinite(k_override) else None
                    ),
                )
            if residual_mode:
                utility = _hermes_combined_utility(
                    model,
                    local_scores,
                    global_scores,
                    mixed_scores,
                    layer_idx,
                    kv_head,
                    frame_ids,
                )

            protect_table = config.get("protect_latest")
            protect_head = (
                bool(protect_table[layer_idx, kv_head])
                if protect_table is not None else True
            )
            if grace_frames > 0 and protect_head:
                protected = visual_indices[
                    frame_ids.index_select(0, visual_indices) >= protected_floor
                ]
            else:
                protected = visual_indices[:0]
            history = visual_indices[
                frame_ids.index_select(0, visual_indices) < protected_floor
            ]
            budget = int(budget_table[layer_idx, kv_head])
            if protected.numel() > budget:
                raise RuntimeError(
                    "Timestamp-v8 protected Frame tokens exceed the physical Head "
                    f"budget at L{layer_idx}-H{kv_head}: protected="
                    f"{int(protected.numel())}, budget={budget}. Increase kv_size or "
                    "reduce visual resolution."
                )

            if config.get("policy_mode") == "prospective_lifecycle":
                _, admission_scores = layer.get_horizon_aux_segment(kv_head)
                temporal, zone_counts = select_prospective_lifecycle_tokens(
                    visual_indices=visual_indices,
                    frame_ids=frame_ids,
                    admission_scores=admission_scores,
                    current_frame_id=current_frame_id,
                    sample_fps=float(getattr(model, "sample_fps", 0.0)),
                    budget=budget,
                    content_priors=config["content_priors"],
                    lifecycle_values=config["lifecycle_values"][layer_idx, kv_head],
                    lifecycle_weight=config["lifecycle_weight"],
                )
                keep = torch.unique(
                    torch.cat([text_indices, temporal]), sorted=True
                )
                layer_keep.append(keep)
                visible_lengths.append(int(keep.numel()))
                protected_counts.append(0)
                attention_counts.append(int(temporal.numel()))
                recent_counts.append(zone_counts[0])
                middle_counts.append(zone_counts[1])
                far_counts.append(zone_counts[2])
                continue

            if annealed_mode:
                temporal, zone_counts = select_annealed_recency_tokens(
                    visual_indices=visual_indices,
                    frame_ids=frame_ids,
                    attention=attention_utility,
                    recency=recency_utility,
                    current_frame_id=current_frame_id,
                    sample_fps=float(getattr(model, "sample_fps", 0.0)),
                    budget=budget,
                    alpha_initial=config["alpha_initial"][layer_idx, kv_head],
                    alpha_floor=config["alpha_floor"][layer_idx, kv_head],
                    alpha_tau_sec=config["alpha_tau_sec"][layer_idx, kv_head],
                    alpha_grace_sec=config["alpha_grace_sec"][layer_idx, kv_head],
                    protected=protected,
                )
                keep = torch.unique(
                    torch.cat([text_indices, temporal]), sorted=True
                )
                layer_keep.append(keep)
                visible_lengths.append(int(keep.numel()))
                protected_counts.append(int(protected.numel()))
                attention_counts.append(int(temporal.numel()) - int(protected.numel()))
                recent_counts.append(zone_counts[0])
                middle_counts.append(zone_counts[1])
                far_counts.append(zone_counts[2])
                continue

            if config.get("policy_mode") == "temporal_utility":
                profile_mode = str(config.get("profile_mode", ""))
                survival_mode = profile_mode.startswith(("survival_", "relative_"))
                if survival_mode:
                    _, admission_scores = layer.get_horizon_aux_segment(kv_head)
                    local_probe = local_scores[layer_idx][kv_head].to(
                        device=frame_ids.device
                    )[:frame_ids.numel()]
                    global_probe = global_scores[layer_idx][kv_head].to(
                        device=frame_ids.device
                    )[:frame_ids.numel()]
                    probe = 0.5 * (
                        _percentile_rank_tensor(local_probe)
                        + _percentile_rank_tensor(global_probe)
                    )
                    temporal, zone_counts, updated_scores = (
                        select_temporal_survival_tokens(
                            visual_indices=visual_indices,
                            frame_ids=frame_ids,
                            admission_scores=admission_scores,
                            probe_scores=probe,
                            current_frame_id=current_frame_id,
                            sample_fps=float(getattr(model, "sample_fps", 0.0)),
                            budget=budget,
                            survival_values=config["temporal_values"][layer_idx, kv_head],
                            temporal_weight=config["temporal_weight"],
                            global_survival_values=config["global_temporal_values"],
                            relative_temporal=profile_mode.startswith("relative_"),
                        )
                    )
                    layer.set_horizon_attention_ema(
                        kv_head,
                        torch.arange(frame_ids.numel(), device=frame_ids.device),
                        updated_scores,
                    )
                    keep = torch.unique(
                        torch.cat([text_indices, temporal]), sorted=True
                    )
                    layer_keep.append(keep)
                    visible_lengths.append(int(keep.numel()))
                    protected_counts.append(0)
                    attention_counts.append(int(temporal.numel()))
                    recent_counts.append(zone_counts[0])
                    middle_counts.append(zone_counts[1])
                    far_counts.append(zone_counts[2])
                    continue
                selector = (
                    select_temporal_residual_tokens
                    if residual_mode else select_temporal_utility_tokens
                )
                selector_args = dict(
                    visual_indices=visual_indices,
                    frame_ids=frame_ids,
                    current_frame_id=current_frame_id,
                    sample_fps=float(getattr(model, "sample_fps", 0.0)),
                    budget=budget,
                    temporal_values=config["temporal_values"][layer_idx, kv_head],
                )
                if residual_mode:
                    selector_args.update(
                        hermes_utility=utility,
                        global_temporal_values=config["global_temporal_values"],
                        residual_weight=config["temporal_weight"],
                    )
                else:
                    selector_args.update(
                        utility=utility,
                        current_affinity=config["current_affinity"][layer_idx, kv_head],
                        temporal_weight=config["temporal_weight"],
                        novelty_weight=config["novelty_weight"],
                        novelty_tau_sec=config["novelty_tau_sec"],
                    )
                temporal, zone_counts = selector(**selector_args)
                keep = torch.unique(
                    torch.cat([text_indices, temporal]), sorted=True
                )
                layer_keep.append(keep)
                visible_lengths.append(int(keep.numel()))
                protected_counts.append(0)
                attention_counts.append(int(temporal.numel()))
                recent_counts.append(zone_counts[0])
                middle_counts.append(zone_counts[1])
                far_counts.append(zone_counts[2])
                continue

            if config.get("policy_mode") == "temporal_zones":
                temporal, zone_counts = select_temporal_zone_tokens(
                    visual_indices=visual_indices,
                    frame_ids=frame_ids,
                    utility=utility,
                    current_frame_id=current_frame_id,
                    sample_fps=float(getattr(model, "sample_fps", 0.0)),
                    budget=budget,
                    ratios=config["temporal_ratios"][layer_idx, kv_head],
                    protected=protected,
                )
                keep = torch.unique(
                    torch.cat([text_indices, temporal]), sorted=True
                )
                layer_keep.append(keep)
                visible_lengths.append(int(keep.numel()))
                protected_counts.append(int(protected.numel()))
                attention_counts.append(int(temporal.numel()) - int(protected.numel()))
                recent_counts.append(zone_counts[0])
                middle_counts.append(zone_counts[1])
                far_counts.append(zone_counts[2])
                continue

            history_budget = min(
                max(0, budget - int(protected.numel())),
                int(history.numel()),
            )
            ratio = float(config["attention_ratios"][layer_idx, kv_head])
            attention_k = int(round(history_budget * ratio))
            recent_k = history_budget - attention_k
            recent = select_recent_frames(history, frame_ids, utility, recent_k)

            available_mask = torch.ones(
                frame_ids.numel(), device=frame_ids.device, dtype=torch.bool
            )
            if recent.numel():
                available_mask[recent] = False
            attention_candidates = history[
                available_mask.index_select(0, history)
            ]
            attention = _topk_indices(
                attention_candidates, utility, attention_k
            )
            keep = torch.unique(
                torch.cat([text_indices, protected, recent, attention]), sorted=True
            )
            layer_keep.append(keep)
            visible_lengths.append(int(keep.numel()))
            protected_counts.append(int(protected.numel()))
            attention_counts.append(int(attention.numel()))
            recent_counts.append(int(recent.numel()))
        keep_indices.append(layer_keep)

    model._timestamp_v8_last_stats = {
        "protected_min": min(protected_counts) if protected_counts else 0,
        "protected_max": max(protected_counts) if protected_counts else 0,
        "attention_min": min(attention_counts) if attention_counts else 0,
        "attention_max": max(attention_counts) if attention_counts else 0,
        "recent_min": min(recent_counts) if recent_counts else 0,
        "recent_max": max(recent_counts) if recent_counts else 0,
        "middle_min": min(middle_counts) if middle_counts else 0,
        "middle_max": max(middle_counts) if middle_counts else 0,
        "far_min": min(far_counts) if far_counts else 0,
        "far_max": max(far_counts) if far_counts else 0,
    }
    return keep_indices, visible_lengths
