"""
Per-KV-head HERMES pruning prototype.

This module lets each KV head choose a different visual-token keep set. The
first implementation is intentionally conservative: the physical cache remains
dense and stores the union of all KV-head keep sets, while a per-query-head
attention mask makes each query-head group see only the tokens selected for its
own KV head.

It is a stepping stone toward HybridKV-style flat ragged caches. It validates
head-wise eviction quality without replacing Qwen's full attention/cache stack.
"""

import numpy as np
import torch

from head_analysis.hermes_head_budget import load_head_scores


def _finite_positive_weights(values):
    values = np.asarray(values, dtype=np.float64)
    return np.where(np.isfinite(values) & (values > 0), values, 1.0)


def _allocate_integer_budget(weights, total_budget, min_budget=0, max_budget=None):
    weights = _finite_positive_weights(weights)
    n_items = int(weights.size)
    total_budget = int(total_budget)
    if n_items == 0:
        return np.zeros(0, dtype=int)
    if total_budget <= 0:
        return np.zeros(n_items, dtype=int)

    min_budget = max(0, int(min_budget))
    if min_budget * n_items > total_budget:
        min_budget = total_budget // n_items

    if max_budget is None:
        max_budget = total_budget
    max_budget = max(int(max_budget), min_budget)

    raw = weights / weights.sum() * total_budget
    raw = np.clip(raw, min_budget, max_budget)
    if raw.sum() > 0:
        raw = raw * (total_budget / raw.sum())

    budgets = np.floor(raw).astype(int)
    budgets = np.clip(budgets, min_budget, max_budget)
    frac = raw - np.floor(raw)

    diff = total_budget - int(budgets.sum())
    while diff > 0:
        candidates = np.where(budgets < max_budget)[0]
        if candidates.size == 0:
            break
        idx = candidates[np.argmax(frac[candidates])]
        budgets[idx] += 1
        diff -= 1

    while diff < 0:
        candidates = np.where(budgets > min_budget)[0]
        if candidates.size == 0:
            break
        idx = candidates[np.argmin(frac[candidates])]
        budgets[idx] -= 1
        diff += 1

    return budgets.astype(int)


def _apply_budget_floor(budgets, num_keep, floor_ratio):
    """Raise low head budgets while preserving the original global total."""
    budgets = np.asarray(budgets, dtype=int)
    floor_ratio = float(floor_ratio)
    if floor_ratio <= 0.0:
        return budgets
    if not 0.0 <= floor_ratio <= 1.0:
        raise ValueError("budget floor ratio must be in [0, 1]")

    floor_budget = int(round(int(num_keep) * floor_ratio))
    total_budget = int(budgets.sum())
    count = int(budgets.size)
    if floor_budget * count > total_budget:
        raise ValueError(
            f"budget floor {floor_budget} is infeasible for total {total_budget} "
            f"across {count} heads"
        )

    surplus_total = total_budget - floor_budget * count
    weights = np.maximum(budgets.reshape(-1) - floor_budget, 0).astype(np.float64)
    if weights.sum() <= 1e-12:
        weights = np.ones_like(weights)
    raw_surplus = weights / weights.sum() * surplus_total
    surplus = np.floor(raw_surplus).astype(int)
    remainder = surplus_total - int(surplus.sum())
    if remainder > 0:
        order = np.argsort(-(raw_surplus - surplus), kind="stable")
        surplus[order[:remainder]] += 1
    adjusted = floor_budget + surplus
    return adjusted.reshape(budgets.shape).astype(int)


def _get_language_config(model):
    language_model = getattr(model, "language_model", None)
    config = getattr(language_model, "config", None)
    if config is None and hasattr(language_model, "model"):
        config = getattr(language_model.model, "config", None)
    return config


def _get_attention_shape(model, num_query_heads=None, num_kv_heads=None):
    config = _get_language_config(model)
    if num_query_heads is None:
        num_query_heads = getattr(config, "num_attention_heads", None) if config is not None else None
    if num_kv_heads is None:
        num_kv_heads = getattr(config, "num_key_value_heads", None) if config is not None else None

    num_query_heads = int(num_query_heads or 28)
    num_kv_heads = int(num_kv_heads or 4)
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_query_heads={num_query_heads} must be divisible by "
            f"num_kv_heads={num_kv_heads}"
        )
    return num_query_heads, num_kv_heads, num_query_heads // num_kv_heads


def _aggregate_query_scores_to_kv(scores, num_layers, num_query_heads, num_kv_heads):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (num_layers, num_query_heads):
        raise ValueError(
            f"scores shape must be {(num_layers, num_query_heads)}, got {scores.shape}"
        )
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    groups = num_query_heads // num_kv_heads
    return np.abs(scores).reshape(num_layers, num_kv_heads, groups).mean(axis=-1)


def load_kv_head_memory_scores(csv_path, num_layers=28, num_kv_heads=4):
    """Load per-KV-head memory scores directly from a CSV file.

    Supported schema: layer,kv_head,<score-column>

    Unlike load_head_scores (which expands KV scores to query heads and requires
    re-aggregation), this reads KV-head scores directly and returns a
    [num_layers, num_kv_heads] array.

    The preferred score column order:
      internal_top100_mean_attention, score, spiky_readout_score, ...
    """
    import csv

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if "layer" not in fieldnames:
        raise ValueError(f"{csv_path} must contain a 'layer' column")
    if "kv_head" not in fieldnames:
        raise ValueError(
            f"{csv_path} must contain a 'kv_head' column. "
            f"Use load_head_scores for per-query-head files."
        )

    preferred = [
        "observed_memory_ratio",
        "memory_readout_share",
        "internal_top100_mean_attention",
        "score",
        "spiky_readout_score",
        "readout_shape_score",
    ]
    score_col = next((c for c in preferred if c in fieldnames), None)
    if score_col is None:
        numeric = [c for c in fieldnames if c not in ("layer", "kv_head", "head", "q_heads")]
        if not numeric:
            raise ValueError(f"{csv_path} has no recognizable score column")
        score_col = numeric[0]

    kv_scores = np.zeros((int(num_layers), int(num_kv_heads)), dtype=np.float64)
    for row in rows:
        layer = int(row["layer"])
        kv_head = int(row["kv_head"])
        if 0 <= layer < num_layers and 0 <= kv_head < num_kv_heads:
            kv_scores[layer, kv_head] = float(row[score_col])

    # Detect if the loaded column is already a ratio in [0, 1].
    vmin, vmax = kv_scores.min(), kv_scores.max()
    is_direct_ratio = bool(
        score_col in ("observed_memory_ratio", "memory_readout_share")
        or (0.0 <= vmin and vmax <= 1.0)
    )
    return kv_scores, is_direct_ratio


def compute_memory_current_split_ratios(kv_scores, base_memory_ratio=0.7,
                                         dynamic_scale=0.3,
                                         min_ratio=0.1, max_ratio=0.9):
    """Compute per-KV-head memory budget ratios from KV-head scores.

    Higher kv_score -> more memory-oriented -> larger memory_ratio.

    Normalization:
      norm = (score - min) / (max - min)                              # [0, 1]
      memory_ratio = base_memory_ratio + dynamic_scale * (2*norm - 1)
      memory_ratio = clip(memory_ratio, min_ratio, max_ratio)

    Args:
        kv_scores: np.ndarray [num_layers, num_kv_heads], per-KV-head scores.
        base_memory_ratio: center of the memory-ratio range (default 0.7).
        dynamic_scale: half-width around base; max ratio = base+scale,
                       min ratio = base-scale (before clipping).
        min_ratio, max_ratio: hard floor / ceiling for safety.

    Returns:
        np.ndarray [num_layers, num_kv_heads] of memory ratios.
    """
    kv_scores = np.asarray(kv_scores, dtype=np.float64)
    kv_scores = np.nan_to_num(kv_scores, nan=0.0, posinf=0.0, neginf=0.0)
    vmin = kv_scores.min()
    vmax = kv_scores.max()
    if vmax - vmin < 1e-12:
        norm = np.full_like(kv_scores, 0.5)
    else:
        norm = (kv_scores - vmin) / (vmax - vmin + 1e-12)
    ratios = base_memory_ratio + dynamic_scale * (2.0 * norm - 1.0)
    return np.clip(ratios, float(min_ratio), float(max_ratio))


def build_kv_head_split_budget_table(total_budget_table, memory_ratios):
    """Split a per-KV-head total budget table into memory / current parts.

    Args:
        total_budget_table: np.ndarray [num_layers, num_kv_heads] of integers,
            the total visual-token budget per head (output of
            build_kv_head_budget_table).
        memory_ratios: np.ndarray [num_layers, num_kv_heads] of floats in
            (0, 1); memory_ratios[l, h] is the fraction of the head's budget
            allocated to memory (older) tokens.

    Returns:
        np.ndarray [num_layers, num_kv_heads, 2] of int32:
          [..., 0] = memory_budget
          [..., 1] = current_budget
        Guarantees at least 1 token for each category when total >= 2.
    """
    total = np.asarray(total_budget_table, dtype=np.int64)
    ratios = np.asarray(memory_ratios, dtype=np.float64)
    if total.shape != ratios.shape:
        raise ValueError(
            f"Shape mismatch: total_budget_table {total.shape} vs "
            f"memory_ratios {ratios.shape}"
        )
    num_layers, num_kv_heads = total.shape
    split = np.zeros((num_layers, num_kv_heads, 2), dtype=np.int32)
    for l in range(num_layers):
        for h in range(num_kv_heads):
            t = int(total[l, h])
            if t <= 0:
                split[l, h, 0] = 0
                split[l, h, 1] = 0
            elif t == 1:
                split[l, h, 0] = int(round(float(t) * float(ratios[l, h])))
                split[l, h, 1] = t - split[l, h, 0]
            else:
                mem_b = max(1, int(round(float(t) * float(ratios[l, h]))))
                mem_b = min(mem_b, t - 1)
                split[l, h, 0] = mem_b
                split[l, h, 1] = t - mem_b
    return split


def _select_split_budget_tokens(combined_score, previous_positions,
                                  current_positions, visual_positions,
                                  memory_budget, current_budget):
    """Select tokens independently from memory and current regions.

    This is a shared helper used by both the dense-union and ragged-prefill
    selection paths.

    Args:
        combined_score: 1-D torch.Tensor [num_visual_tokens], the per-token
            attention + recency score (higher = more worth keeping).
        previous_positions: 1-D torch.LongTensor, indices (into the visual range)
            of tokens that belong to the *memory* region (pre-chunk).
        current_positions: 1-D torch.LongTensor, indices (into the visual range)
            of tokens that belong to the *current* region (latest chunk).
        visual_positions: 1-D torch.LongTensor, indices of all visual tokens
            (used for overflow filling when a region has fewer tokens than
            its budget).
        memory_budget: int, how many tokens to select from the memory region.
        current_budget: int, how many tokens to select from the current region.

    Returns:
        torch.LongTensor of selected visual-relative indices (sorted, unique).
    """
    device = combined_score.device
    memory_budget = int(memory_budget)
    current_budget = int(current_budget)

    # --- select from memory region ---
    mem_budget = min(memory_budget, int(previous_positions.numel()))
    if mem_budget <= 0:
        mem_selected = torch.empty(0, device=device, dtype=torch.long)
    else:
        _, topk_mem = torch.topk(
            combined_score.index_select(0, previous_positions),
            mem_budget, sorted=False,
        )
        mem_selected = previous_positions.index_select(0, topk_mem)

    # --- select from current region ---
    cur_budget = min(current_budget, int(current_positions.numel()))
    if cur_budget <= 0:
        cur_selected = torch.empty(0, device=device, dtype=torch.long)
    else:
        _, topk_cur = torch.topk(
            combined_score.index_select(0, current_positions),
            cur_budget, sorted=False,
        )
        cur_selected = current_positions.index_select(0, topk_cur)

    selected = torch.cat([mem_selected, cur_selected])

    # --- overflow: use remaining budget on any unfilled visual token ---
    total_budget = memory_budget + current_budget
    remaining = total_budget - int(selected.numel())
    remaining = min(remaining, int(visual_positions.numel()) - int(selected.numel()))
    if remaining > 0:
        free_score = torch.full_like(
            combined_score, -torch.finfo(combined_score.dtype).max
        )
        if visual_positions.numel() > 0:
            free_score[visual_positions] = combined_score[visual_positions]
        if selected.numel() > 0:
            free_score[selected] = -torch.finfo(free_score.dtype).max
        _, overflow = torch.topk(free_score, remaining, sorted=False)
        selected = torch.cat([selected, overflow])

    return torch.unique(selected, sorted=True)


def build_kv_head_budget_table(scores, num_keep, num_layers=28,
                               num_query_heads=28, num_kv_heads=4,
                               strength=0.75, min_ratio=0.75,
                               max_ratio=1.25,
                               scheme="relative",
                               sparsemm_ratio=0.1,
                               sparsemm_window_size=32,
                               floor_ratio=0.0):
    """
    Build [layer, kv_head] budgets.

    scheme="relative" keeps the old HERMES prototype behavior: each layer has
    average num_keep and heads are clipped by min_ratio/max_ratio.

    scheme="sparsemm" follows SparseMM's static visual-score formula:
        base = num_keep - window_size
        min_cache = base * ratio
        head_budget = score * ((base - min_cache) * L * H_kv) + min_cache
        final_budget = head_budget + window_size

    During pruning, the window_size portion is treated as protected recent
    cache, while the remaining head_budget is used for historical top-k
    selection.

    SparseMM normalizes visual scores globally over all layers/KV-heads, so the
    average budget is num_keep per layer-KV-head.

    scheme="sparsemm_layer_total" keeps the same SparseMM score ranking and
    per-head minimum/window, but constrains the global total visual budget to
    num_keep * L. For Qwen2.5-VL-7B with 4 KV heads, num_keep=6000 means an
    average of 1500 visual tokens per layer-KV-head, or 6000 per layer in total.

    scheme="sparsemm_per_layer_total" constrains each layer separately: the
    layer's KV-head budgets sum to num_keep exactly, while SparseMM scores still
    decide how the layer budget is split among that layer's KV heads.
    """
    kv_scores = _aggregate_query_scores_to_kv(
        scores, num_layers, num_query_heads, num_kv_heads
    )
    scheme = str(scheme or "relative").lower()
    if scheme == "sparsemm":
        base_capacity = max(1, int(num_keep) - int(sparsemm_window_size))
        min_cache = int(base_capacity * float(sparsemm_ratio))
        remain_capacity = (
            max(base_capacity - min_cache, 0)
            * int(num_layers)
            * int(num_kv_heads)
        )
        weights = np.nan_to_num(np.abs(kv_scores), nan=0.0, posinf=0.0, neginf=0.0)
        if weights.sum() <= 1e-12:
            weights = np.ones_like(weights, dtype=np.float64)
        weights = weights / weights.sum()
        budgets = np.round(weights * remain_capacity + min_cache).astype(int)
        budgets = budgets + int(sparsemm_window_size)
        budgets = np.maximum(budgets, 1).astype(int)
        return _apply_budget_floor(budgets, num_keep, floor_ratio)

    if scheme in ("sparsemm_layer_total", "sparsemm_layer", "sparsemm_total"):
        base_capacity = max(1, int(num_keep) - int(sparsemm_window_size))
        min_cache = int(base_capacity * float(sparsemm_ratio))
        min_budget = max(1, min_cache + int(sparsemm_window_size))
        target_total = int(num_keep) * int(num_layers)
        weights = np.nan_to_num(np.abs(kv_scores), nan=0.0, posinf=0.0, neginf=0.0)
        budgets_flat = _allocate_integer_budget(
            weights.reshape(-1),
            total_budget=target_total,
            min_budget=min_budget,
            max_budget=None,
        )
        budgets = budgets_flat.reshape(int(num_layers), int(num_kv_heads)).astype(int)
        return _apply_budget_floor(budgets, num_keep, floor_ratio)

    if scheme in ("sparsemm_per_layer_total", "sparsemm_layer_exact", "sparsemm_per_layer"):
        base_capacity = max(1, int(num_keep) - int(sparsemm_window_size))
        min_cache = int(base_capacity * float(sparsemm_ratio))
        min_budget = max(1, min_cache + int(sparsemm_window_size))
        weights = np.nan_to_num(np.abs(kv_scores), nan=0.0, posinf=0.0, neginf=0.0)
        budgets = np.zeros((int(num_layers), int(num_kv_heads)), dtype=int)
        for layer_idx in range(int(num_layers)):
            budgets[layer_idx] = _allocate_integer_budget(
                weights[layer_idx],
                total_budget=int(num_keep),
                min_budget=min_budget,
                max_budget=None,
            )
        return _apply_budget_floor(budgets, num_keep, floor_ratio)

    if scheme != "relative":
        raise ValueError(f"Unsupported KV-head budget scheme: {scheme}")

    if kv_scores.max() < 1e-8:
        weights = np.ones_like(kv_scores)
    else:
        global_mean = kv_scores.mean() + 1e-8
        weights = 1.0 + strength * (kv_scores / global_mean - 1.0)
        weights = np.clip(weights, min_ratio, max_ratio)

    budgets = np.zeros((num_layers, num_kv_heads), dtype=int)
    total_per_layer = int(num_keep) * int(num_kv_heads)
    min_h = max(1, int(round(num_keep * min_ratio)))
    max_h = max(min_h, int(round(num_keep * max_ratio)))
    for layer_idx in range(num_layers):
        budgets[layer_idx] = _allocate_integer_budget(
            weights[layer_idx],
            total_per_layer,
            min_budget=min_h,
            max_budget=max_h,
        )
    return _apply_budget_floor(budgets, num_keep, floor_ratio)


def _get_layer_past_len(past_key_values, layer_idx):
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "layers"):
        if layer_idx >= len(past_key_values.layers):
            return 0
        layer = past_key_values.layers[layer_idx]
        if not getattr(layer, "is_initialized", False):
            return 0
        return layer.keys.shape[-2]
    return past_key_values[layer_idx][0].shape[-2]


def _build_group_mask(batch, q_len, past_len, hidden_states, layer_idx,
                      model, num_query_heads, num_kv_heads, group_size):
    device = hidden_states.device
    mask = torch.zeros(
        (batch, num_query_heads, q_len, past_len + q_len),
        dtype=torch.bool,
        device=device,
    )

    layer_positions = getattr(model, "_kv_head_budget_cache_positions", {}).get(layer_idx)
    union_len = getattr(model, "_kv_head_budget_union_lens", {}).get(layer_idx, past_len)

    if layer_positions is None:
        mask[:, :, :, :past_len] = True
    else:
        for q_head in range(num_query_heads):
            kv_head = q_head // group_size
            allowed = layer_positions[kv_head]
            if not isinstance(allowed, torch.Tensor):
                allowed = torch.as_tensor(allowed, device=device, dtype=torch.long)
            else:
                allowed = allowed.to(device=device, dtype=torch.long)
            allowed = allowed[(allowed >= 0) & (allowed < past_len)]
            if allowed.numel() > 0:
                mask[:, q_head, :, allowed] = True

        # HERMES long-term layers can append summary tokens after the union.
        # Let every head see those extra tail tokens.
        if union_len < past_len:
            mask[:, :, :, union_len:past_len] = True

    if q_len > 0:
        causal_current = torch.tril(
            torch.ones((q_len, q_len), dtype=torch.bool, device=device)
        )
        mask[:, :, :, past_len:past_len + q_len] = causal_current.view(1, 1, q_len, q_len)

    attn_impl = getattr(_get_language_config(model), "_attn_implementation", "sdpa")
    if attn_impl == "eager":
        additive = torch.zeros(mask.shape, dtype=hidden_states.dtype, device=device)
        additive = additive.masked_fill(~mask, torch.finfo(hidden_states.dtype).min)
        return additive

    return mask


def _install_kv_head_mask_hooks(model, num_query_heads, num_kv_heads, group_size):
    if not hasattr(model.language_model, "layers"):
        raise ValueError("KV-head budget masks require Qwen-style language_model.layers")

    old_handles = getattr(model, "_kv_head_budget_hook_handles", [])
    for handle in old_handles:
        handle.remove()

    handles = []

    def make_hook(layer_idx):
        def hook(module, args, kwargs):
            hidden_states = args[0] if args else kwargs["hidden_states"]
            batch, q_len = hidden_states.shape[:2]
            past_key_values = kwargs.get("past_key_values", None)
            past_len = _get_layer_past_len(past_key_values, layer_idx)

            max_mask_q_len = getattr(model, "_kv_head_budget_max_mask_q_len", 128)
            if q_len <= max_mask_q_len:
                kwargs["attention_mask"] = _build_group_mask(
                    batch=batch,
                    q_len=q_len,
                    past_len=past_len,
                    hidden_states=hidden_states,
                    layer_idx=layer_idx,
                    model=model,
                    num_query_heads=num_query_heads,
                    num_kv_heads=num_kv_heads,
                    group_size=group_size,
                )

            if layer_idx in getattr(model, "_layer_position_ids", {}):
                position_ids = model._layer_position_ids[layer_idx]
                kwargs["position_ids"] = position_ids
                kwargs["position_embeddings"] = model.language_model.rotary_emb(
                    hidden_states,
                    position_ids,
                )
            return args, kwargs

        return hook

    for layer_idx, layer in enumerate(model.language_model.layers):
        handles.append(layer.register_forward_pre_hook(make_hook(layer_idx), with_kwargs=True))

    model._kv_head_budget_hook_handles = handles


def apply_kv_head_budget(model, kv_head_scores_path=None, scores=None,
                         num_layers=None, num_query_heads=None,
                         num_kv_heads=None, strength=0.75,
                         min_ratio=0.75, max_ratio=1.25,
                         budget_scheme="relative",
                         sparsemm_ratio=0.1,
                         sparsemm_window_size=32,
                         floor_ratio=0.0,
                         union_cap_ratio=1.0,
                         max_mask_q_len=128,
                         memory_current_split=False,
                         memory_scores_path=None,
                         base_memory_ratio=0.7,
                         split_dynamic_scale=0.3,
                         short_term_frames=0,
                         layerwise_last_frame=False,
                         layerwise_shallow_end=10,
                         layerwise_mid_end=20,
                         layerwise_shallow_ratio=1.0,
                         layerwise_mid_ratio=0.5,
                         layerwise_deep_ratio=0.0):
    """
    Install per-KV-head logical eviction.

    union_cap_ratio caps the dense physical union length relative to num_keep.
    The default 1.0 keeps the physical layer cache close to the original HERMES
    length while still applying different per-head visibility masks.
    """
    num_layers = int(num_layers or model.num_layers)
    num_query_heads, num_kv_heads, group_size = _get_attention_shape(
        model, num_query_heads=num_query_heads, num_kv_heads=num_kv_heads
    )

    if scores is None and kv_head_scores_path is not None:
        scores = load_head_scores(kv_head_scores_path, num_layers, num_query_heads)
    if scores is None:
        scores = np.zeros((num_layers, num_query_heads), dtype=np.float64)
        print("[kv_head_budget] No scores provided, using uniform KV-head budgets")

    scores = np.asarray(scores, dtype=np.float64)
    if not 0.0 <= float(floor_ratio) <= 1.0:
        raise ValueError("floor_ratio must be in [0, 1]")
    print(f"[kv_head_budget] Loaded scores. Range: [{scores.min():.4f}, {scores.max():.4f}]")

    _install_kv_head_mask_hooks(model, num_query_heads, num_kv_heads, group_size)
    model._kv_head_budget_cache_positions = {}
    model._kv_head_budget_union_lens = {}
    model._kv_head_budget_max_mask_q_len = int(max_mask_q_len)
    model._kv_head_budget_scores = scores
    budget_scheme_name = str(budget_scheme or "relative").lower()
    protected_recent_window = (
        max(0, int(sparsemm_window_size))
        if budget_scheme_name.startswith("sparsemm")
        else 0
    )

    # ---- memory / current split setup ----
    memory_ratios = None
    if memory_current_split:
        is_direct = False
        if memory_scores_path is not None:
            memory_kv_scores, is_direct = load_kv_head_memory_scores(
                memory_scores_path,
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
            )
        else:
            memory_kv_scores = _aggregate_query_scores_to_kv(
                scores, num_layers, num_query_heads, num_kv_heads
            )
        if is_direct:
            # Values are already observed ratios in [0,1]; just clip for safety.
            memory_ratios = np.clip(memory_kv_scores, 0.1, 0.9)
            print(
                "[kv_head_budget] memory_current_split enabled (direct observed ratios). "
                f"memory_ratios range: [{memory_ratios.min():.3f}, {memory_ratios.max():.3f}]"
            )
        else:
            memory_ratios = compute_memory_current_split_ratios(
                memory_kv_scores,
                base_memory_ratio=base_memory_ratio,
                dynamic_scale=split_dynamic_scale,
            )
            print(
                "[kv_head_budget] memory_current_split enabled. "
                f"memory_ratios range: [{memory_ratios.min():.3f}, {memory_ratios.max():.3f}], "
                f"base={base_memory_ratio}, scale={split_dynamic_scale}"
            )

        # Install pre/post lens tracking wrapper on encode_video_chunk so
        # kv_head_budget_prune knows where the latest chunk begins.
        if not hasattr(model, "_kv_head_budget_original_encode_video_chunk"):
            _original_encode = model.encode_video_chunk

            def _tracked_encode_video_chunk(self, video_chunk, **kwargs):
                pre_lens = self._get_cache_seq_len_per_layer()
                if pre_lens is None or len(pre_lens) == 0:
                    pre_lens = [0] * self.num_layers
                _original_encode(video_chunk, **kwargs)
                post_lens = self._get_cache_seq_len_per_layer()
                if post_lens is None or len(post_lens) == 0:
                    post_lens = [0] * self.num_layers
                self._kv_head_budget_last_pre_lens = [
                    int(x) for x in pre_lens
                ]
                self._kv_head_budget_last_post_lens = [
                    int(x) for x in post_lens
                ]

            model._kv_head_budget_original_encode_video_chunk = _original_encode
            model.encode_video_chunk = _tracked_encode_video_chunk.__get__(
                model, type(model)
            )
    # ---------------------------------------------------------------

    def kv_head_budget_prune(attn_weights_local, attn_weights_global,
                             attn_weights_mixed, num_keep=3000):
        device = model.device
        visual_start_idx = model.visual_start_idx
        n_layers = len(attn_weights_local)

        if n_layers != num_layers:
            raise ValueError(
                f"KV-head budget configured for {num_layers} layers, got {n_layers}"
            )

        actual_heads = attn_weights_local[0].shape[1]
        if actual_heads != num_query_heads:
            raise ValueError(
                f"Configured num_query_heads={num_query_heads}, attention has {actual_heads}"
            )

        question_len_local = attn_weights_local[0].shape[2]
        question_len_global = attn_weights_global[0].shape[2]
        question_len_mixed = attn_weights_mixed[0].shape[2]

        budget_table = build_kv_head_budget_table(
            scores,
            num_keep,
            num_layers=num_layers,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            strength=strength,
            min_ratio=min_ratio,
            max_ratio=max_ratio,
            scheme=budget_scheme,
            sparsemm_ratio=sparsemm_ratio,
            sparsemm_window_size=sparsemm_window_size,
            floor_ratio=floor_ratio,
        )
        model._kv_head_budget_values = budget_table

        # Build split budget table when memory/current split is enabled.
        split_budget_table = None
        if memory_current_split and memory_ratios is not None:
            split_budget_table = build_kv_head_split_budget_table(
                budget_table, memory_ratios
            )
            print(
                "[kv_head_budget] split_budget_table built: "
                f"mem=[{split_budget_table[:,:,0].min()}, {split_budget_table[:,:,0].max()}], "
                f"cur=[{split_budget_table[:,:,1].min()}, {split_budget_table[:,:,1].max()}]"
            )

        keep_indices_all_layers = []
        cache_positions = {}
        union_lens = {}
        union_lengths_for_log = []
        visible_lengths_for_log = []

        for layer_idx in range(n_layers):
            if layer_idx < model.short_term_threshold:
                layer_type = "short-term"
                layer_attn_weights = attn_weights_local[layer_idx]
                question_len = question_len_local
                recency_alpha = 1.0
                k_decay = 20.0
            elif layer_idx >= model.long_term_threshold:
                layer_type = "long-term"
                layer_attn_weights = attn_weights_global[layer_idx]
                question_len = question_len_global
                recency_alpha = 0.0
                k_decay = 0.0
            else:
                layer_type = "mid-term"
                layer_attn_weights = attn_weights_mixed[layer_idx]
                question_len = question_len_mixed
                progress = (
                    (layer_idx - model.short_term_threshold)
                    / (model.long_term_threshold - model.short_term_threshold)
                )
                recency_alpha = 0.75 - 0.6 * progress
                k_decay = 20.0 - 12.0 * progress

            visual_attn_q = layer_attn_weights[0].mean(dim=1)[:, visual_start_idx:-question_len]
            if visual_attn_q.shape[0] != num_query_heads:
                raise ValueError(
                    f"Layer {layer_idx}: attention heads {visual_attn_q.shape[0]} "
                    f"!= configured {num_query_heads}"
                )

            num_visual_tokens = visual_attn_q.shape[1]
            if num_visual_tokens <= 0:
                keep = torch.arange(visual_start_idx, device=device)
                keep_indices_all_layers.append(keep.tolist())
                all_text = keep.clone()
                cache_positions[layer_idx] = [all_text for _ in range(num_kv_heads)]
                union_lens[layer_idx] = int(keep.numel())
                continue

            visual_attn_kv = visual_attn_q.view(
                num_kv_heads, group_size, num_visual_tokens
            ).mean(dim=1)

            positions = torch.arange(num_visual_tokens, device=device, dtype=torch.float32)
            time_distances = (num_visual_tokens - 1 - positions) / max(num_visual_tokens - 1, 1)
            recency = torch.exp(-k_decay * time_distances)
            recency = (recency - recency.min()) / (recency.max() - recency.min() + 1e-6)

            kv_selected = []
            vote = torch.zeros(num_visual_tokens, device=device)
            score_sum = torch.zeros(num_visual_tokens, device=device)
            effective_budgets = budget_table[layer_idx].copy()
            if layer_type == "long-term":
                effective_budgets = np.maximum(effective_budgets - 1, 0)

            # Region boundaries for memory/current split (dense path:
            # all KV heads in a layer share the same cache length, so the
            # boundary is per-layer.)
            pre_lens = getattr(model, "_kv_head_budget_last_pre_lens", None)
            if split_budget_table is not None and pre_lens is not None:
                pre_vis_len = max(0, int(pre_lens[layer_idx]) - visual_start_idx)
                pre_vis_len = min(pre_vis_len, num_visual_tokens)
            else:
                pre_vis_len = 0

            for kv_head in range(num_kv_heads):
                h_attn = visual_attn_kv[kv_head]
                attn_norm = (h_attn - h_attn.min()) / (h_attn.max() - h_attn.min() + 1e-6)
                score = attn_norm * (1.0 - recency_alpha) + recency * recency_alpha
                score_sum += score

                if split_budget_table is not None:
                    # ---- memory/current split selection ----
                    mem_budget = int(split_budget_table[layer_idx, kv_head, 0])
                    cur_budget = int(split_budget_table[layer_idx, kv_head, 1])

                    # recent window: always keep last N visual tokens
                    recent_k = min(protected_recent_window, num_visual_tokens)
                    if recent_k > 0:
                        protected_recent = torch.arange(
                            num_visual_tokens - recent_k,
                            num_visual_tokens,
                            device=device,
                            dtype=torch.long,
                        )
                    else:
                        protected_recent = torch.empty(0, device=device, dtype=torch.long)

                    history_len = max(0, num_visual_tokens - recent_k)

                    # Split the history portion into previous (memory) and
                    # current regions.
                    split_point = min(pre_vis_len, history_len)
                    prev_positions = torch.arange(0, split_point, device=device, dtype=torch.long)
                    curr_positions = torch.arange(split_point, history_len, device=device, dtype=torch.long)
                    all_history = torch.arange(0, history_len, device=device, dtype=torch.long)

                    # Adjust budgets to account for the reserved recent window.
                    total_avail = mem_budget + cur_budget
                    history_budget = max(0, total_avail - recent_k)
                    if history_budget > 0:
                        mem_frac = mem_budget / max(total_avail, 1)
                        hist_mem_budget = max(0, int(round(history_budget * mem_frac)))
                        hist_cur_budget = history_budget - hist_mem_budget
                    else:
                        hist_mem_budget = 0
                        hist_cur_budget = 0

                    history_selected = _select_split_budget_tokens(
                        combined_score=score[:history_len],
                        previous_positions=prev_positions,
                        current_positions=curr_positions,
                        visual_positions=all_history,
                        memory_budget=hist_mem_budget,
                        current_budget=hist_cur_budget,
                    )
                    selected = torch.cat([history_selected, protected_recent])
                    selected = torch.unique(selected, sorted=True)
                    vote[selected] += 1.0
                    kv_selected.append(selected)
                else:
                    # ---- original top-k selection (unchanged) ----
                    k = min(int(effective_budgets[kv_head]), num_visual_tokens)
                    if k <= 0:
                        selected = torch.empty(0, device=device, dtype=torch.long)
                    else:
                        recent_k = min(protected_recent_window, k, num_visual_tokens)
                        history_len = max(0, num_visual_tokens - recent_k)
                        history_k = min(max(k - recent_k, 0), history_len)

                        pieces = []
                        if history_k > 0:
                            pieces.append(torch.topk(score[:history_len], history_k, sorted=False)[1])
                        if recent_k > 0:
                            pieces.append(
                                torch.arange(
                                    num_visual_tokens - recent_k,
                                    num_visual_tokens,
                                    device=device,
                                    dtype=torch.long,
                                )
                            )
                        selected = torch.unique(torch.cat(pieces), sorted=True)
                        vote[selected] += 1.0
                    kv_selected.append(selected)

            combined = vote + 1e-3 * (
                (score_sum - score_sum.min()) / (score_sum.max() - score_sum.min() + 1e-6)
            )

            union_rel = torch.unique(torch.cat(kv_selected), sorted=True)
            union_cap = int(round(float(num_keep) * float(union_cap_ratio)))
            if layer_type == "long-term":
                union_cap = max(0, union_cap - 1)
            union_cap = min(max(union_cap, 0), num_visual_tokens)

            if union_cap > 0 and union_rel.numel() > union_cap:
                recent_k = min(protected_recent_window, union_cap, num_visual_tokens)
                if recent_k > 0:
                    protected_rel = torch.arange(
                        num_visual_tokens - recent_k,
                        num_visual_tokens,
                        device=device,
                        dtype=torch.long,
                    )
                    remaining_cap = union_cap - int(protected_rel.numel())
                    if remaining_cap > 0:
                        all_rel = torch.arange(num_visual_tokens, device=device, dtype=torch.long)
                        candidate_rel = all_rel[~torch.isin(all_rel, protected_rel)]
                        _, capped_pos = torch.topk(combined[candidate_rel], remaining_cap, sorted=False)
                        capped = candidate_rel[capped_pos]
                        union_rel = torch.unique(torch.cat([protected_rel, capped]), sorted=True)
                    else:
                        union_rel = protected_rel
                else:
                    _, capped = torch.topk(combined, union_cap, sorted=False)
                    union_rel = torch.sort(capped)[0]
            elif union_cap == 0:
                union_rel = torch.empty(0, device=device, dtype=torch.long)

            union_abs = union_rel + visual_start_idx
            text_abs = torch.arange(visual_start_idx, device=device)
            full_keep = torch.cat([text_abs, union_abs]).unique(sorted=True)

            union_lens[layer_idx] = int(full_keep.numel())
            keep_indices_all_layers.append(full_keep.tolist())
            union_lengths_for_log.append(int(full_keep.numel()))

            layer_positions = []
            for kv_head, selected_rel in enumerate(kv_selected):
                selected_abs = selected_rel + visual_start_idx
                selected_abs = selected_abs[torch.isin(selected_abs, union_abs)]
                allowed_abs = torch.cat([text_abs, torch.sort(selected_abs)[0]]).unique(sorted=True)

                # full_keep is sorted, so searchsorted maps old absolute indices to
                # positions in the compacted dense-union cache.
                compact_pos = torch.searchsorted(full_keep, allowed_abs)
                compact_pos = compact_pos[compact_pos < full_keep.numel()]
                layer_positions.append(compact_pos.detach())
                visible_lengths_for_log.append(int(compact_pos.numel()))

            cache_positions[layer_idx] = layer_positions

        model._kv_head_budget_cache_positions = cache_positions
        model._kv_head_budget_union_lens = union_lens
        model._kv_head_budget_last_stats = {
            "union_min": min(union_lengths_for_log) if union_lengths_for_log else 0,
            "union_max": max(union_lengths_for_log) if union_lengths_for_log else 0,
            "visible_min": min(visible_lengths_for_log) if visible_lengths_for_log else 0,
            "visible_max": max(visible_lengths_for_log) if visible_lengths_for_log else 0,
            "budget_min": int(budget_table.min()),
            "budget_max": int(budget_table.max()),
        }
        print(
            "[kv_head_budget] prune stats: "
            f"kv_budget=[{budget_table.min()}, {budget_table.max()}], "
            f"visible=[{model._kv_head_budget_last_stats['visible_min']}, "
            f"{model._kv_head_budget_last_stats['visible_max']}], "
            f"union=[{model._kv_head_budget_last_stats['union_min']}, "
            f"{model._kv_head_budget_last_stats['union_max']}]"
        )

        return keep_indices_all_layers

    model.prune_kv_cache_by_attention = kv_head_budget_prune
    model._kv_head_budget_config = {
        "num_query_heads": num_query_heads,
        "num_kv_heads": num_kv_heads,
        "group_size": group_size,
        "strength": strength,
        "min_ratio": min_ratio,
        "max_ratio": max_ratio,
        "budget_scheme": str(budget_scheme or "relative").lower(),
        "sparsemm_ratio": float(sparsemm_ratio),
        "sparsemm_window_size": int(sparsemm_window_size),
        "floor_ratio": float(floor_ratio),
        "union_cap_ratio": union_cap_ratio,
        "max_mask_q_len": int(max_mask_q_len),
        "mode": "logical_per_kv_head_dense_union",
        "memory_current_split": bool(memory_current_split),
        "memory_scores_path": memory_scores_path,
        "base_memory_ratio": float(base_memory_ratio),
        "split_dynamic_scale": float(split_dynamic_scale),
        "short_term_frames": int(short_term_frames),
        "layerwise_last_frame": bool(layerwise_last_frame),
        "layerwise_shallow_end": int(layerwise_shallow_end),
        "layerwise_mid_end": int(layerwise_mid_end),
        "layerwise_shallow_ratio": float(layerwise_shallow_ratio),
        "layerwise_mid_ratio": float(layerwise_mid_ratio),
        "layerwise_deep_ratio": float(layerwise_deep_ratio),
    }
    model.short_term_frames = int(short_term_frames)
    if memory_current_split and memory_ratios is not None:
        model._kv_head_budget_config["memory_ratios"] = memory_ratios
    print(
        "[kv_head_budget] Installed logical per-KV-head eviction. "
        f"q_heads={num_query_heads}, kv_heads={num_kv_heads}, "
        f"group={group_size}, budget_scheme={model._kv_head_budget_config['budget_scheme']}, "
        f"sparsemm_ratio={float(sparsemm_ratio):g}, sparsemm_window={int(sparsemm_window_size)}, "
        f"floor_ratio={float(floor_ratio):g}, "
        f"union_cap_ratio={union_cap_ratio}, "
        f"max_mask_q_len={int(max_mask_q_len)}, "
        f"memory_current_split={memory_current_split}, "
        f"layerwise_last_frame={bool(layerwise_last_frame)}."
    )
    if layerwise_last_frame:
        print(
            "[kv_head_budget] Layerwise last-frame protection: "
            f"layers< {int(layerwise_shallow_end)} ratio={float(layerwise_shallow_ratio):g}, "
            f"layers< {int(layerwise_mid_end)} ratio={float(layerwise_mid_ratio):g}, "
            f"deep ratio={float(layerwise_deep_ratio):g}."
        )
    return model
