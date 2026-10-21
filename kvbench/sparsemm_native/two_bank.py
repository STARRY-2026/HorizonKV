"""DP-sized recent sliding windows and SparseMM-scored historical caches.

The recent window includes (rather than adds) the protected W tokens. New
tokens leaving that window compete with retained history at every patch.
No previously evicted token can return, and no head lends its unused quota.
"""

import csv

import torch

from sparsemm_baselines.core import BaselineConfig, _head_scores, _validate_inputs
from sparsemm_native.policy import _integer, load_fixed_head_budgets


def load_two_bank_budgets(path, num_layers, num_kv_heads, budget, window_size=32,
                          *, short_cache_gate="per_head_capacity"):
    if short_cache_gate not in ("per_head_capacity", "legacy_global"):
        raise ValueError("Unknown two-bank short-cache gate")
    capacities, allocation = load_fixed_head_budgets(
        path, num_layers, num_kv_heads, budget, window_size)
    recent = torch.full_like(capacities, -1)
    with open(allocation["budget_path"], newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"recent_r", "history_q"} <= set(reader.fieldnames or []):
            raise ValueError("Two-bank budgets require recent_r and history_q")
        for row in reader:
            layer, head, value = int(row["layer"]), int(row["kv_head"]), int(row["recent_r"])
            if value < window_size:
                raise ValueError("Two-bank recent_r must include the protected window")
            recent[layer, head] = value
    # The common loader independently validates indices, uniqueness, totals,
    # integral splits and recent_r + history_q == capacity_n.
    if bool((recent > capacities).any()) or bool((recent < window_size).any()):
        raise ValueError("Invalid two-bank recent capacity")
    allocation = dict(allocation, method="dp_two_bank_sparsemm_selector",
                      selector="dp_recent_sliding_then_sparsemm_history_attention",
                      recent_budgets=recent.tolist(),
                      history_budgets=(capacities - recent).tolist(),
                      dp_bank_split_used_online=True,
                      protect_window_included_in_recent=True,
                      short_cache_gate=short_cache_gate,
                      unused_capacity_redistribution=False)
    return capacities, recent, allocation


@torch.inference_mode()
def select_two_bank_layer(keys, queries, budgets, recent_budgets, prefix_tokens=0, *,
                          budget=1000, window_size=32, kernel_size=5,
                          pooling="avgpool", gqa_reduce="mean", layer_idx=0,
                          short_cache_gate="per_head_capacity"):
    """Keep prefix + attention top-H older tokens + newest R tokens per head."""
    budget = _integer(budget, "budget", 1)
    window_size = _integer(window_size, "window_size", 1)
    prefix_tokens = _integer(prefix_tokens, "prefix_tokens")
    layer_idx = _integer(layer_idx, "layer_idx")
    if short_cache_gate not in ("per_head_capacity", "legacy_global"):
        raise ValueError("Unknown two-bank short-cache gate")
    if not isinstance(keys, (list, tuple)) or not keys:
        raise ValueError("keys must be a nonempty list of per-head tensors")
    if not isinstance(queries, torch.Tensor):
        raise ValueError("queries must be a tensor [Hq,Q,D]")
    budgets = budgets.tolist() if hasattr(budgets, "tolist") else budgets
    recent_budgets = (recent_budgets.tolist() if hasattr(recent_budgets, "tolist")
                      else recent_budgets)
    if (not isinstance(budgets, (list, tuple))
            or not isinstance(recent_budgets, (list, tuple))
            or len(budgets) != len(keys) or len(recent_budgets) != len(keys)):
        raise ValueError("One total and recent budget is required per KV head")
    budgets = [_integer(n, "head budget", window_size) for n in budgets]
    recent_budgets = [_integer(r, "recent budget", window_size) for r in recent_budgets]
    if any(r > n for n, r in zip(budgets, recent_budgets)):
        raise ValueError("Recent budget cannot exceed total head budget")
    config = BaselineConfig(method="adakv", budget=budget, window_size=window_size,
                           kernel_size=kernel_size, pooling=pooling,
                           gqa_reduce=gqa_reduce, budget_scope="visual",
                           score_precision="float32")
    groups, query_len, _ = _validate_inputs(keys, queries, config, 0, 1, prefix_tokens, None)
    selected, recent_counts, history_counts, bypassed = [], [], [], []
    lengths = [key.shape[0] for key in keys]
    reference_warmup = short_cache_gate == "legacy_global" and all(
        length - prefix_tokens - min(window_size, length - prefix_tokens) < budget - window_size
        for length in lengths)
    history_budgets = [n - r for n, r in zip(budgets, recent_budgets)]
    for head, (key, capacity, recent_capacity, history_capacity) in enumerate(
            zip(keys, budgets, recent_budgets, history_budgets)):
        visual = key.shape[0] - prefix_tokens
        r = min(recent_capacity, visual)
        h = min(history_capacity, visual - r)
        bypass = (reference_warmup if short_cache_gate == "legacy_global"
                  else visual <= capacity)
        if bypass:
            keep = torch.arange(key.shape[0], device=key.device, dtype=torch.long)
            h = visual - r
        else:
            if h:
                score = _head_scores(key, queries[head * groups:(head + 1) * groups],
                                     prefix_tokens, r, config, layer_idx, head)
                ranking = torch.sort(score, descending=True).indices[:h]
            else:
                ranking = torch.empty(0, device=key.device, dtype=torch.long)
            keep = torch.cat((torch.arange(prefix_tokens, device=key.device),
                              ranking + prefix_tokens,
                              torch.arange(key.shape[0] - r, key.shape[0], device=key.device)))
        selected.append(keep)
        recent_counts.append(r)
        history_counts.append(h)
        bypassed.append(bypass)
    after = [keep.numel() for keep in selected]
    return selected, {
        "method": "dp_two_bank", "selector": "recent_sliding_then_sparsemm_history_attention",
        "budget_scope": "visual", "configured_budget": budget, "layer_idx": layer_idx,
        "head_budgets": budgets, "recent_budgets": recent_budgets,
        "history_budgets": history_budgets, "prefix_tokens_kept": prefix_tokens,
        "window_size": window_size, "protect_window_included_in_recent": True,
        "before_head_lengths": lengths, "head_lengths": after,
        "recent_counts": recent_counts, "retained_history_counts": history_counts,
        "proposed_history_counts": history_budgets,
        "retained_visual_counts": [n - prefix_tokens for n in after],
        "observed_queries": min(query_len, window_size), "gqa_groups": groups,
        "gqa_reduce": gqa_reduce, "score_precision": "float32", "pooling": pooling,
        "kernel_size": kernel_size, "target_total_tokens": sum(budgets) + len(keys) * prefix_tokens,
        "actual_total_tokens": sum(after), "bypassed_short_cache": all(bypassed),
        "bypassed_short_cache_per_head": bypassed, "short_cache_gate": short_cache_gate,
        "reference_warmup_bypass": reference_warmup,
        "compressed": any(a < b for a, b in zip(after, lengths)),
        "tie_policy": "upstream_torch_sort", "rounding": "external_exact_capacity",
    }
