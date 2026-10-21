"""Original SparseMM visual-head quotas with native streaming attention scores.

The pinned upstream class is named SparseMM. Its static quotas normalize every
query-head visual score globally and SUM shared query heads into physical KV
heads; its online attention scores use GQA MEAN. The default Qwen2.5 scores are
from the local 1000-image official SynthDOG calibration, rather than the older
Qwen2-VL score file shipped in the pinned checkout. Visual scope protects initial
text separately, and FP32/ragged/short-query scoring is the streaming extension.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
import re

import numpy as np
import torch

from sparsemm_baselines.core import BaselineConfig, UPSTREAM_COMMIT, _head_scores, _validate_inputs


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_SCORE_PATH = (WORKSPACE / "hermes/results/myobservations/sparsemm_official_calibration_20261010"
                      / "calibration/head_scores.csv")
DEFAULT_SCORE_SHA256 = "9cc0f47f24a0af4306ce7367806516d119dc4f57df77d064a3eac59c05b69dab"
DEFAULT_RATIO = .1


def _integer(value, name, minimum=0):
    if not isinstance(value, Integral) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _index(value, name):
    if not isinstance(value, str) or re.fullmatch(r"\d+", value.strip()) is None:
        raise ValueError(f"{name} must contain a nonnegative integer")
    return int(value)


def _scores_array(scores):
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    scores = np.asarray(scores)
    if (scores.ndim != 2 or min(scores.shape) < 1 or not np.issubdtype(scores.dtype, np.number)
            or np.iscomplexobj(scores)):
        raise ValueError("Visual head scores must be a numeric [layers,query_heads] matrix")
    scores = scores.astype(np.float64, copy=False)
    if not np.isfinite(scores).all() or np.any(scores < 0):
        raise ValueError("Visual head scores must be finite and nonnegative")
    # Upstream uses Python's sequential sum over NumPy float64 head means.
    total = sum(scores.reshape(-1))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Visual head scores need a finite positive global sum")
    return scores, total


def load_head_scores(path=DEFAULT_SCORE_PATH, num_layers=28, num_query_heads=28):
    """Load strict query-head means from CSV or original per-image JSON lists."""
    num_layers = _integer(num_layers, "num_layers", 1)
    num_query_heads = _integer(num_query_heads, "num_query_heads", 1)
    path = Path(path or DEFAULT_SCORE_PATH).expanduser().resolve()
    if path == DEFAULT_SCORE_PATH.resolve():
        if (num_layers, num_query_heads) != (28, 28):
            raise ValueError("Default Qwen2.5 visual scores require 28 layers and 28 query heads")
        if hashlib.sha256(path.read_bytes()).hexdigest() != DEFAULT_SCORE_SHA256:
            raise ValueError("Default Qwen2.5 visual-score SHA-256 changed")
    scores = np.full((num_layers, num_query_heads), np.nan, dtype=np.float64)
    seen = set()

    def assign(layer, head, value):
        if not 0 <= layer < num_layers or not 0 <= head < num_query_heads:
            raise ValueError("Visual-score layer/query-head index is outside the configured shape")
        if (layer, head) in seen:
            raise ValueError("Duplicate visual-score layer/query-head entry")
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
            raise ValueError("Visual head means must be finite nonnegative numbers")
        seen.add((layer, head))
        scores[layer, head] = value

    if path.suffix.lower() == ".json":
        def unique_object(pairs):
            result = {}
            for name, value in pairs:
                if name in result:
                    raise ValueError("Duplicate visual-score JSON key")
                result[name] = value
            return result
        records = json.loads(path.read_text(), object_pairs_hook=unique_object)
        if not isinstance(records, dict):
            raise ValueError("Visual-score JSON must map layer-head to per-image score lists")
        lengths = set()
        for name, samples in records.items():
            if not isinstance(name, str) or re.fullmatch(r"\d+-\d+", name) is None:
                raise ValueError("Visual-score JSON keys must be layer-head")
            if not isinstance(samples, list) or not samples:
                raise ValueError("Each visual head requires a nonempty per-image score list")
            if any(isinstance(value, bool) or not isinstance(value, Real) for value in samples):
                raise ValueError("Per-image visual scores must be real numbers")
            sample_array = np.asarray(samples, dtype=np.float64)
            if not np.isfinite(sample_array).all() or np.any(sample_array < 0):
                raise ValueError("Per-image visual scores must be finite and nonnegative")
            lengths.add(len(samples))
            layer, head = map(int, name.split("-"))
            assign(layer, head, np.mean(sample_array))
        if len(lengths) > 1:
            raise ValueError("Visual heads were calibrated on different sample counts")
    elif path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            allowed = {"layer", "head", "score_mean", "score_sum", "nonzero", "max"}
            if (fields is None or len(fields) != len(set(fields))
                    or not {"layer", "head", "score_mean"} <= set(fields) or not set(fields) <= allowed):
                raise ValueError("Visual-score CSV requires layer,head,score_mean columns")
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("Malformed visual-score CSV row")
                layer, head = _index(row["layer"], "layer"), _index(row["head"], "head")
                try:
                    mean = np.float64(row["score_mean"])
                except (ValueError, TypeError) as error:
                    raise ValueError("Invalid visual head mean") from error
                assign(layer, head, mean)
    else:
        raise ValueError("Visual scores must be a CSV or JSON asset")
    if len(seen) != num_layers * num_query_heads:
        raise ValueError("Visual-score asset is missing layer/query-head entries")
    scores, _ = _scores_array(scores)
    return torch.from_numpy(scores.copy())


def allocate_head_budgets(scores, budget=1000, window_size=32, ratio=DEFAULT_RATIO, num_kv_heads=4):
    """Return original static visual capacities, preserving upstream rounding."""
    budget = _integer(budget, "budget", 1)
    window_size = _integer(window_size, "window_size", 1)
    num_kv_heads = _integer(num_kv_heads, "num_kv_heads", 1)
    if budget <= window_size:
        raise ValueError("SparseMM requires budget > window_size")
    if isinstance(ratio, bool) or not isinstance(ratio, Real) or not math.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError("SparseMM ratio must be finite and in [0,1]")
    scores, total = _scores_array(scores)
    layers, query_heads = scores.shape
    if query_heads % num_kv_heads:
        raise ValueError("Query heads must be divisible by physical KV heads")
    groups = query_heads // num_kv_heads
    normalized = torch.tensor(scores.reshape(-1) / total, dtype=torch.float64)
    grouped = normalized.view(layers, num_kv_heads, groups).sum(dim=-1)
    historical = budget - window_size
    floor = int(historical * ratio)
    remaining = (historical - floor) * layers * query_heads // groups
    raw = grouped * remaining + floor
    if bool((raw > torch.iinfo(torch.int32).max).any()):
        raise ValueError("SparseMM quotas exceed the upstream int32 capacity range")
    # The source constructs a float64 tensor from NumPy means, then rounds and
    # casts to int32. Return long for native indexing without changing decisions.
    history_budgets = torch.round(raw).int()
    capacities = history_budgets.long() + window_size
    nominal = layers * num_kv_heads * budget
    metadata = {
        "method": "sparsemm", "selector": "static_visual_head_scores_then_attention",
        "shape": [layers, num_kv_heads], "num_query_heads": query_heads,
        "gqa_groups": groups, "score_grouping": "sum_after_global_query_head_normalization",
        "score_dtype": "float64", "ratio": float(ratio), "budget": budget,
        "window_size": window_size, "history_base_capacity": historical,
        "minimum_history_quota": floor, "remaining_history_slots": remaining,
        "head_budgets": capacities.tolist(), "history_budgets": history_budgets.tolist(),
        "target_slots": nominal, "allocated_slots": int(capacities.sum()),
        "rounding_drift_tokens": int(capacities.sum()) - nominal,
        "rounding": "upstream_float64_round_to_even", "exact_total_correction": False,
        "uniform_blend": False, "capacity_clip": False, "upstream_commit": UPSTREAM_COMMIT,
    }
    return capacities, metadata


def load_fixed_head_budgets(path, num_layers, num_kv_heads, budget, window_size=32):
    """Load an exact external per-KV-head capacity table for an allocation ablation.

    DP recent/history splits are checked when present, but SparseMM's online
    selector continues to protect its own fixed recent window.
    """
    num_layers = _integer(num_layers, "num_layers", 1)
    num_kv_heads = _integer(num_kv_heads, "num_kv_heads", 1)
    budget = _integer(budget, "budget", 1)
    window_size = _integer(window_size, "window_size", 1)
    path = Path(path).expanduser().resolve()
    table = [[None] * num_kv_heads for _ in range(num_layers)]
    splits = [[None] * num_kv_heads for _ in range(num_layers)]
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if (len(fields) != len(set(fields)) or not {"layer", "kv_head", "capacity_n"} <= set(fields)
                or ("recent_r" in fields) != ("history_q" in fields)):
            raise ValueError("External budget CSV requires layer,kv_head,capacity_n and paired recent_r/history_q")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Malformed external budget CSV row")
            layer = _index(row["layer"], "layer")
            head = _index(row["kv_head"], "kv_head")
            capacity = _index(row["capacity_n"], "capacity_n")
            if not 0 <= layer < num_layers or not 0 <= head < num_kv_heads:
                raise ValueError("External budget head index is outside model shape")
            if table[layer][head] is not None:
                raise ValueError("Duplicate external budget head")
            if capacity < window_size:
                raise ValueError("External budget cannot protect SparseMM's recent window")
            table[layer][head] = capacity
            if "recent_r" in fields:
                recent = _index(row["recent_r"], "recent_r")
                history = _index(row["history_q"], "history_q")
                if recent + history != capacity:
                    raise ValueError("External DP bank split disagrees with capacity_n")
                splits[layer][head] = [recent, history]
    if any(value is None for row in table for value in row):
        raise ValueError("External budget CSV is missing model KV heads")
    allocated = sum(map(sum, table))
    target = num_layers * num_kv_heads * budget
    if allocated != target:
        raise ValueError(f"External budget total {allocated} differs from {target}")
    metadata = {"method": "dp_budget_sparsemm_selector", "selector": "sparsemm_attention",
                "budget_source": "external_capacity_n", "budget_path": str(path),
                "budget_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "shape": [num_layers, num_kv_heads], "budget": budget,
                "window_size": window_size, "head_budgets": table,
                "history_budgets": [[n - window_size for n in row] for row in table],
                "target_slots": target, "allocated_slots": allocated,
                "dp_bank_split_present": "recent_r" in fields,
                "dp_bank_split_used_online": False,
                "recent_below_sparsemm_window_heads": sum(
                    split is not None and split[0] < window_size for row in splits for split in row)}
    return torch.tensor(table, dtype=torch.long), metadata


@torch.inference_mode()
def select_layer(keys, queries, budgets, prefix_tokens=0, *, budget=1000,
                 window_size=32, kernel_size=5, pooling="avgpool", gqa_reduce="mean", layer_idx=0,
                 external_budget=False):
    """Original SparseMM within-head selection, with explicit visual protection.

    The global base-history threshold controls short-cache bypass, independently
    of individual static head quotas. Once active, history is sorted descending
    with torch.sort and each head retains its own quota plus the recent window.
    Ragged streaming heads clamp unavailable slots; spare capacity is not moved.
    """
    budget = _integer(budget, "budget", 1)
    window_size = _integer(window_size, "window_size", 1)
    kernel_size = _integer(kernel_size, "kernel_size", 1)
    prefix_tokens = _integer(prefix_tokens, "prefix_tokens")
    layer_idx = _integer(layer_idx, "layer_idx")
    if not isinstance(keys, (list, tuple)) or not keys or not all(isinstance(key, torch.Tensor) for key in keys):
        raise ValueError("keys must be a nonempty list of per-head tensors")
    if not isinstance(queries, torch.Tensor):
        raise ValueError("queries must be a tensor [Hq,Q,D]")
    if hasattr(budgets, "tolist"):
        budgets = budgets.tolist()
    if not isinstance(budgets, (list, tuple)) or len(budgets) != len(keys):
        raise ValueError("One static integer visual budget is required per KV head")
    budgets = [_integer(value, "head budget", window_size) for value in budgets]
    config = BaselineConfig(method="adakv", budget=budget, window_size=window_size,
        kernel_size=kernel_size, pooling=pooling, gqa_reduce=gqa_reduce,
        budget_scope="visual", score_precision="float32")
    groups, query_len, _ = _validate_inputs(keys, queries, config, 0, 1, prefix_tokens, None)
    lengths = [key.shape[0] for key in keys]
    recent = [min(window_size, length - prefix_tokens) for length in lengths]
    histories = [length - prefix_tokens - r for length, r in zip(lengths, recent)]
    base_history = budget - window_size
    bypass = all(length < base_history for length in histories)
    selected, retained = [], []
    for head, (key, available, r, capacity) in enumerate(zip(keys, histories, recent, budgets)):
        if bypass:
            keep = torch.arange(key.shape[0], device=key.device, dtype=torch.long)
            historical_count = available
        else:
            score = _head_scores(key, queries[head * groups:(head + 1) * groups],
                                 prefix_tokens, r, config, layer_idx, head)
            historical_count = min(capacity - window_size, score.numel())
            ranking = torch.sort(score, descending=True).indices[:historical_count]
            keep = torch.cat((torch.arange(prefix_tokens, device=key.device), ranking + prefix_tokens,
                              torch.arange(key.shape[0] - r, key.shape[0], device=key.device)))
        selected.append(keep)
        retained.append(historical_count)
    after = [keep.numel() for keep in selected]
    nominal = len(keys) * (budget + prefix_tokens)
    metadata = {
        "method": "sparsemm", "selector": ("external_capacity_then_attention" if external_budget
                                            else "static_visual_head_scores_then_attention"),
        "budget_scope": "visual", "configured_budget": budget, "layer_idx": layer_idx,
        "base_history_capacity": base_history, "head_budgets": budgets,
        "prefix_tokens_kept": prefix_tokens, "window_size": window_size,
        "before_head_lengths": lengths, "head_lengths": after,
        "recent_counts": recent, "proposed_history_counts": [value - window_size for value in budgets],
        "retained_history_counts": retained, "retained_visual_counts": [n - prefix_tokens for n in after],
        "observed_queries": min(query_len, window_size), "gqa_groups": groups,
        "gqa_reduce": gqa_reduce, "score_precision": "float32", "pooling": pooling, "kernel_size": kernel_size,
        "target_total_tokens": sum(budgets) + len(keys) * prefix_tokens,
        "nominal_total_tokens": nominal, "actual_total_tokens": sum(after),
        "allocation_drift_tokens": sum(after) - nominal,
        "calibrated_drift_tokens": sum(budgets) - len(keys) * budget,
        "bypassed_short_cache": bypass, "compressed": any(a < b for a, b in zip(after, lengths)),
        "rounding": "external_exact_capacity" if external_budget else "upstream_float64_round_to_even",
        "tie_policy": "upstream_torch_sort",
        "upstream_commit": UPSTREAM_COMMIT,
    }
    return selected, metadata
