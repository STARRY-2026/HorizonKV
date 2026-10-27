"""Fixed LU head capacities with SparseMM direct-attention token ranking.

Online scores use the current visual queries and cached keys, with no proxy
questions or recency score mixing. The existing balanced capacity table was
calibrated with HERMES ranking: using it here transfers budgets to a different
selector, rather than recalibrating LU or reproducing the old LU accuracy.
Legacy HERMES helpers below remain solely for historical reference tests.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path
import re

import torch

from sparsemm_baselines.core import BaselineConfig, _head_scores, _validate_inputs


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_BUDGET_PATH = (
    WORKSPACE / "hermes/results/myobservations/lu_kv_balanced100v_20260915/traces"
    / "lu_raw_balanced100v_blend05_clip075_150_budget.csv"
)
DEFAULT_BUDGET_SHA256 = "cd3d6e0f42d0c35ffa0b4aa84635b9c35443ebb47bac6b8b6ca89bad4d7a65fe"


def _integer(value, name, minimum=0):
    if not isinstance(value, Integral) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _csv_integer(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[+-]?\d+", value.strip()) is None:
        raise ValueError(f"{name} must contain an integer, not {value!r}")
    return int(value)


def load_budget_table(path=DEFAULT_BUDGET_PATH, num_layers=28, num_kv_heads=4,
                      budget=1000, window=32):
    """Load a complete CPU int64 capacity table, without any redistribution.

    The constrained Raw-LU contract is .75B <= capacity <= 1.5B, capacity >= W,
    and sum(capacity) == L*Hkv*B. Text tokens are separate from these visual
    capacities. The default balanced calibration asset is pinned by SHA-256.
    """
    num_layers = _integer(num_layers, "num_layers", 1)
    num_kv_heads = _integer(num_kv_heads, "num_kv_heads", 1)
    budget = _integer(budget, "budget", 1)
    window = _integer(window, "window")
    low, high = max(window, math.ceil(.75 * budget)), math.floor(1.5 * budget)
    if low > high:
        raise ValueError("Recent-window protection is incompatible with the capacity bounds")
    path = Path(path or DEFAULT_BUDGET_PATH).expanduser().resolve()
    if path == DEFAULT_BUDGET_PATH.resolve():
        if (num_layers, num_kv_heads, budget, window) != (28, 4, 1000, 32):
            raise ValueError("Default balanced LU asset requires 28 layers, 4 KV heads, B=1000, W=32")
        if hashlib.sha256(path.read_bytes()).hexdigest() != DEFAULT_BUDGET_SHA256:
            raise ValueError("Default balanced LU calibration asset SHA-256 changed")
    table = [[None] * num_kv_heads for _ in range(num_layers)]
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or len(reader.fieldnames) != 3 or set(reader.fieldnames) != {"layer", "kv_head", "budget"}:
            raise ValueError("LU capacity CSV must contain exactly layer,kv_head,budget columns")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Malformed LU capacity CSV row")
            layer = _csv_integer(row["layer"], "layer")
            head = _csv_integer(row["kv_head"], "kv_head")
            capacity = _csv_integer(row["budget"], "budget")
            if not 0 <= layer < num_layers or not 0 <= head < num_kv_heads:
                raise ValueError(f"LU capacity index ({layer},{head}) is outside [{num_layers},{num_kv_heads}]")
            if table[layer][head] is not None:
                raise ValueError(f"Duplicate LU capacity for layer {layer}, KV head {head}")
            if not low <= capacity <= high:
                raise ValueError(f"LU capacity {capacity} is outside [{low},{high}]")
            table[layer][head] = capacity
    if any(value is None for row in table for value in row):
        raise ValueError("LU capacity CSV is missing one or more layer/KV-head entries")
    if sum(sum(row) for row in table) != num_layers * num_kv_heads * budget:
        raise ValueError("LU capacities do not sum to the exact global L*Hkv*B budget")
    return torch.tensor(table, dtype=torch.long)


def budget_asset_metadata(path=DEFAULT_BUDGET_PATH, num_layers=28, num_kv_heads=4,
                          budget=1000, window=32):
    """Validate capacities and report their calibration provenance for a run.

    Custom CSVs have no implied calibration provenance. For the default table,
    both export/calibration manifests and their declared annotation sources are
    hashed; original source hashes and calibration/evaluation separation are
    checked before the default asset is accepted.
    """
    path = Path(path or DEFAULT_BUDGET_PATH).expanduser().resolve()
    table = load_budget_table(path, num_layers, num_kv_heads, budget, window)
    result = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "shape": list(table.shape), "sum": int(table.sum()),
              "minimum": int(table.min()), "maximum": int(table.max()),
              "calibration_manifests": {}, "calibration_sources": {}, "calibration_summary": {},
              "runtime_selector": "sparsemm_attention", "recalibrated_for_runtime_selector": None,
              "prior_accuracy_reused": False}
    if path != DEFAULT_BUDGET_PATH.resolve():
        return result
    result.update(usage="budget_transfer", calibration_selector="hermes_deployed",
                  recalibrated_for_runtime_selector=False)
    deployment_path = path.with_name("lu_raw_balanced100v_blend05_clip075_150_manifest.json")
    calibration_path = path.with_name("manifest.json")
    deployment = json.loads(deployment_path.read_text())
    calibration = json.loads(calibration_path.read_text())
    if (deployment.get("K") != budget or deployment.get("budget_sum") != result["sum"]
            or deployment.get("budget_range") != [result["minimum"], result["maximum"]]
            or deployment.get("blend_with_uniform") != .5 or deployment.get("bounds") != [750, 1500]
            or deployment.get("num_videos") != 100 or deployment.get("num_observations") != 150
            or calibration.get("evaluation_overlap") != 0):
        raise ValueError("Default LU calibration/export manifest does not match the pinned capacity table")
    hermes = WORKSPACE / "hermes"
    for manifest_path in (deployment_path, calibration_path):
        result["calibration_manifests"][str(manifest_path)] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    for name, expected in calibration["source_sha256"].items():
        source = Path(name)
        if not source.is_absolute():
            source = hermes / source
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Default LU calibration annotation source changed: {source}")
        result["calibration_sources"][str(source.resolve())] = actual
    # ID exclusion is checked independently of the stored overlap declaration.
    config = calibration["config"]
    excluded_path = Path(config["exclude_annotation"])
    if not excluded_path.is_absolute():
        excluded_path = hermes / excluded_path
    excluded = json.loads(excluded_path.read_text())
    selected_ids = {str(video["video_id"]) for video in calibration["videos"]}
    excluded_ids = {str(video["video_id"]) for video in excluded}
    if selected_ids & excluded_ids:
        raise ValueError("Default LU calibration overlaps the evaluation annotation IDs")
    result["calibration_summary"] = {
        "name": deployment["name"], "num_videos": deployment["num_videos"],
        "num_observations": deployment["num_observations"],
        "blend_with_uniform": deployment["blend_with_uniform"], "bounds": deployment["bounds"],
        "evaluation_overlap": 0, "scope": calibration["scope"],
        "utility": calibration["utility"], "adaptations": calibration["adaptations"],
        "ranking_selector": "hermes_deployed", "runtime_selector": "sparsemm_attention",
        "usage": "budget_transfer", "recalibrated_for_runtime_selector": False,
        "prior_accuracy_reused": False,
    }
    return result



DEFAULT_PROFILE_PATH = WORKSPACE / "sparsemm_lukv/assets/attention_native10v_20261014/profile.json"


def load_budget_profile(path=DEFAULT_PROFILE_PATH, num_layers=28, num_kv_heads=4, window=32):
    """Load a scorer-specific profile; no fallback to the legacy CSV is allowed."""
    from sparsemm_lukv.profile import validate_profile
    path = Path(path or DEFAULT_PROFILE_PATH).expanduser().resolve()
    profile = json.loads(path.read_text())
    validate_profile(profile, num_layers=num_layers, num_kv_heads=num_kv_heads, window_size=window)
    return profile


def profile_asset_metadata(path=DEFAULT_PROFILE_PATH, num_layers=28, num_kv_heads=4,
                           budget=1000, window=32):
    path = Path(path or DEFAULT_PROFILE_PATH).expanduser().resolve()
    profile = load_budget_profile(path, num_layers, num_kv_heads, window)
    calibration = profile.get("calibration", {})
    if calibration.get("selector") != "sparsemm_attention":
        raise ValueError("LU profile must be calibrated with the deployed visual-attention scorer")
    if calibration.get("utility_mode") not in ("projected", "attention"):
        raise ValueError("LU profile must declare its long-horizon utility target")
    protocol = calibration.get("token_window_protocol", {})
    calibration_model = Path(calibration.get("model_path", "")) / "config.json"
    model_kind = json.loads(calibration_model.read_text())["model_type"]
    native_resolution = model_kind != "llava_onevision"
    required = {"window_size": window, "kernel_size": 5, "pooling": "avgpool",
                "gqa_reduce": "mean", "score_precision": "float32",
                "native_resolution": native_resolution}
    if any(protocol.get(key) != value for key, value in required.items()):
        raise ValueError("LU calibration protocol differs from native online attention scoring")
    sources = calibration.get("source_sha256", {})
    if not sources:
        raise ValueError("LU profile has no bound calibration annotation sources")
    for name, expected in sources.items():
        source = Path(name)
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f"LU calibration source changed: {source}")
    selected = calibration.get("selected_videos", [])
    excluded_path = Path(calibration["exclude_annotation"])
    excluded = json.loads(excluded_path.read_text())
    excluded_ids = {str(video["video_id"]) for video in excluded}
    excluded_names = {Path(video["video_path"]).name for video in excluded}
    if (not selected or len({str(video["video_id"]) for video in selected}) != len(selected)
            or any(str(video["video_id"]) in excluded_ids or Path(video["video_path"]).name in excluded_names
                   for video in selected)):
        raise ValueError("LU calibration videos overlap evaluation or have duplicate IDs")
    selected_hash = hashlib.sha256(json.dumps(selected, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if selected_hash != calibration.get("selected_sha256"):
        raise ValueError("LU calibration selected video fingerprint changed")
    evidence_path = Path(calibration["manifest_path"])
    if hashlib.sha256(evidence_path.read_bytes()).hexdigest() != calibration.get("manifest_sha256"):
        raise ValueError("LU calibration evidence manifest changed")
    evidence = json.loads(evidence_path.read_text())
    if evidence.get("status") != "complete" or evidence.get("selected_sha256") != selected_hash:
        raise ValueError("LU calibration evidence is incomplete or belongs to another subset")
    observations = calibration.get("observations", [])
    if not observations or observations != evidence.get("observations"):
        raise ValueError("LU calibration observation evidence is missing or inconsistent")
    for observation in observations:
        observation_path = Path(observation["path"])
        if hashlib.sha256(observation_path.read_bytes()).hexdigest() != observation["sha256"]:
            raise ValueError(f"LU calibration observation changed: {observation_path}")
    summary = {"num_videos": len(selected), "num_observations": len(observations),
               "selector": "sparsemm_attention", "utility_mode": calibration["utility_mode"],
               "native_resolution": native_resolution, "evaluation_overlap": 0,
               "usage": "attention_calibrated_profile", "recalibrated_for_runtime_selector": True,
               "prior_accuracy_reused": False, "uniform_blend": 0, "legacy_capacity_clip": False,
               "budget_mode": "compression_ratio_lookup_history_with_protected_window"}
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "shape": [num_layers, num_kv_heads], "sum": None,
            "global_visual_capacity": num_layers * num_kv_heads * budget,
            "asset_type": "profile", "calibration_manifests": {str(evidence_path): calibration["manifest_sha256"]},
            "calibration_sources": sources, "calibration_summary": summary,
            "runtime_selector": "sparsemm_attention", "usage": "attention_calibrated_profile",
            "recalibrated_for_runtime_selector": True, "prior_accuracy_reused": False}


def configured_asset_metadata(config, num_layers=28, num_kv_heads=4, model_path=None):
    if config.lukv_budget_path:
        return budget_asset_metadata(config.lukv_budget_path, num_layers, num_kv_heads,
                                     config.budget, config.window_size)
    path = config.lukv_profile_path or DEFAULT_PROFILE_PATH
    metadata = profile_asset_metadata(path, num_layers, num_kv_heads, config.budget, config.window_size)
    if model_path is not None:
        calibration = load_budget_profile(path, num_layers, num_kv_heads, config.window_size)["calibration"]
        for name, expected in calibration["model_files_sha256"].items():
            if hashlib.sha256((Path(model_path) / name).read_bytes()).hexdigest() != expected:
                raise ValueError("LU calibration model/tokenizer differs from the runtime model")
    return metadata


@torch.inference_mode()
def attention_select_layer(keys, queries, budgets, prefix_tokens, window_size=32,
                           pooling="avgpool", kernel_size=5, gqa_reduce="mean"):
    """Select one layer using fixed LU capacities and SparseMM attention.

    ``keys`` contains ragged ``[N_h,D]`` heads, including protected prefix text
    and the just-appended visual chunk. ``queries`` is that chunk's rotated
    ``[Hq,Q,D]`` tensor. Only its final min(W,Q) positions score history, using
    FP32 causal softmax, mean query pooling, GQA reduction, and token pooling
    exactly as in :func:`sparsemm_baselines.core._head_scores`.

    Each integer budget counts visual tokens, including the latest W. Prefix
    text is additional. Short heads retain all tokens and spare capacity stays
    unused. Compressed heads follow SparseMM's order: prefix, history in score
    order (including Torch's tie ordering), then the recent causal window.
    Neither AdaKV allocation nor the PyramidKV layer schedule is applied.
    """
    prefix_tokens = _integer(prefix_tokens, "prefix_tokens")
    window_size = _integer(window_size, "window_size", 1)
    kernel_size = _integer(kernel_size, "kernel_size", 1)
    if not isinstance(keys, (list, tuple)) or not keys:
        raise ValueError("keys must be a nonempty list of per-head tensors")
    if not isinstance(queries, torch.Tensor):
        raise ValueError("queries must be a tensor [Hq,Q,D]")
    if not all(isinstance(key, torch.Tensor) for key in keys):
        raise ValueError("Each key must be a tensor [N_h,D]")
    if hasattr(budgets, "tolist"):
        budgets = budgets.tolist()
    if not isinstance(budgets, (list, tuple)) or len(budgets) != len(keys):
        raise ValueError("budgets must contain exactly one integer per KV head")
    budgets = [_integer(capacity, "head budget", 1) for capacity in budgets]
    if min(budgets) < window_size:
        raise ValueError("LU visual capacities must cover recent-window protection")
    # This config is used only by input validation and the shared score kernel.
    # Its scalar budget and method never determine any LU capacity or schedule.
    config = BaselineConfig(
        method="adakv", budget=max(max(budgets), window_size + 1),
        window_size=window_size, kernel_size=kernel_size, pooling=pooling,
        gqa_reduce=gqa_reduce, budget_scope="visual", score_precision="float32",
    )
    groups, query_len, _ = _validate_inputs(
        keys, queries, config, 0, 1, prefix_tokens, None,
    )
    lengths = [key.shape[0] for key in keys]
    selected, recents, history_counts, after = [], [], [], []
    for head, (key, capacity) in enumerate(zip(keys, budgets)):
        length = key.shape[0]
        visual = length - prefix_tokens
        recent = min(window_size, visual)
        recents.append(recent)
        if visual <= capacity:
            keep = torch.arange(length, device=key.device, dtype=torch.long)
            history_counts.append(visual - recent)
        else:
            score = _head_scores(
                key, queries[head * groups:(head + 1) * groups],
                prefix_tokens, recent, config, 0, head,
            )
            history_count = min(capacity - recent, score.numel())
            # SparseMM AdaKV's within-head ordering is torch.sort; only its
            # cross-head allocation is replaced by the supplied fixed budget.
            historical = torch.sort(score, descending=True).indices[:history_count]
            keep = torch.cat((
                torch.arange(prefix_tokens, device=key.device, dtype=torch.long),
                historical + prefix_tokens,
                torch.arange(length - recent, length, device=key.device, dtype=torch.long),
            ))
            history_counts.append(history_count)
        selected.append(keep)
        after.append(keep.numel())
    retained_visual = [length - prefix_tokens for length in after]
    metadata = {
        "method": "lukv", "selector": "sparsemm_attention", "budget_scope": "visual",
        "head_budgets": budgets, "before_head_lengths": lengths, "head_lengths": after,
        "prefix_tokens": prefix_tokens, "prefix_tokens_kept": prefix_tokens,
        "window_size": window_size, "observed_queries": min(window_size, query_len),
        "gqa_groups": groups, "gqa_reduce": gqa_reduce, "pooling": pooling,
        "kernel_size": kernel_size, "score_precision": "float32",
        "recent_counts": recents, "retained_history_counts": history_counts,
        "retained_visual_counts": retained_visual,
        "total_visual_capacity": sum(budgets), "actual_visual_slots": sum(retained_visual),
        "target_total_tokens": sum(budgets) + len(keys) * prefix_tokens,
        "actual_total_tokens": sum(after),
        "compressed": any(new < old for old, new in zip(lengths, after)),
        "indices_order": "prefix_score_descending_recent",
    }
    return selected, metadata


# Legacy HERMES selector helpers. The native LU plugin uses only
# attention_select_layer; these are retained for old source-parity tests.
def layer_policy(layer_idx, num_layers, short_term_ratio=.1, long_term_ratio=.3):
    """Return (probe name, recency alpha, recency decay) for a decoder layer.

    Defaults follow the deployed loader, which overrides the QwenVL_Hermes
    constructor's initial short-term ratio to .1 (28 layers: thresholds 2/19).
    """
    num_layers = _integer(num_layers, "num_layers", 1)
    layer_idx = _integer(layer_idx, "layer_idx")
    if layer_idx >= num_layers:
        raise ValueError("layer_idx is outside num_layers")
    for name, value in (("short_term_ratio", short_term_ratio), ("long_term_ratio", long_term_ratio)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be finite and in [0,1]")
    if short_term_ratio + long_term_ratio > 1:
        raise ValueError("Short and long layer regions must not overlap")
    short = int(num_layers * short_term_ratio)
    long = int(num_layers * (1 - long_term_ratio))
    if layer_idx < short:
        return "local", 1.0, 20.0
    if layer_idx >= long:
        return "global", 0.0, 0.0
    progress = (layer_idx - short) / (long - short)
    return "mixed", .75 - .6 * progress, 20.0 - 12.0 * progress


def combined_score(score, length, prefix_tokens, recency_alpha, k_decay):
    """Return visual-token scores with the original normalization arithmetic.

    Input scores include text positions. Short score vectors are zero-padded as
    in HERMES; excess positions are discarded. Computation preserves the input
    score dtype, while recency uses float32, matching the original selector.
    """
    length = _integer(length, "length")
    prefix_tokens = _integer(prefix_tokens, "prefix_tokens")
    if not isinstance(score, torch.Tensor) or score.ndim != 1 or not score.is_floating_point():
        raise ValueError("A probe score must be a one-dimensional floating tensor")
    if not math.isfinite(recency_alpha) or not 0 <= recency_alpha <= 1 or not math.isfinite(k_decay) or k_decay < 0:
        raise ValueError("Invalid recency mixing parameters")
    if score.numel() < length:
        score = torch.cat((score, score.new_zeros(length - score.numel())))
    score = score[:length]
    if not bool(torch.isfinite(score).all()):
        raise FloatingPointError("Non-finite HERMES probe scores")
    text = min(prefix_tokens, length)
    visual = score[text:]
    if visual.numel() == 0:
        return torch.empty(0, dtype=score.dtype, device=score.device)
    positions = torch.arange(visual.numel(), device=score.device, dtype=torch.float32)
    distance = (visual.numel() - 1 - positions) / max(visual.numel() - 1, 1)
    recency = torch.exp(-k_decay * distance)
    recency = (recency - recency.min()) / (recency.max() - recency.min() + 1e-6)
    attention = (visual - visual.min()) / (visual.max() - visual.min() + 1e-6)
    return attention * (1.0 - recency_alpha) + recency * recency_alpha


def _matrix(values, name):
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{name} must be a nonempty layer/head matrix")
    matrix = []
    heads = None
    for row in values:
        if not isinstance(row, (list, tuple)) or not row:
            raise ValueError(f"{name} must be a rectangular layer/head matrix")
        if heads is None:
            heads = len(row)
        if len(row) != heads:
            raise ValueError(f"{name} must be rectangular")
        matrix.append([_integer(value, name) for value in row])
    return matrix


@torch.inference_mode()
def select_keep_indices(local_scores, global_scores, mixed_scores, lengths, budgets,
                        prefix_tokens, window_size=32, short_term_ratio=.1,
                        long_term_ratio=.3):
    """Select chronological per-head indices under fixed visual capacities.

    Each score argument is nested [layer][KV head] of one-dimensional tensors.
    Prefix text is retained separately; the most recent W visual tokens count
    inside the fixed head capacity. Early heads use available tokens only and
    spare capacity is never transferred to another head. Optional HERMES role,
    HE, frame-protection and history-uniform variants are deliberately absent.
    """
    prefix_tokens = _integer(prefix_tokens, "prefix_tokens")
    window_size = _integer(window_size, "window_size")
    lengths, budgets = _matrix(lengths, "lengths"), _matrix(budgets, "budgets")
    layers, heads = len(lengths), len(lengths[0])
    if len(budgets) != layers or len(budgets[0]) != heads:
        raise ValueError("Budget shape differs from layer/head lengths")
    probes = {"local": local_scores, "global": global_scores, "mixed": mixed_scores}
    for name, value in probes.items():
        if len(value) != layers or any(len(row) != heads for row in value):
            raise ValueError(f"{name} scores differ from layer/head lengths")
    selected, after, retained, policies = [], [], [], []
    for layer in range(layers):
        name, alpha, decay = layer_policy(layer, layers, short_term_ratio, long_term_ratio)
        policies.append({"probe": name, "recency_alpha": alpha, "k_decay": decay})
        layer_selected, layer_after, layer_retained = [], [], []
        for head in range(heads):
            capacity = budgets[layer][head]
            if capacity < max(window_size, 1):
                raise ValueError("LU visual capacity must be positive and cover recent-window protection")
            length = lengths[layer][head]
            score = probes[name][layer][head]
            combined = combined_score(score, length, prefix_tokens, alpha, decay)
            text = min(prefix_tokens, length)
            visual = length - text
            effective = min(capacity, visual)
            recent = min(window_size, visual)
            history = torch.arange(visual - recent, device=score.device, dtype=torch.long)
            pieces = [torch.arange(text, device=score.device, dtype=torch.long)]
            history_k = min(max(effective - recent, 0), history.numel())
            if history_k:
                top = torch.topk(combined[history], history_k, sorted=False).indices
                pieces.append(history[top] + text)
            if recent:
                pieces.append(torch.arange(length - recent, length, device=score.device, dtype=torch.long))
            keep = torch.unique(torch.cat(pieces), sorted=True)
            layer_selected.append(keep)
            layer_after.append(keep.numel())
            layer_retained.append(keep.numel() - text)
        selected.append(layer_selected)
        after.append(layer_after)
        retained.append(layer_retained)
    metadata = {
        "method": "lukv", "selector": "hermes_deployed", "budget_scope": "visual",
        "head_budgets": budgets, "before_head_lengths": lengths, "head_lengths": after,
        "retained_visual_counts": retained, "prefix_tokens": prefix_tokens,
        "window_size": window_size, "short_term_ratio": short_term_ratio,
        "long_term_ratio": long_term_ratio, "short_term_threshold": int(layers * short_term_ratio),
        "long_term_threshold": int(layers * (1 - long_term_ratio)), "layer_policies": policies,
        "total_visual_capacity": sum(sum(row) for row in budgets),
        "actual_visual_slots": sum(sum(row) for row in retained),
        "compressed": any(new < old for old_row, new_row in zip(lengths, after)
                          for old, new in zip(old_row, new_row)),
    }
    return selected, metadata
