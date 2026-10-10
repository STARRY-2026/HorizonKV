"""Question-conditioned answer-time attention-head output gating."""

import csv
import math
from pathlib import Path

import numpy as np
import torch


MEMORY_TERMS = (
    "at the beginning", "at the start", "previously", "earlier", "before ",
    " after ", "so far", "in total", "why ", "what happened", "just now",
    "did ", "was ", "were ", "might ", "next", "sequence", "summar",
    "according to", "first shown", "initially",
)
CURRENT_TERMS = (
    "right now", "currently", " now", "wearing", "holding", "visible",
    "what color", "where is", "where are", "doing right now", "shown right now",
)


def _memory_demand(question):
    text = " " + str(question).lower().strip() + " "
    memory = sum(term in text for term in MEMORY_TERMS)
    current = sum(term in text for term in CURRENT_TERMS)
    if memory == current:
        return 0.0
    return float(math.tanh(0.8 * (memory - current)))


def _layer_center(values):
    centered = np.zeros_like(values)
    for layer in range(values.shape[0]):
        row = values[layer]
        scale = float(row.std())
        if scale > 1e-12:
            centered[layer] = np.tanh((row - row.mean()) / scale)
    return centered


def _query_head_quantile_roles(values, quantile=0.2):
    """Map low current/previous density to memory and high density to current."""
    roles = np.zeros_like(values)
    for layer in range(values.shape[0]):
        row = values[layer]
        finite = np.isfinite(row)
        if not finite.any():
            continue
        low = float(np.quantile(row[finite], quantile))
        high = float(np.quantile(row[finite], 1.0 - quantile))
        roles[layer, finite & (row <= low)] = 1.0
        roles[layer, finite & (row >= high)] = -1.0
    return roles


def _load_head_roles(profile_path, num_layers, num_query_heads, num_kv_heads):
    with Path(profile_path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        rows = list(reader)

    if {"layer", "head", "b_log_per_token_ratio"} <= fields:
        values = np.full((num_layers, num_query_heads), np.nan, dtype=np.float64)
        covered = np.zeros((num_layers, num_query_heads), dtype=bool)
        for row in rows:
            layer, head = int(row["layer"]), int(row["head"])
            if 0 <= layer < num_layers and 0 <= head < num_query_heads:
                covered[layer, head] = True
                raw_value = row["b_log_per_token_ratio"].strip()
                if raw_value:
                    values[layer, head] = float(raw_value)
        if not covered.all():
            raise ValueError("Query-head profile does not cover every query head")
        return _query_head_quantile_roles(values), "query_head_b_topbottom20"

    required = {"layer", "kv_head", "far_share", "recent_share"}
    missing = required - fields
    if missing:
        raise ValueError(f"Answer-head profile missing columns: {sorted(missing)}")
    values = np.full((num_layers, num_kv_heads), np.nan, dtype=np.float64)
    for row in rows:
        layer, head = int(row["layer"]), int(row["kv_head"])
        if 0 <= layer < num_layers and 0 <= head < num_kv_heads:
            values[layer, head] = float(row["far_share"]) - float(row["recent_share"])
    if not np.isfinite(values).all():
        raise ValueError("Answer-head profile does not cover every physical KV head")
    group_size = num_query_heads // num_kv_heads
    return np.repeat(_layer_center(values), group_size, axis=1), "kv_head_far_recent_broadcast"


def install_answer_head_router(model, profile_path, strength=0.2,
                               min_gate=0.8, max_gate=1.2,
                               start_layer=8, mode="question"):
    language_model = model.language_model
    config = getattr(language_model, "config", None)
    if config is None and hasattr(language_model, "model"):
        config = language_model.model.config
    if config is None:
        raise AttributeError("Cannot locate the language-model attention configuration")
    num_query_heads = int(config.num_attention_heads)
    num_kv_heads = int(config.num_key_value_heads)
    query_roles, profile_kind = _load_head_roles(
        profile_path, model.num_layers, num_query_heads, num_kv_heads
    )
    query_roles[:max(0, int(start_layer))] = 0.0
    model._answer_head_router_config = {
        "profile_path": str(profile_path), "strength": float(strength),
        "min_gate": float(min_gate), "max_gate": float(max_gate),
        "start_layer": int(start_layer), "mode": str(mode),
        "query_roles": query_roles, "profile_kind": profile_kind,
    }
    model._answer_head_gates = None
    model._answer_head_router_last_stats = None
    print(
        "[AnswerHeadRouter] Installed: "
        f"mode={mode}, strength={strength:g}, bounds=[{min_gate:g},{max_gate:g}], "
        f"start_layer={start_layer}, profile_kind={profile_kind}, "
        f"profile={profile_path}"
    )
    return model


def activate_answer_head_router(model, question):
    config = getattr(model, "_answer_head_router_config", None)
    if not config:
        return None
    demand = _memory_demand(question)
    if config["mode"] == "inverted":
        demand = -demand
    elif config["mode"] == "static_memory":
        demand = 1.0
    elif config["mode"] not in ("question", "baseline"):
        raise ValueError(f"Unknown answer-head router mode: {config['mode']}")
    if config["mode"] == "baseline":
        demand = 0.0

    raw = np.exp(float(config["strength"]) * demand * config["query_roles"])
    raw /= np.maximum(raw.mean(axis=1, keepdims=True), 1e-12)
    raw = np.clip(raw, float(config["min_gate"]), float(config["max_gate"]))
    raw /= np.maximum(raw.mean(axis=1, keepdims=True), 1e-12)
    gates = torch.as_tensor(raw, device=model.device, dtype=torch.float32)
    model._answer_head_gates = gates
    model._answer_head_router_last_stats = {
        "question": str(question), "memory_demand": float(demand),
        "gate_min": float(gates.min().item()), "gate_max": float(gates.max().item()),
        "gate_mean": float(gates.mean().item()), "mode": config["mode"],
    }
    print(f"[AnswerHeadRouter] {model._answer_head_router_last_stats}")
    return gates


def deactivate_answer_head_router(model):
    model._answer_head_gates = None
