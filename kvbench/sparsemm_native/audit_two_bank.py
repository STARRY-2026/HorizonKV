"""Independently audit recorded DP two-bank allocation and pruning invariants.

Run with ``python -m sparsemm_native.audit_two_bank OUTPUT_ROOT/dp_dual``.
The auditor uses only Python's standard library and never imports a selector.
It checks recorded token identities and counts, but cannot reconstruct QK
attention scores or prove the retained historical tokens have optimal scores:
the trace intentionally does not store model Q/K tensors.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path


LAYERS, HEADS, BUDGET, WINDOW = 28, 4, 500, 32
TOTAL = LAYERS * HEADS * BUDGET


class AuditError(ValueError):
    """A trace contradicts its pinned protocol or a cache invariant."""


def require(condition, message):
    if not condition:
        raise AuditError(message)


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum,
            f"{label} must be an integer >= {minimum}")
    return value


def vector(value, label, length=HEADS, minimum=0):
    require(isinstance(value, list) and len(value) == length,
            f"{label} must contain {length} entries")
    return [integer(item, f"{label}[{i}]", minimum) for i, item in enumerate(value)]


def matrix(value, label, minimum=0):
    require(isinstance(value, list) and len(value) == LAYERS,
            f"{label} must contain {LAYERS} layers")
    return [vector(row, f"{label}[{i}]", minimum=minimum) for i, row in enumerate(value)]


def bool_vector(value, label):
    require(isinstance(value, list) and len(value) == HEADS
            and all(type(item) is bool for item in value),
            f"{label} must contain {HEADS} booleans")
    return value


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protocol_tables(protocol):
    require(isinstance(protocol, dict), "Protocol must be a JSON object")
    require(protocol.get("arm") == "dp_dual", "Protocol arm is not dp_dual")
    require(protocol.get("cache_policy") == "two_bank", "Protocol cache policy is not two_bank")
    require(protocol.get("budget") == BUDGET and protocol.get("protect_window") == WINDOW,
            "Protocol must use K=500 and protect window=32")
    require(protocol.get("attention_observation_window") == WINDOW,
            "Protocol attention observation window must be 32")
    require(protocol.get("protect_window_included_in_recent") is True,
            "Protocol adds protection outside the Recent budget")
    allocation = protocol.get("dp_allocation")
    require(isinstance(allocation, dict), "Protocol has no DP allocation table")
    capacity = matrix(allocation.get("head_budgets"), "capacity", minimum=WINDOW)
    recent = matrix(allocation.get("recent_budgets"), "recent", minimum=WINDOW)
    history = matrix(allocation.get("history_budgets"), "history")
    require(sum(map(sum, capacity)) == TOTAL, "Protocol does not allocate exactly 56,000 visual positions")
    for layer in range(LAYERS):
        for head in range(HEADS):
            require(375 <= capacity[layer][head] <= 750,
                    f"Protocol head {layer}:{head} violates original 375..750 bounds")
            require(recent[layer][head] + history[layer][head] == capacity[layer][head],
                    f"Protocol head {layer}:{head} has R+H != N")
    require(allocation.get("dp_bank_split_used_online") is True,
            "Protocol does not execute the DP bank split online")
    require(allocation.get("protect_window_included_in_recent") is True,
            "Allocation does not include protection inside Recent")
    require(allocation.get("allocated_slots") == TOTAL
            and allocation.get("target_slots") == TOTAL, "Allocation totals contradict the tables")
    gate = protocol.get("short_cache_gate", allocation.get("short_cache_gate"))
    require(gate in ("legacy_global", "per_head_capacity"), "Unknown protocol short-cache gate")
    require(allocation.get("short_cache_gate") == gate, "Protocol and allocation gates disagree")
    return capacity, recent, history, allocation, gate


def audit_layer(layer, expected_index, prefix, input_tokens, tables, gate, allocation):
    require(isinstance(layer, dict), "Layer metadata must be a JSON object")
    capacity, recent, history = (table[expected_index] for table in tables)
    require(layer.get("layer_idx") == expected_index, "Layer index is missing, duplicated, or out of order")
    require(layer.get("method") == "dp_two_bank", "Layer does not execute the two-bank selector")
    require(layer.get("budget_scope") == "visual" and layer.get("configured_budget") == BUDGET,
            "Layer budget/scope differs from protocol")
    require(layer.get("head_budgets") == capacity and layer.get("recent_budgets") == recent
            and layer.get("history_budgets") == history, "Layer head quotas differ from protocol")
    require(layer.get("allocation") == allocation, "Layer allocation provenance differs from protocol")
    require(layer.get("window_size") == WINDOW
            and layer.get("protect_window_included_in_recent") is True,
            "Layer protect window differs from protocol")
    require(layer.get("short_cache_gate") == gate, "Layer short-cache gate differs from protocol")
    require(layer.get("prefix_tokens") == prefix and layer.get("prefix_tokens_kept") == prefix,
            "Layer text prefix differs from initialization")
    require(layer.get("query_len") == input_tokens
            and layer.get("observed_queries") == min(input_tokens, WINDOW),
            "Layer attention observation count differs from min(32, current queries)")
    require(layer.get("score_precision") == "float32" and layer.get("gqa_reduce") == "mean"
            and layer.get("gqa_groups") == 7 and layer.get("pooling") == "avgpool"
            and layer.get("kernel_size") == 5, "Layer attention scoring configuration changed")
    before = vector(layer.get("before_head_lengths"), "before_head_lengths", minimum=prefix)
    after = vector(layer.get("head_lengths"), "head_lengths", minimum=prefix)
    observed_recent = vector(layer.get("recent_counts"), "recent_counts")
    observed_history = vector(layer.get("retained_history_counts"), "retained_history_counts")
    bypassed = bool_vector(layer.get("bypassed_short_cache_per_head"), "bypassed_short_cache_per_head")
    visual = [length - prefix for length in before]
    require(all(length >= input_tokens for length in before), "Current query chunk is absent from a KV head")
    # Reconstruct the OLD reference gate from visual history beyond Recent-32,
    # not beyond the DP Recent quota and not from a per-head capacity check.
    legacy_gate = all(v - min(WINDOW, v) < BUDGET - WINDOW for v in visual)
    warmup = gate == "legacy_global" and legacy_gate
    require(layer.get("reference_warmup_bypass", False) is warmup,
            "Reference warmup bypass contradicts independently recomputed legacy gate")
    expected_bypass = [warmup] * HEADS if gate == "legacy_global" else [
        v <= n for v, n in zip(visual, capacity)]
    require(bypassed == expected_bypass and layer.get("bypassed_short_cache") is all(expected_bypass),
            "Recorded bypass decision differs from its exact gate")
    require(layer.get("proposed_history_counts") == history, "Proposed history quotas changed")
    require(layer.get("retained_visual_counts") == [length - prefix for length in after],
            "Retained visual counts differ from actual lengths")
    kept_all = layer.get("kept_indices")
    require(isinstance(kept_all, list) and len(kept_all) == HEADS, "Missing per-head retained token indices")
    checks = layer.get("position_checks")
    require(isinstance(checks, dict) and checks and all(value is True for value in checks.values()),
            "Runtime position/invariant checks failed or are absent")
    required_checks = {"prefix_preserved", "recent_preserved", "indices_unique", "indices_in_range",
                       "positions_match_kept", "recent_bank_preserved", "bank_counts_valid"}
    require(required_checks <= set(checks), "Runtime omitted required position checks")
    require(checks.get("quota_or_reference_warmup", checks.get("capacity_respected")) is True,
            "Runtime omitted the capacity/warmup check")
    over_quota = 0
    for head in range(HEADS):
        context = f"head {head}"
        r = min(recent[head], visual[head])
        h = (visual[head] - r if warmup else min(history[head], visual[head] - r))
        require(observed_recent[head] == r and observed_history[head] == h,
                f"{context}: bank counts do not follow quotas and available candidates")
        require(after[head] == prefix + r + h, f"{context}: actual cache length differs from bank counts")
        require(warmup or after[head] <= prefix + capacity[head],
                f"{context}: cache exceeds quota outside reference warmup")
        indices = vector(kept_all[head], f"{context} kept indices", length=after[head])
        require(len(indices) == len(set(indices)), f"{context}: duplicate/cross-bank token indices")
        require(all(index < before[head] for index in indices), f"{context}: retained token index out of range")
        require(indices[:prefix] == list(range(prefix)), f"{context}: text prefix is not retained exactly")
        if r:
            require(indices[-r:] == list(range(before[head] - r, before[head])),
                    f"{context}: Recent cache is not the exact latest-R sliding window")
        protected = min(WINDOW, visual[head])
        if protected:
            require(indices[-protected:] == list(range(before[head] - protected, before[head])),
                    f"{context}: protected last-32 window was lost")
        historical = indices[prefix:len(indices) - r if r else len(indices)]
        require(len(historical) == h and all(prefix <= index < before[head] - r for index in historical),
                f"{context}: History contains a prefix/recent token or has the wrong count")
        if expected_bypass[head]:
            require(indices == list(range(before[head])), f"{context}: bypass reordered or removed tokens")
        over_quota += int(after[head] > prefix + capacity[head])
    require(layer.get("target_total_tokens") == sum(capacity) + HEADS * prefix,
            "Layer target token total contradicts quotas")
    require(layer.get("actual_total_tokens") == sum(after), "Layer actual total contradicts head lengths")
    require(layer.get("compressed") is any(a < b for a, b in zip(after, before)),
            "Layer compressed flag contradicts retained lengths")
    return after, int(warmup), over_quota


def audit(output_dir):
    output = Path(output_dir).expanduser().resolve()
    protocol_path, trace_path = output / "protocol.json", output / "cache_trace.jsonl"
    result = {"output_dir": str(output), "passed": False, "event_counts": {},
              "auditor_sha256": file_digest(Path(__file__).resolve()),
              "layer_records": 0, "head_records": 0, "reference_warmup_layer_records": 0,
              "reference_warmup_heads_over_capacity": 0,
              "limits": ["Historical attention rankings cannot be independently recomputed: Q/K tensors are not stored.",
                         "Absolute KV positions are not stored; positions_match_kept is checked as a recorded runtime assertion.",
                         "This audit establishes recorded cache invariants, not answer correctness or comparative QA accuracy."]}
    digest = hashlib.sha256()
    counters = collections.Counter()
    prefix = None
    line_number, layer_index = 0, None
    complete_stream = False
    trace_stat = None
    try:
        protocol_bytes = protocol_path.read_bytes()
        result["protocol_sha256"] = hashlib.sha256(protocol_bytes).hexdigest()
        protocol = json.loads(protocol_bytes)
        capacity, recent, history, allocation, gate = protocol_tables(protocol)
        tables = capacity, recent, history
        result.update(short_cache_gate=gate, total_visual_slots=TOTAL,
                      recent_slots=sum(map(sum, recent)), history_slots=sum(map(sum, history)))
        budget_path = Path(protocol["dp_budget_path"])
        if budget_path.is_file():
            require(file_digest(budget_path) == protocol["dp_budget_sha256"], "Pinned budget CSV SHA-256 changed")
            with budget_path.open(newline="", encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            require(len(rows) == LAYERS * HEADS, "Pinned budget CSV has an unexpected head count")
            seen = set()
            for row in rows:
                l, h = int(row["layer"]), int(row["kv_head"])
                require(0 <= l < LAYERS and 0 <= h < HEADS and (l, h) not in seen,
                        "Pinned budget CSV contains duplicate or invalid heads")
                seen.add((l, h))
                require([int(row[k]) for k in ("capacity_n", "recent_r", "history_q")]
                        == [capacity[l][h], recent[l][h], history[l][h]],
                        "Protocol allocation differs from the pinned budget CSV")
            result["budget_csv_verified"] = True
        else:
            result["budget_csv_verified"] = False
            result["limits"].append("Original budget CSV is unavailable; tables were checked against the protocol only.")
        for name, expected_hash in protocol.get("code_sha256", {}).items():
            snapshot_path = output / "source_snapshot" / name
            require(snapshot_path.is_file() and file_digest(snapshot_path) == expected_hash,
                    f"Pinned source snapshot changed or is absent: {name}")
        result["source_snapshot_files_verified"] = len(protocol.get("code_sha256", {}))
        trace_stat = trace_path.stat()
        with trace_path.open("rb") as stream:
            for line_number, raw in enumerate(stream, 1):
                digest.update(raw)
                if not raw.strip():
                    continue
                event = json.loads(raw)
                require(isinstance(event, dict), "Trace event must be a JSON object")
                name = event.get("event")
                require(isinstance(name, str), "Trace event has no event name")
                counters[name] += 1
                if name == "init":
                    config = event.get("config", {})
                    require(isinstance(config, dict), "Initialization config must be a JSON object")
                    require(config.get("method") == "sparsemm" and config.get("sparsemm_cache_policy") == "two_bank",
                            "Initialization uses a different cache method/policy")
                    require(config.get("budget") == BUDGET and config.get("window_size") == WINDOW
                            and config.get("budget_scope") == "visual", "Initialization changes budget/protection")
                    require(config.get("sparsemm_two_bank_gate", "per_head_capacity") == gate,
                            "Initialization gate differs from protocol")
                    require(config.get("sparsemm_budget_path") == protocol["dp_budget_path"],
                            "Initialization budget path differs from protocol")
                    require(event.get("num_layers") == LAYERS and event.get("num_kv_heads") == HEADS
                            and event.get("num_query_heads") == HEADS * 7, "Initialization model shape differs")
                    prefix = integer(event.get("text_tokens"), "initialized prefix", 1)
                elif name == "video_prefill":
                    require(prefix is not None, "Video prefill occurred without an initialization event")
                    input_tokens = integer(event.get("input_tokens"), "input_tokens", 1)
                    layers = event.get("layers")
                    require(isinstance(layers, list) and len(layers) == LAYERS,
                            "Video prefill does not contain all 28 decoder layers")
                    after = []
                    for layer_index, layer in enumerate(layers):
                        length, warmup, over = audit_layer(layer, layer_index, prefix, input_tokens,
                                                          tables, gate, allocation)
                        after.append(length)
                        result["layer_records"] += 1
                        result["head_records"] += HEADS
                        result["reference_warmup_layer_records"] += warmup
                        result["reference_warmup_heads_over_capacity"] += over
                    layer_index = None
                    require(event.get("head_lengths") == after, "Root prefill lengths differ from layer lengths")
                    method = event.get("method_metadata")
                    require(isinstance(method, dict) and method.get("layers") == layers
                            and method.get("head_lengths") == after,
                            "Root method metadata contradicts the audited layer records")
        complete_stream = True
        current_stat = trace_path.stat()
        require(current_stat.st_size == trace_stat.st_size and current_stat.st_mtime_ns == trace_stat.st_mtime_ns,
                "Trace changed during audit; retry after inference has finished writing")
        require(counters["init"] > 0 and counters["video_prefill"] > 0,
                "Trace contains no initialized video prefills")
        result["passed"] = True
    except (AuditError, OSError, ValueError, KeyError, TypeError, IndexError) as error:
        result["error"] = str(error)
        result["error_context"] = {"trace_line": line_number, "layer_index": layer_index}
    result["event_counts"] = dict(counters)
    result["trace_lines_read"] = line_number
    result["trace_sha256"] = digest.hexdigest() if complete_stream else None
    result["trace_hash_complete"] = complete_stream
    if trace_stat is not None:
        result["trace_bytes_at_start"] = trace_stat.st_size
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="Directory containing protocol.json and cache_trace.jsonl")
    parser.add_argument("--report", type=Path, help="Default: OUTPUT_DIR/two_bank_audit.json")
    args = parser.parse_args(argv)
    result = audit(args.output_dir)
    destination = args.report or args.output_dir / "two_bank_audit.json"
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
