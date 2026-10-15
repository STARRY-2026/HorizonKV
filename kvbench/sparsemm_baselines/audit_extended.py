"""Independently audit completed multi-video SparseMM streaming results.

Uses only the standard library and the runner's established CSV validator.
It never imports torch, loads weights, acquires a GPU, or changes run sources.
Logged dimensions and counts can be checked; attention rankings, tensor values,
and exact position IDs cannot be reconstructed from this trace format.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct

from sparsemm_baselines.run_streamingbench import load_resume, sha256, write_json


CONFIG_KEYS = ("method", "budget", "window_size", "kernel_size", "pooling", "floor_alpha",
               "normalize", "beta", "gqa_reduce", "budget_scope", "score_precision")


class AuditError(RuntimeError):
    pass


def check(condition, message):
    if not condition:
        raise AuditError(message)


def finite(value, location="value"):
    if isinstance(value, float):
        check(math.isfinite(value), f"Non-finite number at {location}")
    elif isinstance(value, dict):
        for key, child in value.items():
            finite(child, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            finite(child, f"{location}[{index}]")


def read_json(path):
    check(path.is_file(), f"Required file missing: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    finite(result, path.name)
    return result


def integer(value, label, minimum=0):
    check(isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
          f"{label} must be an integer >= {minimum}")
    return value


def integer_list(value, size, label):
    check(isinstance(value, list) and len(value) == size, f"Wrong {label} length")
    for item in value:
        integer(item, label)
    return value


def head_lengths(value, layers, heads, label):
    check(isinstance(value, list) and len(value) == layers, f"Wrong {label} layer count")
    return [integer_list(row, heads, label) for row in value]


def layer_capacity(protocol, index, layers, seen, prefix):
    if protocol["method"] == "adakv":
        return protocol["budget"]
    window = protocol["window_size"]
    base = protocol["budget"] - window
    low = base // protocol["beta"]
    high = 2 * base - low
    protected = prefix if protocol["budget_scope"] == "visual" else 0
    available = max(0, seen - protected - window)
    if high >= available:
        high = available
        low = 2 * base - high
    return high - index * ((high - low) // (layers - 1)) + window


def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def upstream_round(allocation, alpha, floor):
    # The upstream integer allocation multiplied by a Python float produces
    # float32 torch values. Reproduce those round-to-even decisions with stdlib.
    return round(f32(f32(f32(allocation) * f32(1 - alpha)) + f32(floor)))


def validate_adaptive_counts(proposed, histories, capacity, alpha, label):
    """Check whether counts can result from upstream integer floor + rounding.

    Trace does not contain global top-k winner IDs. Inverting its monotone
    rounding formula gives feasible allocation intervals per head; their sum
    must contain the exact number of globally selected historical positions.
    """
    floor = int(capacity * alpha)
    total = min(len(histories) * capacity, sum(histories))
    factor = 1 - alpha
    lower_sum = upper_sum = 0
    for head, (count, available) in enumerate(zip(proposed, histories)):
        if factor == 0:
            check(count == floor, f"{label}: invalid floor-only allocation in head {head}")
            low, high = 0, available
        else:
            center = (count - floor) / factor
            radius = .5001 / factor + 2
            low = max(0, math.floor(center - radius))
            high = min(available, math.ceil(center + radius))
            candidates = [a for a in range(low, high + 1) if upstream_round(a, alpha, floor) == count]
            check(bool(candidates), f"{label}: count cannot arise from upstream floor/round in head {head}")
            low, high = candidates[0], candidates[-1]
        lower_sum += low
        upper_sum += high
    check(lower_sum <= total <= upper_sum,
          f"{label}: rounded counts cannot come from the global top-k budget {total}")


def accuracy_statistics(rows):
    correct = sum(float(row["qa_acc"]) == 100 for row in rows)
    count = len(rows)
    p = correct / count
    z = 1.959963984540054
    denominator = 1 + z * z / count
    midpoint = (p + z * z / (2 * count)) / denominator
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return {"questions": count, "correct": correct, "accuracy_percent": 100 * p,
            "wilson_95_percent": [100 * (midpoint - half), 100 * (midpoint + half)]}


def audit_run(output_dir, *, require_compression=False, min_compression_events_per_video=0):
    out = Path(output_dir).resolve()
    protocol = read_json(out / "protocol.json")
    annotations = read_json(out / "annotations.json")
    summary = read_json(out / "summary.json")
    dataset = read_json(out / "dataset_audit.json")
    manifest = read_json(out / "run_manifest.json")
    check((out / "run.status").is_file() and (out / "run.status").read_text().strip() == "complete",
          "Runner has not marked the run complete")
    check(isinstance(annotations, list) and annotations, "No selected annotations")
    video_ids = [str(video["video_id"]) for video in annotations]
    check(len(set(video_ids)) == len(video_ids), "Duplicate annotation video IDs")
    expected_questions = sum(len(video["conversations"]) for video in annotations)
    canonical = json.dumps(annotations, sort_keys=True, ensure_ascii=False).encode()
    check(hashlib.sha256(canonical).hexdigest() == protocol["subset_sha256"], "Selected annotation hash differs from protocol")
    check(dataset["subset_sha256"] == protocol["subset_sha256"] and dataset["source_sha256"] == protocol["source_sha256"],
          "Dataset audit and protocol hashes disagree")
    check(dataset["videos"] == len(annotations) and dataset["questions"] == expected_questions
          and [str(v) for v in dataset["video_ids"]] == video_ids, "Dataset audit selection counts or IDs disagree")
    source = Path(dataset["source"])
    check(source.is_file(), "Original annotation source is unavailable for hash verification")
    check(sha256(source) == protocol["source_sha256"], "Original annotation source hash changed")
    check(protocol["upstream_commit"] == manifest["upstream_commit"], "Manifest/protocol upstream commits disagree")
    check(protocol["method"] in ("adakv", "pyramidkv"), "Unsupported method")
    check(protocol["budget_scope"] in ("prompt", "visual"), "Unsupported budget scope")
    check(protocol["question_cache"] == "temporary_then_restore_video_cache", "Unexpected question cache protocol")
    integer(protocol["window_size"], "window_size", 1)
    integer(protocol["budget"], "budget", protocol["window_size"] + 1)
    integer(protocol["beta"], "beta", 1)
    check(0 <= protocol["floor_alpha"] <= 1, "Invalid floor_alpha")

    result_path = out / "1_0.csv"
    check(result_path.is_file(), "Result CSV is missing")
    try:
        rows, completed = load_resume(result_path, annotations, require_complete=True)
    except (RuntimeError, ValueError, KeyError) as exc:
        raise AuditError(f"CSV validation failed: {exc}") from exc
    check(completed == len(annotations) and len(rows) == expected_questions, "CSV is incomplete")
    check((out / "results.csv").is_file() and (out / "results.csv").read_bytes() == result_path.read_bytes(),
          "Evaluator CSV differs from committed runner CSV")
    for row, video in zip(rows, [v for v in annotations for _ in v["conversations"]]):
        check(row.get("duration_category", "") == (video.get("duration_category", "") or ""),
              "CSV duration_category differs from annotation")
    overall = accuracy_statistics(rows)
    check(summary["videos"] == len(annotations) and summary["questions"] == len(rows), "Summary count mismatch")
    check(math.isclose(summary["accuracy_percent"], overall["accuracy_percent"], abs_tol=1e-9), "Summary accuracy mismatch")
    if summary.get("is_first250_complete"):
        check(len(annotations) == 250 and len(rows) == 1255, "Subset was incorrectly labeled a completed first250 benchmark")

    model_path = Path(protocol["model_path"]) / "config.json"
    model = read_json(model_path)
    if "config.json" in manifest.get("model_files_sha256", {}):
        check(sha256(model_path) == manifest["model_files_sha256"]["config.json"], "Model configuration changed since this run")
    text = model.get("text_config", model)
    layers = integer(text["num_hidden_layers"], "model layers", 1)
    heads = integer(text["num_key_value_heads"], "model KV heads", 1)
    query_heads = integer(text["num_attention_heads"], "model query heads", 1)
    hidden = integer(text["hidden_size"], "model hidden size", 1)
    check(hidden % query_heads == 0 and query_heads % heads == 0, "Invalid model head dimensions")
    dim = hidden // query_heads
    check(protocol["method"] != "pyramidkv" or layers > 1, "Pyramid schedule requires multiple layers")
    element_bytes = {"float16": 2, "bfloat16": 2, "float32": 4}.get(protocol["dtype"])
    check(element_bytes is not None, "Unsupported protocol dtype")
    bytes_per_token = dim * 2 * element_bytes
    expected_config = {key: protocol[key] for key in CONFIG_KEYS}
    trace_path = out / "cache_trace.jsonl"
    check(trace_path.is_file(), "Cache trace is missing")
    trace = []
    for line_number, line in enumerate(trace_path.read_text().splitlines(), 1):
        check(bool(line.strip()), f"Empty trace line {line_number}")
        event = json.loads(line)
        finite(event, f"trace line {line_number}")
        trace.append(event)
    check(trace, "Cache trace is empty")
    states = []
    state = None
    initial_prefix = None
    zero_mass = Counter()
    drifts, means, target_means = [], [], []
    observed_tokens = 0

    def validate_cache(event, expected, label):
        actual = head_lengths(event.get("head_lengths"), layers, heads, label)
        check(actual == expected, f"{label}: cache head lengths changed unexpectedly")
        expected_bytes = sum(sum(row) for row in expected) * bytes_per_token
        check(event.get("kv_bytes") == expected_bytes, f"{label}: KV bytes do not match physical token counts")
        return expected_bytes

    def finish_video():
        if state is None:
            return
        expected = len(annotations[len(states)]["conversations"])
        check(state["questions"] == expected,
              f"Video {state['video_id']}: expected {expected} restored answers, saw {state['questions']}")
        threshold = max(min_compression_events_per_video, int(require_compression))
        check(state["compression_events"] >= threshold,
              f"Video {state['video_id']}: insufficient compression events for requested threshold {threshold}")
        state["sustained_compression"] = state["compression_events"] >= 2
        states.append({key: value for key, value in state.items() if key not in ("lengths", "seen", "last_answer_prefill", "cursor")})

    for line_number, event in enumerate(trace, 1):
        kind = event.get("event")
        label = f"Trace line {line_number} ({kind})"
        vid = str(event.get("video_id"))
        if kind == "init":
            finish_video()
            check(len(states) < len(annotations) and vid == video_ids[len(states)], f"{label}: extra or out-of-order video init")
            prefix = integer(event.get("text_tokens"), "initial text tokens", 1)
            if initial_prefix is None:
                initial_prefix = prefix
            check(prefix == initial_prefix, f"{label}: per-video system prefix did not reset consistently")
            check(event.get("config") == expected_config, f"{label}: selector config differs from protocol")
            check(event.get("num_layers") == layers and event.get("num_kv_heads") == heads,
                  f"{label}: architecture differs from model config")
            state = {"video_id": vid, "prefix_tokens": prefix, "prefills": 0, "questions": 0,
                     "compression_events": 0, "compressed_layer_events": 0, "seen": prefix,
                     "lengths": [[prefix] * heads for _ in range(layers)], "cursor": prefix,
                     "last_answer_prefill": None, "final_next_position": prefix,
                     "video_input_tokens": 0, "zero_mass_head_events": 0,
                     "repeated_answers_without_prefill": 0}
            continue
        check(state is not None and vid == state["video_id"], f"{label}: event has no matching active video")
        if kind == "video_prefill":
            state["prefills"] += 1
            check(event.get("prefill") == state["prefills"], f"{label}: prefill counter did not reset or increment")
            q = integer(event.get("input_tokens"), "prefill input tokens", 1)
            state["seen"] += q
            state["video_input_tokens"] += q
            observed_tokens += q
            metadata = event.get("layers")
            check(isinstance(metadata, list) and len(metadata) == layers, f"{label}: missing layer metadata")
            output_lengths = []
            event_compressed = False
            protected = state["prefix_tokens"] if protocol["budget_scope"] == "visual" else 0
            for index, meta in enumerate(metadata):
                name = f"{label}, layer {index}"
                capacity = layer_capacity(protocol, index, layers, state["seen"], state["prefix_tokens"])
                historical_capacity = max(0, capacity - protocol["window_size"])
                before = [n + q for n in state["lengths"][index]]
                recent = [min(protocol["window_size"], n - protected) for n in before]
                history = [n - protected - r for n, r in zip(before, recent)]
                after = integer_list(meta.get("head_lengths"), heads, name)
                proposed = integer_list(meta.get("proposed_history_counts"), heads, name)
                retained = integer_list(meta.get("retained_history_counts"), heads, name)
                bypass = all(n < historical_capacity for n in history) if protocol["method"] == "adakv" else all(n - protected < capacity for n in before)
                expected_fields = {
                    "method": protocol["method"], "budget_scope": protocol["budget_scope"],
                    "configured_budget": protocol["budget"], "layer_capacity": capacity,
                    "history_capacity_per_head": historical_capacity, "prefix_tokens_kept": protected,
                    "seen_tokens": state["seen"], "observed_queries": min(q, protocol["window_size"]),
                    "gqa_groups": query_heads // heads, "gqa_reduce": protocol["gqa_reduce"],
                    "score_precision": protocol["score_precision"], "before_head_lengths": before,
                    "recent_counts": recent, "bypassed_short_cache": bypass, "rounding": "upstream",
                    "tie_policy": "upstream_torch", "zero_mass_normalization": 0,
                    "upstream_commit": protocol["upstream_commit"],
                }
                for key, value in expected_fields.items():
                    check(meta.get(key) == value, f"{name}: invalid {key}")
                if bypass:
                    check(proposed == history, f"{name}: bypass changed historical capacity")
                elif protocol["method"] == "pyramidkv":
                    check(proposed == [historical_capacity] * heads, f"{name}: Pyramid historical budget mismatch")
                else:
                    validate_adaptive_counts(proposed, history, historical_capacity, protocol["floor_alpha"], name)
                expected_retained = [min(p, h) for p, h in zip(proposed, history)]
                check(retained == expected_retained, f"{name}: historical availability was not capped correctly")
                check(after == [protected + p + r for p, r in zip(retained, recent)], f"{name}: retained length decomposition differs")
                compressed = any(a < b for a, b in zip(after, before))
                check(meta.get("compressed") == compressed, f"{name}: compression flag is wrong")
                target = heads * (capacity + protected)
                actual = sum(after)
                drift = actual - target
                check(meta.get("target_total_tokens") == target and meta.get("actual_total_tokens") == actual,
                      f"{name}: total capacity mismatch")
                check(meta.get("actual_mean_tokens") == actual / heads and meta.get("allocation_drift_tokens") == drift,
                      f"{name}: mean capacity or integer drift mismatch")
                zeros = meta.get("zero_mass_heads")
                check(isinstance(zeros, list) and len(set(zeros)) == len(zeros)
                      and all(isinstance(h, int) and not isinstance(h, bool) and 0 <= h < heads for h in zeros),
                      f"{name}: invalid zero-mass head indices")
                check(not zeros or protocol["method"] == "adakv" and protocol["normalize"] and not bypass,
                      f"{name}: unexpected zero-mass normalization")
                for head in zeros:
                    zero_mass[f"layer_{index}_head_{head}"] += 1
                state["zero_mass_head_events"] += len(zeros)
                state["compressed_layer_events"] += int(compressed)
                event_compressed |= compressed
                drifts.append(drift)
                means.append(actual / heads)
                target_means.append(capacity + protected)
                output_lengths.append(after)
            validate_cache(event, output_lengths, label)
            state["lengths"] = output_lengths
            state["compression_events"] += int(event_compressed)
        elif kind == "answer_restored":
            validate_cache(event, state["lengths"], label)
            current = integer(event.get("next_position"), "answer cursor", state["prefix_tokens"])
            check(current >= state["cursor"], f"{label}: next_position moved backward")
            if state["last_answer_prefill"] == state["prefills"]:
                check(current == state["cursor"], f"{label}: repeated QA changed the cursor without new video")
                state["repeated_answers_without_prefill"] += 1
            video = annotations[len(states)]
            qi = state["questions"]
            check(qi < len(video["conversations"]), f"{label}: extra restored answer")
            if qi:
                now = math.ceil(float(video["conversations"][qi]["end_time"]) * protocol["sample_fps"])
                prev = math.ceil(float(video["conversations"][qi - 1]["end_time"]) * protocol["sample_fps"])
                if now == prev:
                    check(state["last_answer_prefill"] == state["prefills"], f"{label}: equal video endpoints unexpectedly re-encoded frames")
            state["questions"] += 1
            state["cursor"] = current
            state["final_next_position"] = current
            state["last_answer_prefill"] = state["prefills"]
        else:
            raise AuditError(f"{label}: unknown trace event")
    finish_video()
    check(len(states) == len(annotations), "Cache trace is missing selected videos")
    check(sum(v["questions"] for v in states) == len(rows), "Trace/CSV question count mismatch")
    by_task = defaultdict(list)
    for row in rows:
        by_task[row.get("task") or "(unspecified)"].append(row)
    chance = []
    for video in annotations:
        chance.extend(100 / len(q["choices"]) for q in video["conversations"])
    report = {
        "passed": True, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(out), "method": protocol["method"], "budget_scope": protocol["budget_scope"],
        "configured_budget": protocol["budget"], "scope": "selected subset; not a full benchmark claim",
        "is_first250_complete": bool(summary.get("is_first250_complete")),
        "videos": len(states), "questions": len(rows), "overall": overall,
        "task_accuracy": {task: accuracy_statistics(task_rows) for task, task_rows in sorted(by_task.items())},
        "calibration": {"confidence_scores_available": False, "ece": None, "brier_score": None,
                        "reason": "Runner stores greedy answers only; probability calibration is unavailable.",
                        "mean_uniform_choice_chance_percent": sum(chance) / len(chance),
                        "interval_note": "Wilson intervals describe selected questions; videos and questions are correlated."},
        "architecture": {"num_layers": layers, "num_kv_heads": heads, "num_query_heads": query_heads,
                         "head_dim": dim, "kv_element_bytes": element_bytes, "bytes_per_head_token": bytes_per_token},
        "trace": {"events": len(trace), "prefills": sum(v["prefills"] for v in states),
                  "compression_events": sum(v["compression_events"] for v in states),
                  "compressed_layer_events": sum(v["compressed_layer_events"] for v in states),
                  "videos_with_sustained_compression": sum(v["sustained_compression"] for v in states),
                  "repeated_answers_without_prefill": sum(v["repeated_answers_without_prefill"] for v in states),
                  "observed_visual_tokens": observed_tokens,
                  "zero_mass_head_events": sum(zero_mass.values()), "zero_mass_counts": dict(sorted(zero_mass.items())),
                  "allocation_drift_tokens_range": [min(drifts), max(drifts)] if drifts else None,
                  "actual_mean_tokens_range": [min(means), max(means)] if means else None,
                  "scheduled_mean_tokens_range": [min(target_means), max(target_means)] if target_means else None},
        "per_video": states,
        "checked": ["complete CSV and exact annotation order/ground truth/prediction consistency",
                    "protocol/subset/source/model-config hashes and summary counts/accuracy",
                    "per-video init config, architecture, prefix reset and prefill counter reset",
                    "every layer's cumulative lengths, upstream capacity schedule and integer rounding feasibility",
                    "ragged historical availability caps, recent/prefix counts and logged physical KV byte accounting",
                    "question restoration of cache lengths/bytes and stable repeated-question cursor",
                    "nondecreasing question cursor, finite logged metadata and compression counts"],
        "limitations": ["Trace contains counts, not K/V contents or token position IDs; tensor identity and exact attention rankings cannot be re-audited.",
                        "Answer events lack question IDs; answers are correlated by video and annotation order, with exact per-video counts.",
                        "Prefill events lack absolute position cursors; monotonicity is checked at answer boundaries only.",
                        "AdaKV trace lacks global top-k winner IDs; integer floor/round feasibility is verified, not attention-score optimality.",
                        "This audit cannot establish full benchmark accuracy or confidence calibration from a small selected subset."],
    }
    finite(report, "report")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-compression", action="store_true", help="Require at least one compression event in every selected video")
    parser.add_argument("--min-compression-events-per-video", type=int, default=0)
    args = parser.parse_args(argv)
    check(args.min_compression_events_per_video >= 0, "Compression threshold must be nonnegative")
    destination = args.output or args.output_dir / "audited_report.json"
    try:
        result = audit_run(args.output_dir, require_compression=args.require_compression,
                           min_compression_events_per_video=args.min_compression_events_per_video)
    except (AuditError, KeyError, ValueError, TypeError, OSError) as exc:
        result = {"passed": False, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                  "output_dir": str(args.output_dir.resolve()), "error": str(exc)}
        write_json(destination, result)
        print(json.dumps(result, ensure_ascii=False))
        return 1
    write_json(destination, result)
    print(json.dumps({"passed": True, "videos": result["videos"], "questions": result["questions"],
                      "compression_events": result["trace"]["compression_events"], "report": str(destination.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
