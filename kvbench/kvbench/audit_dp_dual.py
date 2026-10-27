"""Read-only audit of a bounded StreamingBench DP two-bank run."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from sparsemm_baselines.run_streamingbench import load_resume, prepare_annotations, sha256, write_json
from sparsemm_native.audit_two_bank import (BUDGET, HEADS, LAYERS, WINDOW,
                                            audit_layer, protocol_tables, require)


def audit(output_dir, min_compression_events_per_video=0):
    out = Path(output_dir).resolve()
    protocol = json.loads((out / "protocol.json").read_text())
    require((out / "run.status").read_text().strip() == "complete", "DP two-bank run is incomplete")
    require(protocol.get("method") == "sparsemm" and protocol.get("sparsemm_cache_policy") == "two_bank"
            and protocol.get("sparsemm_two_bank_gate") == "legacy_global"
            and protocol.get("budget_scope") == "visual" and protocol.get("window_size") == WINDOW,
            "Run did not execute the registered DP two-bank protocol")
    allocation = protocol["sparsemm_allocation"]
    proxy = {"arm": "dp_dual", "cache_policy": "two_bank", "budget": protocol["budget"],
             "protect_window": WINDOW, "attention_observation_window": WINDOW,
             "protect_window_included_in_recent": True, "dp_allocation": allocation,
             "short_cache_gate": "legacy_global"}
    capacity, recent, history, allocation, gate = protocol_tables(proxy)
    budget_path = Path(allocation["budget_path"])
    require(budget_path.is_file() and sha256(budget_path) == allocation["budget_sha256"],
            "Pinned DP two-bank budget CSV is missing or changed")
    with budget_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == LAYERS * HEADS, "DP two-bank CSV must contain 112 heads")
    seen = set()
    for row in rows:
        layer, head = int(row["layer"]), int(row["kv_head"])
        require(0 <= layer < LAYERS and 0 <= head < HEADS and (layer, head) not in seen,
                "DP two-bank CSV contains an invalid or duplicate head")
        seen.add((layer, head))
        require([int(row[key]) for key in ("capacity_n", "recent_r", "history_q")]
                == [capacity[layer][head], recent[layer][head], history[layer][head]],
                "DP two-bank CSV differs from the recorded allocation")
    annotation = Path(protocol["annotation_source"])
    require(annotation.is_file() and sha256(annotation) == protocol["source_sha256"],
            "Annotation source is missing or changed")
    data = json.loads((out / "annotations.json").read_text())
    expected = prepare_annotations(annotation, len(data),
                                   max(len(video["conversations"]) for video in data))
    digest = hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    require(data == expected and digest == protocol["subset_sha256"],
            "Selected annotations differ from the pinned source")
    rows, videos = load_resume(out / "1_0.csv", data, require_complete=True)
    require(videos == len(data) and len(rows) == sum(len(v["conversations"]) for v in data),
            "Result CSV does not cover every question")
    summary = json.loads((out / "summary.json").read_text())
    correct = sum(float(row["qa_acc"]) == 100.0 for row in rows)
    require(summary.get("method") == "sparsemm" and summary.get("budget") == BUDGET
            and summary.get("videos") == videos and summary.get("questions") == len(rows)
            and summary.get("correct") == correct
            and summary.get("accuracy_percent") == 100 * correct / len(rows),
            "Summary differs from the checked answer rows")
    asset = protocol["sparsemm_asset_metadata"]
    require(Path(asset["path"]).is_file() and sha256(asset["path"]) == asset["sha256"]
            and asset["allocation"] == allocation
            and protocol["sparsemm_asset_sha256"] == asset["sha256"],
            "SparseMM score asset changed or disagrees with its allocation")
    manifest = json.loads((out / "run_manifest.json").read_text())
    source_hashes = manifest["code_sha256"]
    require(isinstance(source_hashes, dict) and source_hashes, "Run has no pinned source hashes")
    for name, fingerprint in source_hashes.items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts,
                "Source snapshot path escapes the run")
        snapshot = out / "source_snapshot" / relative
        require(snapshot.is_file() and sha256(snapshot) == fingerprint,
                f"Pinned source snapshot is missing or changed: {name}")
    ids = {str(video["video_id"]): len(video["conversations"]) for video in data}
    prefixes, prefills, compression, restored = {}, set(), Counter(), Counter()
    layer_records = warmup_layers = over_quota = 0
    trace_path = out / "cache_trace.jsonl"
    stat = trace_path.stat()
    with trace_path.open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            video = str(event.get("video_id"))
            require(video in ids, "Trace contains an unknown video")
            if event.get("event") == "init":
                config = event.get("config", {})
                require(video not in prefixes and config.get("method") == "sparsemm"
                        and config.get("sparsemm_cache_policy") == "two_bank"
                        and config.get("sparsemm_two_bank_gate") == gate
                        and config.get("sparsemm_budget_path") == str(budget_path)
                        and config.get("budget") == BUDGET and config.get("window_size") == WINDOW
                        and config.get("budget_scope") == "visual"
                        and event.get("num_layers") == LAYERS and event.get("num_kv_heads") == HEADS
                        and event.get("num_query_heads") == HEADS * 7,
                        "Trace initialization differs from the DP two-bank protocol")
                prefix = event.get("text_tokens")
                require(type(prefix) is int and prefix > 0, "Invalid text prefix length")
                prefixes[video] = prefix
            elif event.get("event") == "video_prefill":
                identity = (video, event.get("prefill"))
                require(video in prefixes and identity not in prefills and type(identity[1]) is int
                        and identity[1] > 0, "Prefill has no initialization or is duplicated")
                prefills.add(identity)
                inputs = event.get("input_tokens")
                layers = event.get("layers")
                require(type(inputs) is int and inputs > 0 and isinstance(layers, list)
                        and len(layers) == LAYERS, "Prefill has invalid decoder layers")
                after = []
                compressed = False
                for index, layer in enumerate(layers):
                    lengths, warmup, excess = audit_layer(layer, index, prefixes[video], inputs,
                                                          (capacity, recent, history), gate, allocation)
                    after.append(lengths)
                    compressed |= layer["compressed"]
                    layer_records += 1
                    warmup_layers += warmup
                    over_quota += excess
                require(event.get("head_lengths") == after
                        and event.get("method_metadata", {}).get("layers") == layers
                        and event.get("method_metadata", {}).get("head_lengths") == after,
                        "Root prefill metadata differs from audited layers")
                compression[video] += int(compressed)
            elif event.get("event") == "answer_restored":
                require(video in prefixes and event.get("query_check", {}).get("restored") is True,
                        "Question trace did not restore the persistent cache")
                restored[video] += 1
    require(trace_path.stat().st_size == stat.st_size and trace_path.stat().st_mtime_ns == stat.st_mtime_ns,
            "Trace changed during the audit")
    require(set(prefixes) == set(ids) and prefills and layer_records == LAYERS * len(prefills),
            "Trace does not cover all selected videos and decoder layers")
    require(all(restored[video] == questions for video, questions in ids.items()),
            "Trace does not cover every question's cache restoration")
    require(all(compression[video] >= min_compression_events_per_video for video in ids),
            "A video has fewer compression events than required")
    checks = json.loads((out / "runtime_checks.json").read_text())
    require(checks.get("status") == "complete" and not checks.get("failures")
            and checks.get("qa_restore_checks") == len(rows)
            and len(checks.get("qa_checks", [])) == len(rows)
            and all(item.get("returned") is True and item.get("restored") is True
                    for item in checks["qa_checks"])
            and len(checks.get("selector_checks", [])) == layer_records
            and all(item.get("validation_passed") is True for item in checks["selector_checks"]),
            "Native DP two-bank runtime checks are incomplete")
    report = {"passed": True, "method": "dp_dual", "videos": videos, "questions": len(rows),
              "budget_scope": "visual", "budget": BUDGET,
              "trace": {"compression_events": sum(compression.values()),
                        "qa_restore_checks": len(rows), "layer_records": layer_records,
                        "reference_warmup_layer_records": warmup_layers,
                        "reference_warmup_heads_over_capacity": over_quota},
              "limitations": ["Historical Q/K attention rankings cannot be recomputed from the saved trace.",
                              "This audit checks recorded token indices and cache invariants; it cannot inspect tensor values."]}
    write_json(out / "audit_report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--min-compression-events-per-video", type=int, default=0)
    args = parser.parse_args(argv)
    print(json.dumps(audit(args.output_dir, args.min_compression_events_per_video), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
