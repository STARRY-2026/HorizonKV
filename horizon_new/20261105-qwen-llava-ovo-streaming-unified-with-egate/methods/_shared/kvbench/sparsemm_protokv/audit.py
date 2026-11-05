"""Audit native lmms-eval streaming outputs without importing a model or torch.

Checks recorded dimensions, budget transitions, source hashes, CSV ground truth
and native attention/restoration instrumentation. Tensor values and attention
rankings cannot be reconstructed independently from the JSON trace.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

from sparsemm_baselines.audit_extended import (
    AuditError, accuracy_statistics, check, finite, head_lengths, integer,
    integer_list, layer_capacity, read_json, validate_adaptive_counts,
)
from sparsemm_baselines.run_streamingbench import load_resume, sha256, write_json


WORKSPACE = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "33f2f0ab96d1d99516bcd2092caccab60516ff2c"
CONFIG_KEYS = (
    "method", "budget", "budget_scope", "window_size", "kernel_size", "pooling",
    "floor_alpha", "normalize", "beta", "gqa_reduce", "score_precision", "modes",
    "subquantizers", "codewords", "beam", "lambda_sp", "lambda_idle", "idle_threshold",
    "alpha", "proto_beta", "eta", "gamma", "merge_key", "merge_value",
    "warmup_residuals", "pq_smoothing", "covariance_ridge", "kmeans_iterations",
    "kmeans_batch_size",
)
MEMORY_KEYS = (
    "persistent_kv_bytes", "bank_bytes", "raw_near_key_bytes", "metadata_bytes",
    "persistent_total_bytes", "query_context_kv_bytes",
)
REQUIRED_CODE_FILES = (
    "sparsemm_protokv/config.py", "sparsemm_protokv/cache.py",
    "sparsemm_protokv/qwen2_5.py", "sparsemm_protokv/streaming.py",
    "sparsemm_protokv/monkeypatch.py", "protokv/full_state.py",
    "sparsemm_baselines/core.py", "sparsemm_baselines/cache.py",
)


def source_file(output, name, expected_hash):
    """Resolve a hash against current source or this run's own saved snapshot."""
    relative = Path(name)
    check(not relative.is_absolute() and ".." not in relative.parts,
          f"Source hash path escapes workspace: {name}")
    check(isinstance(expected_hash, str) and re.fullmatch(r"[0-9a-f]{64}", expected_hash),
          f"Invalid source SHA-256: {name}")
    current = (WORKSPACE / relative).resolve()
    check(current.is_relative_to(WORKSPACE), f"Source hash path escapes workspace: {name}")
    if current.is_file() and sha256(current) == expected_hash:
        return current, False
    snapshot_directory = output / "source_snapshot"
    walked = snapshot_directory
    check(not walked.is_symlink(), f"Run source snapshot escapes its regular run-owned tree: {name}")
    for part in relative.parts:
        walked = walked / part
        check(not walked.is_symlink(), f"Run source snapshot escapes its regular run-owned tree: {name}")
    snapshot_root = snapshot_directory.resolve()
    candidate = (snapshot_root / relative).resolve()
    check(snapshot_root.is_relative_to(output) and candidate.is_relative_to(snapshot_root)
          and candidate.is_relative_to(output), f"Run source snapshot escapes output: {name}")
    check(candidate.is_file() and sha256(candidate) == expected_hash, f"Source hash changed: {name}")
    return candidate, True


def sparsemm_table(path, layers, heads, query_heads, budget, window, ratio):
    """Independent stdlib implementation of the original float64 static quotas."""
    check(query_heads % heads == 0 and budget > window, "Invalid SparseMM head/budget dimensions")
    check(isinstance(ratio, (int, float)) and not isinstance(ratio, bool)
          and math.isfinite(ratio) and 0 <= ratio <= 1, "Invalid SparseMM ratio")
    scores = [None] * (layers * query_heads)
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        check(fields is not None and len(fields) == len(set(fields))
              and {"layer", "head", "score_mean"} <= set(fields), "SparseMM score CSV columns are invalid")
        for row in reader:
            check(None not in row and all(value is not None for value in row.values()), "Malformed SparseMM score row")
            check(re.fullmatch(r"\d+", row["layer"].strip()) and re.fullmatch(r"\d+", row["head"].strip()),
                  "Invalid SparseMM layer/query-head index")
            layer, head = int(row["layer"]), int(row["head"])
            check(0 <= layer < layers and 0 <= head < query_heads, "SparseMM score index outside model shape")
            index = layer * query_heads + head
            check(scores[index] is None, "Duplicate SparseMM score index")
            try:
                score = float(row["score_mean"])
            except (TypeError, ValueError) as error:
                raise AuditError("Invalid SparseMM visual score") from error
            check(math.isfinite(score) and score >= 0, "Non-finite/negative SparseMM visual score")
            scores[index] = score
    check(all(value is not None for value in scores), "SparseMM scores are missing layer/query-head entries")
    total = sum(scores)
    check(math.isfinite(total) and total > 0, "SparseMM visual scores have no finite positive mass")
    normalized = [value / total for value in scores]
    group = query_heads // heads
    historical = budget - window
    floor = int(historical * ratio)
    remaining = (historical - floor) * layers * query_heads // group
    history = [[round(sum(normalized[(layer * query_heads + head * group):
                                    (layer * query_heads + (head + 1) * group)]) * remaining + floor)
                for head in range(heads)] for layer in range(layers)]
    table = [[value + window for value in row] for row in history]
    allocated, nominal = sum(map(sum, table)), layers * heads * budget
    allocation = {
        "method": "sparsemm", "selector": "static_visual_head_scores_then_attention",
        "shape": [layers, heads], "num_query_heads": query_heads, "gqa_groups": group,
        "score_grouping": "sum_after_global_query_head_normalization", "score_dtype": "float64",
        "ratio": ratio, "budget": budget, "window_size": window,
        "history_base_capacity": historical, "minimum_history_quota": floor,
        "remaining_history_slots": remaining, "head_budgets": table, "history_budgets": history,
        "target_slots": nominal, "allocated_slots": allocated, "rounding_drift_tokens": allocated - nominal,
        "rounding": "upstream_float64_round_to_even", "exact_total_correction": False,
        "uniform_blend": False, "capacity_clip": False, "upstream_commit": UPSTREAM_COMMIT,
    }
    return table, allocation


def _sparsemm_asset(output, protocol, manifest, layers, heads, query_heads, code):
    metadata = protocol.get("sparsemm_asset_metadata")
    check(isinstance(metadata, dict) and metadata == manifest.get("sparsemm_asset_metadata"),
          "SparseMM protocol/manifest asset metadata differ")
    path = Path(protocol.get("sparsemm_score_path", "")).expanduser().resolve()
    check(isinstance(metadata.get("path"), str)
          and path == Path(metadata["path"]).expanduser().resolve(),
          "SparseMM score path differs from its provenance")
    expected = protocol.get("sparsemm_asset_sha256")
    check(expected == manifest.get("sparsemm_asset_sha256") == metadata.get("sha256"),
          "SparseMM asset SHA-256 bindings differ")
    if path.is_relative_to(WORKSPACE):
        name = str(path.relative_to(WORKSPACE))
        check(code.get(name) == expected, "SparseMM score asset is missing from source hash manifest")
        verified, _ = source_file(output, name, expected)
    else:
        check(path.is_file() and sha256(path) == expected, "SparseMM score asset hash changed")
        verified = path
    check(metadata.get("shape") == [layers, heads], "SparseMM asset physical-head shape differs from model")
    check(metadata.get("query_head_shape") == [layers, query_heads], "SparseMM asset query-head shape differs from model")
    # Provenance file bindings are independently checked without importing the
    # runtime assets module (which imports torch through the policy constants).
    resolved = {str(path): verified}
    for field in ("calibration_manifests", "calibration_sources", "asset_files_sha256", "files_sha256"):
        bindings = metadata.get(field, {})
        check(isinstance(bindings, dict), f"Invalid SparseMM {field}")
        for name, expected_hash in bindings.items():
            source = Path(name).resolve()
            if source.is_relative_to(WORKSPACE):
                relative = str(source.relative_to(WORKSPACE))
                check(code.get(relative) == expected_hash, f"SparseMM provenance file is unbound: {name}")
                resolved[str(source)] = source_file(output, relative, expected_hash)[0]
            else:
                check(source.is_file() and sha256(source) == expected_hash, f"SparseMM provenance file changed: {name}")
                resolved[str(source)] = source
    if metadata.get("asset_type") == "visual_head_scores":
        default_csv = WORKSPACE / "hermes/results/myobservations/sparsemm_official_calibration_20261010/calibration/head_scores.csv"
        if path == default_csv.resolve():
            check(expected == "9cc0f47f24a0af4306ce7367806516d119dc4f57df77d064a3eac59c05b69dab"
                  and metadata.get("calibration_manifest_sha256") == "254af2300f1c6ea742a4d05f24d003ee6d9599849454f9a7581b7307d653d52d"
                  and metadata.get("raw_sha256") == "1e02713bc3e582e4a8adde7eaa27188c657361cbc6603463cd67eee224a5cd5f"
                  and metadata.get("protocol_canonical_sha256") == "9ba783db0468b2bd7696b5279ca620b3101ffecacd1d2bb80b5785fb8802d1a4",
                  "SparseMM pinned default calibration fingerprints changed")
        check(metadata.get("num_images") == 1000 and (layers, query_heads, heads) == (28, 28, 4),
              "SparseMM official calibration requires 1000 images and the Qwen2.5 head shape")
        check(metadata.get("static_gqa_reduction") == "sum" and metadata.get("online_gqa_reduction") == "mean"
              and metadata.get("exact_total_correction") is False, "SparseMM calibration quota protocol changed")
        check(metadata.get("official_commit") == UPSTREAM_COMMIT, "SparseMM calibration upstream commit changed")
        check(Path(metadata.get("model_path", "")).resolve() == Path(protocol["model_path"]).resolve()
              and metadata.get("model_config_sha256") == sha256(Path(protocol["model_path"]) / "config.json"),
              "SparseMM calibration belongs to another model/configuration")
        def bound(name, hash_field):
            original = Path(metadata.get(name, "")).resolve()
            check(str(original) in resolved, f"SparseMM calibration evidence is not hash-bound: {name}")
            selected = resolved[str(original)]
            check(sha256(selected) == metadata.get(hash_field), f"SparseMM calibration evidence hash differs: {name}")
            return selected
        calibration = read_json(bound("calibration_manifest_path", "calibration_manifest_sha256"))
        calibration_protocol = read_json(bound("protocol_path", "protocol_file_sha256"))
        raw = read_json(bound("raw_path", "raw_sha256"))
        canonical = hashlib.sha256(json.dumps(calibration_protocol, sort_keys=True, separators=(",", ":"),
                                              ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        check(canonical == metadata.get("protocol_canonical_sha256") == calibration.get("protocol_sha256"),
              "SparseMM canonical calibration protocol SHA-256 differs")
        check(calibration.get("status") == "completed" and calibration.get("count") == 1000
              and calibration.get("requested_count") == 1000 and calibration.get("official_full_calibration") is True
              and calibration.get("last_completed_image_index") == 999, "SparseMM calibration is incomplete")
        check(calibration.get("head_scores_sha256") == expected
              and calibration.get("raw_head_scores_sha256") == metadata.get("raw_sha256"),
              "SparseMM score/raw fingerprints differ from completed calibration")
        check(calibration_protocol.get("official_commit") == UPSTREAM_COMMIT
              and calibration_protocol.get("official_dataset_verified") is True
              and calibration_protocol.get("official_full_calibration_requested") is True,
              "SparseMM calibration official-flow evidence is missing")
        check(isinstance(raw, dict) and set(raw) == {f"{layer}-{head}" for layer in range(layers)
                                                   for head in range(query_heads)}, "SparseMM raw calibration head shape differs")
        csv_means = {}
        with verified.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                csv_means[f"{int(row['layer'])}-{int(row['head'])}"] = float(row["score_mean"])
        for key, samples in raw.items():
            check(isinstance(samples, list) and len(samples) == 1000
                  and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                          and math.isfinite(value) and value >= 0 for value in samples),
                  "SparseMM raw head does not contain 1000 finite calibration scores")
            check(math.isclose(csv_means[key], math.fsum(samples) / 1000, rel_tol=1e-12, abs_tol=1e-15),
                  "SparseMM score_mean differs from raw calibration observations")
    else:
        check(metadata.get("asset_type") == "synthetic_test_scores" and metadata.get("official_calibration") is False,
              "SparseMM asset lacks official calibration provenance")
    table, allocation = sparsemm_table(verified, layers, heads, query_heads,
                                      protocol["budget"], protocol["window_size"], protocol["sparsemm_ratio"])
    check(protocol.get("sparsemm_allocation") == allocation, "SparseMM static allocation differs from original formula")
    if "allocation" in metadata:
        check(metadata["allocation"] == allocation, "SparseMM asset metadata allocation differs from original formula")
    return table, allocation


def _validate_sparsemm(event, state, config, layers, heads, query_heads, table, allocation, expected_asset, label):
    metadata = event.get("layers")
    check(isinstance(metadata, list) and len(metadata) == layers, f"{label}: missing SparseMM layers")
    q, prefix, window = event["input_tokens"], state["prefix"], config["window_size"]
    output, selectors, compressed = [], [], False
    for layer, item in enumerate(metadata):
        name = f"{label}, layer {layer}"
        before = [value + q for value in state["lengths"][layer]]
        recent = [min(window, value - prefix) for value in before]
        histories = [value - prefix - r for value, r in zip(before, recent)]
        bypass = all(value < config["budget"] - window for value in histories)
        proposed = [value - window for value in table[layer]]
        retained = histories if bypass else [min(a, b) for a, b in zip(histories, proposed)]
        after = [prefix + r + h for r, h in zip(recent, retained)]
        did_compress = any(a < b for a, b in zip(after, before))
        nominal, target = heads * (config["budget"] + prefix), sum(table[layer]) + heads * prefix
        fields = {
            "method": "sparsemm", "selector": "static_visual_head_scores_then_attention",
            "budget_scope": "visual", "configured_budget": config["budget"], "layer_idx": layer,
            "base_history_capacity": config["budget"] - window, "head_budgets": table[layer],
            "prefix_tokens_kept": prefix, "prefix_tokens": prefix, "window_size": window,
            "before_head_lengths": before, "head_lengths": after, "recent_counts": recent,
            "proposed_history_counts": proposed, "retained_history_counts": retained,
            "retained_visual_counts": [value - prefix for value in after], "observed_queries": min(q, window),
            "query_len": q, "gqa_groups": query_heads // heads, "gqa_reduce": config["gqa_reduce"],
            "score_precision": config["score_precision"], "pooling": config["pooling"], "kernel_size": config["kernel_size"],
            "target_total_tokens": target, "nominal_total_tokens": nominal, "actual_total_tokens": sum(after),
            "allocation_drift_tokens": sum(after) - nominal, "calibrated_drift_tokens": sum(table[layer]) - heads * config["budget"],
            "bypassed_short_cache": bypass, "compressed": did_compress,
            "rounding": "upstream_float64_round_to_even", "tie_policy": "upstream_torch_sort",
            "upstream_commit": UPSTREAM_COMMIT, "allocation": allocation,
        }
        for key, expected in fields.items():
            check(item.get(key) == expected, f"{name}: invalid SparseMM {key}")
        asset = item.get("asset")
        check(asset == expected_asset, f"{name}: SparseMM cache asset differs from protocol")
        checks = item.get("position_checks")
        check(isinstance(checks, dict) and all(checks.get(key) is True for key in (
            "prefix_preserved", "recent_preserved", "indices_unique", "indices_in_range", "positions_match_kept")),
              f"{name}: SparseMM index/position preservation failed")
        indices = item.get("kept_indices")
        check(isinstance(indices, list) and len(indices) == heads, f"{name}: missing kept indices")
        for head, keep in enumerate(indices):
            integer_list(keep, after[head], f"{name} kept indices")
            check(len(set(keep)) == len(keep) and all(value < before[head] for value in keep), f"{name}: invalid kept indices")
            check(keep[:prefix] == list(range(prefix)), f"{name}: prefix indices were lost")
            if recent[head]:
                check(keep[-recent[head]:] == list(range(before[head] - recent[head], before[head])), f"{name}: recent indices were lost")
            if bypass:
                check(keep == list(range(before[head])), f"{name}: bypass reordered/dropped tokens")
            else:
                middle = keep[prefix:len(keep) - recent[head]] if recent[head] else keep[prefix:]
                check(all(prefix <= value < before[head] - recent[head] for value in middle), f"{name}: historical index is outside history")
        selectors.append({"video_id": event["video_id"], "prefill_index": event["prefill"], "layer_idx": layer,
            "validation_passed": True, "compressed": did_compress, "bypassed_short_cache": bypass,
            "before_head_lengths": before, "head_lengths": after, "head_budgets": table[layer],
            "asset": asset, "allocation": allocation, **checks})
        output.append(after)
        compressed |= did_compress
    return output, compressed, selectors


def bank_nbytes(groups, dim, config, prefix):
    """Compute every FullProtoBank tensor's storage from its public dimensions."""
    visual = config["budget"] - (prefix if config["budget_scope"] == "prompt" else 0)
    modes = integer(config["modes"], "modes", 1)
    check(visual > modes, "Visual budget cannot fit prototype and near memory")
    slots = max(1, (visual - visual // 4) // modes)
    near = visual - slots * modes
    sub = integer(config["subquantizers"], "subquantizers", 1)
    words = integer(config["codewords"], "codewords", 1)
    warm = integer(config["warmup_residuals"], "warmup_residuals", words)
    check(dim % sub == 0 and near > 0, "Invalid PQ head dimension or near budget")
    total = (
        groups * 8 + slots * 8 + sub * 8 + 16  # group/slot/sub indices and I_2
        + 2 * groups * slots * dim * 4  # K/V centers
        + groups * slots * 4  # mass
        + groups * slots * 2 * 4  # spatial mean
        + 2 * groups * slots * 2 * 2 * 4  # covariance and inverse
        + groups * slots * 8  # last
        + 3 * groups * slots * 4  # anchors
        + 2 * groups * slots  # active and ever_used
        + groups * slots * slots  # cached merge pairs
        + 2 * groups * slots * sub * words * 4  # residual histograms
        + groups * slots * 8  # n_res
        + 2 * groups * warm * dim * 4  # warm-up residuals
        + groups * warm * 8 + groups * 8  # warm-slot and warm-count
        + 2 * groups * sub * words * (dim // sub) * 4  # codebooks
        + groups  # code_ready
    )
    return total, visual, slots, near


def expected_memory(lengths, layers, heads, dim, element_bytes, bank_bytes=0, near=0):
    tokens = sum(sum(row) for row in lengths)
    kv = tokens * dim * 2 * element_bytes
    metadata = tokens * 3 * 4 + layers * (heads + heads + 1) * 4 + near * 2 * 4
    raw = layers * heads * near * dim * element_bytes
    return {"persistent_kv_bytes": kv, "bank_bytes": bank_bytes,
            "raw_near_key_bytes": raw, "metadata_bytes": metadata,
            "persistent_total_bytes": kv + bank_bytes + raw + metadata,
            "query_context_kv_bytes": 0}


def validate_memory(actual, expected, label):
    check(isinstance(actual, dict), f"{label}: missing memory accounting")
    for key, value in expected.items():
        check(actual.get(key) == value, f"{label}: {key} differs from physical dimensions")


def _source_path(output, protocol, manifest):
    for key in ("annotation_source", "anno_path", "source_path", "source"):
        if protocol.get(key):
            return Path(protocol[key])
    dataset_file = output / "dataset_audit.json"
    if dataset_file.is_file():
        dataset = read_json(dataset_file)
        if dataset.get("source"):
            return Path(dataset["source"])
    argv = manifest.get("argv", [])
    if "--anno-path" in argv:
        return Path(argv[argv.index("--anno-path") + 1])
    raise AuditError("Annotation source path is missing from protocol/manifest")


def _validate_source_selection(source, annotations):
    original = read_json(source)
    check(isinstance(original, list), "Original annotation source is not a video list")
    originals = {str(video["video_id"]): video for video in original}
    check(len(originals) == len(original), "Original annotation source has duplicate video IDs")
    for selected in annotations:
        video = originals.get(str(selected["video_id"]))
        check(video is not None, "Selected video does not occur in original annotation source")
        for key, value in selected.items():
            if key not in ("video_path", "conversations"):
                check(video.get(key) == value, f"Selected video metadata changed: {key}")
        # Selection may retain an ordered question subset; paths are resolved by
        # prepare_annotations, and may be explicitly remapped to a video root.
        cursor = 0
        originals_qa = video["conversations"]
        for question in selected["conversations"]:
            while cursor < len(originals_qa) and originals_qa[cursor] != question:
                cursor += 1
            check(cursor < len(originals_qa), "Selected question/ground truth differs from original annotation source")
            cursor += 1


def _validate_baseline(event, state, config, layers, heads, query_heads, label):
    q = event["input_tokens"]
    metadata = event.get("layers")
    check(isinstance(metadata, list) and len(metadata) == layers, f"{label}: missing selector layer metadata")
    output, compressed = [], False
    protected = state["prefix"] if config["budget_scope"] == "visual" else 0
    for i, meta in enumerate(metadata):
        name = f"{label}, layer {i}"
        # Uniform uses one configured capacity at every layer and KV head.
        # The legacy layer_capacity helper treats non-Ada methods as PyramidKV.
        capacity = (config["budget"] if config["method"] == "uniform" else
                    layer_capacity(config, i, layers, state["seen"] + state["prefix"], state["prefix"]))
        historical_capacity = max(0, capacity - config["window_size"])
        before = [n + q for n in state["lengths"][i]]
        recent = [min(config["window_size"], n - protected) for n in before]
        histories = [n - protected - r for n, r in zip(before, recent)]
        proposed = integer_list(meta.get("proposed_history_counts"), heads, name)
        retained = integer_list(meta.get("retained_history_counts"), heads, name)
        after = integer_list(meta.get("head_lengths"), heads, name)
        bypass = (all(n < historical_capacity for n in histories) if config["method"] == "adakv"
                  else all(n - protected < capacity for n in before))
        fields = {"method": config["method"], "budget_scope": config["budget_scope"],
                  "configured_budget": config["budget"], "layer_capacity": capacity,
                  "history_capacity_per_head": historical_capacity, "prefix_tokens_kept": protected,
                  "seen_tokens": state["seen"] + state["prefix"],
                  "observed_queries": min(q, config["window_size"]),
                  "gqa_groups": query_heads // heads, "gqa_reduce": config["gqa_reduce"],
                  "score_precision": config["score_precision"], "before_head_lengths": before,
                  "recent_counts": recent, "bypassed_short_cache": bypass,
                  "rounding": "upstream", "tie_policy": "upstream_torch",
                  "zero_mass_normalization": 0, "upstream_commit": UPSTREAM_COMMIT}
        for key, value in fields.items():
            check(meta.get(key) == value, f"{name}: invalid {key}")
        if config["method"] == "uniform":
            check(meta.get("zero_mass_heads") == [], f"{name}: Uniform must not adapt budgets to score mass")
        if bypass:
            check(proposed == histories, f"{name}: bypass changed historical allocation")
        elif config["method"] in ("pyramidkv", "uniform"):
            check(proposed == [historical_capacity] * heads,
                  f"{name}: {'Uniform fixed budget' if config['method'] == 'uniform' else 'Pyramid schedule'} mismatch")
        else:
            validate_adaptive_counts(proposed, histories, historical_capacity, config["floor_alpha"], name)
        check(retained == [min(p, h) for p, h in zip(proposed, histories)], f"{name}: history availability mismatch")
        check(after == [protected + r + h for r, h in zip(recent, retained)], f"{name}: retained length mismatch")
        did_compress = any(a < b for a, b in zip(after, before))
        check(meta.get("compressed") == did_compress, f"{name}: incorrect compression flag")
        target, actual = heads * (capacity + protected), sum(after)
        check(meta.get("target_total_tokens") == target and meta.get("actual_total_tokens") == actual,
              f"{name}: total-token accounting mismatch")
        check(meta.get("allocation_drift_tokens") == actual - target and meta.get("actual_mean_tokens") == actual / heads,
              f"{name}: integer allocation drift mismatch")
        output.append(after)
        compressed |= did_compress
    return output, compressed


def _lmms_results(output, summary, accuracy, require):
    paths = []
    for name in ("lmms_results_path", "lmms_result_file", "lmms_results_file"):
        if summary.get(name):
            path = Path(summary[name])
            paths.append(path if path.is_absolute() else output / path)
    paths.extend(sorted((output / "lmms").rglob("*results*.json")))
    checked = []
    for path in dict.fromkeys(paths):
        document = read_json(path)
        result = document.get("results", {}).get("streamingbench_local", {})
        if not result:
            continue
        metric = result.get("mcqa_accuracy,none", result.get("mcqa_accuracy"))
        check(isinstance(metric, (int, float)) and not isinstance(metric, bool), "lmms-eval weighted accuracy is missing")
        check(math.isclose(metric, accuracy, rel_tol=0, abs_tol=1e-8), "lmms-eval accuracy differs from question-weighted CSV accuracy")
        checked.append(str(path.resolve()))
    check(checked or not require, "No native lmms-eval streamingbench_local result JSON found")
    return checked


def audit_run(output_dir, *, min_compression_events_per_video=2, require_lmms=True):
    out = Path(output_dir).resolve()
    integer(min_compression_events_per_video, "minimum compression events")
    protocol = read_json(out / "protocol.json")
    manifest = read_json(out / "run_manifest.json")
    annotations = read_json(out / "annotations.json")
    summary = read_json(out / "summary.json")
    runtime = read_json(out / "runtime_checks.json")
    check((out / "run.status").is_file() and (out / "run.status").read_text().strip() == "complete", "Run is not marked complete")
    check(runtime.get("status") == "complete" and runtime.get("failures") == [], "Runtime validation did not complete cleanly")
    check(isinstance(annotations, list) and annotations, "Selected annotations are empty")
    vids = [str(video["video_id"]) for video in annotations]
    check(len(vids) == len(set(vids)), "Duplicate annotation video IDs")
    total_questions = sum(len(video["conversations"]) for video in annotations)
    canonical = json.dumps(annotations, sort_keys=True, ensure_ascii=False).encode()
    check(hashlib.sha256(canonical).hexdigest() == protocol.get("subset_sha256"), "Annotation subset hash mismatch")
    source = _source_path(out, protocol, manifest)
    check(source.is_file() and sha256(source) == protocol.get("source_sha256"), "Original annotation source hash mismatch")
    _validate_source_selection(source, annotations)
    check(protocol.get("upstream_commit") == manifest.get("upstream_commit") == UPSTREAM_COMMIT, "Upstream commit mismatch")
    code = manifest.get("code_sha256")
    check(isinstance(code, dict) and code, "Source manifest has no code hashes")
    check(set(REQUIRED_CODE_FILES) <= code.keys(), "Source manifest omits required inference files")
    snapshot_files = []
    for name, expected_hash in code.items():
        _, historical = source_file(out, name, expected_hash)
        if historical:
            snapshot_files.append(name)
    config = {key: protocol[key] for key in CONFIG_KEYS}
    # Shared configurations now also record LU options for other methods.
    # Older saved protocols do not have these optional fields.
    for key in ("lukv_budget_path", "lukv_profile_path", "lukv_history", "lukv_selector",
                "sparsemm_score_path", "sparsemm_ratio"):
        if key in protocol:
            config[key] = protocol[key]
    check(config["method"] in ("protokv", "adakv", "pyramidkv", "sparsemm", "uniform"), "Unsupported method")
    check(config["budget_scope"] in ("visual", "prompt"), "Unsupported budget scope")
    model_path = Path(protocol["model_path"])
    model = read_json(model_path / "config.json")
    model_hashes = manifest.get("model_files_sha256", manifest.get("model_files", {}))
    check(isinstance(model_hashes, dict) and "config.json" in model_hashes, "Model config hash is missing")
    for name, expected_hash in model_hashes.items():
        check((model_path / name).is_file() and sha256(model_path / name) == expected_hash, f"Model file hash changed: {name}")
    text = model.get("text_config", model)
    layers = integer(text["num_hidden_layers"], "num layers", 1)
    heads = integer(text["num_key_value_heads"], "num KV heads", 1)
    query_heads = integer(text["num_attention_heads"], "num query heads", 1)
    dim = integer(text.get("head_dim", text["hidden_size"] // query_heads), "head dim", 1)
    check(query_heads % heads == 0, "Nonintegral GQA ratio")
    check(config["method"] != "pyramidkv" or layers > 1, "Pyramid schedule requires more than one layer")
    if config["method"] == "uniform":
        check(config["budget"] > config["window_size"] > 0, "Uniform requires budget > window_size > 0")
        check(config["score_precision"] == "float32" and config["pooling"] == "avgpool"
              and config["gqa_reduce"] == "mean", "Uniform validation requires FP32/mean/avgpool scoring")
        uniform_fields = {
            "uniform_budget_mode": "uniform_per_layer_per_kv_head",
            "uniform_selector": "attention_topk", "uniform_offline_assets": False,
            "uniform_scoring": "FP32 causal Q-to-K attention, avgpool, GQA mean",
            "uniform_probe_count": 0,
        }
        for key, value in uniform_fields.items():
            check(protocol.get(key) == value, f"Uniform protocol mismatch: {key}")
    sparsemm_budgets = sparsemm_allocation = sparsemm_layer_asset = None
    if config["method"] == "sparsemm":
        check(config["budget_scope"] == "visual" and config["score_precision"] == "float32"
              and config["gqa_reduce"] == "mean" and config["pooling"] == "avgpool",
              "SparseMM native validation requires visual/FP32/mean/avgpool scoring")
        required_sparsemm = {f"sparsemm_native/{name}" for name in ("policy.py", "cache.py", "streaming.py", "assets.py")}
        check(required_sparsemm <= code.keys(), "SparseMM source hash manifest omits native method files")
        sparsemm_budgets, sparsemm_allocation = _sparsemm_asset(out, protocol, manifest, layers, heads, query_heads, code)
        sparsemm_layer_asset = {key: value for key, value in protocol["sparsemm_asset_metadata"].items() if key != "allocation"}
        if sparsemm_layer_asset.get("asset_type") == "synthetic_test_scores":
            sparsemm_layer_asset = {key: sparsemm_layer_asset[key] for key in ("path", "sha256", "shape", "query_head_shape")}
    element = {"float16": 2, "bfloat16": 2, "float32": 4}.get(protocol["dtype"])
    check(element is not None, "Unsupported dtype")
    try:
        rows, completed = load_resume(out / "1_0.csv", annotations, require_complete=True)
    except (RuntimeError, ValueError, KeyError) as error:
        raise AuditError(f"CSV validation failed: {error}") from error
    check(completed == len(annotations) and len(rows) == total_questions, "Result CSV is incomplete")
    check((out / "results.csv").is_file() and (out / "results.csv").read_bytes() == (out / "1_0.csv").read_bytes(), "Result CSV copies differ")
    for row, video in zip(rows, [video for video in annotations for _ in video["conversations"]]):
        check(row.get("duration_category", "") == (video.get("duration_category", "") or ""), "CSV duration metadata mismatch")
    accuracy = accuracy_statistics(rows)
    check(summary.get("videos") == len(annotations) and summary.get("questions") == len(rows), "Summary count mismatch")
    check(summary.get("correct") == accuracy["correct"], "Summary correct-answer count mismatch")
    check(math.isclose(summary.get("accuracy_percent", -1), accuracy["accuracy_percent"], rel_tol=0, abs_tol=1e-8), "Summary accuracy mismatch")
    for name in ("method", "budget", "budget_scope"):
        check(summary.get(name) == config[name], f"Summary {name} mismatch")
    if summary.get("is_first250_complete"):
        check(len(annotations) == 250 and len(rows) == 1255, "Subset is incorrectly labeled a complete first250 run")
    trace_path = out / "cache_trace.jsonl"
    check(trace_path.is_file(), "Cache trace missing")
    trace = []
    for number, line in enumerate(trace_path.read_text().splitlines(), 1):
        check(bool(line.strip()), f"Empty trace line {number}")
        event = json.loads(line)
        finite(event, f"trace line {number}")
        trace.append(event)
    check(trace, "Cache trace is empty")
    states, qa_checks, selector_checks = [], [], []
    state = None

    def finish_state():
        if state is None:
            return
        expected = len(annotations[len(states)]["conversations"])
        check(state["questions"] == expected, f"Video {state['video_id']}: question trace count mismatch")
        check(state["compressions"] >= min_compression_events_per_video, f"Video {state['video_id']}: insufficient repeated compression")
        states.append({key: value for key, value in state.items() if key not in ("lengths", "memory", "last_answer_prefill")})

    for number, event in enumerate(trace, 1):
        kind = event.get("event")
        label = f"Trace line {number} ({kind})"
        vid = str(event.get("video_id"))
        if kind == "init":
            finish_state()
            check(len(states) < len(vids) and vid == vids[len(states)], f"{label}: unexpected video/reset order")
            prefix = integer(event.get("text_tokens"), "text tokens", 1)
            check(event.get("config") == config and event.get("num_layers") == layers and event.get("num_kv_heads") == heads, f"{label}: initial config/architecture mismatch")
            lengths = [[prefix] * heads for _ in range(layers)]
            bank_bytes, visual, slots, near = (bank_nbytes(layers * heads, dim, config, prefix)
                                               if config["method"] == "protokv" else (0, 0, 0, 0))
            memory = expected_memory(lengths, layers, heads, dim, element, bank_bytes)
            validate_memory(event.get("memory"), memory, label)
            state = {"video_id": vid, "prefix": prefix, "prefills": 0, "compressions": 0,
                     "questions": 0, "seen": 0, "absorbed": 0, "active": 0,
                     "bank_bytes": bank_bytes, "visual_budget": visual, "prototype_slots": slots,
                     "near_capacity": near, "lengths": lengths, "memory": memory,
                     "last_answer_prefill": None, "cursor": prefix, "native_attention_calls": layers}
            if config["method"] == "sparsemm":
                state["sparsemm_asset_sha256"] = protocol["sparsemm_asset_sha256"]
            continue
        check(state is not None and vid == state["video_id"], f"{label}: missing/mismatched video init")
        if kind == "video_prefill":
            q = integer(event.get("input_tokens"), "video input tokens", 1)
            state["prefills"] += 1
            state["seen"] += q
            check(event.get("prefill") == state["prefills"], f"{label}: prefill counter did not reset/increment")
            meta = event.get("method_metadata")
            check(isinstance(meta, dict), f"{label}: method metadata missing")
            if config["method"] == "protokv":
                near = min(state["seen"], state["near_capacity"])
                absorbed = state["seen"] - near
                outgoing = absorbed - state["absorbed"]
                expected_active = layers * heads * min(absorbed, state["prototype_slots"])
                expected_pq = layers * heads if absorbed >= state["prototype_slots"] + config["warmup_residuals"] else 0
                expected_meta = {
                    "absorbed_tokens": absorbed, "outgoing_tokens": outgoing,
                    "active_prototypes": expected_active, "pq_ready_groups": expected_pq,
                    "total_groups": layers * heads, "near_tokens": near,
                    "near_capacity": state["near_capacity"], "prototype_capacity": state["prototype_slots"],
                    "pseudo_tokens_per_head": state["prototype_slots"] * config["modes"],
                    "visual_budget": state["visual_budget"],
                }
                for key, value in expected_meta.items():
                    check(meta.get(key) == value, f"{label}: ProtoKV {key} mismatch")
                check(event.get("bank_total_absorbed") == absorbed and event.get("active_prototypes") == expected_active
                      and event.get("pq_ready_groups") == expected_pq, f"{label}: bank trace counters mismatch")
                state["absorbed"], state["active"] = absorbed, expected_active
                lengths = [[state["prefix"] + near] * heads for _ in range(layers)]
                compressed = outgoing > 0
            else:
                if config["method"] == "sparsemm":
                    lengths, compressed, selectors = _validate_sparsemm(event, state, config, layers, heads, query_heads,
                                                                        sparsemm_budgets, sparsemm_allocation, sparsemm_layer_asset, label)
                    selector_checks.extend(selectors)
                else:
                    lengths, compressed = _validate_baseline(event, state, config, layers, heads, query_heads, label)
                near = 0
                check(event.get("bank_total_absorbed") == 0 and event.get("active_prototypes") == 0
                      and event.get("pq_ready_groups") == 0, f"{label}: baseline unexpectedly has prototype state")
            state["compressions"] += int(compressed)
            check(meta.get("compressed") == compressed and meta.get("compressions") == state["compressions"]
                  and meta.get("prefill") == state["prefills"] and meta.get("input_tokens") == q
                  and meta.get("logical_tokens") == state["prefix"] + state["seen"], f"{label}: committed method counters mismatch")
            actual_lengths = head_lengths(event.get("head_lengths"), layers, heads, label)
            check(actual_lengths == lengths and meta.get("head_lengths") == lengths, f"{label}: physical near lengths mismatch")
            memory = expected_memory(lengths, layers, heads, dim, element, state["bank_bytes"], near)
            validate_memory(event.get("memory"), memory, label)
            validate_memory(meta, memory, f"{label} method metadata")
            check(event.get("kv_bytes") == memory["persistent_kv_bytes"], f"{label}: physical KV bytes mismatch")
            native = integer(event.get("native_attention_calls"), "native attention calls", layers)
            check(native >= state["native_attention_calls"] + layers and native % layers == 0, f"{label}: native layer-call counter mismatch")
            state.update(lengths=lengths, memory=memory, native_attention_calls=native)
        elif kind == "answer_restored":
            check(head_lengths(event.get("head_lengths"), layers, heads, label) == state["lengths"]
                  and event.get("kv_bytes") == state["memory"]["persistent_kv_bytes"], f"{label}: QA changed persistent cache dimensions")
            validate_memory(event.get("memory"), state["memory"], label)
            qa = event.get("query_check")
            check(isinstance(qa, dict), f"{label}: query instrumentation missing")
            check(str(qa.get("video_id")) == vid and qa.get("question_index") == state["questions"], f"{label}: question identity/order mismatch")
            check(qa.get("returned") is True and qa.get("restored") is True, f"{label}: query did not return and restore")
            check(qa.get("first_answer_native_layers") == layers, f"{label}: first answer skipped native decoder layers")
            expected_pseudo = layers if config["method"] == "protokv" and state["active"] else 0
            check(qa.get("first_answer_pseudo_layers") == expected_pseudo, f"{label}: first answer did not use expected prototype memory")
            query_memory = dict(state["memory"])
            pseudo = state["prototype_slots"] * config["modes"] if config["method"] == "protokv" else 0
            query_memory["query_context_kv_bytes"] = state["memory"]["persistent_kv_bytes"] + layers * heads * pseudo * dim * 2 * element
            validate_memory(qa.get("query_memory"), query_memory, f"{label} query view")
            cursor = integer(event.get("next_position"), "query cursor", state["prefix"])
            check(cursor >= state["cursor"], f"{label}: position cursor moved backward")
            if state["last_answer_prefill"] == state["prefills"]:
                check(cursor == state["cursor"], f"{label}: repeated QA advanced the video cursor")
            state["cursor"] = cursor
            state["last_answer_prefill"] = state["prefills"]
            state["questions"] += 1
            qa_checks.append(qa)
        else:
            raise AuditError(f"{label}: unknown event")
    finish_state()
    check(len(states) == len(annotations) and len(qa_checks) == len(rows), "Trace selection/question count mismatch")
    check(runtime.get("qa_restore_checks") == len(rows) and runtime.get("qa_checks") == qa_checks, "Runtime and trace query checks disagree")
    if config["method"] == "sparsemm":
        check(runtime.get("probe_checks") == [] and runtime.get("probe_restore_checks") == 0,
              "SparseMM must not execute HERMES probes")
        check(runtime.get("selector_checks") == selector_checks, "SparseMM runtime selector checks differ from every native trace layer")
    if config["method"] == "uniform":
        check(runtime.get("probe_checks", []) == [] and runtime.get("probe_restore_checks", 0) == 0,
              "Uniform must not execute HERMES probes")
    prefills = sum(state["prefills"] for state in states)
    hidden_checks = integer(runtime.get("finite_hidden_checks"), "finite hidden checks", len(annotations) + prefills + len(rows))
    logits_checks = integer(runtime.get("finite_logits_checks"), "finite logits checks", len(rows))
    check(hidden_checks == len(annotations) + prefills + logits_checks, "Finite hidden/logits forward counts are inconsistent")
    lmms_files = _lmms_results(out, summary, accuracy["accuracy_percent"], require_lmms)
    by_task = defaultdict(list)
    for row in rows:
        by_task[row.get("task") or "(unspecified)"].append(row)
    return {
        "passed": True, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(out), "method": config["method"], "budget": config["budget"],
        "budget_scope": config["budget_scope"], "videos": len(annotations), "questions": len(rows),
        "overall": accuracy, "task_accuracy": {key: accuracy_statistics(value) for key, value in sorted(by_task.items())},
        "trace": {"events": len(trace), "prefills": prefills,
                  "compression_events": sum(state["compressions"] for state in states),
                  "qa_restore_checks": len(qa_checks), "finite_hidden_checks": hidden_checks,
                  "finite_logits_checks": logits_checks},
        "architecture": {"num_layers": layers, "num_kv_heads": heads,
                         "num_query_heads": query_heads, "head_dim": dim, "element_bytes": element},
        "per_video": states, "lmms_results_checked": lmms_files,
        "source_snapshot_files": sorted(snapshot_files),
        "source_validation": "current_or_run_owned_snapshot" if snapshot_files else "current",
        "sparsemm_allocation": sparsemm_allocation,
        "uniform_allocation": ({"budget_per_layer_per_kv_head": config["budget"],
                                "history_per_layer_per_kv_head": config["budget"] - config["window_size"],
                                "protected_recent_tokens": config["window_size"],
                                "target_visual_slots": layers * heads * config["budget"]}
                               if config["method"] == "uniform" and config["budget_scope"] == "visual" else None),
        "sparsemm_calibration_verified": (protocol.get("sparsemm_asset_metadata", {}).get("asset_type") == "visual_head_scores"
                                           if config["method"] == "sparsemm" else None),
        "checks": ["complete ordered CSV with annotation ground truth and predictions",
                   "matching runtime source (current or own run snapshot), model and annotation hashes",
                   "per-video resets and repeated online compression",
                   "method budget transitions and all persistent KV/bank/raw/metadata bytes",
                   "first answer uses every native decoder layer and active ProtoKV memory",
                   "query restoration and finite decoder/logit instrumentation",
                   "question-weighted native lmms-eval accuracy matches CSV"],
        "limitations": ["Trace stores counts and runtime check results, not independently replayable tensor contents.",
                        "AdaKV checks establish feasible integer allocations rather than recomputing attention rankings.",
                        "SparseMM index/position checks do not independently replay attention ranks or tensor values.",
                        "Selected stability tests do not establish full benchmark accuracy or official numerical equivalence."],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-compression-events-per-video", type=int, default=2)
    parser.add_argument("--allow-missing-lmms-results", action="store_true")
    args = parser.parse_args(argv)
    destination = args.output or args.output_dir / "audit_report.json"
    try:
        result = audit_run(args.output_dir,
                           min_compression_events_per_video=args.min_compression_events_per_video,
                           require_lmms=not args.allow_missing_lmms_results)
    except (AuditError, KeyError, ValueError, TypeError, OSError) as error:
        result = {"passed": False, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                  "output_dir": str(args.output_dir.resolve()), "error": str(error)}
        write_json(destination, result)
        print(json.dumps(result, ensure_ascii=False))
        return 1
    write_json(destination, result)
    print(json.dumps({"passed": True, "videos": result["videos"], "questions": result["questions"],
                      "compression_events": result["trace"]["compression_events"],
                      "report": str(destination.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
