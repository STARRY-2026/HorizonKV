"""Independent stdlib audit of native SparseMM LU-KV validation outputs.

This checks LU profile lookup or historical fixed capacities, direct-attention
selection, prefix/recent protection, QA transactions, calibration evidence,
data provenance, and question-weighted accuracy.
Numeric attention parity is covered separately by tiny native CPU tests.
"""
from __future__ import annotations

import argparse
import ast
from bisect import bisect_left
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct
import zipfile

from sparsemm_baselines.audit_extended import (
    AuditError, accuracy_statistics, check, finite, head_lengths, integer, read_json,
)
from sparsemm_baselines.run_streamingbench import load_resume, sha256, write_json
from sparsemm_protokv.audit import (
    UPSTREAM_COMMIT, WORKSPACE, _lmms_results, _source_path, _validate_source_selection,
    expected_memory, validate_memory,
)


DEFAULT_BUDGET_PATH = WORKSPACE / (
    "hermes/results/myobservations/lu_kv_balanced100v_20260915/traces/"
    "lu_raw_balanced100v_blend05_clip075_150_budget.csv"
)
DEFAULT_BUDGET_SHA256 = "cd3d6e0f42d0c35ffa0b4aa84635b9c35443ebb47bac6b8b6ca89bad4d7a65fe"
DEFAULT_PROFILE_PATH = WORKSPACE / "sparsemm_lukv/assets/attention_native10v_20261014/profile.json"
REQUIRED_CODE_FILES = (
    "sparsemm_lukv/cache.py", "sparsemm_lukv/policy.py", "sparsemm_lukv/streaming.py",
    "sparsemm_protokv/config.py", "sparsemm_protokv/cache.py", "sparsemm_protokv/qwen2_5.py",
    "sparsemm_protokv/streaming.py", "sparsemm_protokv/monkeypatch.py",
    "sparsemm_protokv/models/qwen2_5_vl_stream.py", "sparsemm_protokv/run_streamingbench.py",
    "sparsemm_protokv/tasks/streamingbench_local.py", "sparsemm_baselines/cache.py",
    "sparsemm_baselines/adapter.py", "sparsemm_baselines/core.py",
)


def read_budget_table(path, layers, heads, budget, window):
    """Parse the asset independently from the torch-based runtime loader."""
    table = [[None] * heads for _ in range(layers)]
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        check(reader.fieldnames is not None and len(reader.fieldnames) == 3
              and set(reader.fieldnames) == {"layer", "kv_head", "budget"},
              "LU asset has invalid columns")
        for row in reader:
            check(None not in row and all(value is not None for value in row.values()),
                  "LU asset has malformed rows")
            values = []
            for key in ("layer", "kv_head", "budget"):
                value = row[key].strip()
                check(value.lstrip("+-").isdigit(), f"LU asset {key} is not an integer")
                values.append(int(value))
            layer, head, capacity = values
            check(0 <= layer < layers and 0 <= head < heads, "LU asset index is out of range")
            check(table[layer][head] is None, "LU asset has duplicate head capacity")
            check(max(window, math.ceil(.75 * budget)) <= capacity <= math.floor(1.5 * budget),
                  "LU asset capacity is outside constrained bounds")
            table[layer][head] = capacity
    check(all(value is not None for row in table for value in row), "LU asset is missing head capacities")
    check(sum(map(sum, table)) == layers * heads * budget, "LU asset global budget is not exact")
    return table


def read_profile(path, layers, heads, window):
    """Validate a profile without importing numpy, torch, or runtime policy."""
    profile = read_json(path)
    required = {"schema": "sparsemm_lukv.history_profile", "version": 1,
                "selector": "sparsemm_attention", "profile_domain": "unprotected_history",
                "budget_scope": "visual", "window_size": window,
                "num_layers": layers, "num_kv_heads": heads}
    check(isinstance(profile, dict) and all(profile.get(key) == value for key, value in required.items()),
          "LU profile schema, selector, architecture, or protection window mismatch")
    grid = profile.get("retention_grid")
    check(isinstance(grid, list) and len(grid) >= 2
          and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                  and math.isfinite(value) for value in grid)
          and grid[0] == 0 and grid[-1] == 1
          and all(left < right for left, right in zip(grid, grid[1:])),
          "LU profile retention grid must strictly increase from zero to one")
    ratios = profile.get("history_retention")
    check(isinstance(ratios, list) and len(ratios) == len(grid), "LU profile ratio grid count mismatch")
    for point, allocation in enumerate(ratios):
        check(isinstance(allocation, list) and len(allocation) == layers,
              "LU profile ratio layer count mismatch")
        for row in allocation:
            check(isinstance(row, list) and len(row) == heads,
                  "LU profile ratio head count mismatch")
            for value in row:
                check(isinstance(value, (int, float)) and not isinstance(value, bool)
                      and math.isfinite(value) and 0 <= value <= 1,
                      "LU profile history retention must be a finite ratio")
                check(point not in (0, len(grid) - 1) or value == (0 if point == 0 else 1),
                      "LU profile endpoints must retain no/all history")
    check(all(ratios[point][layer][head] <= ratios[point + 1][layer][head]
              for point in range(len(grid) - 1) for layer in range(layers) for head in range(heads)),
          "LU profile head retention must not decrease across the grid")
    check(profile.get("online_utility") is False, "LU profile must not use online utility")
    return profile


def allocate_profile_independent(profile, seen, budget, window):
    """Recompute interpolation, box/sum projection and exact rounding in stdlib."""
    integer(seen, "uncompressed visual length")
    integer(budget, "visual budget", window)
    layers, heads = profile["num_layers"], profile["num_kv_heads"]
    total_heads = layers * heads
    target = total_heads * min(seen, budget)
    rho = min(1., (budget - window) / (seen - window)) if seen > window else None
    if seen <= budget:
        capacities = [seen] * total_heads
        endpoints, weight = ([1., 1.] if rho is not None else None), 0.
    else:
        grid, ratios = profile["retention_grid"], profile["history_retention"]
        upper = bisect_left(grid, rho)
        lower = upper if grid[upper] == rho else upper - 1
        weight = 0. if lower == upper else (rho - grid[lower]) / (grid[upper] - grid[lower])
        raw = [window + (ratios[lower][layer][head] * (1. - weight)
                         + ratios[upper][layer][head] * weight) * (seen - window)
               for layer in range(layers) for head in range(heads)]
        left, right = window - max(raw), seen - min(raw)
        for _ in range(80):
            shift = (left + right) / 2.
            projected = [min(seen, max(window, value + shift)) for value in raw]
            if math.fsum(projected) < target:
                left = shift
            else:
                right = shift
        projected = [min(seen, max(window, value + (left + right) / 2.)) for value in raw]
        capacities = [math.floor(value) for value in projected]
        missing = target - sum(capacities)
        eligible = [index for index, value in enumerate(capacities) if value < seen]
        eligible.sort(key=lambda index: (-(projected[index] - capacities[index]), index))
        check(0 <= missing <= len(eligible), "Independent LU integer projection failed")
        for index in eligible[:missing]:
            capacities[index] += 1
        endpoints = [float(grid[lower]), float(grid[upper])]
    rows = [capacities[layer * heads:(layer + 1) * heads] for layer in range(layers)]
    check(sum(capacities) == target and all(min(window, seen) <= value <= seen for value in capacities),
          "Independent LU profile allocation violates its visual budget/window")
    metadata = {
        "source_T": seen, "rho": rho, "grid_endpoints": endpoints,
        "interpolation_weight": weight, "target_slots": target,
        "allocated_slots": sum(capacities), "budget": budget, "window_size": window,
        "protected_tokens_per_head": min(window, seen), "profile_domain": "unprotected_history",
        "budget_scope": "visual", "profile_bypassed": seen <= budget, "no_online_utility": True,
        "actual_occupancy_may_be_lower": True,
        "projection": "bounded_euclidean_shift_largest_remainder", "head_capacities": rows,
    }
    return rows, metadata


def validate_observation_arrays(path, layers, heads, visual_tokens, window):
    """Read only ZIP/NPY headers; large calibration tensors stay off memory."""
    expected = {
        "ranking.npy": ((layers * heads, visual_tokens - window), "integer"),
        "score.npy": ((layers * heads, visual_tokens - window), "float32"),
        "utility_attention.npy": ((layers * heads, visual_tokens), "float32"),
        "utility_projected.npy": ((layers * heads, visual_tokens), "float32"),
    }
    try:
        with zipfile.ZipFile(path) as archive:
            check(set(expected) <= set(archive.namelist()), "LU calibration observation omits attention/utility arrays")
            for name, (shape, kind) in expected.items():
                with archive.open(name) as handle:
                    check(handle.read(6) == b"\x93NUMPY", "LU observation array is not a NPY tensor")
                    version = tuple(handle.read(2))
                    check(version in ((1, 0), (2, 0), (3, 0)), "Unsupported LU observation NPY version")
                    size_bytes = 2 if version == (1, 0) else 4
                    header_size = struct.unpack("<H" if size_bytes == 2 else "<I", handle.read(size_bytes))[0]
                    check(header_size <= 65536, "LU observation NPY header is unreasonably large")
                    header = ast.literal_eval(handle.read(header_size).decode("utf-8" if version == (3, 0) else "latin1"))
                check(isinstance(header, dict) and header.get("shape") == shape
                      and header.get("fortran_order") is False, "LU calibration observation tensor shape/order mismatch")
                dtype = header.get("descr")
                allowed = ("<i8", "<i4", "=i8", "=i4") if kind == "integer" else ("<f4", "=f4")
                check(dtype in allowed, "LU calibration observation tensor dtype mismatch")
    except (zipfile.BadZipFile, KeyError, ValueError, SyntaxError, struct.error, UnicodeError) as error:
        raise AuditError(f"Invalid LU calibration observation archive: {error}") from error


def _video_content_evidence(videos, label):
    """Fingerprint actual transfer videos without loading complete files into RAM."""
    records = []
    for video in videos:
        path = Path(video["video_path"])
        check(path.is_absolute() and path.is_file(), f"LU {label} transfer video is missing: {path}")
        path = path.resolve()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        records.append({"video_id": str(video["video_id"]), "path": str(path),
                        "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    return records


def validate_profile_calibration(profile, asset_meta, protocol, model_path, model_hashes, evaluation_source,
                                 *, allow_calibration_transfer=False, evaluation_videos=None,
                                 transfer_evidence=None):
    """Validate calibration inputs/evidence independently of the asset loader."""
    calibration = profile.get("calibration")
    check(isinstance(calibration, dict) and calibration.get("selector") == "sparsemm_attention"
          and calibration.get("utility_mode") in ("projected", "attention"),
          "LU profile has no declared attention calibration objective")
    scorer = calibration.get("token_window_protocol", {})
    expected_scorer = {"window_size": protocol["window_size"], "kernel_size": protocol["kernel_size"],
                       "pooling": protocol["pooling"], "gqa_reduce": protocol["gqa_reduce"],
                       "score_precision": "float32",
                       "native_resolution": protocol["native_resolution"]}
    check(isinstance(scorer, dict) and all(scorer.get(key) == value for key, value in expected_scorer.items()),
          "LU calibration attention/window protocol differs from online scoring")
    if "sample_fps" in scorer:
        check(scorer["sample_fps"] == protocol.get("sample_fps", .5), "LU calibration sample FPS mismatch")
    sources = calibration.get("source_sha256")
    check(isinstance(sources, dict) and len(sources) >= 2
          and sources == asset_meta.get("calibration_sources")
          and sources == protocol.get("lukv_calibration_sources"),
          "LU profile calibration annotation sources are missing or disagree")
    for name, expected in sources.items():
        path = Path(name)
        check(path.is_absolute() and path.is_file() and sha256(path) == expected,
              f"LU profile calibration source hash changed: {name}")
    exclude_path = Path(calibration.get("exclude_annotation", ""))
    check(exclude_path.is_absolute() and str(exclude_path) in sources,
          "LU profile exclusion annotation is not source-bound")
    selected = calibration.get("selected_videos")
    check(isinstance(selected, list) and 0 < len(selected) <= 10, "LU calibration is not bounded to ten videos")
    selected_ids = [str(video["video_id"]) for video in selected]
    selected_names = [Path(video["video_path"]).name for video in selected]
    selected_paths = {Path(video["video_path"]).resolve() for video in selected}
    check(len(set(selected_ids)) == len(selected_ids) and len(set(selected_names)) == len(selected_names),
          "LU calibration has duplicate IDs or video basenames")
    excluded = read_json(exclude_path)
    check(isinstance(excluded, list), "LU calibration exclusion source is not a video list")
    excluded_ids = {str(video["video_id"]) for video in excluded}
    excluded_names = {Path(video["video_path"]).name for video in excluded}
    excluded_paths = {Path(video["video_path"]).resolve() for video in excluded}
    evaluation = read_json(evaluation_source)
    check(isinstance(evaluation, list) and evaluation, "LU evaluation source is not a nonempty video list")
    evaluation_ids = {str(video["video_id"]) for video in evaluation}
    evaluation_names = {Path(video["video_path"]).name for video in evaluation}
    evaluation_paths = {Path(video["video_path"]).resolve() for video in evaluation}
    check(not set(selected_ids) & excluded_ids and not set(selected_names) & excluded_names
          and not selected_paths & excluded_paths
          and not set(selected_ids) & evaluation_ids and not set(selected_names) & evaluation_names
          and not selected_paths & evaluation_paths,
          "LU profile calibration overlaps evaluation videos")
    exclusion_covers_evaluation = evaluation_ids <= excluded_ids and evaluation_names <= excluded_names
    check(exclusion_covers_evaluation or allow_calibration_transfer,
          "LU calibration exclusion source omits evaluation videos; cross-dataset transfer requires "
          "explicit --allow-calibration-transfer")
    if not exclusion_covers_evaluation:
        actual_evaluation = evaluation if evaluation_videos is None else evaluation_videos
        check(isinstance(actual_evaluation, list) and actual_evaluation,
              "LU calibration transfer has no selected evaluation videos")
        # audit_run already binds its selected annotations to the source. Keep
        # this public helper equally strict when called independently.
        _validate_source_selection(evaluation_source, actual_evaluation)
        check(not set(selected_ids) & {str(video["video_id"]) for video in actual_evaluation}
              and not set(selected_names) & {Path(video["video_path"]).name for video in actual_evaluation}
              and not selected_paths & {Path(video["video_path"]).resolve() for video in actual_evaluation},
              "LU profile calibration overlaps selected evaluation videos")
        calibration_files = _video_content_evidence(selected, "calibration")
        evaluation_files = _video_content_evidence(actual_evaluation, "evaluation")
        check(not {record["sha256"] for record in calibration_files}
              & {record["sha256"] for record in evaluation_files},
              "LU calibration transfer overlaps evaluation video content (SHA256)")
        if transfer_evidence is not None:
            transfer_evidence.update({
                "enabled": True, "explicit_opt_in": True, "mode": "cross_dataset_profile_transfer",
                "original_exclusion_covers_evaluation": False,
                "no_evaluation_recalibration": True, "content_overlap": 0,
                "calibration_annotation_sources": dict(sources),
                "original_exclusion_annotation": {"path": str(exclude_path),
                                                  "sha256": sources[str(exclude_path)]},
                "evaluation_annotation_source": {"path": str(Path(evaluation_source).resolve()),
                                                 "sha256": sha256(evaluation_source)},
                "calibration_model_path": str(model_path.resolve()),
                "calibration_model_files_sha256": calibration.get("model_files_sha256"),
                "calibration_videos": calibration_files, "evaluation_videos": evaluation_files,
            })
    source_candidates = [Path(name) for name in sources if Path(name) != exclude_path]
    matched_source = False
    for path in source_candidates:
        try:
            _validate_source_selection(path, selected)
        except AuditError:
            continue
        matched_source = True
        break
    check(matched_source, "LU calibration selected original questions differ from their bound source")
    selected_hash = hashlib.sha256(json.dumps(selected, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    check(calibration.get("selected_sha256") == selected_hash, "LU calibration selected subset hash changed")
    expected_observations = {}
    for video in selected:
        questions = video.get("conversations")
        check(isinstance(questions, list) and 0 < len(questions) <= 2,
              "LU calibration exceeds two original questions per video")
        for question in questions:
            source_index = integer(question.get("source_question_index"), "calibration source question index")
            end = question.get("end_time")
            check(isinstance(end, (int, float)) and not isinstance(end, bool) and 0 < end <= 120,
                  "LU calibration question exceeds the bounded original timestamp")
            key = (str(video["video_id"]), source_index)
            check(key not in expected_observations, "Duplicate LU calibration original question")
            expected_observations[key] = question
    observations = calibration.get("observations")
    check(isinstance(observations, list) and len(observations) == len(expected_observations),
          "LU calibration observation evidence is missing or incomplete")
    observed = set()
    for observation in observations:
        source_index = integer(observation.get("source_question_index"), "observation source question index")
        key = (str(observation.get("video_id")), source_index)
        check(key in expected_observations and key not in observed
              and observation.get("end_time") == expected_observations[key]["end_time"],
              "LU calibration observation does not match an original selected question")
        observed.add(key)
        tokens = integer(observation.get("visual_tokens"), "calibration visual tokens", protocol["window_size"] + 1)
        check(observation.get("qa_restore_passed") is True, "LU calibration QA transaction was not restored")
        check(observation.get("additional_scoring_forwards") == 0, "LU calibration executed auxiliary scoring forwards")
        path = Path(observation.get("path", ""))
        check(path.is_absolute() and path.is_file() and sha256(path) == observation.get("sha256"),
              "LU calibration observation hash changed")
        validate_observation_arrays(path, profile["num_layers"], profile["num_kv_heads"], tokens, protocol["window_size"])
        metadata_path = Path(observation.get("metadata_path", ""))
        check(metadata_path.is_absolute() and metadata_path.is_file()
              and sha256(metadata_path) == observation.get("metadata_sha256"),
              "LU calibration observation metadata hash changed")
        metadata = read_json(metadata_path)
        expected_metadata = {
            "collector": "native_full_prefix", "selector": "sparsemm_visual_attention",
            "video_id": observation["video_id"], "source_question_index": source_index,
            "question": expected_observations[key], "visual_tokens": tokens,
            "num_layers": profile["num_layers"], "num_kv_heads": profile["num_kv_heads"],
            "recent": protocol["window_size"], "window_size": protocol["window_size"],
            "kernel_size": protocol["kernel_size"], "pooling": protocol["pooling"],
            "gqa_reduce": protocol["gqa_reduce"], "score_precision": "float32",
            "ranking_indices": "visual_history_local", "projected_norm_normalization": "layer_sum",
            "additional_scoring_forwards": 0, "compressed": False, "query_restored": True,
        }
        check(all(metadata.get(name) == value for name, value in expected_metadata.items()),
              "LU calibration observation did not use matching native attention/utility protocol")
        integer(metadata.get("prefix_tokens"), "calibration protected text prefix", 1)
        integer(metadata.get("answer_prediction_forwards"), "calibration answer prediction forwards", 1)
        check(metadata["answer_prediction_forwards"] <= 16, "LU calibration exceeds its 16-token QA observation limit")
    evidence_path = Path(calibration.get("manifest_path", ""))
    check(evidence_path.is_absolute() and evidence_path.is_file()
          and sha256(evidence_path) == calibration.get("manifest_sha256"),
          "LU calibration manifest hash changed")
    evidence = read_json(evidence_path)
    check(evidence.get("status") == "complete" and evidence.get("selected_sha256") == selected_hash
          and evidence.get("selected_videos") == selected and evidence.get("observations") == observations,
          "LU calibration evidence is incomplete or belongs to a different subset")
    evidence_checks = evidence.get("runtime_checks", {})
    check(isinstance(evidence_checks, dict) and evidence_checks.get("failures") == []
          and evidence_checks.get("qa_restore_checks") == len(observations),
          "LU calibration runtime checks report failures or incomplete QA restoration")
    manifests = {str(evidence_path): calibration["manifest_sha256"]}
    check(asset_meta.get("calibration_manifests") == manifests
          and protocol.get("lukv_calibration_manifests") == manifests,
          "LU profile calibration evidence manifest provenance disagrees")
    calibration_model = calibration.get("model_files_sha256")
    check(isinstance(calibration_model, dict) and "config.json" in calibration_model
          and evidence.get("model_files_sha256") == calibration_model,
          "LU calibration model fingerprints are missing or disagree")
    check(Path(calibration.get("model_path", "")).resolve() == model_path.resolve(),
          "LU calibration used a different model path")
    for name, expected in calibration_model.items():
        path = (model_path / name).resolve()
        check(path.is_relative_to(model_path.resolve()) and path.is_file() and sha256(path) == expected
              and (name not in model_hashes or model_hashes[name] == expected),
              f"LU calibration model hash differs from online model: {name}")
    code_hashes = evidence.get("code_sha256")
    check(isinstance(code_hashes, dict)
          and {"sparsemm_lukv/run_calibration.py", "sparsemm_lukv/profile.py"} <= code_hashes.keys(),
          "LU calibration evidence omits calibration/profile source fingerprints")
    for name, expected in code_hashes.items():
        path = (WORKSPACE / name).resolve()
        check(not Path(name).is_absolute() and path.is_relative_to(WORKSPACE),
              f"LU calibration source path escapes workspace: {name}")
        if path.is_file() and sha256(path) == expected:
            continue
        snapshot_root = (evidence_path.parent / "source_snapshot").resolve()
        snapshot = (snapshot_root / name).resolve()
        check(snapshot.is_relative_to(snapshot_root) and snapshot.is_file() and sha256(snapshot) == expected,
              f"LU calibration source code and archived snapshot hash changed: {name}")
    summary = asset_meta.get("calibration_summary", {})
    expected_summary = {"num_videos": len(selected), "num_observations": len(observations),
        "selector": "sparsemm_attention", "utility_mode": calibration["utility_mode"],
        "native_resolution": protocol["native_resolution"], "evaluation_overlap": 0,
        "usage": "attention_calibrated_profile", "recalibrated_for_runtime_selector": True,
        "prior_accuracy_reused": False, "uniform_blend": 0, "legacy_capacity_clip": False,
        "budget_mode": "compression_ratio_lookup_history_with_protected_window"}
    check(summary == expected_summary and protocol.get("lukv_calibration_summary") == summary,
          "LU profile calibration summary disagrees with its evidence")
    check(profile.get("calibration_summary", {}).get("num_videos") == len(selected)
          and profile.get("calibration_summary", {}).get("num_observations") == len(observations),
          "LU profile solver calibration counts disagree with its evidence")
    return summary


def validate_kept_indices(kept, before, after, budgets, prefix, window, label):
    check(isinstance(kept, list) and len(kept) == len(before), f"{label}: missing kept indices")
    for layer, row in enumerate(before):
        check(isinstance(kept[layer], list) and len(kept[layer]) == len(row), f"{label}: wrong kept head count")
        for head, old in enumerate(row):
            indices = kept[layer][head]
            check(isinstance(indices, list), f"{label}: kept indices must be lists")
            for index in indices:
                integer(index, f"{label}: kept index")
            check(len(indices) == len(set(indices)) and all(index < old for index in indices),
                  f"{label}: kept indices are not unique valid positions")
            text = min(prefix, old)
            visual = old - text
            expected = text + min(budgets[layer][head], visual)
            check(len(indices) == after[layer][head] == expected, f"{label}: target head capacity mismatch")
            check(indices[:text] == list(range(text)), f"{label}: initial text prefix was not preserved")
            recent = min(window, visual)
            check(not recent or indices[-recent:] == list(range(old - recent, old)),
                  f"{label}: recent visual window was not preserved")


def audit_run(output_dir, *, min_compression_events_per_video=1, require_lmms=True,
              allow_calibration_transfer=False):
    out = Path(output_dir).resolve()
    integer(min_compression_events_per_video, "minimum compression events")
    protocol = read_json(out / "protocol.json")
    manifest = read_json(out / "run_manifest.json")
    annotations = read_json(out / "annotations.json")
    summary = read_json(out / "summary.json")
    runtime = read_json(out / "runtime_checks.json")
    check((out / "run.status").is_file() and (out / "run.status").read_text().strip() == "complete",
          "Run is not marked complete")
    check(runtime.get("status") == "complete" and runtime.get("failures") == [],
          "Runtime validation did not complete cleanly")
    check(protocol.get("method") == "lukv" and protocol.get("budget_scope") == "visual",
          "LU audit requires visual LU-KV capacities")
    check(not (protocol.get("lukv_profile_path") and protocol.get("lukv_budget_path")),
          "LU profile and legacy fixed capacity CSV are mutually exclusive")
    profile_mode = bool(protocol.get("lukv_profile_path"))
    check(isinstance(annotations, list) and annotations, "Selected annotations are empty")
    vids = [str(video["video_id"]) for video in annotations]
    check(len(set(vids)) == len(vids), "Duplicate selected video IDs")
    canonical = json.dumps(annotations, sort_keys=True, ensure_ascii=False).encode()
    check(hashlib.sha256(canonical).hexdigest() == protocol.get("subset_sha256"),
          "Annotation subset hash mismatch")
    source = _source_path(out, protocol, manifest)
    check(source.is_file() and sha256(source) == protocol.get("source_sha256"),
          "Annotation source hash mismatch")
    _validate_source_selection(source, annotations)
    check(protocol.get("upstream_commit") == manifest.get("upstream_commit") == UPSTREAM_COMMIT,
          "Original lmms-eval commit mismatch")
    code = manifest.get("code_sha256")
    check(isinstance(code, dict) and set(REQUIRED_CODE_FILES) <= code.keys(),
          "Source manifest omits required native LU inference files")
    for name, expected in code.items():
        path = (WORKSPACE / name).resolve()
        check(not Path(name).is_absolute() and path.is_relative_to(WORKSPACE),
              f"Source path escapes workspace: {name}")
        if path.is_file() and sha256(path) == expected:
            continue
        snapshot_root = (out / "source_snapshot").resolve()
        snapshot = (snapshot_root / name).resolve()
        check(snapshot.is_relative_to(snapshot_root) and snapshot.is_file() and sha256(snapshot) == expected,
              f"Source and run snapshot hash changed: {name}")
    model_path = Path(protocol["model_path"])
    hashes = manifest.get("model_files_sha256", {})
    check(isinstance(hashes, dict) and "config.json" in hashes, "Model config hash is missing")
    for name, expected in hashes.items():
        path = (model_path / name).resolve()
        check(path.is_relative_to(model_path.resolve()) and path.is_file() and sha256(path) == expected,
              f"Model hash changed: {name}")
    model = read_json(model_path / "config.json")
    text = model.get("text_config", model)
    layers = integer(text["num_hidden_layers"], "num layers", 1)
    heads = integer(text["num_key_value_heads"], "num KV heads", 1)
    query_heads = integer(text["num_attention_heads"], "num query heads", 1)
    dim = integer(text.get("head_dim", text["hidden_size"] // query_heads), "head dim", 1)
    check(query_heads % heads == 0, "Nonintegral GQA ratio")
    element = {"float16": 2, "bfloat16": 2, "float32": 4}.get(protocol["dtype"])
    check(element is not None, "Unsupported model dtype")
    budget = integer(protocol["budget"], "budget", 1)
    window = integer(protocol["window_size"], "recent window", 1)
    asset_path = Path((protocol.get("lukv_profile_path") or DEFAULT_PROFILE_PATH) if profile_mode
                      else (protocol.get("lukv_budget_path") or DEFAULT_BUDGET_PATH)).resolve()
    check(asset_path.is_file(), "LU capacity asset is missing")
    asset_hash = sha256(asset_path)
    if not profile_mode and asset_path == DEFAULT_BUDGET_PATH.resolve():
        check(asset_hash == DEFAULT_BUDGET_SHA256, "Default LU calibration asset hash changed")
    profile = read_profile(asset_path, layers, heads, window) if profile_mode else None
    budgets = None if profile_mode else read_budget_table(asset_path, layers, heads, budget, window)
    check(protocol.get("lukv_asset_sha256") == manifest.get("lukv_asset_sha256") == asset_hash,
          "LU asset provenance hash mismatch")
    asset_meta = manifest.get("lukv_asset_metadata", {})
    check(isinstance(asset_meta.get("path"), str)
          and Path(asset_meta["path"]).expanduser().resolve() == asset_path
          and asset_meta.get("sha256") == asset_hash
          and asset_meta.get("shape") == [layers, heads], "LU asset metadata mismatch")
    if profile_mode:
        check(asset_meta.get("sum") is None and asset_meta.get("global_visual_capacity") == layers * heads * budget
              and asset_meta.get("asset_type") == "profile"
              and asset_meta.get("runtime_selector") == "sparsemm_attention"
              and asset_meta.get("usage") == "attention_calibrated_profile"
              and asset_meta.get("recalibrated_for_runtime_selector") is True
              and asset_meta.get("prior_accuracy_reused") is False,
              "LU profile asset is not labeled as recalibrated dynamic capacity lookup")
        check("sparsemm_lukv/profile.py" in code, "LU source manifest omits profile allocation code")
    else:
        check(asset_meta.get("sum") == layers * heads * budget
              and asset_meta.get("minimum") == min(map(min, budgets))
              and asset_meta.get("maximum") == max(map(max, budgets)), "LU fixed asset metadata mismatch")
    for key in ("calibration_manifests", "calibration_sources"):
        declared = protocol.get("lukv_" + key, {})
        check(declared == asset_meta.get(key, {}), "LU calibration provenance disagrees across files")
        for name, expected in declared.items():
            path = Path(name)
            check(path.is_file() and sha256(path) == expected, f"LU calibration hash changed: {name}")
    calibration_transfer = {}
    calibration_summary = (validate_profile_calibration(profile, asset_meta, protocol, model_path, hashes, source,
                               allow_calibration_transfer=allow_calibration_transfer,
                               evaluation_videos=annotations, transfer_evidence=calibration_transfer)
                           if profile_mode else asset_meta.get("calibration_summary", {}))
    if not profile_mode and asset_path == DEFAULT_BUDGET_PATH.resolve():
        calibration_path = asset_path.with_name("manifest.json")
        check(str(calibration_path) in protocol.get("lukv_calibration_manifests", {}),
              "Default LU calibration provenance is missing")
        calibration = read_json(calibration_path)
        calibration_videos = calibration.get("videos", [])
        calibration_ids = {str(video["video_id"]) for video in calibration_videos}
        calibration_names = {Path(video["video_path"]).name for video in calibration_videos}
        full_evaluation = read_json(source)
        check(not calibration_ids & {str(video["video_id"]) for video in full_evaluation}
              and not calibration_names & {Path(video["video_path"]).name for video in full_evaluation},
              "LU calibration overlaps evaluation videos")
    try:
        rows, completed = load_resume(out / "1_0.csv", annotations, require_complete=True)
    except (RuntimeError, ValueError, KeyError) as error:
        raise AuditError(f"CSV validation failed: {error}") from error
    total_questions = sum(len(video["conversations"]) for video in annotations)
    check(completed == len(annotations) and len(rows) == total_questions, "Result CSV is incomplete")
    check((out / "results.csv").is_file()
          and (out / "results.csv").read_bytes() == (out / "1_0.csv").read_bytes(), "Result CSV copies differ")
    accuracy = accuracy_statistics(rows)
    check(summary.get("videos") == len(annotations) and summary.get("questions") == total_questions,
          "Summary count mismatch")
    check(summary.get("correct") == accuracy["correct"] and math.isclose(
        summary.get("accuracy_percent", -1), accuracy["accuracy_percent"], rel_tol=0, abs_tol=1e-8),
        "Summary accuracy mismatch")
    check(all(summary.get(name) == protocol.get(name) for name in ("method", "budget", "budget_scope")),
          "Summary method or budget mismatch")
    check(protocol.get("lukv_selector") == "visual_attention" and protocol.get("lukv_history") == "none",
          "LU audit requires the direct visual-attention selector without history probes")
    check(protocol.get("lukv_probe_count") == 0 and protocol.get("lukv_recency_bonus") is False,
          "LU protocol unexpectedly enables probes or recency scores")
    if profile_mode:
        check(protocol.get("lukv_budget_recalibrated") is True
              and protocol.get("lukv_budget_transfer") == "attention_calibrated_profile"
              and protocol.get("lukv_budget_mode") == "compression_ratio_lookup",
              "LU profile must be labeled as attention-recalibrated ratio lookup")
    else:
        check(protocol.get("lukv_budget_recalibrated") is False
              and protocol.get("lukv_budget_transfer") == "legacy_calibrated_transfer",
              "Existing LU capacities must be labeled transferred without recalibration")
    check(runtime.get("probe_checks") == [] and runtime.get("probe_restore_checks") == 0,
          "Direct-attention LU unexpectedly executed proxy probes")
    trace_path = out / "cache_trace.jsonl"
    check(trace_path.is_file(), "Cache trace missing")
    trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
    check(trace, "Cache trace is empty")
    finite(trace, "cache trace")
    states, qa_checks, selector_checks, allocations = [], [], [], []
    state = None
    position_fields = ("prefix_preserved", "recent_preserved", "indices_unique", "indices_in_range", "positions_match_kept")

    def finish_state():
        if state is None:
            return
        check(state["questions"] == len(annotations[len(states)]["conversations"]), "Per-video question count mismatch")
        check(state["compressions"] >= min_compression_events_per_video, "Insufficient LU compression events for video")
        states.append({key: value for key, value in state.items()
                       if key not in ("lengths", "memory", "last_answer_prefill")})

    for number, event in enumerate(trace, 1):
        kind = event.get("event")
        label = f"Trace line {number} ({kind})"
        vid = str(event.get("video_id"))
        if kind == "init":
            finish_state()
            check(len(states) < len(vids) and vid == vids[len(states)], f"{label}: video reset/order mismatch")
            prefix = integer(event.get("text_tokens"), "prefix tokens", 1)
            config = event.get("config")
            check(isinstance(config, dict) and all(protocol.get(name) == value for name, value in config.items()),
                  f"{label}: initial configuration mismatch")
            check(event.get("num_layers") == layers and event.get("num_kv_heads") == heads, f"{label}: architecture mismatch")
            lengths = [[prefix] * heads for _ in range(layers)]
            memory = expected_memory(lengths, layers, heads, dim, element)
            validate_memory(event.get("memory"), memory, label)
            state = {"video_id": vid, "prefix": prefix, "prefills": 0, "compressions": 0,
                     "questions": 0, "seen": 0, "lengths": lengths, "memory": memory,
                     "last_answer_prefill": None, "cursor": prefix}
            continue
        check(state is not None and vid == state["video_id"], f"{label}: missing matching video init")
        if kind == "video_prefill":
            count = integer(event.get("input_tokens"), "input video tokens", 1)
            state["prefills"] += 1
            state["seen"] += count
            check(event.get("prefill") == state["prefills"], f"{label}: prefill counter mismatch")
            allocation = None
            if profile_mode:
                budgets, allocation = allocate_profile_independent(profile, state["seen"], budget, window)
                allocations.append({"video_id": vid, "prefill_index": state["prefills"], **allocation})
            before = [[old + count for old in row] for row in state["lengths"]]
            expected_after = [[state["prefix"] + min(capacity, old - state["prefix"])
                               for capacity, old in zip(capacities, previous)]
                              for capacities, previous in zip(budgets, before)]
            method = event.get("method_metadata")
            check(isinstance(method, dict), f"{label}: missing selection metadata")
            layer_meta = event.get("layers")
            check(isinstance(layer_meta, list) and len(layer_meta) == layers and layer_meta == method.get("layers"),
                  f"{label}: selection did not cover every native layer")
            for layer_index, meta in enumerate(layer_meta):
                layer_label = f"{label} layer {layer_index}"
                check(meta.get("method") == "lukv" and meta.get("selector") == "visual_attention",
                      f"{layer_label}: selector is not direct visual attention")
                if profile_mode:
                    actual_allocation = meta.get("allocation")
                    check(isinstance(actual_allocation, dict)
                          and all(actual_allocation.get(key) == value for key, value in allocation.items()),
                          f"{layer_label}: profile lookup interpolation or exact integer allocation mismatch")
                    aliases = {"seen_visual_tokens": state["seen"], "target_visual_slots": allocation["target_slots"],
                               "retention_fraction": allocation["rho"]}
                    for key, value in aliases.items():
                        check(key not in actual_allocation or actual_allocation[key] == value,
                              f"{layer_label}: profile allocation {key} mismatch")
                expected_fields = dict(layer_idx=layer_index, query_len=count,
                    before_head_lengths=before[layer_index], head_lengths=expected_after[layer_index],
                    head_budgets=budgets[layer_index], prefix_tokens=state["prefix"], window_size=window,
                    observed_queries=min(window, count), gqa_groups=query_heads // heads,
                    gqa_reduce=protocol["gqa_reduce"], pooling=protocol["pooling"],
                    kernel_size=protocol["kernel_size"], score_precision="float32",
                    indices_order="prefix_score_descending_recent")
                for key, expected in expected_fields.items():
                    check(meta.get(key) == expected, f"{layer_label}: invalid {key}")
                recent = [min(window, old - state["prefix"]) for old in before[layer_index]]
                visual = [length - state["prefix"] for length in expected_after[layer_index]]
                history = [length - rec for length, rec in zip(visual, recent)]
                check(meta.get("recent_counts") == recent and meta.get("retained_visual_counts") == visual
                      and meta.get("retained_history_counts") == history, f"{layer_label}: retained counts mismatch")
                validate_kept_indices([meta.get("kept_indices")], [before[layer_index]], [expected_after[layer_index]],
                                      [budgets[layer_index]], state["prefix"], window, layer_label)
                positions = meta.get("position_checks")
                check(isinstance(positions, dict) and all(positions.get(name) is True for name in position_fields),
                      f"{layer_label}: selected positions failed validation")
                did_prune = any(new < old for old, new in zip(before[layer_index], expected_after[layer_index]))
                check(meta.get("compressed") is did_prune, f"{layer_label}: actual compression flag mismatch")
                check(meta.get("total_visual_capacity") == sum(budgets[layer_index])
                      and meta.get("actual_visual_slots") == sum(visual)
                      and meta.get("target_total_tokens") == sum(budgets[layer_index]) + heads * state["prefix"]
                      and meta.get("actual_total_tokens") == sum(expected_after[layer_index]),
                      f"{layer_label}: visual capacity accounting mismatch")
                selector_check = dict(video_id=event.get("video_id"), prefill_index=state["prefills"],
                    layer_idx=layer_index, validation_passed=True, compressed=did_prune,
                    before_head_lengths=before[layer_index], head_lengths=expected_after[layer_index],
                    head_budgets=budgets[layer_index], **positions)
                if profile_mode:
                    selector_check["allocation"] = actual_allocation
                selector_checks.append(selector_check)
            compressed = any(new < old for old_row, new_row in zip(before, expected_after)
                             for old, new in zip(old_row, new_row))
            state["compressions"] += int(compressed)
            check(method.get("compressed") is compressed and method.get("compressions") == state["compressions"]
                  and method.get("logical_tokens") == state["prefix"] + state["seen"],
                  f"{label}: patch commit counters mismatch")
            check(head_lengths(event.get("head_lengths"), layers, heads, label) == expected_after
                  and method.get("head_lengths") == expected_after, f"{label}: head capacities mismatch")
            check(event.get("bank_total_absorbed") == event.get("active_prototypes") == event.get("pq_ready_groups") == 0,
                  f"{label}: LU unexpectedly has prototype state")
            check(event.get("native_attention_calls", 0) >= layers * (state["prefills"] + 1),
                  f"{label}: video prefill skipped native layers")
            memory = expected_memory(expected_after, layers, heads, dim, element)
            validate_memory(event.get("memory"), memory, label)
            validate_memory(method, memory, f"{label} committed memory")
            check(event.get("kv_bytes") == memory["persistent_kv_bytes"], f"{label}: KV byte count mismatch")
            state.update(lengths=expected_after, memory=memory)
        elif kind == "answer_restored":
            check(head_lengths(event.get("head_lengths"), layers, heads, label) == state["lengths"], f"{label}: QA changed persistent lengths")
            validate_memory(event.get("memory"), state["memory"], label)
            qa = event.get("query_check")
            check(isinstance(qa, dict) and str(qa.get("video_id")) == vid
                  and qa.get("question_index") == state["questions"], f"{label}: QA order/identity mismatch")
            check(qa.get("returned") is True and qa.get("restored") is True, f"{label}: QA did not return and restore")
            check(qa.get("first_answer_native_layers") == layers and qa.get("first_answer_pseudo_layers") == 0,
                  f"{label}: first answer native layer mismatch")
            query_memory = dict(state["memory"], query_context_kv_bytes=state["memory"]["persistent_kv_bytes"])
            validate_memory(qa.get("query_memory"), query_memory, f"{label} query memory")
            cursor = integer(event.get("next_position"), "QA cursor", state["prefix"])
            check(cursor >= state["cursor"], f"{label}: video cursor moved backward")
            if state["last_answer_prefill"] == state["prefills"]:
                check(cursor == state["cursor"], f"{label}: repeated QA advanced video clock")
            state.update(cursor=cursor, last_answer_prefill=state["prefills"])
            state["questions"] += 1
            qa_checks.append(qa)
        else:
            raise AuditError(f"{label}: unexpected event for direct-attention LU")
    finish_state()
    check(len(states) == len(annotations) and len(qa_checks) == len(rows), "Trace and CSV counts differ")
    check(runtime.get("qa_checks") == qa_checks and runtime.get("qa_restore_checks") == len(qa_checks),
          "Runtime and trace QA restoration checks disagree")
    runtime_selector_checks = runtime.get("selector_checks")
    if not profile_mode and isinstance(runtime_selector_checks, list):
        check(all(check_item.get("allocation") is None for check_item in runtime_selector_checks),
              "Legacy fixed LU unexpectedly logged dynamic profile allocation")
        # Current legacy runs include allocation=None; earlier records omit it.
        runtime_selector_checks = [{key: value for key, value in check_item.items() if key != "allocation"}
                                   for check_item in runtime_selector_checks]
    check(runtime_selector_checks == selector_checks, "Runtime and trace selector validation checks disagree")
    prefills = sum(state["prefills"] for state in states)
    hidden_checks = integer(runtime.get("finite_hidden_checks"), "finite hidden checks")
    logits_checks = integer(runtime.get("finite_logits_checks"), "finite logits checks", len(rows))
    check(hidden_checks == len(annotations) + prefills + logits_checks,
          "Finite hidden/logits forward counts indicate unexpected auxiliary forwards")
    lmms_files = _lmms_results(out, summary, accuracy["accuracy_percent"], require_lmms)
    by_task = defaultdict(list)
    for row in rows:
        by_task[row.get("task") or "(unspecified)"].append(row)
    return {
        "passed": True, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(out), "method": "lukv", "selector": "visual_attention", "budget": budget, "budget_scope": "visual",
        "videos": len(annotations), "questions": len(rows), "overall": accuracy,
        "task_accuracy": {key: accuracy_statistics(value) for key, value in sorted(by_task.items())},
        "trace": {"events": len(trace), "prefills": prefills,
                  "compression_events": sum(state["compressions"] for state in states),
                  "qa_restore_checks": len(qa_checks), "probe_restore_checks": 0,
                  "selector_checks": len(selector_checks),
                  "finite_hidden_checks": hidden_checks, "finite_logits_checks": logits_checks},
        "asset": {"path": str(asset_path), "sha256": asset_hash,
                  "head_budgets": None if profile_mode else budgets,
                  "total_visual_capacity": layers * heads * budget,
                  "usage": "attention_calibrated_profile" if profile_mode else "legacy_calibrated_transfer",
                  "recalibrated_for_runtime_selector": profile_mode,
                  "budget_mode": "compression_ratio_lookup" if profile_mode else "fixed_csv",
                  "calibration_summary": calibration_summary,
                  **({"calibration_transfer": calibration_transfer} if calibration_transfer else {})},
        "dynamic_allocation": {"enabled": profile_mode, "patches_checked": len(allocations),
            "uncompressed_length_used": profile_mode, "online_utility_used": False,
            "capacity_snapshots": allocations},
        "architecture": {"num_layers": layers, "num_kv_heads": heads,
                         "num_query_heads": query_heads, "head_dim": dim, "element_bytes": element},
        "per_video": states, "lmms_results_checked": lmms_files,
        "limitations": ["Trace checks selected indices and runtime position validation; score rankings are tested separately.",
                        ("Offline attention-calibrated profiles supply target capacities; streaming eviction may leave some heads below target occupancy."
                         if profile_mode else
                         "Existing LU budgets are transferred from the previous calibration and were not recalibrated for direct attention."),
                        ("The streaming profile protects the fixed recent window; it omits the paper's one-percent per-head floor for long fixed-budget video streams."
                         if profile_mode else "Historical fixed capacity constraints remain specific to the legacy CSV adaptation."),
                        "Small validation does not establish full benchmark accuracy or official LU-KV equivalence."],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-compression-events-per-video", type=int, default=1)
    parser.add_argument("--allow-missing-lmms-results", action="store_true")
    parser.add_argument("--allow-calibration-transfer", action="store_true",
                        help="Permit an external dataset only after checking actual calibration/evaluation "
                             "video hashes for overlap; do not recalibrate the existing LU profile.")
    args = parser.parse_args(argv)
    destination = args.output or args.output_dir / "audit_report.json"
    try:
        result = audit_run(args.output_dir,
                           min_compression_events_per_video=args.min_compression_events_per_video,
                           require_lmms=not args.allow_missing_lmms_results,
                           allow_calibration_transfer=args.allow_calibration_transfer)
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
