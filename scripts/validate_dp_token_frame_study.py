#!/usr/bin/env python3
"""Independently validate matched DP/HERMES immutable-token/frame traces."""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "results/myobservations/dp_token_frame_study_20261010"
HEADS = {(layer, head) for layer in range(28) for head in range(4)}


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def csv_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT)
    parser.add_argument("--ledger", type=Path,
                        default=ROOT / "results/myobservations/dp_failure_audit_20261010/case_ledger.csv")
    args = parser.parse_args()
    study = args.study.resolve()
    errors = []
    checks = Counter()

    def check(condition, label, detail=""):
        checks[label] += 1
        if not condition:
            errors.append({"check": label, "detail": str(detail)})

    annotation = json.loads((study / "annotation.json").read_text())
    expected = {(str(v["video_id"]), qi): q for v in annotation
                for qi, q in enumerate(v["conversations"])}
    ledger = {(r["video_id"], int(r["question_index"])): r for r in csv_rows(args.ledger)}
    profile = ROOT / "results/myobservations/k1000_dp_cachecomp_score_20260924/recent_ratio_profile.csv"
    capacities = {(int(r["layer"]), int(r["kv_head"])): int(r["capacity_n"])
                  for r in csv_rows(profile)}
    reports = {}
    manifests = {}
    artifact_hashes = {str(study / "annotation.json"): digest(study / "annotation.json"),
                       str(args.ledger): digest(args.ledger), str(profile): digest(profile)}

    for policy in ("dp", "hermes"):
        directory = study / policy
        needed = [directory / name for name in
                  ("command.json", "source_manifest.jsonl", "tokens.jsonl", "readout.jsonl")]
        if not all(path.exists() for path in needed):
            check(False, "required_files", policy)
            continue
        command = json.loads(needed[0].read_text())
        for source, sha in command["sources_sha256"].items():
            check(Path(source).exists() and digest(Path(source)) == sha,
                  "source_sha256_unchanged", f"{policy}: {source}")
        for path in needed:
            artifact_hashes[str(path)] = digest(path)
        manifest = jsonl(directory / "source_manifest.jsonl")
        manifests[policy] = [r for r in manifest if r.get("type") != "run"]
        videos = {}
        encodes = defaultdict(list)
        for record in manifests[policy]:
            vid = str(record["video_id"])
            if record["type"] == "video":
                check(vid not in videos, "one_video_manifest", (policy, vid))
                videos[vid] = record
            elif record["type"] == "encode":
                encodes[vid].append(record)
        check(set(videos) == {vid for vid, _ in expected}, "manifest_video_set", policy)
        uid_frames = {}
        question_generated = {}
        question_sample_count = {}
        encode_counts = {}
        for vid, calls in encodes.items():
            video = videos[vid]
            check(len(video["sample_indices"]) == len(video["sample_pts"]) == video["loaded_shape"][0],
                  "loader_sample_lengths", (policy, vid))
            check(video["sample_indices"] == sorted(set(video["sample_indices"])),
                  "loader_unique_ordered_indices", (policy, vid))
            if video.get("loader_rounded_fps") is not None:
                rounded = int(video["loader_rounded_fps"])
                clip = video.get("clip")
                first = max(0, int(clip[0] * rounded)) if clip is not None else 0
                stop = min(video["source_frame_count"], int(clip[1] * rounded) + 1) if clip is not None else video["source_frame_count"]
                step = int(rounded / float(video["sample_fps"]))
                check(video["sample_indices"] == list(range(first, stop, step)),
                      "sample_indices_match_original_loader_rule", (policy, vid))
            video_path = Path(video["video_path"])
            if str(video_path) not in artifact_hashes:
                artifact_hashes[str(video_path)] = digest(video_path)
            next_uid = next_frame = 0
            uid_map = {}
            for call_index, call in enumerate(calls):
                start, count = int(call["first_sample_ordinal"]), int(call["raw_frame_count"])
                t, h, w = map(int, call["grid_thw"])
                merge = int(call["merge_size"])
                temporal = int(call["temporal_patch_size"])
                q_len = int(call["q_len"])
                u0 = int(call["uid_start"])
                check(call["encode_call"] == call_index, "encode_call_sequence", (policy, vid, call_index))
                check(u0 == next_uid, "continuous_uid_allocation", (policy, vid, u0, next_uid))
                check(start == next_frame, "continuous_sample_allocation", (policy, vid, start, next_frame))
                check(temporal == 2 and t == math.ceil(count / temporal), "qwen_temporal_pair_grid", (policy, vid, call_index))
                check(h % merge == w % merge == 0 and q_len == t * (h // merge) * (w // merge),
                      "qwen_spatial_merged_grid", (policy, vid, call_index))
                check(call["uid_stop_exclusive"] == u0 + q_len, "uid_interval_length", (policy, vid, call_index))
                check(start + count <= len(video["sample_indices"]), "encode_inside_loaded_video", (policy, vid, call_index))
                samples = call["samples"]
                check([s["sample_ordinal"] for s in samples] == list(range(start, start + count)),
                      "encode_sample_ordinals", (policy, vid, call_index))
                check([s["source_frame_index"] for s in samples] == video["sample_indices"][start:start + count],
                      "encode_decoder_indices", (policy, vid, call_index))
                check([s["decoder_pts"] for s in samples] == video["sample_pts"][start:start + count],
                      "encode_decoder_pts", (policy, vid, call_index))
                spatial = (h // merge) * (w // merge)
                for local in range(q_len):
                    tubelet = local // spatial
                    right = start + min(temporal * tubelet + temporal - 1, count - 1)
                    # Check the historical trace sidecar's formula against the
                    # actual Qwen temporal-pair construction, including padding.
                    core_right = start + min(((tubelet + 1) * count - 1) // t, count - 1)
                    if right != core_right:
                        check(False, "core_frame_label_matches_temporal_pair", (policy, vid, call_index, tubelet))
                    uid_map[u0 + local] = right
                checks["core_frame_label_matches_temporal_pair"] += t
                next_uid += q_len
                next_frame += count
                qi = int(call["question_index"])
                question_generated[(vid, qi)] = next_uid
                question_sample_count[(vid, qi)] = next_frame
            uid_frames[vid] = uid_map
            encode_counts[vid] = {"encode_calls": len(calls), "generated_uids": next_uid,
                                  "encoded_raw_samples": next_frame}
        token_records = jsonl(directory / "tokens.jsonl")
        readouts = {(str(r["video_id"]), int(r["question_index"])): r
                    for r in jsonl(directory / "readout.jsonl")}
        tokens = {(str(r["video_id"]), int(r["question_index"])): r for r in token_records}
        check(len(tokens) == len(token_records), "unique_token_questions", policy)
        check(set(tokens) == set(readouts) == set(expected), "six_question_trace_set", policy)
        per_question = []
        for key, record in tokens.items():
            if key not in expected or key not in readouts:
                continue
            q = expected[key]
            check(record["question"] == q["question"] and float(record["end_time"]) == float(q["end_time"]),
                  "trace_annotation_identity", (policy, key))
            check(int(record["score_last_n"]) == 0, "true_question_prompt_pooling", (policy, key))
            head_map = {(int(h["layer"]), int(h["kv_head"])): h for h in record["heads"]}
            readout_map = {(int(h["layer"]), int(h["kv_head"])): h for h in readouts[key]["heads"]}
            check(set(head_map) == set(readout_map) == HEADS and len(record["heads"]) == 112,
                  "all_112_heads", (policy, key))
            union = set()
            physical = 0
            lengths = []
            generated = question_generated[key]
            sampled = question_sample_count[key]
            expected_samples = min(len(videos[key[0]]["sample_indices"]), math.ceil(float(q["end_time"]) * record["sample_fps"]))
            check(sampled == expected_samples, "question_encoded_prefix", (policy, key, sampled, expected_samples))
            for hk, head in head_map.items():
                uids = list(map(int, head["visual_token_uids"]))
                frames = list(map(int, head["visual_frame_ids"]))
                attention = head["visual_attention"]
                n = len(uids)
                detail = (policy, key, hk)
                check(n == len(frames) == len(attention), "aligned_head_arrays", detail)
                check(uids == sorted(set(uids)), "unique_ordered_uid_per_head", detail)
                check(all(0 <= uid < generated and uid in uid_frames[key[0]] for uid in uids), "retained_uid_legal", detail)
                check(all(uid_frames[key[0]].get(uid) == frame for uid, frame in zip(uids, frames)),
                      "retained_frame_is_pair_right_endpoint", detail)
                check(all(0 <= frame < sampled for frame in frames), "retained_frame_not_future", detail)
                check(all(math.isfinite(float(v)) and -1e-8 <= float(v) <= 1.0001 for v in attention),
                      "finite_attention_probabilities", detail)
                check(sum(attention) <= 1.0001, "visual_attention_mass_bounded", detail)
                capacity = capacities[hk] if policy == "dp" else 1000
                check(n == min(capacity, generated), "head_visual_capacity", (detail, n, capacity, generated))
                r = readout_map[hk]
                check(int(r["visual_tokens"]) == n and sum(map(int, r["frame_tokens"].values())) == n,
                      "readout_token_count_agrees", detail)
                check(Counter(frames) == Counter({int(k): int(v) for k, v in r["frame_tokens"].items()}),
                      "readout_frame_histogram_agrees", detail)
                check(abs(sum(attention) - float(r["visual_mass"])) < 1e-5,
                      "readout_attention_sum_agrees", detail)
                union.update(uids)
                physical += n
                lengths.append(n)
            pts = videos[key[0]]["sample_pts"][sampled - 1] if sampled else None
            per_question.append({"video_id": key[0], "question_index": key[1],
                                 "generated_source_uids": generated, "physical_kv_copies": physical,
                                 "unique_retained_source_uids": len(union), "min_visual_tokens_per_head": min(lengths),
                                 "max_visual_tokens_per_head": max(lengths), "last_sample_pts": pts})
        prediction_files = sorted((directory / "predictions").glob("*.csv"))
        check(len(prediction_files) == 1, "one_prediction_file", policy)
        predictions = []
        if len(prediction_files) == 1:
            prediction_path = prediction_files[0]
            artifact_hashes[str(prediction_path)] = digest(prediction_path)
            predictions = csv_rows(prediction_path)
        key_order = [(str(v["video_id"]), qi) for v in annotation for qi, _ in enumerate(v["conversations"])]
        check(len(predictions) == len(expected) == 6, "six_prediction_rows", policy)
        prediction_checks = []
        for key, pred in zip(key_order, predictions):
            q = expected[key]
            choices = ast.literal_eval(pred["choices"])
            check(str(pred["video_id"]) == key[0] and pred["question"] == q["question"]
                  and choices == q["choices"] and pred["answer"] == q["answer"],
                  "prediction_annotation_identity", (policy, key))
            old = ledger[key]
            prefix = policy + "1000"
            matches = pred["pred_choice"] == old[prefix + "_choice"]
            check(matches, "prediction_reproduces_case_ledger", (policy, key, pred["pred_choice"], old[prefix + "_choice"]))
            check(float(pred["qa_acc"]) == 100 * int(old[prefix + "_correct"]),
                  "prediction_accuracy_reproduces_ledger", (policy, key))
            prediction_checks.append({"video_id": key[0], "question_index": key[1],
                                      "pred_choice": pred["pred_choice"], "historical_choice": old[prefix + "_choice"],
                                      "matches_historical": matches})
        reports[policy] = {"videos": len(videos), "token_questions": len(tokens),
                           "encode_counts": encode_counts, "per_question": per_question,
                           "predictions": prediction_checks}
    if set(manifests) == {"dp", "hermes"}:
        check(manifests["dp"] == manifests["hermes"], "dp_hermes_identical_source_manifests")
    preview_validation = {}
    preview_path = study / "input_frame_previews_manifest.json"
    if preview_path.exists() and set(manifests) == {"dp", "hermes"}:
        previews = json.loads(preview_path.read_text())
        artifact_hashes[str(preview_path)] = digest(preview_path)
        for policy in ("dp", "hermes"):
            vmap = {str(r["video_id"]): r for r in manifests[policy] if r["type"] == "video"}
            matched = 0
            for frame in previews["frames"]:
                vid = str(frame["video_id"])
                ordinal = int(frame["sample_ordinal"])
                video = vmap[vid]
                match = (video["sample_indices"][ordinal] == frame["source_frame_index"]
                         and video["sample_pts"][ordinal] == frame["decoder_pts"]
                         and video["source_shape"] == frame["source_rgb_shape"])
                check(match, "preview_matches_both_actual_loader_manifests", (policy, vid, ordinal))
                matched += int(match)
                if policy == "dp":
                    image_path = study / frame["preview_image"]
                    check(digest(image_path) == frame["preview_jpeg_sha256"],
                          "preview_jpeg_sha256_unchanged", (vid, ordinal))
            preview_validation[policy] = {"frames_checked": len(previews["frames"]), "frames_matched": matched}
        preview_validation["v124_previous_pending_status"] = ("confirmed against both completed loader manifests by index, PTS and source shape"
            if all(preview_validation[p]["frames_matched"] == len(previews["frames"]) for p in ("dp", "hermes"))
            else "mismatch found; inspect preview checks")
        preview_validation["pixel_limit"] = "Raw RGB hashes belong to independent CPU preview re-decodes; original inference loader did not record pixel hashes, so this validation confirms decoder index/PTS/shape rather than bitwise original input equality."
    output = {"schema_version": 1, "passed": not errors,
              "checks": dict(checks), "errors": errors, "policies": reports,
              "artifact_sha256": artifact_hashes, "preview_validation": preview_validation,
              "interpretation": "UID is an immutable visual output position scoped to a video; one source tubelet covers two sampled frames (last-frame replication when odd). Source patch geometry is not an exclusive semantic receptive field. Head copies are counted separately from unique source UIDs."}
    (study / "validation.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": output["passed"], "checks_executed": sum(checks.values()),
                      "errors": errors, "validation": str(study / "validation.json")}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if output["passed"] else 1)


if __name__ == "__main__":
    main()
