"""CPU-only audit of the default-input 27-question paired OVO diagnostic.

This audit deliberately does not require equal recent token selections: the
history arm reserves 28 fewer recent slots on each of its 56 one-frame heads.
Usage: python3 sparsemm_native/audit_aligned_pseudo_pair.py --output-root RUN \
    --reference-root PREVIOUS_PIXEL_CAPPED_RUN
"""

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("recent_only", "mild_history")
TASKS = ("EPM", "HLD", "ASI", "STU", "OJR", "ATR", "FPD", "ACR", "OCR")
IDENTITY = ("query_id", "source_id", "probe_id", "task", "group", "timestamp", "clip_end", "target")
QUOTA_PATH = ROOT / "results/ovobench_simplestream_uniform_history_budget500_proposal_20261025/per_head_allocation.csv"


def read_json(path):
    return json.loads(path.read_text())


def rows_at(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(value):
    return hashlib.sha256(value).hexdigest()


def digest(value):
    return sha(json.dumps(value, sort_keys=True).encode())


def scores(rows):
    """Independent implementation of the official nine-task raw MCQ rule."""
    result = {}
    for arm in ARMS:
        task_correct, task_count = collections.Counter(), collections.Counter()
        budgets = collections.defaultdict(list)
        for row in rows:
            item = row["arms"][arm]
            correct = item["response"] is not None and row["target"] in item["response"]
            task_correct[row["task"]] += int(correct)
            task_count[row["task"]] += 1
            for field in ("recent", "history", "total"):
                key = "actual_visual_slots" if field == "total" else f"actual_{field}_visual_slots"
                budgets[field].append(item["budget_audit"][key] / 112)
        accuracy = {task: task_correct[task] / task_count[task] for task in task_count}
        groups = ([accuracy[task] for task in TASKS[:3] if task in accuracy],
                  [accuracy[task] for task in TASKS[3:] if task in accuracy])
        result[arm] = dict(
            questions=len(rows), correct=sum(task_correct.values()),
            question_weighted_accuracy=sum(task_correct.values()) / len(rows) if rows else None,
            nine_task_equal_mean=sum(accuracy.values()) / 9 if len(accuracy) == 9 else None,
            two_group_equal_mean=sum(sum(group) / len(group) for group in groups) / 2
            if len(accuracy) == 9 else None,
            per_task={task: dict(questions=task_count[task], correct=task_correct[task],
                                 accuracy=accuracy[task]) for task in accuracy},
            **{f"mean_{field}_tokens_per_head": sum(values) / len(values) if values else None
               for field, values in budgets.items()})
    return result


def audit(output_root, reference_root, *, allow_partial=False):
    protocol = read_json(output_root / "protocol.json")
    reference = read_json(reference_root / "protocol.json")
    manifest = read_json(output_root / "query_manifest.json")
    reference_manifest = read_json(reference_root / "query_manifest.json")
    rows = rows_at(output_root / "queries.jsonl")
    reference_rows = rows_at(reference_root / "queries.jsonl")
    reference_by_id = {str(row["query_id"]): row for row in reference_rows}
    checks, failures = {}, []

    def check(name, passed, detail=None):
        checks[name] = bool(passed)
        if not passed:
            failures.append(dict(check=name, detail=detail))

    check("manifest_matches_reference_all_fields_and_order", manifest == reference_manifest)
    check("manifest_contains_same_27_unique_questions",
          len(manifest) == 27 and len({str(q["query_id"]) for q in manifest}) == 27)
    check("three_questions_per_non_forward_task",
          collections.Counter(q["task"] for q in manifest) == collections.Counter({task: 3 for task in TASKS}))
    check("manifest_sha256_matches_protocol", digest(manifest) == protocol["query_sha256"])
    check("protocol_query_ids_match_manifest", protocol["query_ids"] == [q["query_id"] for q in manifest])
    check("removed_extra_video_pixel_cap", protocol.get("video_max_pixels", "missing") is None)
    check("fps_and_K500_unchanged", protocol["fps"] == reference["fps"] == .5
          and protocol["mean_visual_budget"] == reference["mean_visual_budget"] == 500
          and protocol["configured_visual_slots"] == reference["configured_visual_slots"] == 56000)
    stable_fields = ("model", "model_config_sha256", "annotation_sha256", "allocation_sha256",
                     "query_count", "arms", "chunk_size", "resize_policy", "position_scheme",
                     "wrapper", "system_prompt", "question_text", "selector", "max_new_tokens",
                     "greedy", "repetition_penalty", "eos", "decoder", "scoring",
                     "paired_features_and_prefills_shared")
    changed = [key for key in stable_fields if protocol.get(key) != reference.get(key)]
    check("other_scientific_protocol_fields_unchanged", not changed, changed)
    runtime = read_json(output_root / "runtime_processor.json")
    check("runtime_video_cap_is_null", runtime.get("video_max_pixels", "missing") is None)
    check("runtime_processor_size_is_not_overridden",
          runtime.get("size_override_passed_to_processor") is False)
    check("runtime_records_processor_defaults", all(runtime.get(key) is not None for key in
          ("default_video_size", "default_image_size", "patch_size", "temporal_patch_size", "spatial_merge_size")))
    model_config = Path(protocol["model"]) / "config.json"
    check("model_config_matches_protocol", sha(model_config.read_bytes()) == protocol["model_config_sha256"])
    with QUOTA_PATH.open(newline="") as handle:
        quotas = list(csv.DictReader(handle))
    quotas.sort(key=lambda row: (int(row["layer"]), int(row["kv_head"])))
    check("quota_table_matches_both_protocols", sha(QUOTA_PATH.read_bytes()) ==
          protocol["allocation_sha256"] == reference["allocation_sha256"])
    check("quota_table_is_56_one_56_four_and_global_K500", len(quotas) == 112
          and sum(row["recent_group"] == "one_frame" for row in quotas) == 56
          and sum(row["recent_group"] == "four_frames" for row in quotas) == 56
          and sum(int(row["total_quota"]) for row in quotas) == 56000)
    snapshot_failures, live_changes = [], []
    for name, expected_hash in protocol["code_sha256"].items():
        snapshot = output_root / "source_snapshot" / name
        if not snapshot.is_file() or sha(snapshot.read_bytes()) != expected_hash:
            snapshot_failures.append(name)
        current = ROOT / name
        if not current.is_file() or sha(current.read_bytes()) != expected_hash:
            live_changes.append(name)
    check("source_snapshot_hashes_match_protocol", not snapshot_failures, snapshot_failures)
    # Live code can legitimately change after a pinned background run. Report
    # it for traceability, but validate against the actual immutable snapshot.
    check("completed_all_manifest_questions", allow_partial or len(rows) == len(manifest),
          {"completed": len(rows), "expected": len(manifest)})
    check("results_unique_and_in_manifest_order", len(rows) <= len(manifest)
          and [str(row["query_id"]) for row in rows] == [str(q["query_id"]) for q in manifest[:len(rows)]])
    row_failures, different_responses, recent_differences = [], [], []
    frame_mismatches, selection_identical, pruned = [], [], collections.Counter()
    generation_hits = collections.Counter()
    for row, query in zip(rows, manifest):
        qid, row_bad = str(row["query_id"]), []
        for field in IDENTITY:
            if row.get(field) != query[field]:
                row_bad.append(f"identity.{field}")
        expected_prompt = sha(query["prompt"].encode())
        if row.get("official_prompt_sha256") != expected_prompt:
            row_bad.append("official_prompt_sha256")
        previous = reference_by_id[qid]
        if any(row.get(field) != previous.get(field) for field in ("frames", "encoded_frames")):
            frame_mismatches.append(qid)
        paired = row["paired_audit"]
        for field in ("shared_visual_features", "shared_independent_recent_prefills", "shared_question_and_decoder",
                      "shared_absolute_positions", "persistent_state_unchanged"):
            if paired.get(field) is not True:
                row_bad.append(f"paired.{field}")
        if paired.get("video_max_pixels", "missing") is not None or paired.get("processor_size_overridden") is not False:
            row_bad.append("paired.default_processor_input")
        if paired.get("recent_real_frames") != previous["paired_audit"]["recent_real_frames"]:
            row_bad.append("recent_real_frames_changed")
        candidate_slots = 56 * (sum(paired["recent_visual_unit_tokens"]) + paired["single_frame_visual_tokens"])
        records = {}
        for arm in ARMS:
            item = row["arms"][arm]
            a = records[arm] = item["budget_audit"]
            recent = a["actual_recent_visual_slots"]
            historical = a["actual_history_visual_slots"]
            total = a["actual_visual_slots"]
            flat = [value for layer in a["per_head_visual_slots"] for value in layer]
            if not (recent + historical == total == sum(flat) <= 56000 and len(flat) == 112):
                row_bad.append(f"{arm}.global_budget")
            limits = [500] * 112 if arm == "recent_only" else [int(q["total_quota"]) for q in quotas]
            if any(value < 0 or value > limit for value, limit in zip(flat, limits)):
                row_bad.append(f"{arm}.per_head_quota")
            if arm == "recent_only" and historical != 0:
                row_bad.append("recent_only.history_not_empty")
            if arm == "mild_history" and historical > sum(int(q["history_quota"]) for q in quotas):
                row_bad.append("mild_history.historical_quota")
            if a["recent_candidate_visual_slots"] != candidate_slots:
                row_bad.append(f"{arm}.candidate_slots")
            for field in ("persistent_state_unchanged_after_both_answers", "history_recent_independent_prefill",
                          "history_recent_time_ranges_disjoint", "fixed_pseudo_query", "common_absolute_video_positions",
                          "independent_last_raw_frame", "text_and_generation_not_in_visual_budget"):
                if a.get(field) is not True:
                    row_bad.append(f"{arm}.{field}")
            if a["official_prompt_sha256"] != expected_prompt:
                row_bad.append(f"{arm}.official_prompt_sha256")
            correct = item["response"] is not None and row["target"] in item["response"]
            if item.get("correct") != correct:
                row_bad.append(f"{arm}.stored_correct")
            pruned[arm] += a["recent_pruned_heads"]
            generation_hits[arm] += bool(a["hit_generation_limit"])
        for field in ("recent_candidate_visual_slots", "text_prefix_tokens", "question_suffix_tokens", "question_position",
                      "official_prompt_sha256", "shared_wrapper_ids_sha256", "eos_token_ids", "repetition_penalty"):
            if records["recent_only"][field] != records["mild_history"][field]:
                row_bad.append(f"arms.{field}")
        identical = records["recent_only"]["recent_selection_sha256"] == records["mild_history"]["recent_selection_sha256"]
        if paired["recent_selection_identical"] != identical:
            row_bad.append("paired.recent_selection_flag")
        (selection_identical if identical else recent_differences).append(qid)
        if row["arms"][ARMS[0]]["response"] != row["arms"][ARMS[1]]["response"]:
            different_responses.append(qid)
        if row_bad:
            row_failures.append(dict(query_id=qid, fields=row_bad))
    check("same_sampled_frame_counts_as_reference", not frame_mismatches, frame_mismatches)
    check("all_completed_rows_pass_budget_prompt_and_state_checks", not row_failures, row_failures)
    independent = scores(rows)
    summary_path = output_root / "summary.json"
    summary_mismatches = []
    if summary_path.is_file():
        summary = read_json(summary_path)
        for arm in ARMS:
            for key in ("questions", "correct", "question_weighted_accuracy", "nine_task_equal_mean",
                        "two_group_equal_mean", "mean_recent_tokens_per_head", "mean_history_tokens_per_head",
                        "mean_total_tokens_per_head"):
                left, right = independent[arm][key], summary["arms"][arm][key]
                if (left is None) != (right is None) or (left is not None and abs(left - right) > 1e-10):
                    summary_mismatches.append(f"{arm}.{key}")
        check("independent_official_rescoring_matches_report", not summary_mismatches, summary_mismatches)
    else:
        check("summary_report_exists", allow_partial)
    return dict(
        passed=all(checks.values()), checks=checks, failures=failures,
        questions=len(rows), expected_questions=len(manifest), reference_root=str(reference_root),
        live_code_changes_after_snapshot=live_changes,
        recent_selections_identical_questions=len(selection_identical),
        recent_selections_differ_questions=len(recent_differences),
        recent_selections_differ_query_ids=recent_differences,
        recent_selection_difference_expected="Control gives each head 500 recent slots; history arm one-frame heads receive 472 recent slots plus historical quota. Equal recent selections are not required.",
        different_responses=different_responses, recent_pruned_heads_total=dict(pruned),
        generation_limits_hit=dict(generation_hits),
        independent_scores=independent, reference_scores=scores(reference_rows),
        actual_budget_scope="post-assembly visual KV slots per arm / 112; excludes text, generated tokens and temporary prefill",
        full_run_started=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--validation-path", type=Path)
    args = parser.parse_args()
    validation = audit(args.output_root.resolve(), args.reference_root.resolve(), allow_partial=args.allow_partial)
    dest = args.validation_path or args.output_root / "validation.json"
    dest.write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: validation[key] for key in ("passed", "questions", "checks", "failures",
                                                     "recent_selections_differ_questions", "independent_scores")},
                     ensure_ascii=False, indent=2))
    raise SystemExit(0 if validation["passed"] else 1)


if __name__ == "__main__":
    main()
