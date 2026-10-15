"""Deterministic, evaluation-disjoint data selection for bounded LU calibration.

This module only reads annotations. It never loads a model, decodes videos, or
starts evaluation. Original questions, timestamps, and source indices survive
selection unchanged; no question is moved to an earlier video timestamp.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import itertools
import json
import math
from numbers import Integral
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = WORKSPACE / "hermes/results/myobservations/lu_kv_balanced100v_20260915/calibration_100v.json"
DEFAULT_EXCLUDE = WORKSPACE / "hermes/data/streamingbench/streamingbench_realtime_half_userpaths.json"
MAX_CALIBRATION_VIDEOS = 10
MAX_CALIBRATION_TIME = 120.0
TOKEN_WINDOW_PROTOCOL = {
    "model_family": "Qwen2.5-VL",
    "video_resolution": "native_processor",
    "sample_fps": 0.5,
    "temporal_patch_frames": 2,
    "budget_scope": "visual",
    "initial_text_outside_visual_budget": True,
    "protected_recent_visual_tokens": 32,
    "scoring_query_positions": "last_min_32_current_visual_queries",
    "attention_dtype": "float32",
    "attention": "scaled_QK_causal_softmax",
    "query_reduction": "mean",
    "gqa_reduction": "mean",
    "pooling": "avgpool",
    "pooling_kernel_size": 5,
    "hermes_probes": 0,
    "recency_score_mixing": False,
    "previous_qa_in_selector": False,
}


def canonical_sha256(value):
    """Fingerprint JSON values independently of source indentation/key order."""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_annotations(path):
    path = Path(path).expanduser().resolve()
    raw = path.read_bytes()
    values = json.loads(raw)
    if not isinstance(values, list) or any(not isinstance(video, dict) for video in values):
        raise ValueError(f"Annotation source must contain a list of videos: {path}")
    return path, values, hashlib.sha256(raw).hexdigest()


def _identity(video):
    if "video_id" not in video or not isinstance(video.get("video_path"), str):
        raise ValueError("Each video requires video_id and video_path")
    return str(video["video_id"]), Path(video["video_path"]).name


def _question_time(question):
    value = question.get("end_time")
    if isinstance(value, bool):
        raise ValueError("Question end_time must be finite numeric seconds")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Question end_time must be finite numeric seconds") from exc
    if not math.isfinite(result):
        raise ValueError("Question end_time must be finite numeric seconds")
    return result


def _question_subset(questions, covered):
    """Prefer new tasks, then longer original prefixes; select at most two."""
    size = min(2, len(questions))
    choices = itertools.combinations(questions, size)

    def key(items):
        tasks = {str(question.get("task", "")) for question in items}
        times = [_question_time(question) for question in items]
        indices = tuple(sorted(question["source_question_index"] for question in items))
        return (-len(tasks - covered), -max(times), -sum(times), indices)

    selected = min(choices, key=key)
    return sorted(selected, key=lambda question: (_question_time(question), question["source_question_index"]))


def select_calibration(source, exclude, num_videos=10, max_time=120):
    """Return at most ten original-video prefixes and source-bound metadata.

    Greedy selection first adds uncovered tasks and then prefers longer eligible
    prefixes. Ties use video ID and basename, so input ordering has no effect.
    Videos present in the evaluation exclusion by either ID or basename are
    removed. If fewer than requested remain, all eligible videos are returned
    and the metadata records the shortfall.
    """
    if not isinstance(num_videos, Integral) or isinstance(num_videos, bool) or not 1 <= num_videos <= MAX_CALIBRATION_VIDEOS:
        raise ValueError("Bounded LU calibration requires 1 <= num_videos <= 10")
    if isinstance(max_time, bool):
        raise ValueError("Bounded LU calibration requires 0 < max_time <= 120 seconds")
    try:
        max_time = float(max_time)
    except (TypeError, ValueError) as exc:
        raise ValueError("Bounded LU calibration requires 0 < max_time <= 120 seconds") from exc
    if not math.isfinite(max_time) or not 0 < max_time <= MAX_CALIBRATION_TIME:
        raise ValueError("Bounded LU calibration requires 0 < max_time <= 120 seconds")
    source_path, videos, source_hash = _read_annotations(source)
    exclude_path, excluded, exclude_hash = _read_annotations(exclude)
    excluded_ids, excluded_names = set(), set()
    for video in excluded:
        video_id, basename = _identity(video)
        excluded_ids.add(video_id)
        excluded_names.add(basename)
    candidates, seen_ids, seen_names = [], set(), set()
    overlap_count = 0
    eligible_question_count = 0
    for video in videos:
        video_id, basename = _identity(video)
        if video_id in seen_ids or basename in seen_names:
            raise ValueError("Calibration source contains duplicate video IDs or basenames")
        seen_ids.add(video_id)
        seen_names.add(basename)
        if video_id in excluded_ids or basename in excluded_names:
            overlap_count += 1
            continue
        questions = video.get("conversations")
        if not isinstance(questions, list) or any(not isinstance(question, dict) for question in questions):
            raise ValueError("Each calibration video requires a list of conversations")
        eligible, seen_indices = [], set()
        for index, question in enumerate(questions):
            source_index = question.get("source_question_index", index)
            if not isinstance(source_index, Integral) or isinstance(source_index, bool) or source_index < 0:
                raise ValueError("source_question_index must be a non-negative integer")
            if source_index in seen_indices:
                raise ValueError("Duplicate source_question_index in calibration video")
            seen_indices.add(source_index)
            end_time = _question_time(question)
            if 0 < end_time <= max_time:
                copied = deepcopy(question)
                copied.setdefault("source_question_index", int(source_index))
                eligible.append(copied)
        if eligible:
            eligible_question_count += len(eligible)
            candidates.append((deepcopy(video), eligible))
    eligible_video_count = len(candidates)
    selected, covered = [], set()
    while candidates and len(selected) < num_videos:
        choices = []
        for position, (video, questions) in enumerate(candidates):
            chosen = _question_subset(questions, covered)
            tasks = {str(question.get("task", "")) for question in chosen}
            times = [_question_time(question) for question in chosen]
            video_id, basename = _identity(video)
            key = (-len(tasks - covered), -max(times), -sum(times), video_id, basename)
            choices.append((key, position, chosen, tasks))
        _, position, chosen, tasks = min(choices, key=lambda entry: entry[0])
        video, _ = candidates.pop(position)
        video["conversations"] = chosen
        selected.append(video)
        covered.update(tasks)
    times = [_question_time(question) for video in selected for question in video["conversations"]]
    task_counts = Counter(str(question.get("task", "")) for video in selected for question in video["conversations"])
    selected_ids = [str(video["video_id"]) for video in selected]
    selected_names = [Path(video["video_path"]).name for video in selected]
    if set(selected_ids) & excluded_ids or set(selected_names) & excluded_names:
        raise AssertionError("Selected calibration overlaps held-out evaluation")
    metadata = {
        "schema_version": 1,
        "purpose": "bounded_attention_only_LU_calibration",
        "full_evaluation_started": False,
        "source_path": str(source_path),
        "exclude_path": str(exclude_path),
        "source_sha256": source_hash,
        "exclude_sha256": exclude_hash,
        "source_hashes": {str(source_path): source_hash, str(exclude_path): exclude_hash},
        "selected_canonical_sha256": canonical_sha256(selected),
        "source_video_count": len(videos),
        "requested_video_count": int(num_videos),
        "eligible_video_count": eligible_video_count,
        "eligible_question_count": eligible_question_count,
        "excluded_overlap_video_count": overlap_count,
        "selected_video_count": len(selected),
        "selected_question_count": len(times),
        "video_shortfall": max(0, int(num_videos) - len(selected)),
        "maximum_question_time_seconds": max_time,
        "selected_question_time_range_seconds": [min(times), max(times)] if times else [],
        "selected_prefix_seconds_sum": sum(max(_question_time(question) for question in video["conversations"]) for video in selected),
        "video_ids": selected_ids,
        "video_basenames": selected_names,
        "task_counts": dict(sorted(task_counts.items())),
        "evaluation_id_overlap": [],
        "evaluation_basename_overlap": [],
        "selection_policy": "greedy_new_task_coverage_then_longest_prefix_then_video_identity",
        "questions_per_video_limit": 2,
        "original_question_timestamps_preserved": True,
        "original_source_question_indices_preserved": True,
        "token_window_protocol": deepcopy(TOKEN_WINDOW_PROTOCOL),
    }
    return selected, metadata
