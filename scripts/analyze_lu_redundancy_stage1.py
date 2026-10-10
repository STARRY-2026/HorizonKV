#!/usr/bin/env python3
"""Analyze cross-frame Value-vector redundancy in the LU K=1000 replay."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path("results/myobservations/lu_redundancy_stage1_k1000_20260921")
SOURCE = Path("results/myobservations/lu_failure_trace_first150_k1000_20260921")
TRACE = ROOT / "redundancy_trace.jsonl"


def mean(values):
    return float(np.mean(values)) if values else None


def pearson(xs, ys):
    if len(xs) < 3 or np.std(xs) == 0 or np.std(ys) == 0:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


def bootstrap_ci(values, seed=20260921, samples=10000):
    if not values:
        return None
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


manifest = json.loads((SOURCE / "manifest.json").read_text())
target_keys = {
    (str(row["video_id"]), int(row["question_index"]))
    for row in manifest["targets"]
}
records = [
    json.loads(line) for line in TRACE.read_text().splitlines() if line.strip()
]

expected_records = sum(len(video["conversations"]) for video in
                       json.loads((SOURCE / "filtered_anno.json").read_text()))
if len(records) != expected_records:
    raise RuntimeError(
        f"incomplete trace: found {len(records)} records, expected {expected_records}")

head_rows = []
question_rows = []
for record in records:
    key = (str(record["video_id"]), int(record["question_index"]))
    eligible = []
    for head in record["heads"]:
        similarity = head.get("mean_nearest_cross_frame_cosine")
        if similarity is None:
            continue
        row = {
            **head,
            "video_id": key[0],
            "question_index": key[1],
            "is_target": key in target_keys,
            "end_time": float(record["end_time"]),
            "full_sample": int(head["sampled_history_tokens"])
            == int(record["max_tokens_per_head"]),
        }
        row["estimated_duplicate_history_tokens_ge_090"] = (
            float(row["duplicate_fraction_ge_090"])
            * int(row["selected_history_tokens"]))
        head_rows.append(row)
        if row["full_sample"]:
            eligible.append(row)

    # Equal sample counts make nearest-neighbour redundancy comparable across
    # heads; otherwise larger candidate sets mechanically raise the maximum.
    low = [row for row in eligible if float(row["budget"]) < 1000]
    high = [row for row in eligible if float(row["budget"]) > 1000]
    question_rows.append({
        "video_id": key[0],
        "question_index": key[1],
        "is_target": key in target_keys,
        "end_time": float(record["end_time"]),
        "eligible_heads": len(eligible),
        "low_heads": len(low),
        "high_heads": len(high),
        "low_similarity": mean([r["mean_nearest_cross_frame_cosine"] for r in low]),
        "high_similarity": mean([r["mean_nearest_cross_frame_cosine"] for r in high]),
        "low_dup090": mean([r["duplicate_fraction_ge_090"] for r in low]),
        "high_dup090": mean([r["duplicate_fraction_ge_090"] for r in high]),
        "low_novelty": mean([r["mean_content_novelty"] for r in low]),
        "high_novelty": mean([r["mean_content_novelty"] for r in high]),
        "low_estimated_duplicate_tokens": mean([
            r["estimated_duplicate_history_tokens_ge_090"] for r in low]),
        "high_estimated_duplicate_tokens": mean([
            r["estimated_duplicate_history_tokens_ge_090"] for r in high]),
        "low_history_fraction": mean([
            r["selected_history_tokens"] / r["selected_visual_tokens"]
            for r in low if r["selected_visual_tokens"]]),
        "high_history_fraction": mean([
            r["selected_history_tokens"] / r["selected_visual_tokens"]
            for r in high if r["selected_visual_tokens"]]),
        "low_history_tokens": mean([r["selected_history_tokens"] for r in low]),
        "high_history_tokens": mean([r["selected_history_tokens"] for r in high]),
        "budget_similarity_r": pearson(
            [float(r["budget"]) for r in eligible],
            [float(r["mean_nearest_cross_frame_cosine"]) for r in eligible]),
        "budget_dup090_r": pearson(
            [float(r["budget"]) for r in eligible],
            [float(r["duplicate_fraction_ge_090"]) for r in eligible]),
    })


def complete(rows, *fields):
    return [row for row in rows if all(row[field] is not None for field in fields)]


def group_summary(rows):
    paired = complete(rows, "low_similarity", "high_similarity",
                      "low_dup090", "high_dup090")
    similarity_deltas = [
        r["high_similarity"] - r["low_similarity"] for r in paired]
    duplicate_fraction_deltas = [
        r["high_dup090"] - r["low_dup090"] for r in paired]
    duplicate_token_rows = complete(
        rows, "low_estimated_duplicate_tokens", "high_estimated_duplicate_tokens")
    duplicate_token_deltas = [
        r["high_estimated_duplicate_tokens"]
        - r["low_estimated_duplicate_tokens"] for r in duplicate_token_rows]
    history_rows = complete(rows, "low_history_fraction", "high_history_fraction",
                            "low_history_tokens", "high_history_tokens")
    history_fraction_deltas = [
        r["high_history_fraction"] - r["low_history_fraction"]
        for r in history_rows]
    history_token_deltas = [
        r["high_history_tokens"] - r["low_history_tokens"]
        for r in history_rows]
    return {
        "questions": len(rows),
        "paired_questions": len(paired),
        "mean_eligible_heads": mean([r["eligible_heads"] for r in rows]),
        "low_mean_similarity": mean([r["low_similarity"] for r in paired]),
        "high_mean_similarity": mean([r["high_similarity"] for r in paired]),
        "high_minus_low_similarity": mean(similarity_deltas),
        "high_minus_low_similarity_bootstrap_95ci": bootstrap_ci(similarity_deltas),
        "low_duplicate_fraction_ge_090": mean([r["low_dup090"] for r in paired]),
        "high_duplicate_fraction_ge_090": mean([r["high_dup090"] for r in paired]),
        "high_minus_low_duplicate_fraction_ge_090": mean(duplicate_fraction_deltas),
        "high_minus_low_duplicate_fraction_bootstrap_95ci": bootstrap_ci(
            duplicate_fraction_deltas),
        "low_estimated_duplicate_history_tokens_ge_090": mean([
            r["low_estimated_duplicate_tokens"] for r in duplicate_token_rows]),
        "high_estimated_duplicate_history_tokens_ge_090": mean([
            r["high_estimated_duplicate_tokens"] for r in duplicate_token_rows]),
        "high_minus_low_estimated_duplicate_tokens": mean(duplicate_token_deltas),
        "high_minus_low_estimated_duplicate_tokens_bootstrap_95ci": bootstrap_ci(
            duplicate_token_deltas),
        "low_history_fraction": mean([
            r["low_history_fraction"] for r in history_rows]),
        "high_history_fraction": mean([
            r["high_history_fraction"] for r in history_rows]),
        "high_minus_low_history_fraction": mean(history_fraction_deltas),
        "high_minus_low_history_fraction_bootstrap_95ci": bootstrap_ci(
            history_fraction_deltas),
        "low_history_tokens": mean([r["low_history_tokens"] for r in history_rows]),
        "high_history_tokens": mean([r["high_history_tokens"] for r in history_rows]),
        "high_minus_low_history_tokens": mean(history_token_deltas),
        "mean_within_question_budget_similarity_r": mean([
            r["budget_similarity_r"] for r in rows
            if r["budget_similarity_r"] is not None]),
        "mean_within_question_budget_dup090_r": mean([
            r["budget_dup090_r"] for r in rows
            if r["budget_dup090_r"] is not None]),
    }


summary = {
    "trace_records": len(records),
    "target_records": sum(r["is_target"] for r in question_rows),
    "measurement_note": (
        "Primary comparisons retain only heads with exactly 128 sampled "
        "historical tokens and at least two historical frames, preventing "
        "nearest-neighbour candidate-count bias."),
    "all": group_summary(question_rows),
    "target_failures": group_summary([r for r in question_rows if r["is_target"]]),
    "prefix_controls": group_summary([r for r in question_rows if not r["is_target"]]),
    "per_question": question_rows,
}

# Stable head-level profile across all fully sampled observations.
by_head = defaultdict(list)
for row in head_rows:
    if row["full_sample"]:
        by_head[(int(row["layer"]), int(row["kv_head"]))].append(row)
summary["per_head"] = [
    {
        "layer": key[0],
        "kv_head": key[1],
        "budget": rows[0]["budget"],
        "observations": len(rows),
        "mean_similarity": mean([
            r["mean_nearest_cross_frame_cosine"] for r in rows]),
        "mean_duplicate_fraction_ge_090": mean([
            r["duplicate_fraction_ge_090"] for r in rows]),
        "mean_novelty": mean([r["mean_content_novelty"] for r in rows]),
        "mean_distinct_history_frames": mean([
            r["distinct_history_frames"] for r in rows]),
    }
    for key, rows in sorted(by_head.items())
]

(ROOT / "analysis.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({key: value for key, value in summary.items()
                  if key not in {"per_question", "per_head"}},
                 indent=2, ensure_ascii=False))
