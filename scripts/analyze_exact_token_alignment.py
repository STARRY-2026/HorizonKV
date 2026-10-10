#!/usr/bin/env python3
"""Exact HERMES/LU visual-token retention and readout alignment."""

from __future__ import annotations

import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path("results/myobservations/exact_token_alignment_k1000_20260921")
SOURCE = Path("results/myobservations/lu_failure_trace_first150_k1000_20260921")
RESULT_ROOT = Path("results/qwen2.5_vl_7b/streamingbench")


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def keyed(rows):
    return {(str(row["video_id"]), int(row["question_index"])): row for row in rows}


def mean(values):
    return float(np.mean(values)) if values else None


def bootstrap_ci(values, seed=20260921, samples=10000):
    if not values:
        return None
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(1)
    return [float(v) for v in np.quantile(draws, [0.025, 0.975])]


def bootstrap_group_difference(left, right, seed=20260921, samples=50000):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = (
        rng.choice(left, size=(samples, len(left)), replace=True).mean(1)
        - rng.choice(right, size=(samples, len(right)), replace=True).mean(1)
    )
    return {
        "difference": float(left.mean() - right.mean()),
        "bootstrap_95ci": [float(v) for v in np.quantile(draws, [0.025, 0.975])],
        "bootstrap_two_sided_p": float(
            min(1.0, 2.0 * min(np.mean(draws <= 0), np.mean(draws >= 0))))
    }


def load_results(tag):
    paths = list(RESULT_ROOT.glob(f"*{tag}/results.csv"))
    if len(paths) != 1:
        raise RuntimeError((tag, paths))
    grouped = defaultdict(list)
    with paths[0].open() as handle:
        for row in csv.DictReader(handle):
            grouped[str(row["video_id"])].append(row)
    return {(video_id, index): row for video_id, rows in grouped.items()
            for index, row in enumerate(rows)}


manifest = json.loads((SOURCE / "manifest.json").read_text())
target_keys = {
    (str(row["video_id"]), int(row["question_index"]))
    for row in manifest["targets"]
}
h_records = keyed(load_jsonl(ROOT / "hermes_token_trace.jsonl"))
l_records = keyed(load_jsonl(ROOT / "lu_token_trace.jsonl"))
if h_records.keys() != l_records.keys() or len(h_records) != 28:
    raise RuntimeError(
        f"incomplete/misaligned traces: H={len(h_records)}, LU={len(l_records)}")

question_rows = []
head_rows = []
frame_uid_mismatches = 0
for key in sorted(h_records):
    h_record, l_record = h_records[key], l_records[key]
    h_heads = {(int(h["layer"]), int(h["kv_head"])): h for h in h_record["heads"]}
    l_heads = {(int(h["layer"]), int(h["kv_head"])): h for h in l_record["heads"]}
    if h_heads.keys() != l_heads.keys() or len(h_heads) != 112:
        raise RuntimeError(f"head mismatch at {key}")

    for head_key in sorted(h_heads):
        h, l = h_heads[head_key], l_heads[head_key]
        h_uids = [int(v) for v in h["visual_token_uids"]]
        l_uids = [int(v) for v in l["visual_token_uids"]]
        h_frames = [int(v) for v in h["visual_frame_ids"]]
        l_frames = [int(v) for v in l["visual_frame_ids"]]
        h_attention = np.asarray(h["visual_attention"], dtype=np.float64)
        l_attention = np.asarray(l["visual_attention"], dtype=np.float64)
        if not (len(h_uids) == len(h_frames) == h_attention.size):
            raise RuntimeError(f"H length mismatch at {key}/{head_key}")
        if not (len(l_uids) == len(l_frames) == l_attention.size):
            raise RuntimeError(f"LU length mismatch at {key}/{head_key}")

        h_map = dict(zip(h_uids, h_frames))
        l_map = dict(zip(l_uids, l_frames))
        h_set, l_set = set(h_uids), set(l_uids)
        common = h_set & l_set
        frame_uid_mismatches += sum(h_map[uid] != l_map[uid] for uid in common)
        h_only = h_set - l_set
        l_only = l_set - h_set
        h_total_mass = float(h_attention.sum())
        l_total_mass = float(l_attention.sum())
        h_only_mass = float(sum(
            score for uid, score in zip(h_uids, h_attention) if uid in h_only))
        l_only_mass = float(sum(
            score for uid, score in zip(l_uids, l_attention) if uid in l_only))

        top_missing = {}
        for fraction in (0.01, 0.05, 0.10):
            k = max(1, int(np.ceil(len(h_uids) * fraction))) if h_uids else 0
            if k:
                top_idx = np.argpartition(h_attention, -k)[-k:]
                top_missing[str(fraction)] = float(np.mean([
                    h_uids[int(i)] not in l_set for i in top_idx]))
            else:
                top_missing[str(fraction)] = None

        head_rows.append({
            "video_id": key[0],
            "question_index": key[1],
            "is_target": key in target_keys,
            "layer": head_key[0],
            "kv_head": head_key[1],
            "hermes_tokens": len(h_set),
            "lu_tokens": len(l_set),
            "common_tokens": len(common),
            "hermes_only_tokens": len(h_only),
            "lu_only_tokens": len(l_only),
            "hermes_token_recall_in_lu": len(common) / len(h_set) if h_set else None,
            "jaccard": len(common) / len(h_set | l_set) if h_set or l_set else None,
            "hermes_only_attention_fraction": (
                h_only_mass / h_total_mass if h_total_mass > 0 else None),
            "lu_only_attention_fraction": (
                l_only_mass / l_total_mass if l_total_mass > 0 else None),
            "hermes_visual_attention_mass": h_total_mass,
            "hermes_only_attention_mass": h_only_mass,
            "top01_hermes_tokens_missing_in_lu": top_missing["0.01"],
            "top05_hermes_tokens_missing_in_lu": top_missing["0.05"],
            "top10_hermes_tokens_missing_in_lu": top_missing["0.1"],
        })

    rows = [row for row in head_rows
            if row["video_id"] == key[0] and row["question_index"] == key[1]]
    h_mass = sum(row["hermes_visual_attention_mass"] for row in rows)
    h_only_mass = sum(row["hermes_only_attention_mass"] for row in rows)
    question_rows.append({
        "video_id": key[0],
        "question_index": key[1],
        "question": h_record["question"],
        "end_time": h_record["end_time"],
        "is_target": key in target_keys,
        "mean_hermes_token_recall_in_lu": mean([
            row["hermes_token_recall_in_lu"] for row in rows]),
        "mean_jaccard": mean([row["jaccard"] for row in rows]),
        "attention_weighted_hermes_only_fraction": (
            h_only_mass / h_mass if h_mass > 0 else None),
        "mean_head_hermes_only_attention_fraction": mean([
            row["hermes_only_attention_fraction"] for row in rows
            if row["hermes_only_attention_fraction"] is not None]),
        "mean_top01_hermes_tokens_missing_in_lu": mean([
            row["top01_hermes_tokens_missing_in_lu"] for row in rows]),
        "mean_top05_hermes_tokens_missing_in_lu": mean([
            row["top05_hermes_tokens_missing_in_lu"] for row in rows]),
        "mean_top10_hermes_tokens_missing_in_lu": mean([
            row["top10_hermes_tokens_missing_in_lu"] for row in rows]),
    })


def group_summary(rows):
    fields = [
        "mean_hermes_token_recall_in_lu",
        "mean_jaccard",
        "attention_weighted_hermes_only_fraction",
        "mean_head_hermes_only_attention_fraction",
        "mean_top01_hermes_tokens_missing_in_lu",
        "mean_top05_hermes_tokens_missing_in_lu",
        "mean_top10_hermes_tokens_missing_in_lu",
    ]
    return {
        "questions": len(rows),
        **{field: mean([row[field] for row in rows]) for field in fields},
        **{field + "_bootstrap_95ci": bootstrap_ci([row[field] for row in rows])
           for field in fields},
    }


h_result = load_results("exact-token-hermes-native-k1000")
l_result = load_results("exact-token-lu-native-k1000")
result_checks = {
    "hermes_correct": sum(h_result[k]["pred_choice"] == h_result[k]["correct_choice"]
                          for k in h_result),
    "lu_correct": sum(l_result[k]["pred_choice"] == l_result[k]["correct_choice"]
                      for k in l_result),
    "target_hermes_correct": sum(
        h_result[k]["pred_choice"] == h_result[k]["correct_choice"]
        for k in target_keys),
    "target_lu_correct": sum(
        l_result[k]["pred_choice"] == l_result[k]["correct_choice"]
        for k in target_keys),
}

targets = [row for row in question_rows if row["is_target"]]
controls = [row for row in question_rows if not row["is_target"]]
summary = {
    "questions": len(question_rows),
    "heads_per_question": 112,
    "frame_uid_mismatches": frame_uid_mismatches,
    "result_checks": result_checks,
    "all": group_summary(question_rows),
    "target_failures": group_summary(targets),
    "prefix_controls": group_summary(controls),
    "target_minus_control": {
        field: bootstrap_group_difference(
            [r[field] for r in targets], [r[field] for r in controls])
        for field in (
            "mean_hermes_token_recall_in_lu",
            "attention_weighted_hermes_only_fraction",
            "mean_top01_hermes_tokens_missing_in_lu",
            "mean_top05_hermes_tokens_missing_in_lu",
            "mean_top10_hermes_tokens_missing_in_lu",
        )
    },
    "per_question": question_rows,
}
(ROOT / "analysis.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({k: v for k, v in summary.items() if k != "per_question"},
                 ensure_ascii=False, indent=2))
