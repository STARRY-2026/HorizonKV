#!/usr/bin/env python3
"""Evaluate profile-gated pseudo scores against future task-facing reads.

This diagnostic keeps the HERMES pseudo observations but changes the scoring
logic: pseudo attention is treated as online evidence, while temporal lifecycle
curves decide how strongly each physical KV head should retain that evidence.
No layer-depth rule is used by the new scores.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from head_analysis.prospective_lifecycle import (
    DEFAULT_AGE_EDGES,
    age_bin_indices,
    align_predictor_to_future_targets,
    fit_remaining_lifecycle,
    load_trace_jsonl,
    ranking_metrics,
)
from scripts.analyze_prospective_lifecycle import (
    bootstrap_video_ci,
    per_video_metrics,
    weighted_metric,
)


METHODS = (
    "hermes",
    "pseudo",
    "pseudo_global_lifecycle",
    "pseudo_head_lifecycle",
    "pseudo_head_shuffled",
    "pseudo_head_reversed",
    "pseudo_future_gate",
)


def _attention_density(row):
    if "attention_density" in row:
        return max(0.0, float(row["attention_density"]))
    return max(0.0, float(row.get("attention_mass", 0.0))) / max(
        1, int(row.get("token_count", 1))
    )


def _build_shuffled_heads(heads, seed):
    rng = np.random.default_rng(seed)
    keys = sorted(heads)
    if not keys:
        return {}
    permuted = keys.copy()
    rng.shuffle(permuted)
    return {dst: heads[src] for dst, src in zip(keys, permuted)}


def _assign_profile_gated_scores(rows, profiles, lifecycle_weight, gate_weight, seed):
    edges = profiles.get("edges", DEFAULT_AGE_EDGES)
    global_curve = profiles["global"]
    heads = profiles["heads"]
    conditional_global = profiles.get("conditional_global", {})
    conditional_heads = profiles.get("conditional_heads", {})
    shuffled_heads = _build_shuffled_heads(heads, seed)
    shuffled_conditional = _build_shuffled_heads(conditional_heads, seed)

    scored = []
    for row in rows:
        item = dict(row)
        idx = int(age_bin_indices([float(row["age_sec"])], edges)[0])
        head_key = (int(row["layer"]), int(row["kv_head"]))
        content_bin = int(row.get("content_bin", -1))

        if content_bin >= 0 and conditional_global:
            global_life = conditional_global.get(content_bin, global_curve)
            head_life = conditional_heads.get(
                (head_key[0], head_key[1], content_bin), global_life
            )
            shuffled_life = shuffled_conditional.get(
                (head_key[0], head_key[1], content_bin), global_life
            )
        else:
            global_life = global_curve
            head_life = heads.get(head_key, global_life)
            shuffled_life = shuffled_heads.get(head_key, global_life)

        pseudo = math.log(_attention_density(row) + 1e-12)
        item["score_pseudo"] = pseudo
        item["score_pseudo_global_lifecycle"] = (
            pseudo + lifecycle_weight * math.log(float(global_life[idx]) + 1e-12)
        )
        item["score_pseudo_head_lifecycle"] = (
            pseudo + lifecycle_weight * math.log(float(head_life[idx]) + 1e-12)
        )
        item["score_pseudo_head_shuffled"] = (
            pseudo + lifecycle_weight * math.log(float(shuffled_life[idx]) + 1e-12)
        )
        item["score_pseudo_head_reversed"] = (
            pseudo + lifecycle_weight * math.log(float(head_life[-1 - idx]) + 1e-12)
        )

        # A residual gate: only add head-specific temporal protection when this
        # head's lifecycle is above the global lifecycle at the same age.
        advantage = max(
            0.0,
            math.log(float(head_life[idx]) + 1e-12)
            - math.log(float(global_life[idx]) + 1e-12),
        )
        item["score_pseudo_future_gate"] = (
            pseudo
            + lifecycle_weight * math.log(float(global_life[idx]) + 1e-12)
            + gate_weight * advantage
        )
        scored.append(item)
    return scored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--lifecycle_weight", type=float, default=0.1)
    parser.add_argument("--gate_weight", type=float, default=0.5)
    parser.add_argument("--bootstrap_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--strict_future", action="store_true")
    args = parser.parse_args()

    rows = load_trace_jsonl(
        args.trace,
        require_full_cache=True,
        query_sources={"hermes", "true_question"},
    )
    rows = align_predictor_to_future_targets(
        rows, include_same_observation=not args.strict_future
    )
    videos = np.asarray(sorted({row["video_id"] for row in rows}))
    if args.folds < 2 or args.folds > videos.size:
        raise ValueError("--folds must be between 2 and the number of videos")
    rng = np.random.default_rng(args.seed)
    rng.shuffle(videos)
    all_videos = set(videos.tolist())
    splits = [
        (all_videos - set(fold.tolist()), set(fold.tolist()))
        for fold in np.array_split(videos, args.folds)
    ]

    held_out = []
    fold_reports = []
    for fold_index, (train_ids, test_ids) in enumerate(splits):
        profiles = fit_remaining_lifecycle(rows, train_ids)
        fold_rows = [row for row in rows if row["video_id"] in test_ids]
        for item in fold_rows:
            item["score_hermes"] = math.log(_attention_density(item) + 1e-12)
        held_out.extend(
            _assign_profile_gated_scores(
                fold_rows,
                profiles,
                args.lifecycle_weight,
                args.gate_weight,
                args.seed + fold_index,
            )
        )
        fold_reports.append({
            "fold": fold_index,
            "train_videos": sorted(train_ids),
            "test_videos": sorted(test_ids),
        })

    video_metrics = per_video_metrics(held_out, methods=METHODS)
    metrics = {}
    for method in METHODS:
        metrics[method] = {
            "groups": sum(item[method]["groups"] for item in video_metrics.values()),
            "frame_auc": weighted_metric(video_metrics, method, "frame_auc"),
            "ndcg": weighted_metric(video_metrics, method, "ndcg"),
            "future_topk_recall": weighted_metric(
                video_metrics, method, "future_topk_recall"
            ),
            "ndcg_ci95": bootstrap_video_ci(
                video_metrics, method, args.bootstrap_samples, args.seed
            ),
        }

    deltas = {}
    reference = "pseudo"
    for method in METHODS:
        if method == reference:
            continue
        deltas[f"{method}_minus_{reference}"] = {
            "ndcg_delta": metrics[method]["ndcg"] - metrics[reference]["ndcg"],
            "ci95": bootstrap_video_ci(
                video_metrics,
                method,
                args.bootstrap_samples,
                args.seed,
                reference=reference,
            ),
        }

    report = {
        "trace_schema_version": 1,
        "predictor_source": "hermes",
        "target_source": "true_question",
        "target_includes_same_observation": not args.strict_future,
        "note": (
            "Existing pilot traces contain the standard HERMES pseudo score, "
            "not separate local/global pseudo streams. This evaluates the "
            "single-pseudo profile-gated variant."
        ),
        "folds": fold_reports,
        "params": {
            "lifecycle_weight": args.lifecycle_weight,
            "gate_weight": args.gate_weight,
        },
        "metrics": metrics,
        "paired_ndcg_deltas": deltas,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
