#!/usr/bin/env python3
"""Evaluate whether head-specific lifecycle predicts held-out future reads."""

import argparse
import json
from pathlib import Path

import numpy as np

from head_analysis.prospective_lifecycle import (
    add_remaining_future_utility,
    assign_scores,
    fit_remaining_lifecycle,
    load_trace_jsonl,
    ranking_metrics,
)


METHODS = ("content", "global", "head", "shuffled", "reversed")


def split_videos(rows, train_fraction, seed):
    videos = np.asarray(sorted({row["video_id"] for row in rows}))
    if videos.size < 2:
        raise ValueError("At least two videos are required for a held-out evaluation")
    rng = np.random.default_rng(seed)
    rng.shuffle(videos)
    split = min(max(1, int(round(videos.size * train_fraction))), videos.size - 1)
    return set(videos[:split]), set(videos[split:])


def per_video_metrics(rows, methods=METHODS):
    by_video = {}
    for row in rows:
        by_video.setdefault(row["video_id"], []).append(row)
    return {
        video: {
            method: ranking_metrics(video_rows, f"score_{method}")
            for method in methods
        }
        for video, video_rows in by_video.items()
    }


def weighted_metric(per_video, method, metric):
    values = [per_video[video][method][metric] for video in per_video]
    weights = [per_video[video][method]["groups"] for video in per_video]
    return float(np.average(values, weights=weights))


def bootstrap_video_ci(per_video, method, samples, seed, reference=None):
    videos = sorted(per_video)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        chosen = rng.choice(videos, size=len(videos), replace=True)
        weights = np.asarray([
            per_video[video][method]["groups"] for video in chosen
        ], dtype=np.float64)
        method_values = np.asarray([
            per_video[video][method]["ndcg"] for video in chosen
        ], dtype=np.float64)
        value = float(np.average(method_values, weights=weights))
        if reference is not None:
            reference_values = np.asarray([
                per_video[video][reference]["ndcg"] for video in chosen
            ], dtype=np.float64)
            value -= float(np.average(reference_values, weights=weights))
        values.append(value)
    return [float(x) for x in np.percentile(values, [2.5, 97.5])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--query_source", action="append", default=None)
    parser.add_argument("--train_fraction", type=float, default=0.7)
    parser.add_argument("--folds", type=int, default=1,
                        help="Video-level cross-validation folds; 1 uses one train/test split.")
    parser.add_argument("--lifecycle_weight", type=float, default=1.0)
    parser.add_argument("--bootstrap_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--allow_compressed_trace", action="store_true")
    args = parser.parse_args()
    if args.query_source is None:
        args.query_source = ["visual_future"]

    rows = load_trace_jsonl(
        args.trace,
        require_full_cache=not args.allow_compressed_trace,
        query_sources=args.query_source,
    )
    rows = add_remaining_future_utility(rows)
    videos = np.asarray(sorted({row["video_id"] for row in rows}))
    if args.folds < 1 or args.folds > videos.size:
        raise ValueError("--folds must be between 1 and the number of videos")
    fold_reports = []
    if args.folds == 1:
        splits = [split_videos(rows, args.train_fraction, args.seed)]
    else:
        rng = np.random.default_rng(args.seed)
        rng.shuffle(videos)
        test_folds = np.array_split(videos, args.folds)
        all_videos = set(videos.tolist())
        splits = [
            (all_videos - set(fold.tolist()), set(fold.tolist()))
            for fold in test_folds
        ]

    held_out = []
    for fold_index, (train_ids, test_ids) in enumerate(splits):
        profiles = fit_remaining_lifecycle(rows, train_ids)
        fold_rows = [row for row in rows if row["video_id"] in test_ids]
        fold_rows = assign_scores(
            fold_rows, profiles, args.lifecycle_weight, args.seed + fold_index
        )
        held_out.extend(fold_rows)
        fold_reports.append({
            "fold": fold_index,
            "train_videos": sorted(train_ids),
            "test_videos": sorted(test_ids),
        })

    video_metrics = per_video_metrics(held_out)
    metrics = {}
    for method in METHODS:
        result = {
            "groups": sum(item[method]["groups"] for item in video_metrics.values()),
            "frame_auc": weighted_metric(video_metrics, method, "frame_auc"),
            "ndcg": weighted_metric(video_metrics, method, "ndcg"),
            "future_topk_recall": weighted_metric(
                video_metrics, method, "future_topk_recall"
            ),
        }
        result["ndcg_ci95"] = bootstrap_video_ci(
            video_metrics, method, args.bootstrap_samples, args.seed
        )
        metrics[method] = result

    head = metrics["head"]
    paired_deltas = {}
    for control in ("global", "shuffled", "reversed"):
        point = head["ndcg"] - metrics[control]["ndcg"]
        ci = bootstrap_video_ci(
            video_metrics, "head", args.bootstrap_samples, args.seed,
            reference=control,
        )
        paired_deltas[f"head_minus_{control}"] = {
            "ndcg_delta": point,
            "ci95": ci,
        }
    decision = {
        f"head_beats_{control}": paired_deltas[f"head_minus_{control}"]["ci95"][0] > 0
        for control in ("global", "shuffled", "reversed")
    }
    decision["go"] = all(decision.values())
    report = {
        "trace_schema_version": 1,
        "folds": fold_reports,
        "query_sources": args.query_source,
        "metrics": metrics,
        "per_video_metrics": video_metrics,
        "paired_ndcg_deltas": paired_deltas,
        "decision": decision,
        "decision_rule": "paired video-bootstrap 95% CI lower bound must exceed zero for head minus global, within-layer shuffled, and reversed-time NDCG",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
