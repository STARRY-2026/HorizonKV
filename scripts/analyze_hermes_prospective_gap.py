#!/usr/bin/env python3
"""Compare HERMES and lifecycle scores against identical future QA reads."""

import argparse
import json
from pathlib import Path

import numpy as np

from head_analysis.prospective_lifecycle import (
    align_predictor_to_future_targets,
    assign_scores,
    fit_remaining_lifecycle,
    load_trace_jsonl,
)
from scripts.analyze_prospective_lifecycle import (
    bootstrap_video_ci,
    per_video_metrics,
    weighted_metric,
)


METHODS = ("content", "global", "head", "shuffled", "reversed", "hermes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--lifecycle_weight", type=float, default=1.0)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument(
        "--strict_future", action="store_true",
        help="Exclude a true question at the HERMES decision observation.",
    )
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
        held_out.extend(assign_scores(
            fold_rows, profiles, args.lifecycle_weight, args.seed + fold_index
        ))
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

    gaps = {}
    for method in ("content", "global", "hermes"):
        gaps[f"head_minus_{method}"] = {
            "ndcg_delta": metrics["head"]["ndcg"] - metrics[method]["ndcg"],
            "ci95": bootstrap_video_ci(
                video_metrics, "head", args.bootstrap_samples, args.seed,
                reference=method,
            ),
        }
    report = {
        "trace_schema_version": 1,
        "predictor_source": "hermes",
        "target_source": "true_question",
        "target_includes_same_observation": not args.strict_future,
        "folds": fold_reports,
        "metrics": metrics,
        "per_video_metrics": video_metrics,
        "paired_ndcg_deltas": gaps,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
