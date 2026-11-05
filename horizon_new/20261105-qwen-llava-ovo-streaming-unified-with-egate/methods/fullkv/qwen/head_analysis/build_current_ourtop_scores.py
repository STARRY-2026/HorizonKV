#!/usr/bin/env python3
"""Aggregate current-cache true-question traces into an ourtop-style prior."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_trace(paths):
    values = defaultdict(list)
    records = 0
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                records += 1
                for head in record["heads"]:
                    value = head.get("historical_top100_mean_attention")
                    if value is not None:
                        values[(int(head["layer"]), int(head["kv_head"]))].append(float(value))
    return values, records


def load_old_scores(path):
    result = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result[(int(row["layer"]), int(row["kv_head"]))] = float(row["score"])
    return result


def percentile_ranks(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    for rank, idx in enumerate(order):
        ranks[idx] = rank
    return ranks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="append", required=True)
    parser.add_argument("--old_scores", required=True)
    parser.add_argument("--old_policy", required=True)
    parser.add_argument("--output_scores", required=True)
    parser.add_argument("--output_policy", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    grouped, records = read_trace(args.trace)
    expected = [(layer, head) for layer in range(28) for head in range(4)]
    missing = [key for key in expected if not grouped.get(key)]
    if missing:
        raise ValueError(f"Missing observations for {len(missing)} heads: {missing[:8]}")

    scores = {key: float(np.mean(grouped[key])) for key in expected}
    score_path = Path(args.output_scores)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    with score_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["layer", "kv_head", "score", "internal_top100_mean_attention", "num_observations"],
        )
        writer.writeheader()
        for layer, head in expected:
            writer.writerow(
                {
                    "layer": layer,
                    "kv_head": head,
                    "score": scores[(layer, head)],
                    "internal_top100_mean_attention": scores[(layer, head)],
                    "num_observations": len(grouped[(layer, head)]),
                }
            )

    old_policy_rows = []
    with Path(args.old_policy).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["layer"]), int(row["kv_head"]))
            old_policy_rows.append(
                {
                    "layer": key[0],
                    "kv_head": key[1],
                    "long_score": scores[key],
                    "short_score": float(row["short_score"]),
                }
            )
    policy_path = Path(args.output_policy)
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    with policy_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["layer", "kv_head", "long_score", "short_score"])
        writer.writeheader()
        writer.writerows(old_policy_rows)

    old = load_old_scores(args.old_scores)
    old_flat = np.asarray([old[key] for key in expected], dtype=np.float64)
    new_flat = np.asarray([scores[key] for key in expected], dtype=np.float64)
    rho = float(np.corrcoef(percentile_ranks(old_flat), percentile_ranks(new_flat))[0, 1])
    top_count = max(1, round(0.1 * len(expected)))
    old_top = set(np.argsort(old_flat)[-top_count:])
    new_top = set(np.argsort(new_flat)[-top_count:])
    summary = {
        "definition": "mean over real questions of top-100 attention among retained tokens older than the newest frame",
        "query_pool": "mean_all_question_prompt_tokens",
        "num_question_records": records,
        "observations_per_head_min": min(map(len, grouped.values())),
        "observations_per_head_max": max(map(len, grouped.values())),
        "score_min": float(new_flat.min()),
        "score_max": float(new_flat.max()),
        "score_mean": float(new_flat.mean()),
        "spearman_vs_original_ourtop": rho,
        "top10_percent_overlap_count": len(old_top & new_top),
        "top10_percent_count": top_count,
        "scores": str(score_path),
        "policy": str(policy_path),
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
