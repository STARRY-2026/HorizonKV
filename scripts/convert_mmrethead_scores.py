#!/usr/bin/env python3
"""Convert an MMRetHead head_score_list JSON into a Hermes score file.

MMRetHead writes scores at query-head granularity::

    {
      "head_score_list": [["27-10", 1.46875], ...]
    }

Hermes has two compatible consumers:

* ``--head_budget_scores``: expects a CSV with ``layer,head,score``.
* ``--kv_head_budget_scores``: expects a CSV with ``layer,kv_head,score``.

For Qwen2.5-VL-7B the query head -> KV head mapping is::

    kv_head = query_head // 7

Usage examples::

    # query-head scores
    python scripts/convert_mmrethead_scores.py \
        --input /path/to/mmrethead_result.json \
        --output results/myobservations/mmrethead_qwen25_32k_head.csv \
        --mode head

    # KV-head scores, averaged over each group of 7 query heads
    python scripts/convert_mmrethead_scores.py \
        --input /path/to/mmrethead_result.json \
        --output results/myobservations/mmrethead_qwen25_32k_kv_mean.csv \
        --mode kv_head --aggregation mean

    # hard top-40 retrieval heads (other heads get zero score)
    python scripts/convert_mmrethead_scores.py \
        --input /path/to/mmrethead_result.json \
        --output results/myobservations/mmrethead_qwen25_32k_top40_binary.csv \
        --mode kv_head --binary-top-k 40
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict


def _parse_head_id(head_id: str) -> tuple[int, int]:
    layer, head = head_id.split("-")
    return int(layer), int(head)


def _aggregate(values: list[float], method: str) -> float:
    if not values:
        return 0.0
    if method == "mean":
        return sum(values) / len(values)
    if method == "sum":
        return sum(values)
    if method == "max":
        return max(values)
    if method == "min":
        return min(values)
    raise ValueError(f"unknown aggregation: {method}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="MMRetHead result JSON")
    parser.add_argument("--output", required=True, help="output CSV")
    parser.add_argument(
        "--mode",
        choices=["head", "kv_head"],
        default="kv_head",
        help="write query-head or KV-head scores",
    )
    parser.add_argument(
        "--aggregation",
        choices=["mean", "sum", "max", "min"],
        default="mean",
        help="how to collapse query heads into a KV head",
    )
    parser.add_argument("--num-layers", type=int, default=28)
    parser.add_argument("--num-query-heads", type=int, default=28)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument(
        "--binary-top-k",
        type=int,
        default=None,
        help="optional: keep only the top-k heads and set the rest to zero",
    )
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        payload = json.load(f)

    if "head_score_list" not in payload:
        raise ValueError("input JSON does not contain 'head_score_list'")

    query_scores: dict[tuple[int, int], float] = {}
    for item in payload["head_score_list"]:
        if isinstance(item, list) and len(item) == 2:
            layer, head = _parse_head_id(str(item[0]))
            query_scores[(layer, head)] = float(item[1])

    if args.binary_top_k is not None:
        top = sorted(query_scores.items(), key=lambda kv: kv[1], reverse=True)
        selected = {key for key, _ in top[: args.binary_top_k]}
        query_scores = {
            key: (1.0 if key in selected else 0.0) for key in query_scores
        }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if args.mode == "head":
        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["layer", "head", "score"])
            for layer in range(args.num_layers):
                for head in range(args.num_query_heads):
                    score = query_scores.get((layer, head), 0.0)
                    writer.writerow([layer, head, score])
        return

    groups: dict[tuple[int, int], list[float]] = defaultdict(list)
    for (layer, head), score in query_scores.items():
        if head >= args.num_query_heads:
            continue
        kv_head = head // (args.num_query_heads // args.num_kv_heads)
        groups[(layer, kv_head)].append(score)

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "kv_head", "score"])
        for layer in range(args.num_layers):
            for kv_head in range(args.num_kv_heads):
                score = _aggregate(groups.get((layer, kv_head), []), args.aggregation)
                writer.writerow([layer, kv_head, score])


if __name__ == "__main__":
    main()
