#!/usr/bin/env python3
"""Freeze the task-facing prospective lifecycle estimator for online eviction."""

import argparse
import csv
import json
from pathlib import Path

from head_analysis.prospective_lifecycle import (
    align_predictor_to_future_targets,
    fit_remaining_lifecycle,
    load_trace_jsonl,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--exclude_video_ids", default="")
    parser.add_argument("--lifecycle_weight", type=float, default=1.0)
    args = parser.parse_args()

    rows = load_trace_jsonl(
        args.trace,
        require_full_cache=True,
        query_sources={"hermes", "true_question"},
    )
    rows = align_predictor_to_future_targets(rows)
    excluded = {
        value.strip() for value in args.exclude_video_ids.split(",") if value.strip()
    }
    available = {str(row["video_id"]) for row in rows}
    train_ids = available - excluded
    if not train_ids:
        raise ValueError("No calibration videos remain after exclusions")
    profiles = fit_remaining_lifecycle(rows, train_ids)

    content_bins = range(4)
    age_bins = range(8)
    fieldnames = [
        "layer", "kv_head", "profile_mode", "lifecycle_weight",
        *(f"content_prior_{content_bin}" for content_bin in content_bins),
        *(
            f"lifecycle_c{content_bin}_t{age_bin}"
            for content_bin in content_bins
            for age_bin in age_bins
        ),
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for layer in range(28):
            for kv_head in range(4):
                record = {
                    "layer": layer,
                    "kv_head": kv_head,
                    "profile_mode": "prospective_lifecycle",
                    "lifecycle_weight": f"{args.lifecycle_weight:.10g}",
                }
                for content_bin in content_bins:
                    record[f"content_prior_{content_bin}"] = f"{profiles['content'].get(content_bin, 1.0):.10g}"
                    curve = profiles["conditional_heads"].get(
                        (layer, kv_head, content_bin),
                        profiles["conditional_global"].get(
                            content_bin, profiles["global"]
                        ),
                    )
                    for age_bin in age_bins:
                        record[f"lifecycle_c{content_bin}_t{age_bin}"] = (
                            f"{float(curve[age_bin]):.10g}"
                        )
                writer.writerow(record)

    manifest = {
        "trace": [str(Path(path)) for path in args.trace],
        "target_source": "true_question",
        "target_includes_same_observation": True,
        "available_video_ids": sorted(available),
        "excluded_video_ids": sorted(excluded),
        "calibration_video_ids": sorted(train_ids),
        "num_aligned_rows": len(rows),
        "lifecycle_weight": args.lifecycle_weight,
        "content_bins": 4,
        "age_edges_sec": [0, 4, 8, 16, 32, 64, 128, 256, "inf"],
        "output": str(output),
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
