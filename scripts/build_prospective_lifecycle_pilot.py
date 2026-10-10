#!/usr/bin/env python3
"""Build a short, calibration-disjoint manifest for lifecycle tracing."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--temporal_manifest", required=True)
    parser.add_argument("--horizon_manifest", required=True)
    parser.add_argument("--num_videos", type=int, default=6)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    temporal = json.loads(Path(args.temporal_manifest).read_text())
    excluded = {str(value) for value in temporal["combined_video_ids"]}
    horizon = json.loads(Path(args.horizon_manifest).read_text())
    excluded.update(str(row["video_id"]) for row in horizon)

    candidates = []
    for source_index, video in enumerate(json.loads(Path(args.annotation).read_text())):
        conversations = [
            row for row in video.get("conversations", []) if "end_time" in row
        ]
        if not conversations or str(video["video_id"]) in excluded:
            continue
        final_question = max(conversations, key=lambda row: float(row["end_time"]))
        selected = dict(video)
        selected["conversations"] = [final_question]
        candidates.append((float(final_question["end_time"]), source_index, selected))

    candidates.sort(key=lambda item: (item[0], item[1]))
    selected = [item[2] for item in candidates[: args.num_videos]]
    if len(selected) != args.num_videos:
        raise ValueError(f"Only found {len(selected)} eligible disjoint videos")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selected, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "video_ids": [row["video_id"] for row in selected],
                "end_times": [row["conversations"][0]["end_time"] for row in selected],
                "excluded_calibration_videos": len(excluded),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
