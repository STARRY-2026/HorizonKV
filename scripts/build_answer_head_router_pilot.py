#!/usr/bin/env python3
"""Build the complete-video pilot set for answer-time head routing."""

import argparse
import json
from pathlib import Path


DEFAULT_VIDEO_IDS = [34, 98, 232, 266, 360, 496, 3, 6, 50, 56, 129, 225]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--video-ids", type=int, nargs="+", default=DEFAULT_VIDEO_IDS)
    args = parser.parse_args()

    records = json.loads(args.source.read_text())
    by_id = {int(record["video_id"]): record for record in records}
    missing = [video_id for video_id in args.video_ids if video_id not in by_id]
    if missing:
        raise ValueError(f"Missing video ids: {missing}")
    selected = [by_id[video_id] for video_id in args.video_ids]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, indent=2, ensure_ascii=False) + "\n")
    print(
        f"wrote {len(selected)} complete videos / "
        f"{sum(len(item['conversations']) for item in selected)} questions to {args.output}"
    )


if __name__ == "__main__":
    main()
