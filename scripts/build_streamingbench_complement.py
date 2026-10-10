#!/usr/bin/env python3
"""Build the ordered complement of an evaluated StreamingBench subset."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("full", type=Path)
    parser.add_argument("evaluated", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    full = json.loads(args.full.read_text())
    evaluated = json.loads(args.evaluated.read_text())
    evaluated_ids = {int(item["video_id"]) for item in evaluated}
    selected = [item for item in full if int(item["video_id"]) not in evaluated_ids]
    overlap = evaluated_ids & {int(item["video_id"]) for item in selected}
    if overlap:
        raise ValueError(f"Complement unexpectedly overlaps evaluated ids: {sorted(overlap)}")
    if len(selected) + len(evaluated_ids) != len(full):
        raise ValueError("Full/evaluated annotations contain duplicate or mismatched video ids")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n")
    print(
        f"wrote complement: {len(selected)} videos / "
        f"{sum(len(item.get('conversations', [])) for item in selected)} questions"
    )


if __name__ == "__main__":
    main()
