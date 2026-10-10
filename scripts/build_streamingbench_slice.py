#!/usr/bin/env python3
"""Build a contiguous slice of a StreamingBench annotation file."""

import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--start", type=int, required=True)
parser.add_argument("--stop", type=int, required=True)
args = parser.parse_args()

annotations = json.loads(args.source.read_text())
selected = annotations[args.start:args.stop]
if len(selected) != args.stop - args.start:
    raise ValueError(
        f"Requested {args.stop - args.start} videos, found {len(selected)}"
    )
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n")
print(
    f"Wrote {len(selected)} videos / "
    f"{sum(len(item['conversations']) for item in selected)} questions to {args.output}"
)
