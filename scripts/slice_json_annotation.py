#!/usr/bin/env python3
"""Write a deterministic half-open slice of a JSON annotation list."""

import argparse
import json
from pathlib import Path


p = argparse.ArgumentParser()
p.add_argument("source")
p.add_argument("output")
p.add_argument("start", type=int)
p.add_argument("stop", type=int)
a = p.parse_args()
rows = json.loads(Path(a.source).read_text())
selected = rows[a.start:a.stop]
if len(selected) != a.stop - a.start:
    raise ValueError(f"Expected {a.stop-a.start} rows, got {len(selected)}")
Path(a.output).write_text(json.dumps(selected, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({"output": a.output, "videos": len(selected),
                  "questions": sum(len(v.get("conversations", [])) for v in selected),
                  "first_video_id": selected[0].get("video_id"),
                  "last_video_id": selected[-1].get("video_id")}, indent=2))
