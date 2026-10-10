#!/usr/bin/env python3
"""Create an annotation copy with video paths resolved inside a supplied video root."""
import argparse
import collections
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--video-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.video_root.is_dir():
        parser.error("--video-root must be an existing directory")
    if args.output.exists():
        parser.error("--output already exists; choose a new annotation path")
    data = json.loads(args.input.read_text(encoding="utf-8"))
    entries = []

    def collect(value):
        if isinstance(value, dict):
            if isinstance(value.get("video_path"), str):
                entries.append(value)
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(data)
    if not entries:
        parser.error("No video_path fields found")
    wanted = {Path(entry["video_path"]).name for entry in entries}
    matches = collections.defaultdict(list)
    for path in args.video_root.resolve().rglob("*"):
        if path.name in wanted and path.is_file():
            matches[path.name].append(path)
    errors = []
    for entry in entries:
        name = Path(entry["video_path"]).name
        if len(matches[name]) != 1:
            errors.append(f"{name}: {len(matches[name])} matches (expected exactly one)")
        else:
            entry["video_path"] = str(matches[name][0])
    if errors:
        parser.error("\n".join(sorted(set(errors))))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Resolved {len(entries)} video_path fields; wrote {args.output}")


if __name__ == "__main__":
    main()
