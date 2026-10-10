#!/usr/bin/env python3
"""Compare retained evidence-frame coverage for the three diagnostic videos."""

import argparse
import json
from pathlib import Path


EVIDENCE_WINDOWS = {
    (18, 1): (126.0, 140.0, "cutting sequence"),
    (18, 3): (350.0, 364.0, "placing the finished flower"),
    (129, 2): (220.0, 260.0, "books used as steps"),
    (129, 3): (380.0, 416.0, "bird takes the valuable item"),
    (237, 2): (40.0, 242.0, "first three inserted clips"),
    (237, 4): (316.0, 333.0, "fourth inserted clip"),
}


def load_trace(path):
    records = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            records[(int(record["video_id"]), int(record["question_index"]))] = record
    return records


def in_window_count(histogram, sample_fps, start, end):
    return sum(
        int(count)
        for frame_id, count in histogram.items()
        if start <= int(frame_id) / sample_fps <= end
    )


def summarize(record, start, end, role=None):
    heads = record["heads"]
    if role is not None:
        heads = [head for head in heads if head.get("role") == role]
    before = 0
    after = 0
    covered = 0
    for head in heads:
        fps = float(record["sample_fps"])
        before += in_window_count(head["before_frame_tokens"], fps, start, end)
        kept = in_window_count(head["after_frame_tokens"], fps, start, end)
        after += kept
        covered += int(kept > 0)
    return before, after, covered, len(heads)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp_trace", required=True)
    parser.add_argument("--hermes_trace", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    timestamp = load_trace(args.timestamp_trace)
    hermes = load_trace(args.hermes_trace)
    lines = [
        "# Timestamp-v8 cache trace diagnosis",
        "",
        "Counts are KV-head token copies retained inside manually inspected evidence windows.",
        "`Heads` is the number of the 112 physical KV heads retaining at least one evidence token.",
        "",
        "| Video/question | Evidence window | Timestamp-v8 kept | HERMES kept | Timestamp heads | HERMES heads |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    details = []
    for key, (start, end, label) in EVIDENCE_WINDOWS.items():
        if key not in timestamp or key not in hermes:
            lines.append(f"| {key[0]}/{key[1]} {label} | {start:.0f}-{end:.0f}s | missing | missing | - | - |")
            continue
        tb, ta, th, _ = summarize(timestamp[key], start, end)
        hb, ha, hh, _ = summarize(hermes[key], start, end)
        lines.append(
            f"| {key[0]}/{key[1]} {label} | {start:.0f}-{end:.0f}s | "
            f"{ta} ({ta / max(tb, 1):.1%} of pre-prune) | "
            f"{ha} ({ha / max(hb, 1):.1%} of pre-prune) | {th} | {hh} |"
        )
        details.extend(["", f"## Video {key[0]}, question {key[1]}: {label}", ""])
        details.append("| Role | Before | After | Retention | Heads with evidence |")
        details.append("|---|---:|---:|---:|---:|")
        for role in ("memory", "mixed", "current"):
            before, after, covered, total = summarize(timestamp[key], start, end, role)
            details.append(
                f"| {role} | {before} | {after} | {after / max(before, 1):.1%} | {covered}/{total} |"
            )
        details.append("")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines + details) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
