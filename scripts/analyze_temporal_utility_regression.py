#!/usr/bin/env python3
"""Locate cache loss on Hermes-attended frames for temporal-utility regressions."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METHODS = ("hermes", "budgeted_hermes", "content", "full", "shuffled")


def load_jsonl(path):
    records = {}
    with Path(path).open() as handle:
        for line in handle:
            record = json.loads(line)
            records[(int(record["video_id"]), int(record["question_index"]))] = record
    return records


def load_accuracy(path):
    counters = {}
    output = {}
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            video_id = int(row["video_id"])
            question_idx = counters.get(video_id, 0)
            counters[video_id] = question_idx + 1
            output[(video_id, question_idx)] = float(row["qa_acc"]) > 0
    return output


def aggregate_frames(readout):
    masses = {}
    tokens = {}
    for head in readout["heads"]:
        for frame_id, value in head["frame_mass"].items():
            masses[int(frame_id)] = masses.get(int(frame_id), 0.0) + float(value)
        for frame_id, value in head["frame_tokens"].items():
            tokens[int(frame_id)] = tokens.get(int(frame_id), 0) + int(value)
    return masses, tokens


def frame_stats(cache, readout, frame_id, focused_heads):
    by_head_cache = {
        (int(h["layer"]), int(h["kv_head"])): h for h in cache["heads"]
    }
    by_head_readout = {
        (int(h["layer"]), int(h["kv_head"])): h for h in readout["heads"]
    }
    after = sum(
        int(h["after_frame_tokens"].get(str(frame_id), 0)) for h in cache["heads"]
    )
    total_tokens = sum(int(h["after_visual_tokens"]) for h in cache["heads"])
    mass = sum(float(h["frame_mass"].get(str(frame_id), 0.0)) for h in readout["heads"])
    visual_mass = sum(float(h["visual_mass"]) for h in readout["heads"])
    focused_tokens = sum(
        int(by_head_cache[key]["after_frame_tokens"].get(str(frame_id), 0))
        for key in focused_heads
    )
    focused_mass = sum(
        float(by_head_readout[key]["frame_mass"].get(str(frame_id), 0.0))
        for key in focused_heads
    )
    return {
        "tokens": after,
        "token_share": after / max(total_tokens, 1),
        "readout_mass": mass,
        "readout_share": mass / max(visual_mass, 1e-12),
        "focused_tokens": focused_tokens,
        "focused_mass": focused_mass,
    }


def mean(records, key):
    return float(np.mean([record[key] for record in records]))


def pct(value):
    return f"{100.0 * value:.2f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--results", action="append", nargs=2, metavar=("METHOD", "CSV"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    manifest = json.loads((root / "manifest.json").read_text())
    targets = {(int(x["video_id"]), int(x["question_index"])): x for x in manifest["targets"]}
    caches = {name: load_jsonl(root / f"{name}_cache_trace.jsonl") for name in METHODS}
    readouts = {name: load_jsonl(root / f"{name}_readout_trace.jsonl") for name in METHODS}
    accuracy = {name: load_accuracy(path) for name, path in (args.results or [])}

    rows = []
    for key, target in targets.items():
        missing = [name for name in METHODS if key not in caches[name] or key not in readouts[name]]
        if missing:
            raise ValueError(f"Missing target {key} for {missing}")
        hermes_masses, _ = aggregate_frames(readouts["hermes"][key])
        top_frame = max(hermes_masses, key=hermes_masses.get)
        head_scores = []
        for head in readouts["hermes"][key]["heads"]:
            hkey = (int(head["layer"]), int(head["kv_head"]))
            tokens = int(head["frame_tokens"].get(str(top_frame), 0))
            mass = float(head["frame_mass"].get(str(top_frame), 0.0))
            head_scores.append((mass / max(tokens, 1), hkey))
        focused_heads = [hkey for _, hkey in sorted(head_scores, reverse=True)[:14]]
        current_frame = int(caches["hermes"][key]["current_frame_id"])
        fps = float(caches["hermes"][key]["sample_fps"])
        row = {
            **target,
            "top_frame_id": top_frame,
            "top_frame_sec": top_frame / fps,
            "top_frame_age_sec": (current_frame - top_frame) / fps,
            "methods": {},
        }
        for name in METHODS:
            stats = frame_stats(caches[name][key], readouts[name][key], top_frame, focused_heads)
            stats["correct"] = accuracy.get(name, {}).get(key)
            row["methods"][name] = stats
        rows.append(row)

    lines = [
        "# Temporal-utility regression diagnosis",
        "",
        "The evidence patch is the source-frame temporal patch receiving the largest aggregate true-question attention under HERMES. Focused heads are the 14 HERMES heads with the highest attention density on that patch.",
        "",
        "| Target | Ability | Evidence age | H tokens | BH tokens | C tokens | F tokens | S tokens | H readout | BH readout | C readout | F readout | S readout |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        m = row["methods"]
        lines.append(
            f"| v{row['video_id']}/q{row['question_index']} {row['task']} | {row['ability']} | "
            f"{row['top_frame_age_sec']:.1f}s | {m['hermes']['tokens']} | {m['budgeted_hermes']['tokens']} | "
            f"{m['content']['tokens']} | {m['full']['tokens']} | {m['shuffled']['tokens']} | "
            f"{pct(m['hermes']['readout_share'])} | {pct(m['budgeted_hermes']['readout_share'])} | "
            f"{pct(m['content']['readout_share'])} | {pct(m['full']['readout_share'])} | "
            f"{pct(m['shuffled']['readout_share'])} |"
        )

    lines.extend(["", "## Aggregate", ""])
    for name in METHODS:
        stats = [row["methods"][name] for row in rows]
        correct = [s["correct"] for s in stats if s["correct"] is not None]
        lines.append(
            f"- {name}: key-patch tokens={mean(stats, 'tokens'):.1f}, "
            f"token share={pct(mean(stats, 'token_share'))}, "
            f"readout share={pct(mean(stats, 'readout_share'))}, "
            f"focused-head tokens={mean(stats, 'focused_tokens'):.1f}, "
            f"accuracy={sum(correct)}/{len(correct) if correct else 0}."
        )

    for ability in ("working memory", "episodic memory"):
        selected = [row for row in rows if row["ability"] == ability]
        lines.extend(["", f"### {ability}", ""])
        for name in METHODS:
            stats = [row["methods"][name] for row in selected]
            lines.append(
                f"- {name}: tokens={mean(stats, 'tokens'):.1f}, "
                f"readout={pct(mean(stats, 'readout_share'))}."
            )

    Path(args.output).write_text("\n".join(lines) + "\n")
    (root / "analysis.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"wrote {args.output} and {root / 'analysis.json'}")


if __name__ == "__main__":
    main()
