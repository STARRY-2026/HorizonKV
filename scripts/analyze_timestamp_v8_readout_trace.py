#!/usr/bin/env python3
"""Compare true-question attention paid to retained evidence frames."""

import argparse
import csv
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


def load_jsonl(path):
    records = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            records[(int(record["video_id"]), int(record["question_index"]))] = record
    return records


def load_accuracy(path):
    accuracy = {}
    counters = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            video_id = int(row["video_id"])
            question_idx = counters.get(video_id, 0)
            counters[video_id] = question_idx + 1
            accuracy[(video_id, question_idx)] = float(row["qa_acc"]) > 0
    return accuracy


def values_in_window(values, fps, start, end):
    return sum(
        float(value)
        for frame_id, value in values.items()
        if start <= int(frame_id) / fps <= end
    )


def summarize(record, start, end, role=None):
    evidence_mass = 0.0
    visual_mass = 0.0
    evidence_tokens = 0.0
    visual_tokens = 0.0
    focused_heads = 0
    heads = [head for head in record["heads"] if role is None or head.get("role") == role]
    fps = float(record["sample_fps"])
    for head in heads:
        head_evidence_mass = values_in_window(head["frame_mass"], fps, start, end)
        head_evidence_tokens = values_in_window(head["frame_tokens"], fps, start, end)
        head_visual_mass = float(head["visual_mass"])
        head_visual_tokens = float(head["visual_tokens"])
        other_mass = max(0.0, head_visual_mass - head_evidence_mass)
        other_tokens = max(0.0, head_visual_tokens - head_evidence_tokens)
        evidence_density = head_evidence_mass / max(head_evidence_tokens, 1.0)
        other_density = other_mass / max(other_tokens, 1.0)
        focused_heads += int(head_evidence_mass > 0.0 and evidence_density > other_density)
        evidence_mass += head_evidence_mass
        visual_mass += head_visual_mass
        evidence_tokens += head_evidence_tokens
        visual_tokens += head_visual_tokens

    readout_share = evidence_mass / max(visual_mass, 1e-12)
    token_share = evidence_tokens / max(visual_tokens, 1.0)
    lift = readout_share / max(token_share, 1e-12)
    return {
        "evidence_mass": evidence_mass,
        "readout_share": readout_share,
        "token_share": token_share,
        "lift": lift,
        "focused_heads": focused_heads,
        "num_heads": len(heads),
    }


def pct(value):
    return f"{100.0 * value:.2f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp_trace", required=True)
    parser.add_argument("--hermes_trace", required=True)
    parser.add_argument("--timestamp_results", required=True)
    parser.add_argument("--hermes_results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    timestamp = load_jsonl(args.timestamp_trace)
    hermes = load_jsonl(args.hermes_trace)
    timestamp_acc = load_accuracy(args.timestamp_results)
    hermes_acc = load_accuracy(args.hermes_results)
    lines = [
        "# Timestamp-v8 true-question readout diagnosis",
        "",
        "Attention is measured from the final real-question prompt token before answer generation.",
        "`Lift` is evidence attention share divided by evidence token share; values above 1 indicate preferential evidence readout.",
        "",
        "| Video/question | Outcome (T/H) | Timestamp readout | HERMES readout | Timestamp lift | HERMES lift | Focused heads (T/H) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    details = []
    for key, (start, end, label) in EVIDENCE_WINDOWS.items():
        if key not in timestamp or key not in hermes:
            lines.append(f"| {key[0]}/{key[1]} {label} | missing | - | - | - | - | - |")
            continue
        ts = summarize(timestamp[key], start, end)
        hs = summarize(hermes[key], start, end)
        outcome = f"{'C' if timestamp_acc.get(key) else 'W'}/{'C' if hermes_acc.get(key) else 'W'}"
        lines.append(
            f"| {key[0]}/{key[1]} {label} | {outcome} | {pct(ts['readout_share'])} | "
            f"{pct(hs['readout_share'])} | {ts['lift']:.2f} | {hs['lift']:.2f} | "
            f"{ts['focused_heads']}/{hs['focused_heads']} |"
        )

        if any("role" in head for head in timestamp[key]["heads"]):
            details.extend(["", f"## Video {key[0]}, question {key[1]}: {label}", ""])
            details.extend([
                "| Timestamp role | Evidence readout | Token share | Lift | Focused heads |",
                "|---|---:|---:|---:|---:|",
            ])
            for role in ("memory", "mixed", "current"):
                rs = summarize(timestamp[key], start, end, role=role)
                details.append(
                    f"| {role} | {pct(rs['readout_share'])} | {pct(rs['token_share'])} | "
                    f"{rs['lift']:.2f} | {rs['focused_heads']}/{rs['num_heads']} |"
                )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines + details) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
