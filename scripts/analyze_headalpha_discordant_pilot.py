#!/usr/bin/env python3
"""Compare a discordant-subset pilot with full-half v8 and Head-Alpha results."""

import argparse
import csv
from collections import Counter


IDENTITY_FIELDS = ("video_id", "question", "task")


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def is_correct(row):
    return float(row["qa_acc"]) > 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v8_results", required=True)
    parser.add_argument("--headalpha_results", required=True)
    parser.add_argument("--pilot_results", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--output_details")
    args = parser.parse_args()

    v8_rows = load_csv(args.v8_results)
    headalpha_rows = load_csv(args.headalpha_results)
    pilot_rows = load_csv(args.pilot_results)
    target_rows = load_csv(args.targets)
    target_by_source = {int(row["source_row"]): row for row in target_rows}
    target_video_ids = {row["video_id"] for row in target_rows}

    selected = [
        (source_row, row)
        for source_row, row in enumerate(v8_rows)
        if row["video_id"] in target_video_ids
    ]
    if len(selected) != len(pilot_rows):
        raise ValueError(
            f"Pilot length mismatch: expected {len(selected)}, got {len(pilot_rows)}"
        )

    records = []
    for (source_row, v8_row), headalpha_row, pilot_row in zip(
        selected,
        (headalpha_rows[idx] for idx, _ in selected),
        pilot_rows,
    ):
        for reference in (headalpha_row, pilot_row):
            if any(
                v8_row.get(field) != reference.get(field)
                for field in IDENTITY_FIELDS
            ):
                raise ValueError(f"Result identity mismatch at source row {source_row}")
        records.append({
            "source_row": source_row,
            "target": source_row in target_by_source,
            "direction": target_by_source.get(source_row, {}).get("direction", "non_target"),
            "video_id": v8_row.get("video_id", ""),
            "task": v8_row.get("task", ""),
            "question": v8_row.get("question", ""),
            "v8": is_correct(v8_row),
            "headalpha": is_correct(headalpha_row),
            "pilot": is_correct(pilot_row),
            "correct_choice": v8_row.get("correct_choice", ""),
            "v8_pred": v8_row.get("pred_answer", ""),
            "headalpha_pred": headalpha_row.get("pred_answer", ""),
            "pilot_pred": pilot_row.get("pred_answer", ""),
        })

    if args.output_details:
        with open(args.output_details, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    def report(label, rows):
        total = len(rows)
        v8_correct = sum(row["v8"] for row in rows)
        headalpha_correct = sum(row["headalpha"] for row in rows)
        pilot_correct = sum(row["pilot"] for row in rows)
        print(
            f"{label}: n={total}, v8={v8_correct}/{total}, "
            f"Head-Alpha={headalpha_correct}/{total}, kd8={pilot_correct}/{total}, "
            f"kd8-v8={pilot_correct - v8_correct:+d}"
        )

    report("All selected videos", records)
    report("Discordant targets", [row for row in records if row["target"]])
    report("v8 gains (regressions to repair)", [row for row in records if row["direction"] == "v8_gain"])
    report("Head-Alpha gains (benefits to retain)", [row for row in records if row["direction"] == "headalpha_gain"])
    report("Non-target controls", [row for row in records if not row["target"]])

    transitions = Counter(
        (row["v8"], row["pilot"])
        for row in records
    )
    print(
        "kd8 vs v8 transitions: "
        f"wrong->correct={transitions[(False, True)]}, "
        f"correct->wrong={transitions[(True, False)]}"
    )

    losses = [
        row for row in records
        if row["direction"] == "headalpha_gain" and not row["pilot"]
    ]
    retained = [
        row for row in records
        if row["direction"] == "headalpha_gain" and row["pilot"]
    ]
    repairs = [
        row for row in records
        if row["direction"] == "v8_gain" and row["pilot"]
    ]
    print(f"Repaired v8-gain targets by task: {dict(Counter(row['task'] for row in repairs))}")
    print(f"Retained Head-Alpha gains by task: {dict(Counter(row['task'] for row in retained))}")
    print(f"Lost Head-Alpha-gain targets by task: {dict(Counter(row['task'] for row in losses))}")
    for row in retained:
        print(
            f"Retained gain source_row={row['source_row']} "
            f"task={row['task']}: {row['question']}"
        )


if __name__ == "__main__":
    main()
