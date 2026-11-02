"""Check tiered one/four runs against the fixed-896 K500 reference on identical questions."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    return [json.loads(line) for line in path.open() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--budget-map", type=Path, required=True)
    parser.add_argument("--baseline-queries", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    args = parser.parse_args()
    run = rows(args.output_root / "queries.jsonl")
    budget_rows = json.loads(args.budget_map.read_text())["budgets"]
    baseline = {row["query_id"]: row for row in rows(args.baseline_queries)}
    total = sum(sum(row) for row in budget_rows)
    if len(run) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} rows, found {len(run)}")
    if len({row["query_id"] for row in run}) != len(run):
        raise ValueError("Duplicate question")
    for row in run:
        control = baseline[row["query_id"]]
        if row["target"] != control["target"] or row["task"] != control["task"]:
            raise ValueError("Question identity changed")
        for field in ("prompt_sha256", "selected_raw_frame_indices", "fixed_frame_audits",
                      "position_layout", "image_grid_thw", "per_frame_visual_tokens"):
            if row["input_audit"][field] != control["input_audit"][field]:
                raise ValueError(f"Input drift for question {row['query_id']}: {field}")
        audit = row["budget_audit"]
        if (audit["per_head_visual_budgets"] != budget_rows
                or audit["actual_visual_slots"] != total
                or audit["actual_history_visual_slots"] != 0
                or audit["question_text_used_for_selection"]):
            raise ValueError(f"Budget/selector drift for question {row['query_id']}")
    correct = sum(bool(row["correct"]) for row in run)
    reference = sum(bool(baseline[row["query_id"]]["correct"]) for row in run)
    flips = sum(row["correct"] != baseline[row["query_id"]]["correct"] for row in run)
    print(json.dumps(dict(questions=len(run), correct=correct, baseline_correct=reference,
                          paired_flips=flips, actual_visual_slots=total)))


if __name__ == "__main__":
    main()
