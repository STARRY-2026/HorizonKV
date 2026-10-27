"""Report matched-question OVO-Bench progress for the two native SparseMM arms."""

import argparse
import collections
import json
from pathlib import Path

from sparsemm_native.run_ovobench import forward_queries, hermes_queries, queries, summary


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def report(root):
    protocol_path = root / "dp" / "protocol.json"
    dataset = json.loads(protocol_path.read_text()).get("dataset", "official_full") if protocol_path.exists() else "official_full"
    expected = (hermes_queries() if dataset == "hermes9" else
                forward_queries() if dataset == "forward3" else queries())
    dp, native = (read(root / arm / "queries.jsonl") for arm in ("dp", "native"))
    paired = []
    for index in range(min(len(dp), len(native))):
        a, b = dp[index], native[index]
        if a["query_id"] != expected[index]["query_id"] or b["query_id"] != a["query_id"]:
            raise ValueError("Methods do not cover identical ordered questions")
        paired.append((a, b))
    by_task = collections.defaultdict(list)
    for a, b in paired:
        by_task[a["task"]].append((int(a["correct"]), int(b["correct"])))
    tasks = {task: {"matched": len(values), "dp_correct": sum(a for a, _ in values),
                    "sparsemm_correct": sum(b for _, b in values),
                    "dp_minus_sparsemm_percentage_points": 100 * sum(a - b for a, b in values) / len(values)}
             for task, values in sorted(by_task.items())}
    return {"paired_queries": len(paired), "total_queries": len(expected),
            "dp_completed": len(dp), "sparsemm_completed": len(native),
            "paired_dp_correct": sum(int(a["correct"]) for a, _ in paired),
            "paired_sparsemm_correct": sum(int(b["correct"]) for _, b in paired),
            "paired_difference_percentage_points":
                100 * sum(int(a["correct"]) - int(b["correct"]) for a, b in paired) / len(paired)
                if paired else None,
            "tasks": tasks, "dp_summary": summary(dp, len(expected)),
            "sparsemm_summary": summary(native, len(expected))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(report(args.output_root), indent=2, ensure_ascii=False))
