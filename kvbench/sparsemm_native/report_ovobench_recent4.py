"""Matched official OVO-Bench results for SparseMM, old DP, and recent-four DP."""

import argparse
import collections
import json
from pathlib import Path

from sparsemm_native import run_ovobench as legacy
from sparsemm_native.run_ovobench_recent4 import ARMS, COMMON_FIELDS, digest, read_progress, schedule_queries, select_queries


def metrics(items):
    count = len(items)
    return dict(matched_queries=count, **{
        arm: {"correct": sum(int(item[arm]["correct"]) for item in items),
              "accuracy": sum(int(item[arm]["correct"]) for item in items) / count if count else None}
        for arm in ARMS})


def report(root):
    root = Path(root).expanduser().resolve()
    expected = legacy.queries()
    protocols, rows, manifests = {}, {}, {}
    for arm in ARMS:
        directory = root / arm
        if not (directory / "protocol.json").is_file():
            continue
        protocol = json.loads((directory / "protocol.json").read_text())
        if (protocol.get("arm") != arm or protocol.get("dataset") != "official_full"
                or protocol.get("score_mode") != "official_OVOBenchOfflineScore"
                or protocol.get("full_query_list_sha256") != digest(expected)):
            raise ValueError(f"{arm}: changed official protocol or full query manifest")
        ids = protocol.get("query_ids")
        if not isinstance(ids, list) or len(ids) != len(set(ids)) or any(not isinstance(i, str) for i in ids):
            raise ValueError(f"{arm}: invalid selected IDs")
        selected = schedule_queries(select_queries(expected, ",".join(ids)), grouped=protocol.get("grouped"))
        if ([q["query_id"] for q in selected] != ids or protocol.get("query_list_sha256") != digest(selected)
                or protocol.get("query_count") != len(selected)):
            raise ValueError(f"{arm}: changed selected execution manifest")
        protocols[arm], manifests[arm] = protocol, selected
        rows[arm] = read_progress(directory / "queries.jsonl", selected)
    if not protocols:
        raise ValueError("No official recent-four protocols exist")
    reference = next(iter(protocols.values()))
    for arm, protocol in protocols.items():
        for field in COMMON_FIELDS + ("model_path", "query_list_sha256", "full_query_list_sha256",
                                      "processor_files_sha256", "sparsemm_score_sha256", "dependencies",
                                      "code_sha256", "grouped", "decoding", "repetition_penalty",
                                      "concurrent_processes", "concurrency_mode"):
            if protocol.get(field) != reference.get(field):
                raise ValueError(f"{arm}: comparison protocol mismatch: {field}")
    indexes = {arm: {row["query_id"]: row for row in rows.get(arm, [])} for arm in ARMS}
    matched = []
    for query in expected:
        query_id = query["query_id"]
        if all(query_id in indexes[arm] for arm in ARMS):
            item = {arm: indexes[arm][query_id] for arm in ARMS}
            if len({row["frames"] for row in item.values()}) != 1:
                raise ValueError(f"Different sampled frame counts: {query_id}")
            matched.append(item)
    by_task = collections.defaultdict(list)
    for item in matched:
        by_task[item["native"]["task"]].append(item)
    counts = collections.Counter(query["task"] for query in expected)
    tasks = {task: dict(expected=counts[task], **metrics(by_task[task])) for task in sorted(counts)}
    complete = len(matched) == 3035
    summaries = {arm: legacy.summary([item[arm] for item in matched], 3035) for arm in ARMS}
    return dict(
        dataset="official_full", budget=500, fps=0.5, total_queries=3035,
        completed_queries={arm: len(rows.get(arm, [])) for arm in ARMS}, complete=complete,
        tasks=tasks, matched=metrics(matched),
        group_accuracy={arm: summaries[arm]["group_accuracy"] for arm in ARMS},
        official_total_average={arm: summaries[arm]["official_total_average"] if complete else None for arm in ARMS},
        aggregation="Mean of backward (3-task mean), realtime (6-task mean), forward (3-task mean)",
        note="Only matched official queries are compared. Official total is available after all 3035 matched queries.",
    )


def markdown(result):
    def cell(metric):
        return f"{metric['correct']} / {metric['accuracy']:.2%}" if metric["accuracy"] is not None else "—"

    lines = ["# Official OVO-Bench: K500, FPS0.5", "", result["note"], "",
             "| Task | Matched / total | SparseMM correct / accuracy | Old DP correct / accuracy | Recent-four DP correct / accuracy |",
             "|---|---:|---:|---:|---:|"]
    for task, item in result["tasks"].items():
        lines.append(f"| {task} | {item['matched_queries']} / {item['expected']} | "
                     + " | ".join(cell(item[arm]) for arm in ARMS) + " |")
    item = result["matched"]
    lines.extend([f"| **Question-weighted matched subtotal** | **{item['matched_queries']} / 3035** | "
                  + " | ".join(cell(item[arm]) for arm in ARMS) + " |", ""])
    if result["complete"]:
        lines.extend(["Official total: " + "; ".join(
            f"{arm} {result['official_total_average'][arm]:.2%}" for arm in ARMS), ""])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args(argv)
    result = report(args.output_root)
    (args.output_root / "comparison.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    (args.output_root / "live_report.md").write_text(markdown(result))
    print(markdown(result))


if __name__ == "__main__":
    main()
