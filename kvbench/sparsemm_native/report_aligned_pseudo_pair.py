"""Report a paired, question-blind recent-only/history OVO diagnostic run.

Usage::

    python3 sparsemm_native/report_aligned_pseudo_pair.py --output-root RUN_DIR

Inputs are RUN_DIR/queries.jsonl. Each complete row has query_id, task, target,
and arms.recent_only / arms.mild_history, each containing response and
budget_audit. Optional stored correct fields are checked, never trusted.

Scoring imports ``official_correct`` from sparsemm_native/run_ovobench.py,
which reproduces OVO-Bench utils/OVOBenchScore.py. In the nine MCQ tasks that
means the case-sensitive ``target in response`` rule, without extracting or
normalizing a predicted letter. These scores cover nine tasks, not all twelve.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sparsemm_native.run_ovobench import official_correct  # noqa: E402


ARMS = ("recent_only", "mild_history")
TASKS = ("EPM", "HLD", "ASI", "STU", "OJR", "ATR", "FPD", "ACR", "OCR")
GROUPS = {"backward": TASKS[:3], "realtime": TASKS[3:]}
HEADS = 112
LABELS = {"recent_only": "近期控制版", "mild_history": "历史版"}


def mean(values):
    return sum(values) / len(values) if values else None


def integer(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}: expected a nonnegative integer, got {value!r}")
    if not math.isfinite(value) or value < 0 or int(value) != value:
        raise ValueError(f"{label}: expected a nonnegative integer, got {value!r}")
    return int(value)


def budget_values(audit, label):
    fields = ("actual_recent_visual_slots", "actual_history_visual_slots", "actual_visual_slots")
    recent, historical, total = (integer(audit[key], f"{label}.{key}") for key in fields)
    if recent + historical != total:
        raise ValueError(f"{label}: recent + history != total")
    raw = audit["per_head_visual_slots"]
    # Native cache audits use [28 layers][4 KV heads]; a flattened vector is
    # accepted to make the JSON interface independent of a tensor library.
    flat = [value for layer in raw for value in layer] if raw and isinstance(raw[0], list) else raw
    if len(flat) != HEADS:
        raise ValueError(f"{label}: expected {HEADS} KV heads, got {len(flat)}")
    flat = [integer(value, f"{label}.per_head_visual_slots") for value in flat]
    if sum(flat) != total:
        raise ValueError(f"{label}: per-head visual slots do not sum to total")
    if total > HEADS * 500:
        raise ValueError(f"{label}: visual KV exceeds global K=500 mean budget")
    return {"recent_tokens_per_head": recent / HEADS,
            "history_tokens_per_head": historical / HEADS,
            "total_tokens_per_head": total / HEADS,
            "visual_slots_all_heads": total,
            "min_head_visual_tokens": min(flat),
            "max_head_visual_tokens": max(flat)}


def load_rows(path, allow_partial=False):
    rows, seen, skipped = [], set(), []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        qid = str(row["query_id"])
        if qid in seen:
            raise ValueError(f"Duplicate query_id {qid} at {path}:{lineno}")
        seen.add(qid)
        if row["task"] not in TASKS:
            raise ValueError(f"Unsupported task {row['task']}; this report covers nine non-forward tasks")
        if not all(arm in row.get("arms", {}) and "response" in row["arms"][arm]
                   and "budget_audit" in row["arms"][arm] for arm in ARMS):
            if allow_partial:
                skipped.append(qid)
                continue
            raise ValueError(f"Query {qid} does not have two complete arms; use --allow-partial for progress")
        expected_group = "backward" if row["task"] in GROUPS["backward"] else "realtime"
        if "group" in row and row["group"] != expected_group:
            raise ValueError(f"Query {qid} has group {row['group']!r}, expected {expected_group!r}")
        scored = {"query_id": qid, "task": row["task"], "group": expected_group,
                  "target": row["target"], "arms": {}}
        if not isinstance(row["target"], str) or row["target"] not in tuple("ABCDEFGH"):
            raise ValueError(f"Query {qid}: expected a single MCQ target letter")
        for arm in ARMS:
            result = row["arms"][arm]
            response = result["response"]
            if response is not None and not isinstance(response, str):
                raise ValueError(f"Query {qid}/{arm}: response must be string or null")
            correct = bool(official_correct(row["task"], response, row["target"]))
            scored["arms"][arm] = dict(
                response=response, correct=correct,
                stored_correct=result.get("correct"),
                stored_correct_matches_official=(None if "correct" not in result
                                                 else bool(result["correct"]) == correct),
                **budget_values(result["budget_audit"], f"{qid}/{arm}"))
            if arm == "recent_only" and scored["arms"][arm]["history_tokens_per_head"] != 0:
                raise ValueError(f"Query {qid}: recent-only control contains historical visual KV")
        rows.append(scored)
    if not rows:
        raise ValueError(f"No complete paired rows in {path}")
    return rows, skipped


def summarize_arm(rows, arm):
    by_task = {}
    for task in TASKS:
        selected = [row["arms"][arm] for row in rows if row["task"] == task]
        if not selected:
            continue
        by_task[task] = {"questions": len(selected),
                         "correct": sum(row["correct"] for row in selected),
                         "accuracy": mean([int(row["correct"]) for row in selected])}
        for field in ("recent_tokens_per_head", "history_tokens_per_head", "total_tokens_per_head"):
            by_task[task]["mean_" + field] = mean([row[field] for row in selected])
    group_means = {
        group: mean([by_task[task]["accuracy"] for task in tasks])
        if all(task in by_task for task in tasks) else None
        for group, tasks in GROUPS.items()}
    selected = [row["arms"][arm] for row in rows]
    return {"questions": len(rows), "correct": sum(row["correct"] for row in selected),
            "question_weighted_accuracy": mean([int(row["correct"]) for row in selected]),
            "nine_task_equal_mean": mean([by_task[task]["accuracy"] for task in TASKS])
            if len(by_task) == len(TASKS) else None,
            "two_group_equal_mean": mean(list(group_means.values()))
            if all(value is not None for value in group_means.values()) else None,
            "group_equal_task_means": group_means, "per_task": by_task,
            "mean_recent_tokens_per_head": mean([row["recent_tokens_per_head"] for row in selected]),
            "mean_history_tokens_per_head": mean([row["history_tokens_per_head"] for row in selected]),
            "mean_total_tokens_per_head": mean([row["total_tokens_per_head"] for row in selected]),
            "min_query_mean_tokens_per_head": min(row["total_tokens_per_head"] for row in selected),
            "max_query_mean_tokens_per_head": max(row["total_tokens_per_head"] for row in selected),
            "stored_correct_disagreements": sum(row["stored_correct_matches_official"] is False
                                                 for row in selected)}


def pct(value):
    return "未覆盖全部所需题型" if value is None else f"{value * 100:.2f}%"


def make_report(output_root, allow_partial=False):
    source = output_root / "queries.jsonl"
    rows, skipped = load_rows(source, allow_partial=allow_partial)
    arms = {arm: summarize_arm(rows, arm) for arm in ARMS}
    recent_only = sum(row["arms"]["recent_only"]["correct"] and
                      not row["arms"]["mild_history"]["correct"] for row in rows)
    history_only = sum(row["arms"]["mild_history"]["correct"] and
                       not row["arms"]["recent_only"]["correct"] for row in rows)
    both_correct = sum(all(row["arms"][arm]["correct"] for arm in ARMS) for row in rows)
    both_wrong = len(rows) - recent_only - history_only - both_correct
    summary = {"input": str(source.resolve()), "paired_questions": len(rows),
               "excluded_incomplete_query_ids": skipped,
               "scope": "OVO-Bench nine backward/realtime task diagnostic sample; excludes REC/SSR/CRR",
               "scoring": {"function": "sparsemm_native.run_ovobench.official_correct",
                           "source": "sparsemm_native/run_ovobench.py; mirrors OVO-Bench utils/OVOBenchScore.py",
                           "mcq_rule": "case-sensitive target in raw response; no letter extraction",
                           "group_mean": "mean(task accuracies) within each group; mean(two group accuracies)",
                           "official_twelve_task_total": None},
               "budget_unit": "post-assembly visual KV token slots / 112 layer-KV-head pairs; excludes text, QA and temporary prefill",
               "configured_mean_visual_budget": 500,
               "arms": arms,
               "paired_outcomes": {"recent_only_correct": recent_only,
                                   "history_only_correct": history_only,
                                   "both_correct": both_correct, "both_wrong": both_wrong,
                                   "net_correct_history_minus_recent": history_only - recent_only}}
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    fieldnames = ["query_id", "task", "group", "target"]
    for arm in ARMS:
        fieldnames.extend(f"{arm}_{field}" for field in
                          ("response", "correct", "recent_tokens_per_head", "history_tokens_per_head",
                           "total_tokens_per_head", "visual_slots_all_heads", "stored_correct_matches_official"))
    with (output_root / "per_query.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {key: row[key] for key in ("query_id", "task", "group", "target")}
            for arm in ARMS:
                for key, value in row["arms"][arm].items():
                    if f"{arm}_{key}" in fieldnames:
                        flat[f"{arm}_{key}"] = value
            writer.writerow(flat)

    lines = ["# 近期控制版与历史版：同题小样本对比", "",
             f"同题 {len(rows)} 道；覆盖回溯与即时任务，不含 REC、SSR、CRR。小样本结果不能作为全量结论。", "",
             "每条 raw response 均用 `sparsemm_native.run_ovobench.official_correct` 重评分，"
             "与官方 `utils/OVOBenchScore.py` 一致：选择题按区分大小写的 `target in response` 判断。", "",
             "| 任务 | 题数 | 近期控制正确数 | 近期控制正确率 | 历史版正确数 | 历史版正确率 | 近期控制 token/头 | 历史版 token/头 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for task in TASKS:
        if task not in arms["recent_only"]["per_task"]:
            continue
        a, b = [arms[arm]["per_task"][task] for arm in ARMS]
        lines.append(f"| {task} | {a['questions']} | {a['correct']} | {pct(a['accuracy'])} | "
                     f"{b['correct']} | {pct(b['accuracy'])} | {a['mean_total_tokens_per_head']:.2f} | "
                     f"{b['mean_total_tokens_per_head']:.2f} |")
    lines.extend(["", "| 汇总指标 | 近期控制版 | 历史版 |", "|---|---:|---:|"])
    for label, field in (("逐题正确率", "question_weighted_accuracy"),
                         ("九任务等权均值", "nine_task_equal_mean"),
                         ("回溯三类均值", "backward"), ("即时六类均值", "realtime"),
                         ("回溯／即时两组等权均值", "two_group_equal_mean")):
        values = [stats["group_equal_task_means"][field] if field in GROUPS else stats[field]
                  for stats in arms.values()]
        lines.append(f"| {label} | {pct(values[0])} | {pct(values[1])} |")
    lines.extend(["", "| 实际缓存用量，视觉 token/头 | 近期控制版 | 历史版 |", "|---|---:|---:|"])
    for label, field in (("近期均值", "mean_recent_tokens_per_head"),
                         ("历史均值", "mean_history_tokens_per_head"),
                         ("合计均值", "mean_total_tokens_per_head"),
                         ("逐题合计均值最小值", "min_query_mean_tokens_per_head"),
                         ("逐题合计均值最大值", "max_query_mean_tokens_per_head")):
        lines.append(f"| {label} | {arms[ARMS[0]][field]:.2f} | {arms[ARMS[1]][field]:.2f} |")
    lines.extend(["", f"配对结果：仅近期控制答对 {recent_only} 题；仅历史版答对 {history_only} 题；"
                  f"两版都对 {both_correct} 题；两版都错 {both_wrong} 题。历史版净变化 {history_only - recent_only:+d} 题。", "",
                  "K500 表示配置的全局平均视觉配额；实际用量为回答前拼接缓存的视觉 token 槽数除以 "
                  "112（28 层 × 4 KV 头），排除文本前缀、问题、生成 token 与临时完整 prefill。", "",
                  "回溯和即时组分别对各自 3 类、6 类正确率取算术平均，再对两组取平均；"
                  "该九类指标不称作十二类官方全量总分。"])
    if skipped:
        lines.extend(["", f"进度报告排除尚未完成两臂的 {len(skipped)} 道题：{', '.join(skipped)}。"])
    (output_root / "comparison.md").write_text("\n".join(lines) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--allow-partial", action="store_true", help="Skip rows missing one complete arm")
    args = parser.parse_args()
    summary = make_report(args.output_root, allow_partial=args.allow_partial)
    print(json.dumps({"paired_questions": summary["paired_questions"],
                      "arms": {arm: {key: value for key, value in stats.items() if key != "per_task"}
                               for arm, stats in summary["arms"].items()},
                      "report": str((args.output_root / "comparison.md").resolve())},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
