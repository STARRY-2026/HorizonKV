"""Compare tail-only recent KV against the matched full-prefix control.

Only standard-library imports are needed. Scoring and KV accounting reuse the
paired report's checked official rule. CLI::

    python3 sparsemm_native/report_recent_tail27.py --root RUN_DIR \
        --reference-root OLD_PAIR_DIR

RUN_DIR/queries.jsonl has query_id, task, target, response, budget_audit,
seconds, and optionally reference_response, reference_budget_audit,
full_prefix_seconds, full_prefix_response, full_prefix_budget_audit,
tail_audit and timing. Reference fields otherwise come from the old paired
run's arms.recent_only. Historical paired wall time is labelled separately;
it is never treated as a fair single-control speed measurement.
"""

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sparsemm_native.report_aligned_pseudo_pair import (  # noqa: E402
    GROUPS, TASKS, budget_values, official_correct, pct, summarize_arm,
)


def load_jsonl(path):
    rows, seen = [], set()
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        qid = str(row["query_id"])
        if qid in seen:
            raise ValueError(f"Duplicate query_id {qid} at {path}:{lineno}")
        seen.add(qid)
        rows.append(row)
    return rows


def finite_seconds(value, label):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{label}: expected nonnegative finite seconds")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{label}: expected nonnegative finite seconds")
    return float(value)


def percentile(values, percent):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values):
    selected = [value for value in values if value is not None]
    if not selected:
        return None
    return {"queries": len(selected), "mean": statistics.mean(selected),
            "minimum": min(selected), "maximum": max(selected),
            "median": statistics.median(selected), "p90": percentile(selected, 90),
            "counts": dict(sorted(Counter(str(value) for value in selected).items()))}


def time_stats(values):
    selected = [value for value in values if value is not None]
    if not selected:
        return None
    result = distribution(selected)
    result.pop("counts")
    result["total"] = sum(selected)
    return result


def first_present(*values):
    return next((value for value in values if value is not None), None)


def normalize_arm(task, target, response, audit, label):
    if response is not None and not isinstance(response, str):
        raise ValueError(f"{label}: response must be string or null")
    result = dict(response=response, correct=bool(official_correct(task, response, target)),
                  stored_correct_matches_official=None, **budget_values(audit, label))
    if result["history_tokens_per_head"] != 0:
        raise ValueError(f"{label}: recent-only control includes historical visual KV")
    return result


def reference_index(reference_root):
    if reference_root is None:
        return {}
    path = reference_root / "queries.jsonl"
    if not path.exists():
        path = reference_root / "recent_only" / "queries.jsonl"
    return {str(row["query_id"]): row for row in load_jsonl(path)}


def make_report(root, reference_root=None, allow_partial=False):
    raw_rows = load_jsonl(root / "queries.jsonl")
    reference = reference_index(reference_root)
    rows, skipped = [], []
    for raw in raw_rows:
        qid = str(raw["query_id"])
        old = reference.get(qid, {})
        old_arm = old.get("arms", {}).get("recent_only", old)
        if "response" not in raw or "budget_audit" not in raw:
            if allow_partial:
                skipped.append(qid)
                continue
            raise ValueError(f"Query {qid}: missing tail response/budget_audit")
        task, target = raw["task"], raw["target"]
        if task not in TASKS:
            raise ValueError(f"Query {qid}: unsupported task {task}")
        if not isinstance(target, str) or target not in tuple("ABCDEFGH"):
            raise ValueError(f"Query {qid}: target must be a single MCQ letter")
        if old and (old["task"] != task or old["target"] != target):
            raise ValueError(f"Query {qid}: reference task or target differs")
        expected_group = "backward" if task in GROUPS["backward"] else "realtime"
        if raw.get("group", expected_group) != expected_group:
            raise ValueError(f"Query {qid}: group does not match task")
        # Presence, rather than truthiness, matters: null is a scored response.
        ref_response = raw["reference_response"] if "reference_response" in raw else old_arm.get("response")
        ref_budget = raw.get("reference_budget_audit", old_arm.get("budget_audit"))
        if ref_budget is None:
            raise ValueError(f"Query {qid}: missing reference_budget_audit/reference row")
        audit, tail = raw["budget_audit"], raw.get("tail_audit", {})
        paired = old.get("paired_audit", {})
        timing = raw.get("timing", {})
        old_audit = ref_budget
        seconds = finite_seconds(first_present(raw.get("seconds"), timing.get("tail_seconds"),
                                                timing.get("tail_total_seconds")), f"{qid}.seconds")
        full_seconds = finite_seconds(first_present(raw.get("full_prefix_seconds"),
            timing.get("full_prefix_seconds"), timing.get("full_prefix_total_seconds")), f"{qid}.full_prefix_seconds")
        full_budget = raw.get("full_prefix_budget_audit")
        full_response = raw.get("full_prefix_response")
        single_tokens = first_present(tail.get("single_frame_visual_tokens"), tail.get("last1_token_count"),
                                     audit.get("single_frame_visual_tokens"), paired.get("single_frame_visual_tokens"))
        last4_tokens = first_present(tail.get("last4_token_count"), tail.get("four_frame_visual_tokens"))
        unit_tokens = first_present(tail.get("recent_visual_unit_tokens"), paired.get("recent_visual_unit_tokens"))
        if last4_tokens is None and unit_tokens is not None:
            last4_tokens = sum(unit_tokens)
        frames_equal = first_present(tail.get("frame_indices_equal"), tail.get("sampled_indices_match"),
                                     tail.get("sampled_frame_indices_match"))
        positions_equal = first_present(tail.get("positions_equal"), tail.get("position_ids_match"),
                                        tail.get("positions_match"))
        row = {"query_id": qid, "task": task, "group": expected_group, "target": target,
               "arms": {"reference": normalize_arm(task, target, ref_response, ref_budget, f"{qid}/reference"),
                        "tail": normalize_arm(task, target, raw["response"], audit, f"{qid}/tail")},
               "response_identical": ref_response == raw["response"],
               "visual_budget_identical": audit["actual_visual_slots"] == old_audit["actual_visual_slots"],
               "recent_selection_identical": (audit.get("recent_selection_sha256") == old_audit.get("recent_selection_sha256")
                                                if "recent_selection_sha256" in audit and "recent_selection_sha256" in old_audit else None),
               "question_position_identical": (audit.get("question_position") == old_audit.get("question_position")
                                                if "question_position" in audit and "question_position" in old_audit else None),
               "official_prompt_identical": (audit.get("official_prompt_sha256") == old_audit.get("official_prompt_sha256")
                                                if "official_prompt_sha256" in audit and "official_prompt_sha256" in old_audit else None),
               "seconds": seconds, "full_prefix_seconds": full_seconds,
               "single_frame_visual_tokens": single_tokens, "four_frame_visual_tokens": last4_tokens,
               "frames_equal": frames_equal, "positions_equal": positions_equal,
               "tail_audit": tail, "timing": timing}
        if "full_prefix_response" in raw:
            row["fresh_full_prefix_response"] = full_response
            row["fresh_full_prefix_response_identical"] = full_response == raw["response"]
        if full_budget is not None:
            row["fresh_full_prefix_budget_identical"] = full_budget["actual_visual_slots"] == audit["actual_visual_slots"]
        rows.append(row)
    if not rows:
        raise ValueError("No complete tail-only queries")
    arms = {arm: summarize_arm(rows, arm) for arm in ("reference", "tail")}
    checks = {}
    for key in ("response_identical", "visual_budget_identical", "recent_selection_identical",
                "question_position_identical", "official_prompt_identical", "frames_equal", "positions_equal",
                "fresh_full_prefix_response_identical", "fresh_full_prefix_budget_identical"):
        available = [row for row in rows if row.get(key) is not None]
        checks[key] = {"checked": len(available), "matched": sum(bool(row[key]) for row in available),
                       "mismatched_query_ids": [row["query_id"] for row in available if not row[key]]}
    matched_timing = [row for row in rows if row["seconds"] is not None and row["full_prefix_seconds"] is not None]
    timing = {"tail": time_stats([row["seconds"] for row in rows]),
              "fresh_full_prefix": time_stats([row["full_prefix_seconds"] for row in rows]),
              "fair_paired_questions": len(matched_timing)}
    if matched_timing:
        timing["fair_tail_total_seconds"] = sum(row["seconds"] for row in matched_timing)
        timing["fair_full_prefix_total_seconds"] = sum(row["full_prefix_seconds"] for row in matched_timing)
        timing["fair_speedup_full_over_tail"] = (
            timing["fair_full_prefix_total_seconds"] / timing["fair_tail_total_seconds"]
            if timing["fair_tail_total_seconds"] > 0 else None)
    if reference_root is not None and (reference_root / "status.json").exists():
        status = json.loads((reference_root / "status.json").read_text())
        timing["previous_two_arm_elapsed_seconds_not_fair_single_control"] = status.get("elapsed_this_invocation_seconds")
    if (root / "status.json").exists():
        status = json.loads((root / "status.json").read_text())
        timing["current_runner_elapsed_seconds"] = status.get("elapsed_this_invocation_seconds")
    tokens = {"single_frame_visual_tokens": distribution([row["single_frame_visual_tokens"] for row in rows]),
              "four_frame_visual_tokens": distribution([row["four_frame_visual_tokens"] for row in rows])}
    summary = {"input": str((root / "queries.jsonl").resolve()),
               "reference_root": str(reference_root.resolve()) if reference_root is not None else None,
               "paired_questions": len(rows), "excluded_incomplete_query_ids": skipped,
               "scope": "Nine backward/realtime MCQ tasks; excludes REC/SSR/CRR",
               "scoring": "official_correct(task, raw_response, target); case-sensitive target in response",
               "budget_unit": "visual KV slots / 112 layer-KV-heads; excludes text, question, generation and temporary prefill",
               "configured_mean_visual_budget": 500, "arms": arms, "checks": checks,
               "token_inputs": tokens, "timing": timing,
               "paired_outcomes": {
                   "reference_only_correct": sum(row["arms"]["reference"]["correct"] and not row["arms"]["tail"]["correct"] for row in rows),
                   "tail_only_correct": sum(row["arms"]["tail"]["correct"] and not row["arms"]["reference"]["correct"] for row in rows)}}
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    fields = ["query_id", "task", "group", "target", "reference_response", "tail_response", "reference_correct", "tail_correct",
              "response_identical", "visual_budget_identical", "recent_selection_identical", "question_position_identical",
              "official_prompt_identical", "frames_equal", "positions_equal", "single_frame_visual_tokens", "four_frame_visual_tokens",
              "reference_recent_tokens_per_head", "reference_history_tokens_per_head", "reference_total_tokens_per_head",
              "tail_recent_tokens_per_head", "tail_history_tokens_per_head", "tail_total_tokens_per_head", "seconds", "full_prefix_seconds"]
    with (root / "per_query.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {key: row.get(key) for key in fields}
            for arm in ("reference", "tail"):
                for key, value in row["arms"][arm].items():
                    if f"{arm}_{key}" in fields:
                        flat[f"{arm}_{key}"] = value
            writer.writerow(flat)
    lines = ["# 仅处理末尾帧：与近期控制版的同题验证", "",
             f"同题 {len(rows)} 道，九类回溯／即时任务，不含前瞻类 REC、SSR、CRR。", "",
             "保留 fps=0.5、K500、默认视频分辨率、官方问题和评分；逐条原始回答重新按官方规则判断。", "",
             "| 指标 | 原近期控制版 | 末尾帧控制版 |", "|---|---:|---:|"]
    lines.append(f"| 正确数 | {arms['reference']['correct']}/{len(rows)} | {arms['tail']['correct']}/{len(rows)} |")
    for label, key in (("逐题正确率", "question_weighted_accuracy"), ("九类等权均值", "nine_task_equal_mean"),
                       ("回溯／即时两组等权均值", "two_group_equal_mean")):
        lines.append(f"| {label} | {pct(arms['reference'][key])} | {pct(arms['tail'][key])} |")
    for label, key in (("平均近期视觉 token/头", "mean_recent_tokens_per_head"),
                       ("平均历史视觉 token/头", "mean_history_tokens_per_head"),
                       ("平均合计视觉 token/头", "mean_total_tokens_per_head")):
        lines.append(f"| {label} | {arms['reference'][key]:.2f} | {arms['tail'][key]:.2f} |")
    lines.extend(["", f"原始回答逐字相同 {checks['response_identical']['matched']}/{len(rows)}；"
                  f"实际视觉预算一致 {checks['visual_budget_identical']['matched']}/{len(rows)}。", "",
                  "| 任务 | 题数 | 原控制正确数 | 末尾版正确数 |", "|---|---:|---:|---:|"])
    for task in TASKS:
        if task in arms["tail"]["per_task"]:
            a, b = [arms[arm]["per_task"][task] for arm in ("reference", "tail")]
            lines.append(f"| {task} | {a['questions']} | {a['correct']} | {b['correct']} |")
    lines.extend(["", "## 单帧输入的实际视觉 token", "",
                  "单帧指最后一张真实采样帧单独送视频处理器；处理器内部补帧，按一个视觉时间单元输出。"
                  "以下为裁剪前候选数量，回答缓存仍受 K500 配额约束。"])
    for label, key in (("一帧候选", "single_frame_visual_tokens"), ("近期四帧候选", "four_frame_visual_tokens")):
        stats = tokens[key]
        if stats:
            counts = "、".join(f"{count} token × {number} 题" for count, number in stats["counts"].items())
            lines.extend(["", f"{label}：平均 {stats['mean']:.2f}，范围 {stats['minimum']}～{stats['maximum']}；{counts}。"])
    lines.extend(["", "## 耗时口径", ""])
    if timing["tail"]:
        a = timing["tail"]
        lines.append(f"末尾版逐题端到端耗时合计 {a['total']:.2f} 秒，平均 {a['mean']:.2f} 秒，"
                     f"中位数 {a['median']:.2f} 秒，P90 {a['p90']:.2f} 秒。")
    if matched_timing:
        lines.extend(["", f"本轮同模型单控制全前缀基准覆盖 {len(matched_timing)} 题："
                      f"全前缀 {timing['fair_full_prefix_total_seconds']:.2f} 秒，"
                      f"末尾版 {timing['fair_tail_total_seconds']:.2f} 秒；"
                      f"合计耗时加速 {timing['fair_speedup_full_over_tail']:.2f} 倍。"])
    else:
        lines.extend(["", "尚无本轮同模型、单控制的全前缀计时，不能据此给出严格的推理加速倍数。"])
    previous = timing.get("previous_two_arm_elapsed_seconds_not_fair_single_control")
    if previous is not None:
        lines.extend(["", f"旧实验总耗时 {previous:.2f} 秒包含历史维护和两个作答臂，"
                      "不能当作单独近期控制版的公平速度基准。"])
    lines.extend(["", "实际 token 预算仅统计作答前视觉 KV，排除文本、生成与临时 dense prefill。"
                  "两组均分先对回溯三类、即时六类各自等权，再对两组等权，不代表十二类全量总分。"])
    failures = {key: value["mismatched_query_ids"] for key, value in checks.items() if value["mismatched_query_ids"]}
    if failures:
        lines.extend(["", "存在需要核对的差异：", "", "```json", json.dumps(failures, ensure_ascii=False, indent=2), "```"])
    (root / "comparison.md").write_text("\n".join(lines) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", "--output-root", dest="root", required=True, type=Path)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    summary = make_report(args.root, args.reference_root, args.allow_partial)
    print(json.dumps({"paired_questions": summary["paired_questions"], "checks": summary["checks"],
                      "timing": summary["timing"], "report": str(args.root / "comparison.md")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
