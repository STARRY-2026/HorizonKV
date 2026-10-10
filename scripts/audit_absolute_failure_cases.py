#!/usr/bin/env python3
"""Extend the completed-run audit to absolute/common errors, without inference.

The 33 reviewed questions are a purposive, task-diverse sample. Their cause
labels must not be extrapolated to all errors. Frame observations distinguish
source-video visibility, actual sampled-input visibility, and unobserved KV
retention. No full-cache counterfactual is available for these native runs.
"""

import argparse
from collections import Counter
import csv
import hashlib
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "results/myobservations/dp_failure_audit_20261010"


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT)
    args = parser.parse_args()
    out = args.audit / "absolute_errors"
    rows = read_csv(args.audit / "case_ledger.csv")
    selected = read_csv(out / "selected_cases.csv")
    timing = {(r["video_id"], r["question_index"]): r
              for r in read_csv(out / "sampling_timing.csv")}
    large = {r["source_row"]: r for r in read_csv(out / "legacy_large_budget_comparison.csv")}
    review = json.loads((out / "manual_case_review.json").read_text())
    notes = {(str(r["video_id"]), str(r["question_index"])): r for r in review["cases"]}
    anno_path = ROOT / "data/streamingbench/streamingbench_realtime_half_userpaths.json"
    expected = []
    for v in json.loads(anno_path.read_text()):
        for qi, q in enumerate(v["conversations"]):
            expected.append((str(v["video_id"]), str(qi), q["question"],
                             q["choices"], q["answer"], float(q["end_time"])))
    observed = [(r["video_id"], r["question_index"], r["question"],
                 json.loads(r["choices"]), r["answer"], float(r["end_time"])) for r in rows]
    assert observed == expected
    assert len(notes) == len(selected) == 33
    assert all(r["outcome_H_L_D"] == "000" for r in selected)
    assert all(r["dp_regression_vs_lu"] == "0" for r in selected)

    errors = [r for r in rows if r["dp1000_correct"] == "0"]
    common = [r for r in rows if r["outcome_H_L_D"] == "000"]
    strict_methods = [f"{m}{k}" for k in (250, 500, 1000)
                      for m in ("hermes", "lu", "dp")]
    persistent = [r for r in rows if all(r[m + "_correct"] == "0" for m in strict_methods)]
    assert (len(rows), len(errors), len(common), len(persistent)) == (1255, 369, 335, 192)
    task_stats = {}
    for task in sorted({r["task"] for r in rows}):
        group = [r for r in rows if r["task"] == task]
        wrong = [r for r in group if r["dp1000_correct"] == "0"]
        task_stats[task] = {"n": len(group), "dp_wrong": len(wrong),
                            "dp_error_rate": len(wrong) / len(group),
                            "common_H_L_D_wrong": sum(r["outcome_H_L_D"] == "000" for r in group)}

    reviewed_rows = []
    for r in selected:
        key = (r["video_id"], r["question_index"])
        n = notes[key]
        t = timing[key]
        extra = {"review_category": n["category"], "review_evidence": n["evidence"],
                 "interpretation_and_limit": n["interpretation_and_limit"],
                 "last_sample_actual_time": t["last_frame_actual_time_s"],
                 "last_sample_source_index": t["last_source_frame_index"],
                 "sampling_lag_seconds": t["lag_end_minus_actual_s"],
                 "legacy_hermes6000_answer": large[r["source_row"]]["legacy_hermes6000_answer"],
                 "legacy_hermes6000_correct": large[r["source_row"]]["legacy_hermes6000_correct"]}
        reviewed_rows.append(dict(r, **extra))
    write_csv(out / "reviewed_cases.csv", reviewed_rows)
    write_csv(out / "absolute_error_ledger.csv", errors)

    strict_rescue = {}
    for name, subset in (("DP_wrong_first150", [r for r in errors if r["hermes2000_correct"] != ""]),
                         ("common_wrong_first150", [r for r in common if r["hermes2000_correct"] != ""]),
                         ("persistent_nine_configs_first150", persistent)):
        strict_rescue[name] = {"n": len(subset),
                               "hermes2000_repairs": sum(r["hermes2000_correct"] == "1" for r in subset),
                               "lu2000_repairs": sum(r["lu2000_correct"] == "1" for r in subset)}
    summary = {
        "protocol": {"identity": "annotation sequence + video_id + within-video question index + full question/choices/gold/end_time",
                     "inference_launched": False,
                     "selected_sample": "33 purposively selected questions from 18 videos, all ten task labels; all common K1000 errors, none in prior DP-vs-LU regression set",
                     "causal_limit": "Sampled input is not the retained KV cache. No matched native full-KV control. Selection is not representative for estimating cause prevalence."},
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (args.audit / "case_ledger.csv", anno_path, out / "manual_case_review.json")},
        "n": len(rows), "dp_errors": len(errors), "common_H_L_D_errors": len(common),
        "common_fraction_of_dp_errors": len(common) / len(errors),
        "common_errors_same_wrong_option_all_three": sum(len({r[m + "_choice"] for m in ("hermes1000", "lu1000", "dp1000")}) == 1 for r in common),
        "dp_errors_first_question": sum(r["question_index"] == "0" for r in errors),
        "by_task": task_stats, "strict_budget_rescue": strict_rescue,
        "reviewed_n": len(selected), "reviewed_videos": len({r["video_id"] for r in selected}),
        "reviewed_tasks": dict(Counter(r["task"] for r in selected)),
        "review_category_counts_selected_only": dict(Counter(r["category"] for r in notes.values())),
        "v11_sampled_event_audit": review["v11_sampled_event_audit"],
        "next_experiments": review["next_experiments"],
        "supporting_artifacts": ["counting_analysis.json", "counting_error_ledger.csv", "sampling_timing.json",
                                 "legacy_large_budget_summary.json", "visual_second_review.json", "frame_manifest.json",
                                 "additional_frame_manifest.json"],
    }
    (out / "absolute_error_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    esc = html.escape
    body = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>实验共同失败案例扩展审计</title>",
            "<style>body{font:16px/1.65 system-ui;max-width:1250px;margin:36px auto;padding:0 20px;color:#17202a}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:9px;text-align:left;vertical-align:top}th{background:#f4f6f8}img{max-width:100%}small{color:#53616f}a{color:#1565c0}</style>",
            "<h1>更多共同失败案例：离线证据审计</h1>",
            "<p>检查全部1255题、DP的369道错题，新增目视审阅18个视频的33题，覆盖全部10类任务。33题均为HERMES、旧LU、DP在K1000共同答错，不属于此前20个DP对LU回退案例。样本按研究目的选择，不能据其比例估计所有错误的原因分布。</p>",
            "<p><b>主要结论：</b>存在压缩前采样漏证据、累计事件状态失败、已输入的细节对应错误、因果/预测推理困难，以及选项/参照系歧义。已输入并不等于已保留；缺少匹配full-KV对照，不能把后几类直接归因模型固有能力。</p>",
            "<h2>采样缺口的直接证据</h2><img src='sampling_gap_examples.jpg' alt='实际末帧与提问时刻对照'><p>全部33题末输入帧落后截点0.4–2.0秒；按真实源FPS与PTS核验，没有输入帧越过截点。此范围只适用于所审33题。</p>",
            "<h2>累计插片计数</h2><p>计数错误96/182，92题三方共同错。累计插片170题错88题；137个同题相邻转移中gold从不下降，DP预测下降16次。v11截至196秒的全部98个实际采样帧包含五次独立插入，来源只有三部电影；三方均答3。插入事件未被采样整体漏掉，可能混淆计数单位或未保住事件边界/累计状态。</p>",
            "<p>九配置持续错误中的66道累计插片题，在严格配对的HERMES和LU K2000中仍全部错误。大预算旧dense路径只作为辅助比较，不是受控预算消融，也不是full cache。</p>",
            "<h2>33题逐例核对</h2><p><small>q为从0起的题序；表中区分直接观察与推测。H6000列来自旧dense路径。</small></p><table><tr><th>案例/任务</th><th>gold → DP</th><th>可见证据</th><th>解释与限制</th></tr>"]
    for r in reviewed_rows:
        body.append("<tr><td>v%s / q%s / %ss<br>%s<br><small>%s</small></td><td>%s → %s</td><td>%s</td><td>%s</td></tr>" % tuple(esc(str(x)) for x in (
            r["video_id"], r["question_index"], r["end_time"], r["task"], r["review_category"],
            r["answer"], r["dp1000_answer"], r["review_evidence"], r["interpretation_and_limit"])))
    body.append("</table><h2>最小验证实验</h2><ol>")
    for exp in review["next_experiments"]:
        body.append("<li><b>%s：</b>%s</li>" % (esc(exp["name"]), esc(exp["design"])))
    body.append("</ol><p>复查数据：<a href='reviewed_cases.csv'>33题CSV</a> · <a href='absolute_error_ledger.csv'>369道错题</a> · <a href='counting_analysis.json'>计数分析</a> · <a href='sampling_timing.csv'>实际采样时刻</a> · <a href='absolute_error_summary.json'>汇总与来源</a></p></html>")
    (out / "report.html").write_text("\n".join(body))
    print(json.dumps({k: summary[k] for k in ("n", "dp_errors", "common_H_L_D_errors", "common_errors_same_wrong_option_all_three", "reviewed_n", "reviewed_videos")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
