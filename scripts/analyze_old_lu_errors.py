#!/usr/bin/env python3
"""Analyze error modes of the old 50-video-calibrated LU policy on 250 videos."""

import ast
import glob
import json
import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path("results/qwen2.5_vl_7b/streamingbench")
ANNO = Path("data/streamingbench/streamingbench_realtime_half_userpaths.json")
OUT = Path("results/myobservations/old_lu_error_analysis_250v_20260916")
OUT.mkdir(parents=True, exist_ok=True)


def files(patterns):
    result = []
    for pattern in patterns:
        matched = glob.glob(str(ROOT / pattern))
        if len(matched) != 1:
            raise RuntimeError((pattern, matched))
        result.extend(matched)
    return result


def load(name, patterns):
    frame = pd.concat([pd.read_csv(p) for p in files(patterns)], ignore_index=True)
    frame["occurrence"] = frame.groupby(["video_id", "question"]).cumcount()
    frame = frame.set_index(["video_id", "question", "occurrence"]).sort_index()
    frame = frame.rename(columns={"qa_acc": name, "pred_choice": f"{name}_pred_choice",
                                  "pred_answer": f"{name}_pred_answer"})
    return frame


old = load("old", [
    "*lu-raw-blend05-clip075-150-native-first50-k1000/results.csv",
    "*lu-raw-blend05-clip075-150-native-51-100-k1000/results.csv",
    "*lu-raw-blend05-clip075-150-native-101-150-k1000/results.csv",
    "*lu-raw-blend05-clip075-150-native-151-250-k1000/results.csv",
])
uniform = load("uniform", [
    "*hermes-uniform-native-k1000-first50/results.csv",
    "*hermes-uniform-native-k1000-51-150/results.csv",
    "*hermes-uniform-native-k1000-151-250/results.csv",
])
new = load("new", ["*lu-balanced100v-blend05-clip075-150-native-first250-k1000/results.csv"])
assert old.index.equals(uniform.index) and old.index.equals(new.index)

annotation_rows = []
for video in json.loads(ANNO.read_text()):
    seen = Counter()
    conversations = video["conversations"]
    for q_index, question in enumerate(conversations):
        key = question["question"]
        occurrence = seen[key]
        seen[key] += 1
        annotation_rows.append({
            "video_id": video["video_id"], "question": key, "occurrence": occurrence,
            "q_index": q_index, "relative_position": q_index / max(1, len(conversations) - 1),
            "end_time": float(question["end_time"]),
            "required_ability": question.get("required_ability", "unknown"),
        })
meta = pd.DataFrame(annotation_rows).set_index(["video_id", "question", "occurrence"]).sort_index()
assert old.index.equals(meta.index)

df = old.join(uniform[["uniform", "uniform_pred_choice"]]).join(new[["new", "new_pred_choice"]]).join(meta)
df["old_correct"] = df.old > 0
df["uniform_correct"] = df.uniform > 0
df["new_correct"] = df.new > 0
df["time_bin"] = pd.cut(df.end_time, [-1, 60, 120, 240, 480, math.inf],
                        labels=["0-60s", "61-120s", "121-240s", "241-480s", ">480s"])


def selected_text(row):
    try:
        choices = ast.literal_eval(row.choices)
        index = ord(str(row.old_pred_choice)[0]) - ord("A")
        return str(choices[index]) if 0 <= index < len(choices) else ""
    except Exception:
        return ""


df["old_selected_text"] = df.apply(selected_text, axis=1)
answers_by_video_question = {}
for index, row in df.reset_index().iterrows():
    key = (row.video_id, row.question)
    answers_by_video_question.setdefault(key, []).append((row.q_index, str(row.answer)))


def temporal_confusion(row):
    if row.old_correct:
        return ""
    alternatives = answers_by_video_question[(row.name[0], row.name[1])]
    matches = [q_index for q_index, answer in alternatives
               if answer.strip().lower() == row.old_selected_text.strip().lower()
               and q_index != row.q_index]
    if not matches:
        return ""
    return "prior_checkpoint_answer" if min(matches) < row.q_index else "future_checkpoint_answer"


df["temporal_confusion"] = df.apply(temporal_confusion, axis=1)
errors = df[~df.old_correct].copy()
regressions = df[(~df.old_correct) & df.uniform_correct].copy()
improvements = df[df.old_correct & (~df.uniform_correct)].copy()

error_columns = [
    "task", "required_ability", "q_index", "end_time", "time_bin", "question",
    "answer", "correct_choice", "old_pred_choice", "old_selected_text", "old_pred_answer",
    "uniform_pred_choice", "new_pred_choice", "uniform_correct", "new_correct",
    "temporal_confusion", "choices",
]
errors.reset_index()[["video_id", "occurrence", *error_columns]].to_csv(
    OUT / "old_lu_all_errors.csv", index=False)
regressions.reset_index()[["video_id", "occurrence", *error_columns]].to_csv(
    OUT / "old_lu_regressions_vs_uniform.csv", index=False)
improvements.reset_index()[["video_id", "occurrence", *error_columns]].to_csv(
    OUT / "old_lu_improvements_vs_uniform.csv", index=False)


def grouped(frame, column):
    table = frame.groupby(column, observed=True).old_correct.agg(["count", "sum"])
    return {str(index): {"n": int(row["count"]), "correct": int(row["sum"]),
                         "accuracy": float(row["sum"] / row["count"] * 100)}
            for index, row in table.iterrows()}


summary = {
    "samples": len(df), "correct": int(df.old_correct.sum()), "errors": len(errors),
    "accuracy": float(df.old_correct.mean() * 100),
    "pairwise_vs_uniform": {
        "both_correct": int((df.old_correct & df.uniform_correct).sum()),
        "old_only_correct": len(improvements), "uniform_only_correct": len(regressions),
        "both_wrong": int(((~df.old_correct) & (~df.uniform_correct)).sum()),
    },
    "three_way_outcomes": {
        f"old_{int(o)}_uniform_{int(u)}_new_{int(n)}": int(((df.old_correct == o) &
            (df.uniform_correct == u) & (df.new_correct == n)).sum())
        for o in (False, True) for u in (False, True) for n in (False, True)
    },
    "by_task": grouped(df, "task"),
    "by_ability": grouped(df, "required_ability"),
    "by_time": grouped(df, "time_bin"),
    "by_question_index": grouped(df, "q_index"),
    "error_correct_choice": dict(Counter(errors.correct_choice)),
    "error_predicted_choice": dict(Counter(errors.old_pred_choice)),
    "temporal_confusions": dict(Counter(x for x in errors.temporal_confusion if x)),
    "temporal_confusions_by_task": dict(Counter(
        errors.loc[errors.temporal_confusion != "", "task"])),
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(summary, indent=2, ensure_ascii=False))
