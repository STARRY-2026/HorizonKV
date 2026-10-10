#!/usr/bin/env python3
"""Audit aligned, completed QA runs without launching model inference.

Question identity includes its annotation position: repeated text is not unique.
All counterfactual combinations in this report are descriptive oracle bounds,
not deployable selectors or estimates of an achievable accuracy gain.
"""

import argparse
import ast
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/qwen2.5_vl_7b/streamingbench"


def load_run(suffix, sources):
    matches = list(RESULTS.glob("*" + suffix))
    if len(matches) != 1:
        raise ValueError(f"Ambiguous or missing run: {suffix}: {matches}")
    path = matches[0] / "results.csv"
    sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if float(row["qa_acc"]) not in (0.0, 100.0):
            raise ValueError(f"Nonbinary score in {path}")
        if (row["pred_choice"] == row["correct_choice"]) != (float(row["qa_acc"]) == 100):
            raise ValueError(f"Choice and score disagree in {path}")
    return rows


def identity(row):
    return (str(row["video_id"]), row["question"],
            tuple(ast.literal_eval(row["choices"])), row["answer"], row["task"])


def compare(rows, left, right):
    selected = [r for r in rows if r.get(left + "_correct", "") != ""
                and r.get(right + "_correct", "") != ""]
    wins = sum(r[right + "_correct"] > r[left + "_correct"] for r in selected)
    losses = sum(r[right + "_correct"] < r[left + "_correct"] for r in selected)
    return {"n": len(selected), "left_correct": sum(r[left + "_correct"] for r in selected),
            "right_correct": sum(r[right + "_correct"] for r in selected),
            "right_wins": wins, "right_losses": losses,
            "delta_pp": 100 * (wins - losses) / len(selected)}


def grouped(rows, field, methods):
    groups = collections.defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    result = {}
    for key, group in sorted(groups.items()):
        result[key] = {"n": len(group)}
        for method in methods:
            result[key][method + "_correct"] = sum(r[method + "_correct"] for r in group)
        result[key]["DP_vs_LU"] = compare(group, "lu1000", "dp1000")
        result[key]["DP_vs_HERMES"] = compare(group, "hermes1000", "dp1000")
    return result


def time_bin(time):
    for upper in (60, 120, 240, 480):
        if time <= upper:
            return f"<={upper}s"
    return ">480s"


def policy_mapping(output, sources):
    """Reproduce the deployed profile transform without loading the model."""
    folders = {
        100: "two_bank_grid_dp_k100_floor75_20260929",
        250: "k250_cachecomp_score_pilot_20260923",
        500: "two_bank_grid_dp_k500_floor375_20260925",
        1000: "k1000_dp_cachecomp_score_20260924",
    }
    reports, mapped = {}, []
    for budget, folder in folders.items():
        directory = ROOT / "results/myobservations" / folder
        path = directory / "recent_ratio_profile.csv"
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = list(csv.DictReader(path.open()))
        if [(int(r["layer"]), int(r["kv_head"])) for r in rows] != [(l, h) for l in range(28) for h in range(4)]:
            raise ValueError(f"Unexpected head order in {path}")
        zero_mad, distinct_zero_mad = [], []
        head_rows = []
        for layer in range(28):
            group = rows[4 * layer:4 * layer + 4]
            fractions = [int(r["recent_r"]) / int(r["capacity_n"]) for r in group]
            center = statistics.median(fractions)
            scale = 1.4826 * statistics.median(abs(x - center) for x in fractions)
            if scale < 1e-12:
                zero_mad.append(layer)
                if max(fractions) - min(fractions) > 1e-12:
                    distinct_zero_mad.append(layer)
            base_alpha, base_k = ((1.0, 20.0) if layer < 8 else
                                  ((0.0, 0.0) if layer >= 19 else
                                   (0.75 - 0.6 * (layer - 8) / 11, 20 - 12 * (layer - 8) / 11)))
            for head, (row, fraction) in enumerate(zip(group, fractions)):
                z = (fraction - center) / scale if scale >= 1e-12 else 0.0
                alpha = max(0.0, min(1.0, base_alpha + 0.1 * math.tanh(z)))
                decay = max(0.0, min(20.0, base_k + 2.0 * math.tanh(z)))
                head_rows.append({"K": budget, "layer": layer, "kv_head": head,
                                  "capacity": int(row["capacity_n"]), "offline_recent_r": int(row["recent_r"]),
                                  "recent_ratio": fraction, "z": z, "base_alpha": base_alpha,
                                  "alpha": alpha, "base_k": base_k, "k": decay})
        capacities = [r["capacity"] for r in head_rows]
        report = {
            "capacity_sum": sum(capacities),
            "heads_at_floor": sum(n == math.ceil(0.75 * budget) for n in capacities),
            "heads_at_cap": sum(n == int(1.5 * budget) for n in capacities),
            "recent_below32": sum(r["offline_recent_r"] < 32 for r in head_rows),
            "zero_MAD_layers": zero_mad,
            "zero_MAD_layers_with_distinct_profiles": distinct_zero_mad,
            "heads_both_parameters_unchanged": sum(
                math.isclose(r["alpha"], r["base_alpha"], abs_tol=1e-12)
                and math.isclose(r["k"], r["base_k"], abs_tol=1e-12) for r in head_rows),
        }
        score_paths = list(directory.glob("*exact_budget_scores.csv"))
        if len(score_paths) != 1:
            raise ValueError(f"Missing or ambiguous budget scores: {directory}")
        score_path = score_paths[0]
        sources[str(score_path.relative_to(ROOT))] = hashlib.sha256(score_path.read_bytes()).hexdigest()
        scores = [float(r["score"]) for r in csv.DictReader(score_path.open())]
        base = budget - 32
        minimum = int(base * 0.2)
        reconstructed = [round(s / sum(scores) * ((base - minimum) * 112) + minimum) + 32 for s in scores]
        report["runtime_budget_roundtrip_max_abs_error"] = max(abs(a - b) for a, b in zip(capacities, reconstructed))
        if report["runtime_budget_roundtrip_max_abs_error"] != 0 or sum(capacities) != 112 * budget:
            raise ValueError(f"Budget round trip failed at K={budget}")
        reports[str(budget)] = report
        mapped.extend(head_rows)
    with (output / "profile_mapping.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(mapped[0]))
        writer.writeheader()
        writer.writerows(mapped)
    (output / "profile_mapping.json").write_text(json.dumps(reports, indent=2) + "\n")
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = {}
    anno_path = ROOT / "data/streamingbench/streamingbench_realtime_half_userpaths.json"
    sources[str(anno_path.relative_to(ROOT))] = hashlib.sha256(anno_path.read_bytes()).hexdigest()
    annotation = json.loads(anno_path.read_text())
    metadata = []
    expected = []
    for video_index, video in enumerate(annotation):
        for question_index, question in enumerate(video["conversations"]):
            expected.append((str(video["video_id"]), question["question"],
                             tuple(question["choices"]), question["answer"], question["task"]))
            metadata.append({"source_row": len(metadata), "video_order": video_index,
                             "video_id": str(video["video_id"]), "question_index": question_index,
                             "question": question["question"], "task": question["task"],
                             "ability": question.get("required_ability", ""),
                             "end_time": float(question["end_time"]),
                             "time_bin": time_bin(float(question["end_time"])),
                             "video_path": video["video_path"],
                             "choices": json.dumps(question["choices"], ensure_ascii=False),
                             "answer": question["answer"]})
    runs = {
        "hermes1000": sum((load_run("hermes-uniform-native-k1000-" + suffix, sources)
                           for suffix in ("first50", "51-150", "151-250")), []),
        "lu1000": sum((load_run("lu-raw-blend05-clip075-150-native-" + suffix + "-k1000", sources)
                       for suffix in ("first50", "51-100", "101-150", "151-250")), []),
        "dp1000": load_run("dp-cachecomp-alpha-k1000-native-first250-20260924", sources),
        "alpha250": load_run("cachecomp-k250-a01-kd2-native-first150-20260923", sources),
        "hermes2000": load_run("hermes-uniform-native-k2000-first150", sources),
        "lu1200": load_run("lu-raw-blend05-clip075-150-native-first150-k1200", sources),
        "lu2000": sum((load_run("lu-raw-k2000-resolved-all-blend05-clip075-150-native-" + suffix, sources)
                       for suffix in ("first50", "51-150")), []),
    }
    for budget, date in ((100, "20260929"), (250, "20260924"), (500, "20260925")):
        runs[f"hermes{budget}"] = load_run(
            f"hermes-uniform-native-k{budget}-first150" + ("-20260929" if budget == 100 else ""), sources)
        runs[f"dp{budget}"] = load_run(f"dp-cachecomp-alpha-k{budget}-native-first150-{date}", sources)
        if budget != 100:
            runs[f"lu{budget}"] = load_run(f"lu-raw-blend05-clip075-150-native-first150-k{budget}", sources)
    for method, run in runs.items():
        if [identity(r) for r in run] != expected[:len(run)]:
            raise ValueError(f"Run does not match annotation sequence: {method}")
        if len(run) not in (754, 1255):
            raise ValueError(f"Unexpected run length for {method}: {len(run)}")
        for index, row in enumerate(metadata):
            if index < len(run):
                source = run[index]
                row[method + "_correct"] = int(float(source["qa_acc"]) == 100)
                row[method + "_choice"] = source["pred_choice"]
                choices = ast.literal_eval(source["choices"])
                row[method + "_answer"] = choices[ord(source["pred_choice"]) - ord("A")]
            else:
                for suffix in ("correct", "choice", "answer"):
                    row[method + "_" + suffix] = ""
    for row in metadata:
        row["outcome_H_L_D"] = "".join(str(row[m + "_correct"])
                                             for m in ("hermes1000", "lu1000", "dp1000"))
        row["dp_regression_vs_lu"] = int(row["lu1000_correct"] and not row["dp1000_correct"])
        row["dp_regression_vs_hermes"] = int(row["hermes1000_correct"] and not row["dp1000_correct"])
        row["prior_same_question_answer_candidate"] = 0
        row["prior_same_task_answer_candidate"] = 0
        if row["task"] == "Counting" and not row["dp1000_correct"]:
            prior = [p for p in metadata[:row["source_row"]] if p["video_id"] == row["video_id"]
                     and p["task"] == row["task"] and p["answer"] != row["answer"]]
            row["prior_same_task_answer_candidate"] = int(any(
                row["dp1000_answer"] == p["answer"] for p in prior))
            row["prior_same_question_answer_candidate"] = int(any(
                row["dp1000_answer"] == p["answer"] and row["question"] == p["question"] for p in prior))
    main_methods = ["hermes1000", "lu1000", "dp1000"]
    summary = {
        "protocol": {"annotation": str(anno_path), "videos": len(annotation),
                     "questions": len(metadata), "matching": "exact ordered annotation identity",
                     "scope": "completed native-resolution runs; no new model inference",
                     "case_indices": "question_index and source_row are zero based"},
        "source_sha256": sources,
        "accuracies": {m: {"n": len(run), "correct": sum(r[m + "_correct"] for r in metadata[:len(run)]),
                             "accuracy": 100 * sum(r[m + "_correct"] for r in metadata[:len(run)]) / len(run)}
                       for m, run in runs.items()},
        "three_way_outcomes_H_L_D": dict(sorted(collections.Counter(
            row["outcome_H_L_D"] for row in metadata).items())),
        "pairwise": {f"{a}_to_{b}": compare(metadata, a, b)
                     for a, b in (("hermes1000", "dp1000"), ("lu1000", "dp1000"),
                                  ("hermes250", "alpha250"), ("alpha250", "dp250"),
                                  ("lu250", "dp250"), ("lu500", "dp500"))},
        "by_task": grouped(metadata, "task", main_methods),
        "by_ability": grouped(metadata, "ability", main_methods),
        "by_time": grouped(metadata, "time_bin", main_methods),
        "by_question_index": grouped(metadata, "question_index", main_methods),
    }
    first150 = metadata[:754]
    losses = [r for r in metadata if r["dp_regression_vs_lu"]]
    summary["dp_vs_lu_lost_cases"] = {
        "n": len(losses), "first150": sum(r["video_order"] < 150 for r in losses),
        "earlier_prediction_changed_vs_lu": sum(any(
            p["video_id"] == r["video_id"] and p["question_index"] < r["question_index"]
            and p["lu1000_answer"] != p["dp1000_answer"] for p in metadata) for r in losses),
    }
    summary["budget_rescue_first150"] = {}
    groups = {
        "dp1000_wrong": [r for r in first150 if not r["dp1000_correct"]],
        "dp1000_regressions_vs_lu": [r for r in first150 if r["dp_regression_vs_lu"]],
        "all_three_K1000_wrong": [r for r in first150 if r["outcome_H_L_D"] == "000"],
    }
    for name, group in groups.items():
        summary["budget_rescue_first150"][name] = {"n": len(group)}
        for method in ("hermes2000", "lu1200", "lu2000"):
            summary["budget_rescue_first150"][name][method + "_correct"] = sum(
                r[method + "_correct"] for r in group)
    all_low_methods = [f"{m}{k}" for m in ("hermes", "lu", "dp") for k in (250, 500, 1000)]
    persistent = [r for r in first150 if all(not r[m + "_correct"] for m in all_low_methods)]
    summary["persistent_errors_9_policies_first150"] = {
        "n": len(persistent), "by_task": dict(collections.Counter(r["task"] for r in persistent)),
        "rescued_hermes2000": sum(r["hermes2000_correct"] for r in persistent),
        "rescued_lu2000": sum(r["lu2000_correct"] for r in persistent),
    }
    counting = [r for r in metadata if r["task"] == "Counting" and not r["dp1000_correct"]]
    summary["counting_error_candidates"] = {
        "n": len(counting),
        "prior_same_task_answer": sum(r["prior_same_task_answer_candidate"] for r in counting),
        "prior_exact_question_answer": sum(r["prior_same_question_answer_candidate"] for r in counting),
        "caveat": "Candidate stale checkpoint answers, not causal evidence of memory failure",
    }
    summary["policy_mapping"] = policy_mapping(args.output, sources)
    summary["oracle_union"] = {
        "correct_if_any_of_three_correct": sum(r["outcome_H_L_D"] != "000" for r in metadata),
        "dp_errors_common_to_all_three": sum(r["outcome_H_L_D"] == "000" for r in metadata),
        "caveat": "Uses test correctness to select a policy; descriptive bound, not an attainable gain estimate",
    }
    partial_dir = next(RESULTS.glob("*lu-head-budget-defer-r01-native-first50-k1000"))
    partial_path = partial_dir / "1_0.csv"
    if partial_path.exists():
        sources[str(partial_path.relative_to(ROOT))] = hashlib.sha256(partial_path.read_bytes()).hexdigest()
        partial = list(csv.DictReader(partial_path.open()))
        if [identity(r) for r in partial] != expected[:len(partial)]:
            raise ValueError("Partial dual-bank run does not match annotation")
        before = metadata[:len(partial)]
        correct = [int(float(r["qa_acc"]) == 100) for r in partial]
        summary["dual_bank_broader_partial"] = {
            "n": len(partial), "videos": len({r["video_id"] for r in partial}),
            "LU_correct": sum(r["lu1000_correct"] for r in before), "dual_bank_correct": sum(correct),
            "wins": sum(c > r["lu1000_correct"] for c, r in zip(correct, before)),
            "losses": sum(c < r["lu1000_correct"] for c, r in zip(correct, before)),
            "status": "incomplete existing run; not an independent held-out validation",
        }
    for name, rows in (("case_ledger.csv", metadata),
                       ("dp_regressions_vs_lu.csv", losses),
                       ("dp_regressions_vs_hermes.csv", [r for r in metadata if r["dp_regression_vs_hermes"]]),
                       ("persistent_errors.csv", persistent)):
        with (args.output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(metadata[0]))
            writer.writeheader()
            writer.writerows(rows)
    targets = {(r["video_id"], r["question_index"]) for r in losses}
    selected_ids = {video for video, _ in targets}
    (args.output / "dp_regression_full_video_annotation.json").write_text(json.dumps(
        [v for v in annotation if str(v["video_id"]) in selected_ids], ensure_ascii=False, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("source_sha256", "by_task", "by_time", "by_question_index")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
