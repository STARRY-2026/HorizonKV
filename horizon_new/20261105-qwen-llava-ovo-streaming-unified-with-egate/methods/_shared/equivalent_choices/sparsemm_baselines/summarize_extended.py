"""Combine the six completed real-model stability runs without using the GPU.

No combined report is written until every saved audit, a fresh trace audit, and
the required instrumentation reports pass. Accuracy describes this selected
stability subset; instrumented memory is not a performance benchmark.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from sparsemm_baselines.audit_extended import CONFIG_KEYS, audit_run, read_json
from sparsemm_baselines.run_streamingbench import prepare_annotations, sha256, write_json


ROOT = Path(__file__).resolve().parent
CASES = (
    ("k1000_prompt", 1000, "prompt", "annotations.json", 9, 23, True),
    ("k64_prompt", 64, "prompt", "edge_annotations.json", 1, 3, False),
    ("k256_visual", 256, "visual", "edge_annotations.json", 1, 3, True),
)
METHODS = ("adakv", "pyramidkv")


class CombinedReportError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise CombinedReportError(message)


def positive_integer(value, label):
    require(isinstance(value, int) and not isinstance(value, bool) and value > 0,
            f"{label} must be a positive integer")
    return value


def without_method(protocol):
    return {key: value for key, value in protocol.items() if key != "method"}


def check_runtime(report, protocol, manifest, annotations, trace):
    label = f"{protocol['method']}/k{protocol['budget']}_{protocol['budget_scope']}"
    expected = [(str(video["video_id"]), index)
                for video in annotations for index in range(len(video["conversations"]))]
    require(report["status"] == "complete", f"{label}: instrumentation is incomplete")
    require(report["failures"] == [] and report["qa_exceptions"] == 0,
            f"{label}: instrumentation recorded a failure or exception")
    require(report["qa_restore_checks"] == report["qa_successes"] == len(expected),
            f"{label}: instrumentation question counts disagree")
    require(report["runtime_code_sha256"] == manifest["code_sha256"],
            f"{label}: instrumentation/runtime source hashes differ")
    require(report["harness_sha256"] == sha256(ROOT / "run_instrumented.py"),
            f"{label}: instrumentation source hash changed")
    require(report["config"] == {key: protocol[key] for key in CONFIG_KEYS},
            f"{label}: instrumentation configuration differs")
    checks = report["qa_checks"]
    require([(str(check["video_id"]), check["question_index"]) for check in checks] == expected,
            f"{label}: instrumentation question order differs")
    require(all(check["returned"] is True and check["restored"] is True for check in checks),
            f"{label}: a question did not return with its exact cache restored")
    norm = positive_integer(report["finite_norm_checks"], f"{label} norm checks")
    logits = positive_integer(report["finite_lm_head_checks"], f"{label} logits checks")
    require(norm >= trace["prefills"] + len(expected) and logits >= len(expected),
            f"{label}: insufficient finite-value checks")
    return {"enabled": True, "finite_hidden_checks": norm,
            "finite_logits_checks": logits, "bitwise_qa_restore_checks": len(expected),
            "finite_hidden_scope": "decoder final norm output on every forward call",
            "finite_logits_scope": "QA lm_head output on every generation step",
            "qa_exceptions": 0, "failures": [],
            "harness_sha256": report["harness_sha256"],
            "peak_checked_cache_bytes": report["peak_cache_bytes"]}


def build_report(base):
    """Read and verify every artifact; this function writes no files."""
    base = Path(base).resolve()
    plan = read_json(base / "test_plan.json")
    subset = read_json(base / "subset_manifest.json")
    raw_main = read_json(base / "annotations.json")
    raw_edge = read_json(base / "edge_annotations.json")
    tasks = Counter(question["task"] for video in raw_main for question in video["conversations"])
    maximum = max(float(question["end_time"]) for video in raw_main for question in video["conversations"])
    require(len(raw_main) == 9 and sum(tasks.values()) == 23 and len(tasks) == 10 and maximum == 502,
            "Main subset must contain 9 videos, 23 questions, 10 tasks, and a 502-second endpoint")
    require(len(raw_edge) == 1 and len(raw_edge[0]["conversations"]) == 3
            and [float(q["end_time"]) for q in raw_edge[0]["conversations"]] == [2, 80, 80],
            "Edge subset must contain the three original questions at 2/80/80 seconds")
    require(plan["expected_question_runs_per_method"] == 29
            and plan["expected_question_runs_both_methods"] == 58,
            "Test plan question totals disagree")
    require(sha256(base / "annotations.json") == plan["main"]["annotations_sha256"]
            == subset["annotations_sha256"], "Main source annotation hash changed")
    require(dict(tasks) == plan["main"]["tasks"] == subset["task_counts"],
            "Main task counts disagree with the recorded selection")
    require(subset["video_count"] == 9 and subset["question_count"] == 23
            and subset["max_selected_end_time_seconds"] == maximum,
            "Subset manifest coverage differs from actual annotations")
    require(sha256(subset["source_annotations"]) == subset["source_annotations_sha256"],
            "Original dataset annotation hash changed")
    pause_markers = (ROOT / "results/PAUSED", ROOT.parent / "protokv/results/PAUSED")
    require(all(path.is_file() for path in pause_markers),
            "Full benchmark/ProtoKV pause markers must remain present")

    runs, pair_protocols, pair_annotations = [], {}, {}
    shared_hashes = shared_environment = shared_settings = None
    for suffix, budget, scope, annotation_file, videos, questions, instrumented in CASES:
        expected_annotations = prepare_annotations(base / annotation_file, max_videos=videos)
        for method in METHODS:
            out = base / f"{method}_{suffix}"
            summary = read_json(out / "summary.json")
            saved_audit = read_json(out / "audited_report.json")
            protocol = read_json(out / "protocol.json")
            manifest = read_json(out / "run_manifest.json")
            annotations = read_json(out / "annotations.json")
            require(saved_audit.get("passed") is True, f"{out.name}: saved audit did not pass")
            fresh = audit_run(out, min_compression_events_per_video=2)
            for key in ("method", "budget_scope", "configured_budget", "videos", "questions",
                        "overall", "task_accuracy", "architecture", "trace", "per_video"):
                require(saved_audit[key] == fresh[key], f"{out.name}: saved audit is stale at {key}")
            require(protocol["method"] == method and protocol["budget"] == budget
                    and protocol["budget_scope"] == scope, f"{out.name}: wrong method/budget/scope")
            require(summary["videos"] == videos and summary["questions"] == questions
                    and fresh["videos"] == videos and fresh["questions"] == questions,
                    f"{out.name}: unexpected video/question totals")
            require(summary.get("is_first250_complete") is False,
                    f"{out.name}: subset incorrectly labeled a full benchmark")
            require(annotations == expected_annotations, f"{out.name}: selected dataset differs")
            require(protocol["source_sha256"] == sha256(base / annotation_file),
                    f"{out.name}: original subset source differs")
            pair = without_method(protocol)
            if suffix in pair_protocols:
                require(pair_protocols[suffix] == pair and pair_annotations[suffix] == annotations,
                        f"{suffix}: the two methods used different settings or annotations")
            pair_protocols[suffix], pair_annotations[suffix] = pair, annotations

            hashes = manifest["code_sha256"]
            require(hashes, f"{out.name}: missing runtime source hashes")
            for filename, digest in plan["runtime_source_sha256"].items():
                require(hashes.get(filename) == digest, f"{out.name}: runtime source differs from plan: {filename}")
            require(all(sha256(ROOT / filename) == digest for filename, digest in hashes.items()),
                    f"{out.name}: runtime source changed since execution")
            environment = {key: manifest[key] for key in
                           ("python", "python_version", "dependencies", "upstream_commit",
                            "evaluator_sha256", "model_files_sha256", "gpu")}
            settings = {key: value for key, value in pair.items()
                        if key not in ("budget", "budget_scope", "source_sha256", "subset_sha256", "initial_text")}
            if shared_hashes is not None:
                require(hashes == shared_hashes, "The six runs used different runtime sources")
                require(environment == shared_environment, "The six runs used different environments or models")
                require(settings == shared_settings, "The six runs differ beyond their planned budget/scope")
            shared_hashes, shared_environment, shared_settings = hashes, environment, settings

            if instrumented:
                runtime = check_runtime(read_json(out / "runtime_checks.json"), protocol, manifest,
                                        annotations, fresh["trace"])
            else:
                require(not (out / "runtime_checks.json").exists(),
                        f"{out.name}: k64 is designated uninstrumented but a runtime report exists")
                runtime = {"enabled": False, "finite_hidden_checks": None,
                           "finite_logits_checks": None, "bitwise_qa_restore_checks": None,
                           "note": "Only runner safeguards and trace audit; no additional hidden/logits or bitwise QA instrumentation."}
            peak = summary["peak_gpu_allocated_gib"]
            require(isinstance(peak, (float, int)) and math.isfinite(peak) and peak > 0,
                    f"{out.name}: invalid peak allocation")
            runs.append({"method": method, "case": suffix, "budget": budget, "budget_scope": scope,
                         "videos": videos, "questions": questions,
                         "correct": fresh["overall"]["correct"],
                         "accuracy_percent": fresh["overall"]["accuracy_percent"],
                         "peak_gpu_allocated_gib": peak, "trace_audit_passed": True,
                         "compression_events": fresh["trace"]["compression_events"],
                         "compressed_layer_events": fresh["trace"]["compressed_layer_events"],
                         "prefills": fresh["trace"]["prefills"],
                         "repeated_answers_without_prefill": fresh["trace"]["repeated_answers_without_prefill"],
                         "zero_mass_head_events": fresh["trace"]["zero_mass_head_events"],
                         "task_accuracy": {task: {key: values[key] for key in
                                            ("questions", "correct", "accuracy_percent")}
                                           for task, values in fresh["task_accuracy"].items()},
                         "instrumentation": runtime,
                         "artifacts": {name: str(out / name) for name in
                                       ("summary.json", "audited_report.json", "run_manifest.json")}})
    total = sum(run["questions"] for run in runs)
    require(total == 58, "Combined question runs must total 58")
    return {"passed": True, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "bounded selected stability validation; not a full benchmark",
            "methods": list(METHODS), "run_count": len(runs), "question_runs": total,
            "question_runs_per_method": {method: sum(run["questions"] for run in runs if run["method"] == method)
                                         for method in METHODS},
            "main_coverage": {"videos": 9, "questions_per_method": 23, "task_count": len(tasks),
                              "task_counts": dict(sorted(tasks.items())), "max_end_time_seconds": maximum,
                              "raw_sampled_frames_per_method": subset["total_raw_sampled_frames_encoded_per_method"]},
            "runtime_source_sha256": shared_hashes, "shared_environment": shared_environment,
            "source_annotations_sha256": subset["source_annotations_sha256"],
            "instrumentation_totals": {
                "finite_hidden_checks": sum(run["instrumentation"]["finite_hidden_checks"] or 0 for run in runs),
                "finite_logits_checks": sum(run["instrumentation"]["finite_logits_checks"] or 0 for run in runs),
                "bitwise_qa_restore_checks": sum(run["instrumentation"]["bitwise_qa_restore_checks"] or 0 for run in runs),
                "instrumented_question_runs": 52, "uninstrumented_k64_question_runs": 6},
            "compression_events": sum(run["compression_events"] for run in runs),
            "runs": runs, "full_benchmarks_paused": True, "protokv_paused": True,
            "pause_markers": [str(path) for path in pause_markers],
            "limitations": ["Accuracy describes a deliberately selected stability subset, not representative benchmark accuracy.",
                            "The k64 cases were not additionally instrumented; their trace audits do not prove bitwise cache equality.",
                            "Instrumented cases clone cache tensors for equality checks, adding memory/work; peak allocations are diagnostic, not a performance benchmark.",
                            "Finite-value and restoration checks establish observed execution stability, not correctness of every attention ranking."]}


def render_markdown(report):
    lines = ["# SparseMM 扩展验证", "",
             "AdaKV 和 PyramidKV 的 6 组真实 Qwen2.5-VL-7B 测试均完成并通过审计，共运行 58 次问答（每种方法 29 次）。",
             "", "主测试每种方法使用相同的 9 个视频、23 道题，覆盖 10 类任务，最长视频前缀 502 秒；另各测试 64-token prompt 预算和 256-token visual 预算各 3 道题。",
             "", "| 方法 | 预算 / 范围 | 视频 / 问答 | 答对 | 正确率 | 压缩次数 | 峰值分配 GiB | 逐位缓存恢复 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for run in report["runs"]:
        checks = run["instrumentation"]["bitwise_qa_restore_checks"]
        lines.append(f"| {run['method']} | {run['budget']} / {run['budget_scope']} | "
                     f"{run['videos']} / {run['questions']} | {run['correct']}/{run['questions']} | "
                     f"{run['accuracy_percent']:.2f}% | {run['compression_events']} | "
                     f"{run['peak_gpu_allocated_gib']:.3f} | {checks if checks is not None else '未额外检查'} |")
    checks = report["instrumentation_totals"]
    lines += ["", f"主测试和 visual 预算测试共通过 {checks['finite_hidden_checks']} 次 decoder 最终 norm 输出、"
              f"{checks['finite_logits_checks']} 次问答生成 logits 有限值检查，以及 {checks['bitwise_qa_restore_checks']} 次问答前后"
              "缓存引用、K/V、位置、长度和游标恢复检查，未出现记录的异常。64-token 两组通过运行器检查和压缩轨迹审计，未额外检查最终隐藏状态、logits 或逐位缓存一致性。",
              "", "六组使用相同版本的运行源码、模型和依赖；每个方法对照组的数据及配置一致。审计检查真实压缩、容量分配、物理 KV 字节、视频重置、连续问答和重复时间戳。",
              "", "正确率仅描述人为选择的稳定性子集。额外检查会克隆缓存并增加显存及执行开销，表中峰值用于诊断，不能当作性能基准。",
              "", "ProtoKV 和全量评测仍保持暂停。", "", "完整结构化结果：[combined_report.json](combined_report.json)。", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results/extended_validation")
    parser.add_argument("--check-only", action="store_true", help="Validate all inputs without writing either report")
    args = parser.parse_args(argv)
    try:
        report = build_report(args.results_dir)
    except (CombinedReportError, RuntimeError, KeyError, ValueError, TypeError, OSError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, ensure_ascii=False))
        return 1
    if not args.check_only:
        destination = args.results_dir.resolve()
        write_json(destination / "combined_report.json", report)
        markdown = destination / "combined_report.md"
        temporary = markdown.with_suffix(".md.tmp")
        temporary.write_text(render_markdown(report), encoding="utf-8")
        temporary.replace(markdown)
    print(json.dumps({"passed": True, "run_count": report["run_count"],
                      "question_runs": report["question_runs"], "check_only": args.check_only,
                      "results_dir": str(args.results_dir.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
