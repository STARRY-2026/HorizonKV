"""Publish a combined report only after all three native validations verify.

This reporting tool does not load models, use a GPU, resume experiments, or
write partial completion reports. Runtime source hashes intentionally exclude
this file; each inference run is independently re-audited before publication.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys

from sparsemm_baselines.audit_extended import AuditError, check, read_json
from sparsemm_baselines.run_streamingbench import load_resume, sha256, write_json
from sparsemm_protokv.audit import audit_run


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
METHODS = ("protokv", "adakv", "pyramidkv")
LABELS = {"protokv": "ProtoKV", "adakv": "AdaKV", "pyramidkv": "PyramidKV"}
MEMORY_FIELDS = ("persistent_kv_bytes", "bank_bytes", "raw_near_key_bytes",
                 "metadata_bytes", "persistent_total_bytes")


def read_trace(path):
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    check(events, f"Trace is empty: {path}")
    return events


def cpu_validation(path):
    check(path.is_file(), f"Final combined CPU test log is missing: {path}")
    content = path.read_text()
    matches = list(re.finditer(r"^Ran (\d+) tests in [^\n]+$", content, re.MULTILINE))
    check(matches and int(matches[-1].group(1)) == 32, "Final CPU log does not report 32 tests")
    check(content[matches[-1].end():].strip().startswith("OK"), "Final CPU suite did not pass")
    return {"passed": True, "tests": 32, "log": str(path.resolve()), "sha256": sha256(path),
            "groups": {"native": 11, "existing_protokv": 8, "audit": 6, "controller": 7}}


def pause_markers():
    paths = [WORKSPACE / "protokv/results/PAUSED", WORKSPACE / "sparsemm_baselines/results/PAUSED",
             ROOT / "results/PAUSED"]
    for path in paths:
        check(path.is_file() and "paus" in path.read_text().lower(), f"Expected pause marker is missing: {path}")
    return [{"path": str(path), "sha256": sha256(path), "content": path.read_text().strip()} for path in paths]


def first_attempt(base):
    out = base / "protokv_k1000_visual"
    check(out.is_dir(), "Preserved initial ProtoKV attempt is missing")
    status = (out / "run.status").read_text().strip()
    check(status.startswith("failed") and "Expected one native lmms result file; found 0" in status,
          "Initial attempt failure no longer matches the recorded result-file matching error")
    annotations = read_json(out / "annotations.json")
    rows, completed = load_resume(out / "1_0.csv", annotations, require_complete=True)
    checks = read_json(out / "runtime_checks.json")
    stats = read_json(out / "inference_stats.json")
    correct = sum(float(row["qa_acc"]) == 100.0 for row in rows)
    check(completed == 2 and len(rows) == 4 and correct == 4, "Initial attempt CSV is not the expected completed 4/4 inference")
    check(stats.get("status") == "complete" and stats.get("completed_videos") == 2 and stats.get("questions") == 4,
          "Initial attempt inference statistics are incomplete")
    check(checks.get("status") == "complete" and checks.get("qa_restore_checks") == 4 and checks.get("failures") == [],
          "Initial attempt runtime checks did not complete")
    return {"output_dir": str(out), "included_in_verified_comparison": False,
            "controller_status": status, "inference_videos": completed,
            "inference_questions": len(rows), "correct": correct, "accuracy_percent": 100.0,
            "reason": "Inference completed; controller failed to match a timestamp-prefixed native lmms result filename. Preserved separately."}


def _validate_native_command(manifest):
    command = manifest.get("native_command", [])
    check(isinstance(command, list), "Native CLI provenance is missing")
    check(any(command[i:i + 2] == ["-m", "lmms_eval"] for i in range(len(command) - 1)),
          "Verified run did not record the native lmms-eval CLI")
    for key, value in (("--model", "qwen2_5_vl_stream"), ("--tasks", "streamingbench_local")):
        check(key in command and command[command.index(key) + 1] == value, f"Unexpected native CLI {key}")
    return command


def collect(base, cpu_tests_log):
    """Read and validate everything before creating either completion artifact."""
    base = Path(base).resolve()
    paused = pause_markers()
    cpu = cpu_validation(Path(cpu_tests_log))
    preserved_attempt = first_attempt(base)
    runs = []
    reference_protocol = reference_annotations = reference_manifest = None
    for method in METHODS:
        out = base / f"{method}_k1000_visual_verified"
        check(out.is_dir(), f"Verified method output is missing: {out}")
        stored_audit = read_json(out / "audit_report.json")
        check(stored_audit.get("passed") is True, f"Stored method audit did not pass: {method}")
        fresh_audit = audit_run(out)
        summary = read_json(out / "summary.json")
        runtime = read_json(out / "runtime_checks.json")
        protocol = read_json(out / "protocol.json")
        manifest = read_json(out / "run_manifest.json")
        annotations = read_json(out / "annotations.json")
        check(summary.get("method") == protocol.get("method") == fresh_audit.get("method") == method,
              f"Method identity mismatch: {method}")
        check(summary.get("videos") == fresh_audit.get("videos") == 2
              and summary.get("questions") == fresh_audit.get("questions") == 4,
              f"Verified small validation must contain 2 videos/4 questions: {method}")
        check(stored_audit.get("videos") == 2 and stored_audit.get("questions") == 4
              and stored_audit.get("method") == method, f"Stored audit selection differs: {method}")
        expected_protocol = {"budget": 1000, "budget_scope": "visual", "dtype": "bfloat16",
                             "sample_fps": 0.5, "native_resolution": True,
                             "small_validation": True, "full_benchmarks_paused": True}
        for key, value in expected_protocol.items():
            check(protocol.get(key) == value, f"Unexpected common protocol {key}: {method}")
        check(summary.get("full_benchmarks_paused") is True and not summary.get("is_first250_complete"),
              f"Run incorrectly claims resumed/completed full evaluation: {method}")
        normalized_protocol = {key: value for key, value in protocol.items() if key != "method"}
        if reference_protocol is None:
            reference_protocol, reference_annotations, reference_manifest = normalized_protocol, annotations, manifest
        else:
            check(normalized_protocol == reference_protocol, f"Protocols differ beyond method: {method}")
            check(annotations == reference_annotations, f"Selected data differs: {method}")
            for key in ("code_sha256", "model_files_sha256", "dependencies", "upstream_commit", "python", "python_version"):
                check(manifest.get(key) == reference_manifest.get(key), f"Runtime provenance differs in {key}: {method}")
        native_command = _validate_native_command(manifest)
        check(runtime.get("status") == "complete" and runtime.get("qa_restore_checks") == 4
              and len(runtime.get("qa_checks", [])) == 4 and runtime.get("failures") == [],
              f"Incomplete runtime validation: {method}")
        architecture = fresh_audit["architecture"]
        check(architecture.get("num_layers") == 28 and architecture.get("num_kv_heads") == 4,
              f"Validation did not use the expected native 7B architecture: {method}")
        queries = runtime["qa_checks"]
        check(all(qa.get("returned") is True and qa.get("restored") is True
                  and qa.get("first_answer_native_layers") == 28 for qa in queries),
              f"First-answer native attention/restoration proof is incomplete: {method}")
        if method == "protokv":
            check(all(qa.get("first_answer_pseudo_layers") == 28 for qa in queries),
                  "Some first ProtoKV answers did not read every layer's prototype memory")
        check(fresh_audit["trace"]["prefills"] == 24, f"Unexpected temporal-patch prefill count: {method}")
        check(fresh_audit["lmms_results_checked"], f"Native weighted metric was not verified: {method}")
        peak = summary.get("peak_gpu_allocated_gib")
        check(isinstance(peak, (int, float)) and math.isfinite(peak) and peak > 0,
              f"Invalid diagnostic CUDA peak: {method}")
        events = read_trace(out / "cache_trace.jsonl")
        prefill = [event for event in events if event["event"] == "video_prefill"]
        memory = {key: max(event["memory"][key] for event in events) for key in MEMORY_FIELDS}
        memory["query_context_kv_bytes"] = max(qa["query_memory"]["query_context_kv_bytes"] for qa in queries)
        last_by_video = {str(event["video_id"]): event for event in prefill}
        proto_per_video = [{
            "video_id": str(video["video_id"]),
            "absorbed_tokens": last_by_video[str(video["video_id"])]["bank_total_absorbed"],
            "active_prototypes": last_by_video[str(video["video_id"])]["active_prototypes"],
            "pq_ready_groups": last_by_video[str(video["video_id"])]["pq_ready_groups"],
            "head_lengths": last_by_video[str(video["video_id"])]["head_lengths"],
        } for video in annotations]
        runs.append({
            "method": method, "output_dir": str(out), "videos": 2, "questions": 4,
            "correct": summary["correct"], "accuracy_percent": summary["accuracy_percent"],
            "prefills": fresh_audit["trace"]["prefills"],
            "compression_events": fresh_audit["trace"]["compression_events"],
            "qa_restore_checks": runtime["qa_restore_checks"],
            "finite_hidden_checks": runtime["finite_hidden_checks"],
            "finite_logits_checks": runtime["finite_logits_checks"],
            "first_answer_native_layers": [qa["first_answer_native_layers"] for qa in queries],
            "first_answer_pseudo_layers": [qa["first_answer_pseudo_layers"] for qa in queries],
            "peak_gpu_allocated_gib_diagnostic": peak,
            "max_memory_bytes": memory, "prototype_state_per_video": proto_per_video,
            "absorbed_tokens_total": sum(item["absorbed_tokens"] for item in proto_per_video),
            "stored_audit": str(out / "audit_report.json"), "fresh_audit_passed": True,
            "lmms_results_checked": fresh_audit["lmms_results_checked"],
            "native_command": native_command,
        })
    return {
        "passed": True, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "label": "paper-based streaming plugin", "scope": "bounded selected subset; not a full benchmark",
        "native_entrypoint": "python -m lmms_eval; METHOD selects the patched native Qwen2.5-VL attention/cache",
        "unique_videos": 2, "unique_questions": 4, "runs": 3, "question_runs": 12,
        "prefills_total": sum(run["prefills"] for run in runs),
        "compression_events_total": sum(run["compression_events"] for run in runs),
        "qa_restore_checks_total": sum(run["qa_restore_checks"] for run in runs),
        "protocol_except_method": reference_protocol, "annotations_sha256": reference_protocol["subset_sha256"],
        "shared_code_sha256": reference_manifest["code_sha256"],
        "shared_model_files_sha256": reference_manifest["model_files_sha256"],
        "shared_dependencies": reference_manifest.get("dependencies", {}),
        "cpu_validation": cpu, "pause_markers": paused, "full_experiments_paused": True,
        "preserved_first_attempt": preserved_attempt, "methods": runs,
        "limitations": ["Four selected questions establish integration/functionality only; they do not estimate full benchmark accuracy.",
                        "CUDA peaks include state clones for bitwise restoration auditing and are diagnostic measurements.",
                        "Prototype bank/raw-key/metadata storage is reported separately from KV context storage.",
                        "This is a paper-based streaming plugin, without a claim of official numeric or speed reproduction."],
    }


def markdown(report):
    lines = ["# SparseMM 原生插件小规模验证", "",
             "实现标记：**paper-based streaming plugin**。三种方法均通过原生 `python -m lmms_eval` 调用、逐组重新审计，使用完全一致的数据、源码和评测协议。", "",
             "统一配置：Qwen2.5-VL-7B-Instruct，BF16，0.5 FPS，原生分辨率，名义视觉预算 B=1000，系统文本另计。AdaKV 按头分配、PyramidKV 按层调度，保留原算法的整数舍入。每种方法 2 视频 / 4 题 / 24 次 temporal-patch 前向，共 12 次问答。", "",
             "| 方法 | 答对 | 准确率 | 压缩次数 | 精确恢复 | CUDA 分配峰值（诊断） |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for run in report["methods"]:
        lines.append(f"| {LABELS[run['method']]} | {run['correct']}/4 | {run['accuracy_percent']:.1f}% | {run['compression_events']} | {run['qa_restore_checks']}/4 | {run['peak_gpu_allocated_gib_diagnostic']:.3f} GiB |")
    lines += ["", "12 次首答案 prefill 均执行全部 28 层原生 decoder；ProtoKV 的 4 次首答案均在全部 28 层读取原型记忆。持久 KV、原型状态和元数据通过原引用及逐位恢复检查，hidden states / logits 的数值检查全部通过。", "",
              "下表为各项观测最大值，单位 MiB；KV、原型银行、原始近端 key 和元数据分别统计。", "",
              "| 方法 | 常驻 KV | 原型银行 | 原始近端 key | 元数据 | 常驻总计 | 查询 KV 上下文 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for run in report["methods"]:
        values = run["max_memory_bytes"]
        cells = [values[key] / 2**20 for key in (*MEMORY_FIELDS, "query_context_kv_bytes")]
        lines.append("| " + LABELS[run["method"]] + " | " + " | ".join(f"{value:.3f}" for value in cells) + " |")
    proto = next(run for run in report["methods"] if run["method"] == "protokv")
    states = proto["prototype_state_per_video"]
    pq = "、".join(f"视频 {state['video_id']}：{state['pq_ready_groups']}/112" for state in states)
    lines += ["", f"ProtoKV 累计吸收 {proto['absorbed_tokens_total']:,} 个出窗视觉 token；PQ 最终就绪：{pq}。其 1000 视觉预算分配为近端 256 + 93×8 个原型伪 token，银行在每个视频重新初始化。", "",
              "CUDA 峰值包含查询前状态克隆等审计开销，用于诊断；本报告不作吞吐或论文性能比较。4 道选定题的成绩仅验证集成与稳定性，不代表全量准确率。", "",
              f"CPU 验证：32/32 通过（原生路径 11、已有 ProtoKV 8、审计 6、控制器 7），见 [完整日志]({report['cpu_validation']['log']})。", "",
              "首轮 `protokv_k1000_visual/` 已完成 4 次问答且答对 4/4，但控制器未匹配带时间戳前缀的 lmms 结果文件而标记失败。目录完整保留，未并入以上 verified 对照。", "",
              "原 ProtoKV 实验及全部全量评测继续暂停。", "",
              "各方法独立证据：", ""]
    for run in report["methods"]:
        lines.append(f"- [{LABELS[run['method']]} 审计]({run['stored_audit']}) · [原生 lmms 结果]({run['lmms_results_checked'][0]})")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", nargs="?", type=Path, default=ROOT / "results/small_validation")
    parser.add_argument("--cpu-tests-log", type=Path, default=ROOT / "results/validation/all_cpu_tests.log")
    parser.add_argument("--check-only", action="store_true", help="Validate all gates without writing completion files")
    args = parser.parse_args(argv)
    try:
        report = collect(args.output_dir, args.cpu_tests_log)
    except (AuditError, KeyError, ValueError, TypeError, OSError) as error:
        print(json.dumps({"passed": False, "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1
    if not args.check_only:
        rendered = markdown(report)
        report_path = args.output_dir / "report.md"
        temporary = report_path.with_suffix(".md.tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(report_path)
        write_json(args.output_dir / "summary.json", report)
    print(json.dumps({"passed": True, "runs": 3, "question_runs": 12,
                      "checks": report["qa_restore_checks_total"],
                      "written": not args.check_only, "output_dir": str(args.output_dir.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
