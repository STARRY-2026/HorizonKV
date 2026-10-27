"""Small paired official OVO test of coupled versus independent history/recent KV.

Default: 24 a-priori short queries, two per task, chosen without looking at
answers. Both arms share one loaded model and run sequentially on an exclusive
GPU, after warmup. This entry point never starts a full-dataset experiment.
"""

import argparse
import collections
import copy
import csv
import gc
import io
import json
import math
import os
from pathlib import Path
import signal
import time

from sparsemm_native import run_ovobench as official
from sparsemm_native import run_ovobench_recent4 as common
from sparsemm_native.supervise_ovobench_dual import now, write_json, atomic_write


ROOT = official.ROOT
ARMS = ("dp_recent4", "dp_decoupled")
REFERENCE = ROOT / "results/ovobench_recent4_official_full_k500_fps05_p151200_20261021/dp_recent4"
SOURCE_FILES = common.SOURCE_FILES + (
    "sparsemm_protokv/qwen2_5.py", "sparsemm_native/decoupled.py",
    "sparsemm_native/run_ovobench_decoupled.py")


def pilot_queries(queries):
    """Two per task, sampled prefixes 8..32 frames; favor shared video QA."""
    selected = []
    for task in sorted({q["task"] for q in queries}):
        candidates = [q for q in queries if q["task"] == task
                      and 8 <= math.ceil(q["clip_end"] * 0.5) <= 32]
        videos = collections.defaultdict(list)
        for query in candidates:
            videos[query["video_path"]].append(query)
        pairs = [group for group in videos.values()
                 if len(group) > 1 and len({q["clip_end"] for q in group}) > 1]
        if pairs:
            group = min(pairs, key=lambda g: (min(q["clip_end"] for q in g), g[0]["query_id"]))
            chosen = sorted(group, key=lambda q: q["clip_end"])[:2]
        else:
            chosen = sorted(candidates, key=lambda q: (q["clip_end"], q["query_id"]))[:2]
        if len(chosen) != 2:
            raise ValueError(f"Not enough predeclared short queries for {task}")
        selected.extend(chosen)
    ids = {q["query_id"] for q in selected}
    return common.schedule_queries([q for q in queries if q["query_id"] in ids])


def timing_snapshot(bridge):
    return {key: value for key, value in bridge.timings.items() if isinstance(value, (int, float))}


def run_group(bridge, group, arm):
    """Same official video sampling and even/odd transactions as the old run."""
    import torch
    from sparsemm_baselines.run_streamingbench import sampled_indices
    from sparsemm_protokv.models.qwen2_5_vl_stream import BatchScopedVideoReader

    reader = BatchScopedVideoReader(group[0]["video_path"], num_threads=1)
    indices = sampled_indices(len(reader), reader.get_avg_fps(), 0.5)
    targets = [min(len(indices), math.ceil(q["clip_end"] * .5)) for q in group]
    if min(targets) < 1:
        raise ValueError("No sampled frames in an official query")
    bridge._active_video_id = group[0]["query_id"]
    bridge.encode_init_prompt()
    base_frames, answers = 0, {}

    def encode(start, stop):
        for offset in range(start, stop, 16):
            end = min(offset + 16, stop)
            with bridge._measure("video_decode_seconds"):
                frames = torch.from_numpy(reader.get_batch(indices[offset:end]).asnumpy())
            bridge.encode_video_chunk(frames, is_last_chunk=end == stop)
            bridge.predict_and_compress()

    def answer(query):
        length = len(bridge.conv_history)
        try:
            text = str(bridge.question_answering(
                {"question": query["prompt"], "formatted_question": query["prompt"],
                 "prompt": bridge.get_prompt(query["prompt"], mc=False)}, max_new_tokens=16))
            if arm == "dp_recent4":
                common.validate_runtime(bridge, encoded_frames=True)
                cache = bridge.kv_cache
                actual = [[int(n) - cache.prefix_tokens for n in layer.head_lens.tolist()]
                          for layer in cache.layers]
                audit = dict(allocation=dict(cache.allocation), actual_visual_counts=actual,
                             actual_visual_slots=sum(map(sum, actual)), target_visual_slots=56000,
                             recent_tokens_per_head=int(cache.recent_budgets[0, 0]),
                             recent_rebuild_seconds=0.0, fusion_seconds=0.0, recent_reused=False)
                r = audit["recent_tokens_per_head"]
                audit["recent_positions_sha256"] = common.digest(
                    cache.layers[0].get_position_segment(0)[:, -r:].cpu().tolist())
            else:
                checks = bridge.validation_checks
                if (len(checks["qa_checks"]) != 1 or checks["failures"]
                        or checks["qa_checks"][0].get("returned") is not True
                        or checks["qa_checks"][0].get("restored") is not True):
                    raise AssertionError("Independent branch QA did not restore state")
                checks["qa_checks"].clear()
                checks["selector_checks"].clear()
                audit = copy.deepcopy(bridge.last_question_metrics)
                audit["recent_positions_sha256"] = common.digest(torch.cat(
                    [unit.positions[:, 0] for unit in bridge.kv_cache.units], dim=1).cpu().tolist())
            audit["total_seen_visual_tokens"] = bridge._seen_visual_tokens
            return text, audit
        finally:
            del bridge.conv_history[length:]

    for index in sorted(range(len(group)), key=lambda i: (targets[i], i)):
        query, target = group[index], targets[index]
        bridge._active_question_index = index
        bridge._active_question = query["prompt"]
        bridge._active_question_end_time = query["clip_end"]
        before = timing_snapshot(bridge)
        torch.cuda.synchronize()
        started = time.perf_counter()
        even = target - target % 2
        if even < base_frames:
            raise AssertionError("A source video cursor moved backwards")
        encode(base_frames, even)
        encoded_frames = even - base_frames
        base_frames = even
        if target % 2:
            parent = bridge.kv_cache
            fields = ("_next_position", "_seen_visual_tokens", "total_processed_frames", "last_encoded_frames",
                      "_prefills", "_compressions", "last_metadata")
            saved = {field: getattr(bridge, field) for field in fields}
            bridge.kv_cache = copy.deepcopy(parent)
            try:
                encode(even, target)
                encoded_frames += 1
                response, audit = answer(query)
            finally:
                bridge.kv_cache = parent
                for field, value in saved.items():
                    setattr(bridge, field, value)
        else:
            response, audit = answer(query)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        after = timing_snapshot(bridge)
        row = {key: query[key] for key in common.IDENTITY}
        row.update(arm=arm, response=response,
                   correct=official.official_correct(query["task"], response, query["target"]),
                   frames=target, encoded_frames=encoded_frames, seconds=elapsed,
                   prompt_sha256=common.digest(query["prompt"]),
                   timings={key: after[key] - before.get(key, 0) for key in after},
                   cache_audit=audit, execution_mode="grouped_even_prefix_isolated_odd_branch")
        row["sampled_source_indices_sha256"] = common.digest(indices[:target])
        common.validate_row(row, query)
        answers[query["query_id"]] = row
    return [answers[query["query_id"]] for query in group]


def render_report(root, rows_by_arm, phase_stats, reference, validation):
    current, independent = rows_by_arm[ARMS[0]], rows_by_arm[ARMS[1]]
    indexed = {arm: {row["query_id"]: row for row in rows} for arm, rows in rows_by_arm.items()}
    n = len(current)
    task_counts = collections.Counter(row["task"] for row in current)
    correct = {arm: sum(row["correct"] for row in rows) for arm, rows in rows_by_arm.items()}
    guards = sum(bool(row["cache_audit"]["allocation"].get("global_capacity_guard")) for row in independent)
    agreement = sum(reference[row["query_id"]]["response"] == row["response"] for row in current)
    paired = [dict(query_id=row["query_id"], task=row["task"], target=row["target"],
                   old=row["response"], old_correct=row["correct"],
                   decoupled=indexed[ARMS[1]][row["query_id"]]["response"],
                   decoupled_correct=indexed[ARMS[1]][row["query_id"]]["correct"])
              for row in current]
    for row in current:
        counterpart = indexed[ARMS[1]][row["query_id"]]
        old, new = row["cache_audit"], counterpart["cache_audit"]
        for field in ("head_budgets", "recent_budgets", "history_budgets"):
            if old["allocation"][field] != new["allocation"][field]:
                raise AssertionError("Paired nominal DP quotas differ")
        actual_new = [[h + new["recent_tokens_per_head"] for h in layer]
                      for layer in new["actual_history_counts"]]
        if (old["actual_visual_counts"] != actual_new
                or row["sampled_source_indices_sha256"] != counterpart["sampled_source_indices_sha256"]
                or old["recent_positions_sha256"] != new["recent_positions_sha256"]
                or old["total_seen_visual_tokens"] != new["total_seen_visual_tokens"]):
            raise AssertionError("Paired actual budgets, frames or recent positions differ")
    validation["paired_actual_per_head_occupancies_identical"] = True
    validation["paired_nominal_DP_quotas_identical"] = True
    validation["paired_sampled_indices_recent_positions_and_visual_counts_identical"] = True
    report = dict(completed_at=now(), queries=n, task_counts=dict(task_counts),
                  correct=correct, accuracy={arm: correct[arm] / n for arm in ARMS},
                  phase_stats=phase_stats, validation=validation,
                  global_capacity_guard_query_count=guards,
                  original_full_DP_response_agreement=dict(agree=agreement, total=n),
                  paired=paired, complete_small_sample=True, full_dataset_run=False,
                  note="A-priori short balanced pilot; not a full OVO score or a statistically conclusive accuracy estimate. "
                  "Speed compares sequential exclusive-GPU arms after warmup; selector_seconds is a subset "
                  "of language_video_seconds. History scoring uses its own outgoing-unit queries to enforce independence.")
    write_json(root / "comparison.json", report)
    text = (f"更新时间：{now()}。{n} 题、12 类，每类 2 题；预先按视频长度和复用条件选题，不按答案筛选。\n\n"
            "同一 Qwen2.5-VL-7B、BF16、K500、FPS0.5、视频像素上限151200、视频两帧时间单元、"
            "流式绝对位置、DP历史权重、protect32、官方提示与评分、贪心生成16token、重复惩罚1.1。\n\n"
            "独占同一GPU，模型只加载一次，两方法分别预热后顺序测速。历史分支使用自身查询打分，"
            "近期从当前两个视频单元特征重建；问题读取一份前缀加两路KV，作答后恢复状态。\n\n"
            "| 题型 | 题数 | 原耦合 DP | 解耦 DP |\n|---|---:|---:|---:|\n")
    for task in sorted(task_counts):
        a = sum(row["correct"] for row in current if row["task"] == task)
        b = sum(row["correct"] for row in independent if row["task"] == task)
        text += f"| {task} | {task_counts[task]} | {a} | {b} |\n"
    text += (f"| 合计 | {n} | {correct[ARMS[0]]}/{n}（{100*correct[ARMS[0]]/n:.2f}%） | "
             f"{correct[ARMS[1]]}/{n}（{100*correct[ARMS[1]]/n:.2f}%） |\n\n"
             "| 计时/显存 | 原耦合 DP | 解耦 DP |\n|---|---:|---:|\n")
    for label, field in (("样本总耗时（秒，含解码）", "steady_seconds"),
                         ("视觉编码（秒）", "vision_seconds"),
                         ("视频/历史语言前向（秒，含筛选）", "language_video_seconds"),
                         ("近期重建累计（秒）", "recent_rebuild_seconds"),
                         ("KV拼接累计（秒）", "fusion_seconds"),
                         ("问题及回答累计（秒）", "qa_seconds"),
                         ("GPU峰值已分配（MiB）", "peak_allocated_mib")):
        text += f"| {label} | {phase_stats[ARMS[0]].get(field,0):.3f} | {phase_stats[ARMS[1]].get(field,0):.3f} |\n"
    rebuilds = [r["cache_audit"]["recent_rebuild_seconds"] for r in independent
                if not r["cache_audit"]["recent_reused"]]
    text += (f"\n近期重建平均 {1000*sum(rebuilds)/len(rebuilds):.1f} ms/新窗口。"
             f"原耦合方法与此前全量结果回答一致：{agreement}/{n}。\n\n"
             "这是短视频小样本验证，不能据此确定全量准确率提升或推算整场长视频吞吐。"
             "实际逐头容量、位置、历史/近期区间、辅助特征与KV显存均保存于逐题 cache_audit。\n")
    atomic_write(root / "report_zh.md", text)
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(paired[0]))
    writer.writeheader()
    writer.writerows(paired)
    atomic_write(root / "paired.csv", output.getvalue())
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ids", help="Optional fixed small sample; at most 100 official queries")
    args = parser.parse_args(argv)
    root = args.output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    official_queries = official.queries()
    selected = (common.schedule_queries(common.select_queries(official_queries, args.ids))
                if args.ids else pilot_queries(official_queries))
    if not 1 <= len(selected) <= 100:
        raise ValueError("This entry point is a small-sample test, not a full run")
    source_hashes = {name: official.sha256(ROOT / name) for name in SOURCE_FILES}
    reference_manifest = json.loads((REFERENCE / "query_manifest.json").read_text())
    reference_protocol = json.loads((REFERENCE / "protocol.json").read_text())
    reference_rows = common.read_progress(REFERENCE / "queries.jsonl", reference_manifest)
    reference = {row["query_id"]: row for row in reference_rows}
    official_indexed = {q["query_id"]: q for q in official_queries}
    if {q["query_id"]: q for q in reference_manifest} != official_indexed:
        raise ValueError("Official reference query/prompt manifest changed")
    for query in selected:
        common.validate_row(reference[query["query_id"]], query)
    if (reference_protocol["fps"] != .5 or reference_protocol["budget"] != 500
            or reference_protocol["video_max_pixels"] != 151200
            or Path(reference_protocol["model_path"]).resolve() != official.MODEL.resolve()):
        raise ValueError("Previous full DP model/budget/input settings changed")
    for name, checksum in reference_protocol["processor_files_sha256"].items():
        if official.sha256(official.MODEL / name) != checksum:
            raise ValueError("Model/processor assets differ from the official reference")
    write_json(root / "protocol.json", dict(
        experiment="independent_history_recent_KV_paired_pilot", budget=500, fps=.5,
        model_path=str(official.MODEL.resolve()), dtype="bfloat16", video_max_pixels=151200,
        max_new_tokens=16, repetition_penalty=1.1, temperature=0, protect_window=32,
        observation_window=32, pooling="avgpool", kernel_size=5, gqa_reduce="mean",
        score_precision="float32", official_prompt_scoring=True, video_temporal_patch_size=2,
        same_video_features_positions_and_DP_allocation=True,
        recent_window="last_two_video_units_with_identical_odd_padding",
        history_queries="own_outgoing_unit_queries_not_recent_branch_queries",
        short_cache_gate="legacy_global", actual_global_visual_limit=56000,
        overflow_guard="independent_branch_disables_legacy_warmup_only_if_it_would_exceed_global_K500",
        history_weight_sha256=official.sha256(official.DP_DUAL_BUDGET),
        source_annotation_sha256=official.sha256(official.ANNOTATIONS),
        processor_files_sha256=reference_protocol["processor_files_sha256"],
        code_sha256=source_hashes, query_count=len(selected), query_list_sha256=common.digest(selected),
        selection="two_per_task_8_to_32_prefix_frames_predeclared_without_answers" if not args.ids else "explicit_ids",
        timing="one_model_exclusive_gpu_sequential_arms_after_warmup", full_dataset_run=False))
    write_json(root / "query_manifest.json", selected)
    for name in SOURCE_FILES:
        target = root / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    status = dict(state="starting", started_at=now(), pid=os.getpid(), queries_per_arm=len(selected))
    write_json(root / "status.json", status)
    try:
        import torch
        from sparsemm_native.decoupled import DecoupledRecent4StreamingQwen
        from sparsemm_native.recent4 import Recent4StreamingQwen
        from sparsemm_native.gpu_slots import gpu_lease
        from sparsemm_native.assets import asset_metadata
        from sparsemm_protokv.model_loader import load_pretrained
        from sparsemm_protokv.config import StreamingConfig
        config = StreamingConfig(method="sparsemm", budget=500, budget_scope="visual",
            sparsemm_budget_path=str(official.DP_DUAL_BUDGET), sparsemm_cache_policy="two_bank",
            sparsemm_two_bank_gate="legacy_global")
        results, stats, bridges = {}, {}, {}
        with gpu_lease(root):
            model, processor = load_pretrained(official.MODEL, dtype="bfloat16")
            for arm, cls in zip(ARMS, (Recent4StreamingQwen, DecoupledRecent4StreamingQwen)):
                bridges[arm] = cls(model, processor, config, sample_fps=.5, validate_state=True,
                                   video_max_pixels=151200, asset_metadata=asset_metadata())
                bridges[arm].timings.setdefault("video_decode_seconds", 0.0)
                bridges[arm].timing_enabled = True
            # Warm both computation paths before measuring either arm.
            for arm, bridge in bridges.items():
                status.update(state="warmup", arm=arm, updated_at=now())
                write_json(root / "status.json", status)
                run_group(bridge, selected[:1], arm)
                bridge.clear_cache()
            for arm, bridge in bridges.items():
                other = bridges[ARMS[1] if arm == ARMS[0] else ARMS[0]]
                other.clear_cache()
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                status.update(state="running", arm=arm, completed_queries=0, updated_at=now())
                write_json(root / "status.json", status)
                before = timing_snapshot(bridge)
                started = time.perf_counter()
                rows = []
                directory = root / arm
                directory.mkdir()
                for group in official.grouped_batches(selected, len(selected)):
                    batch = run_group(bridge, group, arm)
                    for row in batch:
                        if row["frames"] != reference[row["query_id"]]["frames"]:
                            raise AssertionError("New test used a different source prefix")
                    with (directory / "queries.jsonl").open("a") as stream:
                        for row in batch:
                            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    rows.extend(batch)
                    status.update(completed_queries=len(rows), updated_at=now())
                    write_json(root / "status.json", status)
                    print(f"{arm}: {len(rows)}/{len(selected)} correct={sum(r['correct'] for r in rows)}", flush=True)
                torch.cuda.synchronize()
                stats[arm] = dict(steady_seconds=time.perf_counter() - started,
                    peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                    peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                    **{key: value - before.get(key, 0) for key, value in timing_snapshot(bridge).items()})
                results[arm] = rows
                bridge.clear_cache()
            # The same inherited branch transactions must give identical
            # independent answers when each selected prefix is evaluated fresh.
            validation_ids = [q["query_id"] for q in selected if q["task"] in ("EPM", "CRR")]
            validation = dict(fresh_grouped_checked_ids=validation_ids, fresh_grouped_agreement=True)
            bridge = bridges["dp_decoupled"]
            grouped = {row["query_id"]: row for row in results["dp_decoupled"]}
            for query in selected:
                if query["query_id"] not in validation_ids:
                    continue
                row = run_group(bridge, [query], "dp_decoupled")[0]
                if row["response"] != grouped[query["query_id"]]["response"]:
                    raise AssertionError("Fresh and shared-video independent results disagree")
            # With only one/two video units, there is no historical content:
            # these two arms should reduce to the same visual context.
            null_queries = [q for q in official_queries if q["query_id"] in ("34", "1054")]
            validation["no_history_response_agreement"] = []
            for query in null_queries:
                left = run_group(bridges["dp_recent4"], [query], "dp_recent4")[0]
                right = run_group(bridge, [query], "dp_decoupled")[0]
                if left["response"] != right["response"]:
                    raise AssertionError("No-history sanity answers differ")
                validation["no_history_response_agreement"].append(query["query_id"])
            # A same-cutoff second question can reuse recent KV, but forcing
            # another fresh rebuild must produce exactly the same answer.
            reuse_query = next(q for q in selected if math.ceil(q["clip_end"]*.5) % 2 == 0)
            first = run_group(bridge, [reuse_query], "dp_decoupled")[0]
            prompt = {"question": reuse_query["prompt"], "formatted_question": reuse_query["prompt"],
                      "prompt": bridge.get_prompt(reuse_query["prompt"], mc=False)}
            try:
                reused_answer = bridge.question_answering(prompt, max_new_tokens=16)
                if not bridge.last_question_metrics["recent_reused"] or reused_answer != first["response"]:
                    raise AssertionError("Same-window recent KV reuse changed an answer")
                bridge.kv_cache.recent_dense = None
                rebuilt_answer = bridge.question_answering(prompt, max_new_tokens=16)
                if bridge.last_question_metrics["recent_reused"] or rebuilt_answer != first["response"]:
                    raise AssertionError("Forced same-window rebuild changed an answer")
            finally:
                bridge.conv_history.clear()
                bridge.validation_checks["qa_checks"].clear()
            validation["same_window_reuse_and_forced_rebuild_agree"] = reuse_query["query_id"]
            if any(official.sha256(ROOT / name) != checksum for name, checksum in source_hashes.items()):
                raise AssertionError("Source changed during the measured run")
            render_report(root, results, stats, reference, validation)
        status.update(state="complete", updated_at=now(), arms_complete=list(ARMS))
        write_json(root / "status.json", status)
    except BaseException as error:
        status.update(state="failed", updated_at=now(), error=f"{type(error).__name__}: {error}")
        write_json(root / "status.json", status)
        raise


if __name__ == "__main__":
    main()
