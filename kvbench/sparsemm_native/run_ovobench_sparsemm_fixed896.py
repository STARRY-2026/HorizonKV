"""Native SparseMM OVO nine-task evaluation with an exact fixed RGB canvas.

Only spatial pixels change: the existing sampler, two-real-frame temporal
patches, native positions, visual attention eviction, prompts and QA decoder
are retained. Source-video even prefixes are shared; odd endpoints branch.
"""

import argparse
import collections
import copy
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo

from kvbench.fixed_inputs import fixed_input_metadata
from sparsemm_native import run_ovobench as official
from sparsemm_native.run_ovobench_recent4 import digest, validate_row
from sparsemm_native.run_ovobench_sparsemm_fps1 import selected_queries
from sparsemm_native.supervise_ovobench_dual import atomic_write, write_json


ROOT = official.ROOT
SIZE, FPS, BUDGET, CHUNK, MAX_NEW = 896, .5, 500, 16, 16
TASKS = official.BACKWARD | official.REALTIME
VERSION = "native_sparsemm_fixed896_fps05_k500_v1"
SOURCE_NAMES = (
    "sparsemm_native/run_ovobench_sparsemm_fixed896.py", "kvbench/fixed_inputs.py",
    "sparsemm_native/run_ovobench.py", "sparsemm_native/run_ovobench_recent4.py",
    "sparsemm_native/run_ovobench_sparsemm_fps1.py", "sparsemm_native/video_reader.py",
    "sparsemm_native/gpu_slots.py", "sparsemm_native/supervise_ovobench_dual.py",
    "sparsemm_native/streaming.py", "sparsemm_native/cache.py", "sparsemm_native/policy.py",
    "sparsemm_native/two_bank.py", "sparsemm_native/assets.py", "sparsemm_protokv/config.py",
    "sparsemm_protokv/cache.py", "sparsemm_protokv/streaming.py", "sparsemm_protokv/monkeypatch.py",
    "sparsemm_protokv/model_loader.py", "sparsemm_baselines/cache.py",
    "sparsemm_baselines/core.py", "sparsemm_baselines/adapter.py",
    "sparsemm_baselines/run_streamingbench.py",
)


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_manifest(path):
    expected, scheduled = selected_queries()
    if path is None:
        return expected, scheduled
    manifest = json.loads(Path(path).read_text())
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("The manifest must contain a nonempty query list")
    known = {q["query_id"]: q for q in expected}
    if len({q["query_id"] for q in manifest}) != len(manifest):
        raise ValueError("Duplicate manifest query IDs")
    if any(q != known.get(q["query_id"]) for q in manifest):
        raise ValueError("Manifest differs from an official nine-task query")
    # Grouped execution requires each source video to occupy one adjacent run.
    seen, previous = set(), None
    for q in manifest:
        video = q["video_path"]
        if video != previous:
            if video in seen:
                raise ValueError("Manifest must keep each video's queries adjacent")
            seen.add(video)
            previous = video
    return expected, manifest


def protocol(manifest, expected):
    from sparsemm_native.policy import DEFAULT_SCORE_PATH, allocate_head_budgets, load_head_scores
    capacities, allocation = allocate_head_budgets(load_head_scores(DEFAULT_SCORE_PATH),
                                                  budget=BUDGET, window_size=32)
    model_files = ("config.json", "preprocessor_config.json", "generation_config.json")
    return dict(version=VERSION, model=str(official.MODEL.resolve()),
        model_files_sha256={name: sha(official.MODEL / name) for name in model_files},
        annotation=str(official.ANNOTATIONS.resolve()), annotation_sha256=sha(official.ANNOTATIONS),
        query_count=len(manifest), full_nine_query_count=len(expected),
        query_ids=[q["query_id"] for q in manifest], query_sha256=digest(manifest),
        full_nine_query_sha256=digest(expected), fps=FPS, chunk_size=CHUNK,
        fixed_input=fixed_input_metadata(SIZE), video_max_pixels=None,
        temporal_encoding="unchanged native video temporal patches of two real frames",
        dtype="bfloat16", requested_mean_visual_budget=BUDGET,
        configured_visual_slots=int(capacities.sum()),
        configured_mean_visual_tokens_per_head=float(capacities.double().mean()),
        individual_head_quota_cap=None, protect_window=32,
        sparsemm_score_path=str(DEFAULT_SCORE_PATH.resolve()), sparsemm_score_sha256=sha(DEFAULT_SCORE_PATH),
        allocation=allocation, selector="unchanged native visual attention, no pseudo or true question",
        max_new_tokens=MAX_NEW, greedy=True, repetition_penalty=1.1,
        prompt="unchanged native get_prompt wrapping verbatim official MCQA",
        score="official_OVOBenchOfflineScore", grouped_source_video_prefix_reuse=True,
        odd_endpoint="temporary copy of even base cache; preserve original temporal pairs",
        code_sha256={name: sha(ROOT / name) for name in SOURCE_NAMES})


def install_fixed_input(bridge):
    """Thin pixel wrapper matching the shared plugin helper, without lmms imports."""
    from kvbench.fixed_inputs import assert_qwen_grid, frame_rgb_sha256, prepare_fixed_frames
    metadata = fixed_input_metadata(SIZE)
    if bridge.model_config.model_type != "qwen2_5_vl" or bridge.is_llava:
        raise ValueError("The fixed input protocol requires Qwen2.5-VL")
    vision = bridge.model_config.vision_config
    if (vision.patch_size, vision.spatial_merge_size, vision.temporal_patch_size) != (14, 2, 2):
        raise ValueError("Unexpected Qwen visual patch configuration")
    if bridge.video_max_pixels is not None:
        raise ValueError("Do not combine the exact canvas with a pixel cap")
    bridge.processor.video_processor.do_resize = False
    bridge.fixed_input_checks = []
    original_encode = bridge.encode_video_chunk

    def encode_fixed(frames, is_last_chunk=False):
        if frames is None or not len(frames):
            return original_encode(frames, is_last_chunk=is_last_chunk)
        source_hw = ([int(frames.shape[1]), int(frames.shape[2])] if frames.shape[-1] == 3
                     else [int(frames.shape[-2]), int(frames.shape[-1])])
        prepared = prepare_fixed_frames(frames, canvas_size=SIZE)
        original_forward, consumed = bridge.forward_chunk, 0

        def checked_forward(inputs_embeds, positions, *args, **kwargs):
            nonlocal consumed
            visual = bridge._pending_visual_input
            if not isinstance(visual, dict):
                raise AssertionError("Missing native visual forward metadata")
            raw = int(visual["raw_frames"])
            if raw < 1 or consumed + raw > len(prepared):
                raise AssertionError("Visual forwards do not match prepared frame batch")
            audit = assert_qwen_grid(visual["grid_thw"], int(inputs_embeds.shape[1]), raw,
                                    canvas_size=SIZE)
            if (visual["visual_tokens"] != int(inputs_embeds.shape[1])
                    or visual["resized_hw"] != [SIZE, SIZE]
                    or visual["source_hw"] != [SIZE, SIZE]):
                raise AssertionError("The video processor changed the fixed canvas")
            rgb_sha = frame_rgb_sha256(prepared[consumed:consumed + raw])
            visual.update(original_source_hw=source_hw, prepared_rgb_sha256=rgb_sha,
                          fixed_frame_size=SIZE, tokens_per_temporal_unit=1024)
            bridge.fixed_input_checks.append(dict(audit, prepared_rgb_sha256=rgb_sha,
                original_source_hw=source_hw, sampled_ordinal_start=int(bridge.total_processed_frames),
                position_sha256=hashlib.sha256(positions.detach().cpu().contiguous().numpy().tobytes()).hexdigest()))
            consumed += raw
            return original_forward(inputs_embeds, positions, *args, **kwargs)

        bridge.forward_chunk = checked_forward
        try:
            result = original_encode(prepared, is_last_chunk=is_last_chunk)
            if consumed != len(prepared):
                raise AssertionError("A prepared input frame was omitted")
            return result
        finally:
            bridge.forward_chunk = original_forward

    bridge.encode_video_chunk = encode_fixed
    return bridge


def cache_budget_audit(bridge):
    cache = bridge.kv_cache
    per_head = [[int(n) - cache.prefix_tokens for n in layer.head_lens.tolist()]
                for layer in cache.layers]
    flattened = [n for row in per_head for n in row]
    if min(flattened) < 0 or len(flattened) != 112:
        raise AssertionError("Invalid physical visual KV occupancy")
    return dict(actual_visual_slots=sum(flattened), actual_tokens_per_head=sum(flattened) / 112,
        per_head_visual_slots=per_head, minimum_per_head=min(flattened), maximum_per_head=max(flattened),
        configured_visual_slots=int(bridge.budgets.sum()),
        configured_mean_tokens_per_head=float(bridge.budgets.double().mean()),
        individual_quota_cap=None, protected_prefix_tokens=cache.prefix_tokens,
        actual_includes_question_tokens=False, physical_kv_bytes=cache.nbytes())


def run_group(bridge, group, on_active=None, on_result=None):
    """The existing grouped native flow, with pre-answer audits saved per query."""
    import torch
    from sparsemm_baselines.run_streamingbench import sampled_indices
    from sparsemm_native.video_reader import BatchScopedVideoReader
    if not group or len({q["video_path"] for q in group}) != 1:
        raise ValueError("One source video is required")
    reader = BatchScopedVideoReader(group[0]["video_path"], num_threads=1)
    indices = sampled_indices(len(reader), reader.get_avg_fps(), FPS)
    targets = [min(len(indices), math.ceil(q["clip_end"] * FPS)) for q in group]
    if min(targets) < 1:
        raise ValueError("The source contains no sampled frames")
    bridge._active_video_id = group[0]["query_id"]
    bridge.fixed_input_checks.clear()
    bridge.encode_init_prompt()
    base_frames, answers = 0, {}

    def encode_range(start, end):
        for offset in range(start, end, CHUNK):
            stop = min(offset + CHUNK, end)
            frames = torch.from_numpy(reader.get_batch(indices[offset:stop]).asnumpy())
            bridge.encode_video_chunk(frames, is_last_chunk=stop == end)
            bridge.predict_and_compress()

    def answer(query, target, encoded, started):
        budget = cache_budget_audit(bridge)
        audits = copy.deepcopy(bridge.fixed_input_checks)
        if sum(a["raw_frames"] for a in audits) != target:
            raise AssertionError("Fixed-frame audits do not cover the question prefix")
        encoded_tokens = sum(a["visual_tokens"] for a in audits)
        if encoded_tokens != math.ceil(target / 2) * 1024:
            raise AssertionError("Temporal grid/token totals changed")
        response = official._answer_grouped(bridge, query, MAX_NEW, encoded > 0)
        torch.cuda.synchronize()
        row = {key: query[key] for key in ("query_id", "source_id", "probe_id", "task", "group",
                                          "timestamp", "clip_end", "target")}
        row.update(response=response, correct=official.official_correct(query["task"], response, query["target"]),
            frames=target, seconds=time.monotonic() - started, encoded_frames=encoded,
            prompt_sha256=digest(query["prompt"]), budget_audit=budget,
            input_audit=dict(fixed_input=fixed_input_metadata(SIZE), source_video=query["video_path"],
                source_frames=len(reader), source_fps=reader.get_avg_fps(), sampled_raw_frame_indices=indices[:target],
                sampled_raw_frame_indices_sha256=digest(indices[:target]), video_temporal_units=math.ceil(target / 2),
                tokens_per_temporal_unit=1024, prefix_encoded_visual_tokens=encoded_tokens,
                native_video_forward_audits=audits, encoding="unchanged_two_real_frame_temporal_patch",
                external_chunk_size=CHUNK, processor_resize_enabled=False),
            cuda_peak_allocated_mib=torch.cuda.max_memory_allocated() / 1048576,
            recorded_at=now())
        validate_row(row, query)
        return row

    try:
        for question_index in sorted(range(len(group)), key=lambda i: (targets[i], i)):
            query, target = group[question_index], targets[question_index]
            if on_active:
                on_active(query)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            even_target = target - target % 2
            if even_target < base_frames:
                raise AssertionError("The shared source-video cursor moved backward")
            bridge._active_question_index = question_index
            bridge._active_question = query.get("question", query["prompt"])
            bridge._active_question_end_time = query["clip_end"]
            encode_range(base_frames, even_target)
            newly_encoded = even_target - base_frames
            base_frames = even_target
            if target % 2:
                base_cache = bridge.kv_cache
                cursors = (bridge._next_position, bridge._seen_visual_tokens,
                    bridge.total_processed_frames, bridge.last_encoded_frames,
                    bridge._prefills, bridge._compressions, bridge.last_metadata)
                history_length, audit_length = len(bridge.conv_history), len(bridge.fixed_input_checks)
                bridge.kv_cache = copy.deepcopy(base_cache)
                try:
                    encode_range(even_target, target)
                    row = answer(query, target, newly_encoded + 1, started)
                finally:
                    bridge.kv_cache = base_cache
                    (bridge._next_position, bridge._seen_visual_tokens, bridge.total_processed_frames,
                     bridge.last_encoded_frames, bridge._prefills, bridge._compressions,
                     bridge.last_metadata) = cursors
                    del bridge.conv_history[history_length:]
                    del bridge.fixed_input_checks[audit_length:]
            else:
                row = answer(query, target, newly_encoded, started)
            answers[query["query_id"]] = row
            # Deliver only after the odd endpoint's temporary cache/cursors
            # have been restored. A caller can durably commit each answer
            # without waiting for the rest of this source video's questions.
            if on_result:
                on_result(row)
        return [answers[q["query_id"]] for q in group]
    finally:
        bridge.clear_cache()
        bridge.fixed_input_checks.clear()


def summarize(output, rows, manifest):
    total = len(manifest)
    summary = official.summary(rows, total)
    counts = collections.Counter(q["task"] for q in manifest)
    done = collections.Counter(r["task"] for r in rows)
    correct = collections.Counter(r["task"] for r in rows if r["correct"])
    # Partial scores never count absent tasks as zero or imply a full official result.
    summary["task_counts"] = {task: dict(done=done[task], total=counts[task], correct=correct[task],
        accuracy=correct[task] / done[task] if done[task] else None) for task in sorted(counts)}
    summary["nine_task_equal_mean"] = (sum(summary["task_accuracy"].values()) / 9
        if len(summary["task_accuracy"]) == 9 else None)
    summary["official_two_group_equal_mean"] = summary["hermes_category_average"]
    summary["fixed_input"] = fixed_input_metadata(SIZE)
    values = [r["budget_audit"]["actual_tokens_per_head"] for r in rows]
    summary["actual_tokens_per_head"] = dict(mean=sum(values)/len(values), minimum=min(values), maximum=max(values)) if values else None
    summary["measured_inference_seconds"] = sum(r["seconds"] for r in rows)
    summary["updated_at"] = now()
    write_json(output / "summary.json", summary)
    lines = ["# SparseMM 固定 896×896 OVO-Bench 实验", "", f"更新时间：{now()}", "",
        f"进度：**{len(rows)}/{total}**；fps=0.5，平均预算目标 K500，protect32。", "",
        "原生视频两帧融合：每时间单元1024视觉token；16帧输入块为8192。", "",
        "| 任务 | 完成/总数 | 正确 | 当前正确率 |", "|---|---:|---:|---:|"]
    for task, v in summary["task_counts"].items():
        score = f"{v['accuracy']*100:.2f}%" if v["accuracy"] is not None else "—"
        lines.append(f"| {task} | {v['done']}/{v['total']} | {v['correct']} | {score} |")
    if values:
        lines += ["", f"实际平均视觉缓存：**{sum(values)/len(values):.2f} token/头**。"]
    atomic_write(output / "progress.md", "\n".join(lines) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--concurrent-processes", type=int, choices=(1, 2), default=1)
    args = parser.parse_args(argv)
    expected, manifest = load_manifest(args.manifest)
    output = args.output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    path, pinned = output / "queries.jsonl", output / "protocol.json"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    if len(rows) > len(manifest):
        raise ValueError("Too many existing predictions")
    for row, query in zip(rows, manifest):
        validate_row(row, query)
        if row.get("input_audit", {}).get("fixed_input") != fixed_input_metadata(SIZE):
            raise ValueError("An existing prediction has different spatial inputs")
    pin = protocol(manifest, expected)
    if pinned.exists() and json.loads(pinned.read_text()) != pin:
        raise ValueError("Pinned code, model, score or manifest changed; use a new output directory")
    if rows and not pinned.exists():
        raise ValueError("Existing predictions lack their source protocol")
    write_json(pinned, pin)
    write_json(output / "query_manifest.json", manifest)
    for name in SOURCE_NAMES:
        destination = output / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    summarize(output, rows, manifest)
    if args.prepare_only:
        print(json.dumps(dict(output=str(output), queries=len(manifest), model_loaded=False,
                              configured_visual_slots=pin["configured_visual_slots"])))
        return
    if len(rows) == len(manifest):
        write_json(output / "status.json", dict(state="complete", done=len(rows), total=len(manifest), updated_at=now()))
        return
    from sparsemm_native.gpu_slots import gpu_lease
    from sparsemm_native.streaming import load_model
    from sparsemm_protokv.config import StreamingConfig
    started = time.monotonic()

    def status(state, **extra):
        write_json(output / "status.json", dict(state=state, done=len(rows), total=len(manifest),
                   pid=os.getpid(), updated_at=now(), **extra))

    try:
        with gpu_lease(output, concurrent_processes=args.concurrent_processes) as admission:
            write_json(output / "gpu_admission.json", admission)
            status("loading")
            bridge = load_model(official.MODEL, StreamingConfig(method="sparsemm", budget=BUDGET,
                budget_scope="visual"), sample_fps=FPS, trace_path=None, device="cuda",
                dtype="bfloat16", video_max_pixels=None)
            install_fixed_input(bridge)
            write_json(output / "runtime.json", dict(started_at=now(), gpu_admission=admission,
                dependencies={name: version(name) for name in ("torch", "transformers", "numpy", "decord", "Pillow")},
                fixed_input=fixed_input_metadata(SIZE), video_processor=bridge.processor.video_processor.to_dict(),
                allocation=bridge.allocation, generation_config=bridge.model.generation_config.to_dict()))
            for group in official.grouped_batches(manifest[len(rows):], len(manifest)):
                if (output / "PAUSE").exists():
                    status("paused")
                    return
                def commit(row):
                    query = manifest[len(rows)]
                    validate_row(row, query)
                    with path.open("a") as stream:
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    rows.append(row)
                    summarize(output, rows, manifest)
                    status("running")
                    print(f"SparseMM fixed896 {len(rows)}/{len(manifest)} {row['query_id']} {row['task']} "
                          f"correct={row['correct']} seconds={row['seconds']:.2f} "
                          f"visual_mean={row['budget_audit']['actual_tokens_per_head']:.2f}", flush=True)
                run_group(bridge, group, lambda q: status("running", active_query=q["query_id"],
                                                                   active_task=q["task"]), on_result=commit)
            status("complete", elapsed_this_invocation_seconds=time.monotonic()-started)
    except BaseException as error:
        status("failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
