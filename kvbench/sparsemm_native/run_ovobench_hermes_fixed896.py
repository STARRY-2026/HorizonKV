"""Pinned full OVO HERMES evaluation with shared fixed 896 RGB inputs.

The original FP16 HERMES backend, compression, layer groups, summary token,
positions and decoder are unchanged. Video frames and real QA history persist
between questions of the same video, as in native analyze_video. Compression
may use the previous real question and answer. Resume replays completed question
boundaries and restores their saved answers without regenerating them.
"""

import argparse
import collections
import copy
from datetime import datetime
from functools import lru_cache
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


ROOT = official.ROOT
SIZE, FPS, BUDGET, CHUNK, MAX_NEW = 896, .5, 500, 16, 16
TASKS = official.BACKWARD | official.REALTIME
VERSION = "native_hermes_fixed896_fps05_k500_real_history_v2"
SOURCE_NAMES = (
    "sparsemm_native/run_ovobench_hermes_fixed896.py", "kvbench/fixed_inputs.py",
    "kvbench/hermes_runner.py", "kvbench/hermes_backend/qwenvl_hermes.py",
    "kvbench/hermes_backend/abstract_hermes.py", "kvbench/hermes_backend/reindex_3d.py",
    "kvbench/hermes_backend/__init__.py", "kvbench/__init__.py", "kvbench/api.py",
    "kvbench/paths.py", "sparsemm_native/run_ovobench.py",
    "sparsemm_native/run_ovobench_recent4.py", "sparsemm_native/gpu_slots.py",
    "sparsemm_native/video_reader.py", "sparsemm_baselines/run_streamingbench.py",
)
PACKAGES = ("torch", "transformers", "flash-attn", "numpy", "decord", "Pillow", "qwen-vl-utils")
# Native default paths only mutate the first six fields. The optional fields
# are included so enabling an unexpected transient path cannot leak state.
STATE_FIELDS = (
    "kv_cache", "_position_ids_cache", "_layer_position_ids", "total_processed_frames",
    "last_encoded_frames", "visual_start_idx", "conv_history", "fixed_input_checks",
    "_layered_kv_snap", "_layered_pos_snap", "_layered_shallow_kv_snap",
    "_layered_shallow_pos_snap", "_transient_kv_lens", "_transient_pos_lens",
    "_transient_total_frames", "_transient_last_frame", "_fb_pre_instant_lens",
    "_fb_pre_instant_pos_lens",
)


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def write_json(path, value):
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def load_manifest(path):
    expected = [q for q in official.queries() if q["task"] in TASKS]
    if len(expected) != 1468 or len({q["video_path"] for q in expected}) != 512:
        raise ValueError("The official nine-task source set changed")
    if path is None:
        videos = collections.defaultdict(list)
        for q in expected:
            videos[q["video_path"]].append(q)
        manifest = [q for group in videos.values() for q in sorted(group, key=lambda q: q["clip_end"])]
    else:
        manifest = json.loads(Path(path).read_text())
    known = {q["query_id"]: q for q in expected}
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("A nonempty official query manifest is required")
    if len({q["query_id"] for q in manifest}) != len(manifest):
        raise ValueError("Duplicate query IDs")
    if any(q != known.get(q["query_id"]) for q in manifest):
        raise ValueError("The manifest differs from official queries")
    seen, previous, last_target = set(), None, -1
    for q in manifest:
        video = q["video_path"]
        if video != previous:
            if video in seen:
                raise ValueError("Queries of a source video must be adjacent")
            seen.add(video)
            previous, last_target = video, -1
        target = math.ceil(q["clip_end"] * FPS)
        if target < last_target:
            raise ValueError("Within-video queries must follow sampled prefix length")
        last_target = target
    return expected, manifest


def protocol(manifest, expected):
    model_files = ("config.json", "preprocessor_config.json", "generation_config.json",
                   "tokenizer_config.json", "tokenizer.json", "chat_template.json")
    return dict(version=VERSION, method="hermes", model=str(official.MODEL.resolve()),
        model_files_sha256={name: sha(official.MODEL / name) for name in model_files},
        annotation=str(official.ANNOTATIONS.resolve()), annotation_sha256=sha(official.ANNOTATIONS),
        query_count=len(manifest), full_nine_query_count=len(expected),
        query_ids=[q["query_id"] for q in manifest], query_sha256=digest(manifest),
        full_nine_query_sha256=digest(expected), fps=FPS, chunk_size=CHUNK,
        fixed_input=fixed_input_metadata(SIZE), video_max_pixels=None,
        temporal_encoding="native_video_two_real_frames_per_temporal_patch_16_frame_forwards",
        dtype="float16", dtype_policy="unchanged_original_HERMES_loader",
        requested_mean_visual_budget=BUDGET, kv_size=BUDGET,
        budget_scope="native_HERMES_visual_positions_per_layer_shared_by_four_KV_heads",
        budget_includes_summary_token=True, initial_text_prefix_budgeted_separately=True,
        configured_visual_slots=28 * 4 * BUDGET, configured_mean_visual_tokens_per_head=BUDGET,
        selector="unchanged_HERMES_predicted_queries_attention_recency_cross_layer_fusion",
        layer_groups=dict(shallow=[0, 1], middle=[2, 18], deep=[19, 27]),
        layer_group_ratios=[.1, .6, .3], summary_token="unchanged_deep_layer_pruned_token_summary",
        position_policy="unchanged_HERMES_per_layer_MRoPE_and_native_shrink",
        protect_window=None, max_new_tokens=MAX_NEW, greedy=True, repetition_penalty=1.1,
        prompt="verbatim_official_MCQA_wrapped_by_native_get_prompt_mc_false",
        score="official_OVOBenchOfflineScore", grouped_source_video_prefix_reuse=True,
        history_mode="native_previous_real_qa", real_history_readable=True,
        history_metadata="official_source_question_and_options; formatted_options_are_metadata_not_generation_prompt",
        persistent_prefix_alignment=None,
        endpoint_policy="native_continuous_cursor_partial_chunks_committed_at_each_question",
        question_state="native_visual_KV_and_positions_rollback_keep_real_QA_history",
        resume_policy="replay_completed_same_video_boundaries_then_append_saved_real_QA_without_generation",
        processor_resize_enabled=False,
        dependencies={name: version(name) for name in PACKAGES},
        code_sha256={name: sha(ROOT / name) for name in SOURCE_NAMES})


def _tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def install_fixed_input(model):
    """Wrap pixels and audits, retaining the backend's original video forward."""
    from kvbench.fixed_inputs import frame_rgb_sha256, prepare_fixed_frames
    from kvbench.hermes_runner import configure_fixed_processor, validate_fixed_visual_output
    configure_fixed_processor(model, SIZE)
    if (getattr(model, "fixed_frame_canvas_size", 0) or
            any(getattr(model, name, False) for name in ("multi_scale", "layered_resolution", "transient_highres")) or
            getattr(model, "feature_buffer_frames", 0)):
        raise ValueError("This protocol uses only the default native HERMES video branch")
    seen = set()
    for name in ("image_processor", "video_processor"):
        processor = getattr(model.processor, name, None)
        if processor is not None and id(processor) not in seen:
            processor.do_resize = False
            seen.add(id(processor))
    model.fixed_input_checks = []
    native_encode, native_features = model.encode_video_chunk, model.get_video_features

    def encode_fixed(frames, is_last_chunk=False):
        if frames is None or not len(frames):
            return native_encode(frames, is_last_chunk=is_last_chunk)
        raw_frames = len(frames)
        if raw_frames > CHUNK:
            raise ValueError("A native HERMES forward may contain at most sixteen real frames")
        source_hw = ([int(frames.shape[1]), int(frames.shape[2])] if frames.shape[-1] == 3
                     else [int(frames.shape[-2]), int(frames.shape[-1])])
        prepared = prepare_fixed_frames(frames, canvas_size=SIZE, pad_value=127)
        ordinal = int(model.total_processed_frames)
        observed = []

        def checked_features(pixels, grid, *args, **kwargs):
            if len(grid) != 1:
                raise AssertionError("Expected one native video chunk")
            actual_grid = [int(x) for x in grid[0].tolist()]
            validate_fixed_visual_output(raw_frames, SIZE, actual_grid)
            result = native_features(pixels, grid, *args, **kwargs)
            features = result[0]
            if isinstance(features, (list, tuple)):
                features = features[0]
            count = int(features.shape[0])
            check = validate_fixed_visual_output(raw_frames, SIZE, actual_grid, count)
            observed.append(dict(raw_frames=raw_frames, grid_thw=actual_grid, visual_tokens=count,
                **check, source_hw=[SIZE, SIZE], original_source_hw=source_hw,
                prepared_rgb_sha256=frame_rgb_sha256(prepared), sampled_ordinal_start=ordinal,
                temporal_unit_rgb_sha256=[frame_rgb_sha256(prepared[i:i + 2])
                                          for i in range(0, raw_frames, 2)],
                tokens_per_temporal_unit=1024))
            return result

        model.get_video_features = checked_features
        try:
            result = native_encode(prepared, is_last_chunk=is_last_chunk)
            if len(observed) != 1:
                raise AssertionError("Native HERMES changed the number of video forwards")
            observed[0]["position_sha256_by_layer"] = [
                _tensor_sha(pos[:, -observed[0]["visual_tokens"]:]) for pos in model._position_ids_cache]
            model.fixed_input_checks.append(observed[0])
            return result
        finally:
            model.get_video_features = native_features

    model.encode_video_chunk = encode_fixed
    return model


def snapshot_state(model, *, clone=False):
    state = {name: getattr(model, name) for name in STATE_FIELDS if hasattr(model, name)}
    return copy.deepcopy(state) if clone else state


def restore_state(model, state):
    for name in STATE_FIELDS:
        if name in state:
            setattr(model, name, state[name])
        elif hasattr(model, name):
            delattr(model, name)


def _nested_equal(left, right):
    import torch
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_nested_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_nested_equal(a, b) for a, b in zip(left, right))
    return left == right


def _cache_equal(left, right):
    return _nested_equal(list(left), list(right))


def cache_budget_audit(model):
    prefix = int(model.visual_start_idx)
    per_head, lengths, nbytes = [], [], 0
    for key, value in model.kv_cache:
        if key.ndim != 4 or value.shape != key.shape or key.shape[0] != 1:
            raise AssertionError("Invalid native HERMES cache layout")
        visual = int(key.shape[2]) - prefix
        if visual < 0:
            raise AssertionError("Initial text prefix was lost")
        per_head.append([visual] * int(key.shape[1]))
        lengths.append(int(key.shape[2]))
        nbytes += key.numel() * key.element_size() + value.numel() * value.element_size()
    flattened = [n for row in per_head for n in row]
    if len(per_head) != 28 or any(len(row) != 4 for row in per_head) or len(flattened) != 112:
        raise AssertionError("Expected 28 layers and four KV heads per layer")
    return dict(actual_visual_slots=sum(flattened), actual_tokens_per_head=sum(flattened) / 112,
        per_head_visual_slots=per_head, per_head_visual_tokens=per_head,
        minimum_per_head=min(flattened), maximum_per_head=max(flattened),
        layers=28, heads_per_layer=4, configured_visual_slots=BUDGET * 112,
        configured_mean_tokens_per_head=BUDGET, protected_prefix_tokens=prefix,
        prefix_tokens=prefix, actual_includes_question_tokens=False,
        native_summary_tokens_included=True, physical_kv_bytes=nbytes,
        layer_cache_lengths=lengths, native_budget_semantics="shared_visual_positions_per_layer")


@lru_cache(maxsize=1)
def source_question_metadata():
    records = json.loads(official.ANNOTATIONS.read_text())
    return {item["id"]: (item["question"], tuple(item["options"]))
            for item in records if item["task"] in TASKS}


def question_metadata(query):
    raw_question, options = source_question_metadata()[query["source_id"]]
    # The original backend strips each option when saving conversation metadata.
    return raw_question, [option.strip() for option in options]


def formatted_options(options):
    # This string is only read by the backend's metadata regex. The actual
    # generation input remains get_prompt(verbatim_official_prompt, mc=False).
    return "\n".join(f"({chr(65 + i)}) {option}" for i, option in enumerate(options)) + "\nThe best answer"


def answer_native(model, query):
    """Check native visual rollback while retaining its real conversation log."""
    before = snapshot_state(model, clone=True)
    prompt = model.get_prompt(query["prompt"], mc=False)
    expected = f"\n{query['prompt']}<|im_end|><|im_start|>assistant\n"
    if prompt != expected:
        raise AssertionError("The native official question wrapper changed")
    try:
        raw_question, options = question_metadata(query)
        response = model.question_answering(
            dict(prompt=prompt, question=raw_question, formatted_question=formatted_options(options)),
            max_new_tokens=MAX_NEW, temperature=0, repetition_penalty=1.1, pseudo_forward=False)
        cache_equal = _cache_equal(model.kv_cache, before["kv_cache"])
        positions_equal = (_nested_equal(model._position_ids_cache, before["_position_ids_cache"])
                           and _nested_equal(model._layer_position_ids, before["_layer_position_ids"]))
        cursor_equal = all(_nested_equal(getattr(model, name), before[name]) for name in
                           ("total_processed_frames", "last_encoded_frames", "visual_start_idx"))
        history = model.conv_history
        history_equal = (_nested_equal(history[:-1], before["conv_history"])
                         and len(history) == len(before["conv_history"]) + 1
                         and _nested_equal(history[-1], (raw_question, response, options)))
        if not cache_equal or not positions_equal or not cursor_equal or not history_equal:
            raise AssertionError("Native QA failed to restore its visual cache, positions or frame counters")
        return response, dict(official_prompt_mc_false=True, cache_restored=True, positions_restored=True,
            native_backend_unchanged=True, frame_counters_restored=True,
            real_qa_history_preserved=True, history_length_increased_by_one=True,
            raw_question_and_options_saved=True,
            history_length_before=len(before["conv_history"]), history_length_after=len(history),
            history_mode="native_previous_real_qa", real_history_readable=True,
            question_state_isolation="native_visual_rollback_with_persistent_real_QA")
    except BaseException:
        restore_state(model, before)
        raise


def saved_history_entry(query, response):
    """Mirror native question_answering's real metadata without a model call."""
    raw_question, options = question_metadata(query)
    return raw_question, response, options


def run_group(model, group, on_active=None, on_result=None, replay_rows=()):
    import torch
    from sparsemm_baselines.run_streamingbench import sampled_indices
    from sparsemm_native.video_reader import BatchScopedVideoReader
    if not group or len({q["video_path"] for q in group}) != 1:
        raise ValueError("One source video is required")
    reader = BatchScopedVideoReader(group[0]["video_path"], num_threads=1)
    source_fps = reader.get_avg_fps()
    indices = sampled_indices(len(reader), source_fps, FPS)
    targets = [min(len(indices), math.ceil(q["clip_end"] * FPS)) for q in group]
    if min(targets) < 1:
        raise ValueError("The question has no sampled video frame")
    if targets != sorted(targets):
        raise ValueError("Native continuous questions must follow chronological prefix order")
    if len(replay_rows) > len(group):
        raise ValueError("Replay results exceed the source-video question count")
    for row, q in zip(replay_rows, group):
        validate_row(row, q)
    model.clear_cache()
    model.fixed_input_checks, model.conv_history = [], []
    model._layer_position_ids, model._position_ids_cache = {}, [None] * model.num_layers
    model.last_encoded_frames = model.total_processed_frames = 0
    model.encode_init_prompt()
    current, answers = 0, []

    def encode_range(start, end, *, last_chunk):
        for offset in range(start, end, CHUNK):
            stop = min(offset + CHUNK, end)
            frames = torch.from_numpy(reader.get_batch(indices[offset:stop]).asnumpy())
            model.encode_video_chunk(frames, is_last_chunk=last_chunk and stop == end)
            model.predict_and_compress()

    try:
        for question_index in range(len(group)):
            query, target = group[question_index], targets[question_index]
            if on_active:
                on_active(query)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            if target < current:
                raise AssertionError("The native continuous video cursor moved backward")
            before_audits = len(model.fixed_input_checks)
            previous = current
            encode_range(current, target, last_chunk=True)
            current = target
            budget = cache_budget_audit(model)
            audits = copy.deepcopy(model.fixed_input_checks)
            if sum(a["raw_frames"] for a in audits) != target:
                raise AssertionError("The video audits do not cover the exact sampled prefix")
            encoded_tokens = sum(a["visual_tokens"] for a in audits)
            temporal_units = sum(math.ceil(a["raw_frames"] / 2) for a in audits)
            if encoded_tokens != temporal_units * 1024:
                raise AssertionError("Fixed per-forward temporal-unit token count changed")
            if question_index < len(replay_rows):
                saved = replay_rows[question_index]
                expected_audit = saved.get("input_audit", {})
                if (saved["frames"] != target
                        or expected_audit.get("sampled_raw_frame_indices") != indices[:target]
                        or expected_audit.get("prefix_encoded_visual_tokens") != encoded_tokens
                        or expected_audit.get("native_video_forward_audits") != audits
                        or saved.get("budget_audit") != budget):
                    raise ValueError("Replayed video state differs from a completed query")
                model.conv_history.append(saved_history_entry(query, saved["response"]))
                if len(model.conv_history) != question_index + 1:
                    raise AssertionError("Completed real QA history was not replayed exactly")
                continue
            response, validation = answer_native(model, query)
            torch.cuda.synchronize()
            row = {key: query[key] for key in ("query_id", "source_id", "probe_id", "task", "group",
                                              "timestamp", "clip_end", "target")}
            row.update(response=response, correct=official.official_correct(query["task"], response, query["target"]),
                frames=target, seconds=time.monotonic() - started, encoded_frames=target - previous,
                encoded_temporal_units=sum(a["visual_tokens"] // 1024 for a in audits[before_audits:]),
                persistent_encoded_frames=current, real_history_readable=True,
                history_mode="native_previous_real_qa", question_history_length=len(model.conv_history),
                prompt_sha256=digest(query["prompt"]), wrapped_prompt_sha256=digest(model.get_prompt(query["prompt"], mc=False)),
                budget_audit=budget, validation=validation,
                input_audit=dict(fixed_input=fixed_input_metadata(SIZE), source_video=query["video_path"],
                    source_frames=len(reader), source_fps=source_fps, sampled_raw_frame_indices=indices[:target],
                    sampled_raw_frame_indices_sha256=digest(indices[:target]), video_temporal_units=temporal_units,
                    minimum_temporal_units_for_full_prefix=math.ceil(target / 2),
                    temporal_padding_units_from_question_boundaries=temporal_units - math.ceil(target / 2),
                    tokens_per_temporal_unit=1024, prefix_encoded_visual_tokens=encoded_tokens,
                    native_video_forward_audits=audits, encoding="native_continuous_question_boundary_video_forwards",
                    external_chunk_size=CHUNK, processor_resize_enabled=False),
                cuda_peak_allocated_mib=torch.cuda.max_memory_allocated() / 1048576,
                recorded_at=now())
            validate_row(row, query)
            if model.total_processed_frames != current:
                raise AssertionError("A question changed the persistent video frame cursor")
            answers.append(row)
            if on_result:
                on_result(row)
        return answers
    finally:
        model.clear_cache()
        model.fixed_input_checks.clear()
        model._position_ids_cache = [None] * model.num_layers
        model._layer_position_ids.clear()
        model.total_processed_frames = model.last_encoded_frames = 0


def summarize(output, rows, manifest):
    summary = official.summary(rows, len(manifest))
    counts = collections.Counter(q["task"] for q in manifest)
    done = collections.Counter(r["task"] for r in rows)
    correct = collections.Counter(r["task"] for r in rows if r["correct"])
    summary["task_counts"] = {task: dict(done=done[task], total=counts[task], correct=correct[task],
        accuracy=correct[task] / done[task] if done[task] else None) for task in sorted(counts)}
    summary["nine_task_equal_mean"] = (sum(summary["task_accuracy"].values()) / 9
        if len(summary["task_accuracy"]) == 9 else None)
    summary["official_two_group_equal_mean"] = summary["hermes_category_average"]
    summary["fixed_input"] = fixed_input_metadata(SIZE)
    values = [r["budget_audit"]["actual_tokens_per_head"] for r in rows]
    summary["actual_tokens_per_head"] = dict(mean=sum(values) / len(values), minimum=min(values), maximum=max(values)) if values else None
    summary["measured_inference_seconds"] = sum(r["seconds"] for r in rows)
    summary["encoded_temporal_units"] = sum(r["encoded_temporal_units"] for r in rows)
    summary["updated_at"] = now()
    write_json(output / "summary.json", summary)
    lines = ["# HERMES 固定896×896 OVO-Bench实验", "", f"更新时间：{now()}", "",
        f"进度：**{len(rows)}/{len(manifest)}**；fps=0.5，原生HERMES K500，FP16。", "",
        "原生16帧前向；每两真实帧1024视觉token。保留原层组、摘要和位置算法。", "",
        "沿用原生真实问答历史：同视频后续压缩会读取上一道真实问题与模型回答。", "",
        "| 任务 | 完成/总数 | 正确 | 当前正确率 |", "|---|---:|---:|---:|"]
    for task, v in summary["task_counts"].items():
        score = f"{v['accuracy'] * 100:.2f}%" if v["accuracy"] is not None else "—"
        lines.append(f"| {task} | {v['done']}/{v['total']} | {v['correct']} | {score} |")
    if values:
        lines += ["", f"实际平均视觉缓存：**{sum(values) / len(values):.2f} token/头**（含原生摘要位置，初始文本另计）。"]
    atomic_write(output / "progress.md", "\n".join(lines) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--concurrent-processes", type=int, choices=(1,), default=1)
    args = parser.parse_args(argv)
    expected, manifest = load_manifest(args.manifest)
    output = args.output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    path, pinned = output / "queries.jsonl", output / "protocol.json"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    if len(rows) > len(manifest):
        raise ValueError("Existing result count exceeds the pinned manifest")
    for row, query in zip(rows, manifest):
        validate_row(row, query)
        if row.get("input_audit", {}).get("fixed_input") != fixed_input_metadata(SIZE):
            raise ValueError("Existing predictions use different spatial inputs")
        if not all(row.get("validation", {}).get(name) is True for name in (
                "official_prompt_mc_false", "real_qa_history_preserved", "history_length_increased_by_one",
                "cache_restored", "positions_restored", "native_backend_unchanged")):
            raise ValueError("Existing predictions lack the native real-history protocol")
    pin = protocol(manifest, expected)
    if pinned.exists() and json.loads(pinned.read_text()) != pin:
        raise ValueError("Pinned code, dependencies, model or manifest changed; use a new output directory")
    if rows and not pinned.exists():
        raise ValueError("Existing predictions lack their frozen source protocol")
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
    for key in ("PROTOKV_ENABLE", "PROTOKV_FULL_ENABLE", "INFINIPOT_V_PUBLIC"):
        os.environ[key] = "0"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from sparsemm_native.gpu_slots import gpu_lease
    from kvbench.hermes_backend.qwenvl_hermes import load_model
    started = time.monotonic()

    def status(state, **extra):
        write_json(output / "status.json", dict(state=state, done=len(rows), total=len(manifest),
                   pid=os.getpid(), updated_at=now(), history_mode="native_previous_real_qa",
                   real_history_readable=True, **extra))

    try:
        with gpu_lease(output, concurrent_processes=1) as admission:
            write_json(output / "gpu_admission.json", admission)
            status("loading")
            model, _ = load_model(str(official.MODEL.resolve()), kv_size=BUDGET,
                                  sample_fps=FPS, compress_mode="hermes")
            install_fixed_input(model)
            if model.num_layers != 28 or model.short_term_threshold != 2 or model.long_term_threshold != 19:
                raise AssertionError("Original HERMES layer groups changed")
            write_json(output / "runtime.json", dict(started_at=now(), gpu_admission=admission,
                dependencies={name: version(name) for name in PACKAGES}, dtype=str(model.dtype),
                fixed_input=fixed_input_metadata(SIZE),
                video_processor=model.processor.video_processor.to_dict(),
                layer_groups=pin["layer_groups"], generation_config=model.generation_config.to_dict()))
            completed_by_id = {row["query_id"]: row for row in rows}
            for group in official.grouped_batches(manifest, len(manifest)):
                replay = [completed_by_id[q["query_id"]] for q in group if q["query_id"] in completed_by_id]
                if len(replay) == len(group):
                    continue
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
                    print(f"HERMES fixed896 {len(rows)}/{len(manifest)} {row['query_id']} {row['task']} "
                          f"correct={row['correct']} seconds={row['seconds']:.2f} "
                          f"visual_mean={row['budget_audit']['actual_tokens_per_head']:.2f}", flush=True)

                run_group(model, group, lambda q: status("running", active_query=q["query_id"],
                          active_task=q["task"]), on_result=commit, replay_rows=replay)
            status("complete", elapsed_this_invocation_seconds=time.monotonic() - started)
    except BaseException as error:
        status("failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
