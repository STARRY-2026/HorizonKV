"""Official nine-task OVO-Bench test of K3000 last32-attention dual-cache DP.

The historical HERMES-nine experiment and its resumable source files are never
changed by this entry point. This arm runs the nine backward/realtime tasks
from the official manifest; REC, SSR and CRR are excluded by request.
"""

import argparse
import collections
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time

from sparsemm_native import run_ovobench as legacy


ROOT = legacy.ROOT
ARMS = ("dp_last32_k3000",)
LAST32_BUDGET = ROOT.parent / (
    "hermes/results/myobservations/last32_two_bank_k3000_floor2250_cap4500_20261022/"
    "per_head_allocation.csv")
LAST32_REPORT = LAST32_BUDGET.parent / "report.json"
IDENTITY = ("query_id", "source_id", "probe_id", "task", "group", "timestamp", "clip_end", "target")
COMMON_FIELDS = (
    "source_sha256", "model_config_sha256", "budget", "fps", "dtype", "chunk_size",
    "max_new_tokens", "video_max_pixels", "score_mode", "video_mode",
)
SOURCE_FILES = (
    "sparsemm_native/run_ovobench_last32_dp_k3000.py", "sparsemm_native/run_ovobench.py",
    "sparsemm_native/gpu_slots.py",
    "sparsemm_native/streaming.py", "sparsemm_native/cache.py",
    "sparsemm_native/two_bank.py", "sparsemm_native/policy.py", "sparsemm_native/assets.py",
    "sparsemm_protokv/config.py", "sparsemm_protokv/cache.py", "sparsemm_protokv/streaming.py",
    "sparsemm_protokv/monkeypatch.py", "sparsemm_protokv/model_loader.py",
    "sparsemm_protokv/models/qwen2_5_vl_stream.py", "sparsemm_baselines/cache.py",
    "sparsemm_baselines/core.py", "sparsemm_baselines/adapter.py",
    "sparsemm_baselines/run_streamingbench.py",
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def select_queries(expected, ids=None):
    eligible = [q for q in expected if q["task"] not in legacy.FORWARD]
    if not ids:
        return eligible
    requested = [part.strip() for part in ids.split(",")]
    if any(not part for part in requested) or len(set(requested)) != len(requested):
        raise ValueError("--ids must contain unique, nonempty comma-separated query IDs")
    missing = set(requested) - {q["query_id"] for q in eligible}
    if missing:
        raise ValueError("Unknown or excluded forward-task query IDs: " + ", ".join(sorted(missing)))
    # Preserve official ordering even when the CLI IDs were supplied differently.
    return [q for q in eligible if q["query_id"] in set(requested)]


def schedule_queries(selected, *, grouped=True):
    """Use each source video once, including nonadjacent official queries.

    The official query set/order is retained separately in the protocol. Results
    follow this explicitly pinned execution manifest, with videos ordered by
    first appearance and each video's queries in official relative order.
    """
    if not grouped:
        return list(selected)
    videos = collections.defaultdict(list)
    for query in selected:
        videos[query["video_path"]].append(query)
    return [query for group in videos.values() for query in group]


def validate_row(row, query, *, require_prompt_hash=True):
    for field in IDENTITY:
        if row.get(field) != query[field]:
            raise ValueError(f"Query identity/target changed: {query['query_id']} {field}")
    if (not isinstance(row.get("response"), str) or type(row.get("correct")) is not bool
            or type(row.get("frames")) is not int or row["frames"] < 1):
        raise ValueError(f"Malformed result row: {query['query_id']}")
    if row["correct"] != legacy.official_correct(query["task"], row["response"], query["target"]):
        raise ValueError(f"Official scorer disagrees: {query['query_id']}")
    if require_prompt_hash and row.get("prompt_sha256") != digest(query["prompt"]):
        raise ValueError(f"Official prompt changed: {query['query_id']}")


def read_progress(path, expected):
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) > len(expected):
        raise ValueError("Progress contains more queries than the official manifest")
    for row, query in zip(rows, expected):
        validate_row(row, query)
    return rows


def build_protocol(args, selected, expected):
    from importlib.metadata import version
    from sparsemm_native.policy import DEFAULT_SCORE_PATH

    reusable = {}
    for source in args.reuse_root:
        directory = Path(source).expanduser().resolve() / args.arm
        for name in ("protocol.json", "queries.jsonl"):
            path = directory / name
            reusable[str(path)] = legacy.sha256(path)
    return dict(
        protocol_version="official_ovo_last32_dp_k3000_v1", dataset="official_backward_realtime9", arm=args.arm,
        source_annotation=str(legacy.ANNOTATIONS), source_sha256=legacy.sha256(legacy.ANNOTATIONS),
        model_path=str(legacy.MODEL.resolve()), model_config_sha256=legacy.sha256(legacy.MODEL / "config.json"),
        budget=3000, fps=0.5, dtype="bfloat16", chunk_size=args.chunk_size,
        max_new_tokens=16, video_max_pixels=151200, score_mode="official_OVOBenchOfflineScore",
        video_mode="source_prefix_ceil_timestamp", decoding="greedy", repetition_penalty=1.1,
        protect_window=32, attention_observation_window=32, grouped=args.grouped,
        concurrent_processes=args.concurrent_processes,
        concurrency_mode="exclusive_gpu0" if args.concurrent_processes == 1 else "shared_gpu0_two_slots",
        execution_schedule="global_video_first_appearance_then_official_relative_order" if args.grouped else "official_query_order",
        question_state="isolated_qa_snapshot_restore",
        debug_explain=False, query_count=len(selected), full_query_count=len(expected),
        query_ids=[q["query_id"] for q in selected], query_list_sha256=digest(selected),
        full_query_list_sha256=digest(expected),
        official_aggregation="mean(backward_3_task_mean,realtime_6_task_mean)",
        recent_policy="last32_attention_fitted_DP_per_head_recent_quota",
        history_policy="sparsemm_online_attention", protect_window_included_in_recent=True,
        history_weight_source=str(LAST32_BUDGET.resolve()),
        history_weight_sha256=legacy.sha256(LAST32_BUDGET),
        last32_calibration_report_sha256=legacy.sha256(LAST32_REPORT),
        dp_budget_sha256=legacy.sha256(LAST32_BUDGET),
        sparsemm_score_sha256=legacy.sha256(DEFAULT_SCORE_PATH),
        code_sha256={name: legacy.sha256(ROOT / name) for name in SOURCE_FILES},
        dependencies={name: version(name) for name in ("torch", "transformers", "numpy", "decord", "Pillow")},
        processor_files_sha256={name: legacy.sha256(legacy.MODEL / name) for name in (
            "config.json", "preprocessor_config.json", "tokenizer_config.json", "tokenizer.json",
            "chat_template.json", "generation_config.json")},
        reuse_source_files_sha256=reusable,
    )


def reusable_rows(roots, arm, expected, reference):
    """Adopt only official predictions with a matching complete query/prompt hash."""
    if roots and arm == "dp_last32_k3000":
        raise ValueError("The new last32 DP arm has no reusable prior predictions")
    indexed = {q["query_id"]: q for q in expected}
    result = {}
    for root in roots:
        directory = Path(root).expanduser().resolve() / arm
        source = json.loads((directory / "protocol.json").read_text())
        dataset = source.get("dataset", "official_full")
        if dataset not in ("official_full", "forward3") or source.get("arm") != arm:
            raise ValueError("Only matching-arm official_full/forward3 predictions may be reused")
        manifest = expected if dataset == "official_full" else [q for q in expected if q["task"] in legacy.FORWARD]
        if source.get("query_count") != len(manifest) or source.get("query_list_sha256") != digest(manifest):
            raise ValueError("Reusable source official query/prompt manifest changed")
        for field in COMMON_FIELDS:
            if source.get(field) != reference[field]:
                raise ValueError(f"Reusable source protocol mismatch: {field}")
        if Path(source.get("model_path", "")).resolve() != Path(reference["model_path"]).resolve():
            raise ValueError("Reusable source model path differs")
        if source.get("dp_budget_sha256") != reference["dp_budget_sha256"]:
            raise ValueError("Reusable source DP capacity table changed")
        rows = [json.loads(line) for line in (directory / "queries.jsonl").read_text().splitlines() if line.strip()]
        if len(rows) > len(manifest):
            raise ValueError("Reusable source contains excess rows")
        for row, query in zip(rows, manifest):
            validate_row(row, query, require_prompt_hash=False)
            adopted = dict(row, prompt_sha256=digest(indexed[row["query_id"]]["prompt"]),
                           reused_from=str(directory), original_seconds=row.get("seconds"), seconds=0.0)
            if row["query_id"] in result and result[row["query_id"]]["response"] != row["response"]:
                raise ValueError("Conflicting reusable predictions for one official query")
            result[row["query_id"]] = adopted
    return result


def validate_runtime(bridge, *, encoded_frames):
    """Validate each selection without comparing dynamic quotas to the old table."""
    checks = bridge.validation_checks
    selectors, qa = checks["selector_checks"], checks["qa_checks"]
    if (len(qa) != 1 or any(q.get("returned") is not True or q.get("restored") is not True for q in qa)
            or checks["failures"] or (encoded_frames and not selectors)):
        raise RuntimeError("Official recent-four runtime/QA audit failed")
    dynamic = getattr(bridge, "is_recent4", False) or bridge.__class__.__name__ == "Recent4StreamingQwen"
    for check in selectors:
        if check.get("validation_passed") is not True:
            raise RuntimeError("Native cache selector validation failed")
        if dynamic:
            # Recent-four capacities depend on the current temporal-unit token
            # count.  The bridge/cache validates exact frame membership and the
            # global Hamilton allocation at each update; an old static-table
            # equality would incorrectly reject a correct dynamic run.
            values = check.get("head_budgets")
            if not isinstance(values, list) or len(values) != bridge.num_kv_heads or any(type(v) is not int or v < 32 for v in values):
                raise RuntimeError("Invalid dynamic per-head capacities")
            allocation = check.get("allocation", {})
            matrices = [allocation.get(name) for name in ("head_budgets", "recent_budgets", "history_budgets")]
            if any(not isinstance(matrix, list) or len(matrix) != bridge.num_layers
                   or any(not isinstance(row, list) or len(row) != bridge.num_kv_heads
                          or any(type(v) is not int or v < 0 for v in row) for row in matrix)
                   for matrix in matrices):
                raise RuntimeError("Missing dynamic global allocation snapshot")
            capacity, recent, history = matrices
            units = allocation.get("recent_unit_token_counts")
            r = allocation.get("recent_tokens_per_head")
            if (allocation.get("method") != "dp_recent4_history_proportional"
                    or allocation.get("target_visual_slots") != 336000
                    or allocation.get("allocated_visual_slots") != 336000
                    or allocation.get("protect_window_included_in_recent") is not True
                    or allocation.get("recent_temporal_units") != 2
                    or allocation.get("short_cache_gate") != "legacy_global"
                    or type(r) is not int or r < 32
                    or not isinstance(units, list) or not 1 <= len(units) <= 2
                    or any(type(v) is not int or v <= 0 for v in units) or sum(units) != r
                    or sum(map(sum, capacity)) != 336000
                    or values != capacity[check["layer_idx"]]
                    or any(v != r for row in recent for v in row)
                    or any(n != rr + hh for ns, rs, hs in zip(capacity, recent, history)
                           for n, rr, hh in zip(ns, rs, hs))):
                raise RuntimeError("Dynamic four-frame/global budget audit failed")
        else:
            allocation=check.get("allocation",{})
            recent=allocation.get("recent_budgets",[])
            if (check.get("head_budgets") != bridge.budgets[check["layer_idx"]].tolist()
                    or len(recent) != bridge.num_layers
                    or recent[check["layer_idx"]] != bridge.recent_budgets[check["layer_idx"]].tolist()
                    or allocation.get("protect_window_included_in_recent") is not True):
                raise RuntimeError("K3000 dual-bank capacities, recent split or protection changed")
    selectors.clear()
    qa.clear()


def answer(bridge, query, max_new_tokens, *, encoded_frames):
    history_length = len(bridge.conv_history)
    try:
        response = str(bridge.question_answering(
            {"question": query["prompt"], "formatted_question": query["prompt"],
             "prompt": bridge.get_prompt(query["prompt"], mc=False)}, max_new_tokens=max_new_tokens))
        validate_runtime(bridge, encoded_frames=encoded_frames)
        return response
    finally:
        del bridge.conv_history[history_length:]


def run_grouped_video(bridge, group, fps=0.5, chunk_size=16, max_new_tokens=16):
    """Preserve fresh-query pairing, including temporary odd-endpoint branches."""
    import torch
    from sparsemm_baselines.run_streamingbench import sampled_indices
    from sparsemm_protokv.models.qwen2_5_vl_stream import BatchScopedVideoReader

    if not group or len({q["video_path"] for q in group}) != 1:
        raise ValueError("Grouped evaluation requires one source video")
    if chunk_size % 2:
        raise ValueError("Qwen grouped inference requires an even chunk size")
    reader = BatchScopedVideoReader(group[0]["video_path"], num_threads=1)
    indices = sampled_indices(len(reader), reader.get_avg_fps(), fps)
    targets = [min(len(indices), math.ceil(q["clip_end"] * fps)) for q in group]
    if any(n < 1 for n in targets):
        raise ValueError("An official query contains no sampled frames")
    bridge._active_video_id = group[0]["query_id"]
    bridge.encode_init_prompt()
    base_frames, answers = 0, {}

    def encode(start, stop):
        for offset in range(start, stop, chunk_size):
            end = min(offset + chunk_size, stop)
            frames = torch.from_numpy(reader.get_batch(indices[offset:end]).asnumpy())
            bridge.encode_video_chunk(frames, is_last_chunk=end == stop)
            bridge.predict_and_compress()

    for index in sorted(range(len(group)), key=lambda i: (targets[i], i)):
        query, target = group[index], targets[index]
        started = time.monotonic()
        even_target = target - target % 2
        if even_target < base_frames:
            raise AssertionError("Frame cursor moved backwards")
        bridge._active_question_index = index
        bridge._active_question = query["prompt"]
        bridge._active_question_end_time = query["clip_end"]
        encode(base_frames, even_target)
        newly_encoded = even_target - base_frames
        base_frames = even_target
        if target % 2:
            base_cache = bridge.kv_cache
            fields = ("_next_position", "_seen_visual_tokens", "total_processed_frames",
                      "last_encoded_frames", "_prefills", "_compressions", "last_metadata")
            cursors = {name: getattr(bridge, name) for name in fields}
            history_length = len(bridge.conv_history)
            bridge.kv_cache = copy.deepcopy(base_cache)
            try:
                encode(even_target, target)
                newly_encoded += 1
                response = answer(bridge, query, max_new_tokens, encoded_frames=True)
            finally:
                bridge.kv_cache = base_cache
                for name, value in cursors.items():
                    setattr(bridge, name, value)
                del bridge.conv_history[history_length:]
        else:
            response = answer(bridge, query, max_new_tokens, encoded_frames=newly_encoded > 0)
        answers[query["query_id"]] = (response, target, time.monotonic() - started, newly_encoded)
    bridge.clear_cache()
    return answers


def run_query(bridge, query, fps=0.5, chunk_size=16, max_new_tokens=16):
    # A one-query group has identical even/odd temporal pairing to the fresh
    # path, while using the dynamic-aware selector checks above.
    return run_grouped_video(bridge, [query], fps, chunk_size, max_new_tokens)[query["query_id"]]


def append_row(path, rows, row, expected):
    from sparsemm_baselines.run_streamingbench import write_json

    validate_row(row, expected[len(rows)])
    with path.open("a") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    rows.append(row)
    result = legacy.summary(rows, len(expected))
    result["official_total_average"] = None
    if len(expected) != 1468 or len(rows) != 1468 or len(result["task_accuracy"]) != 9:
        result["hermes_category_average"] = None
    write_json(path.parent / "summary.json", result)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=3000)
    parser.add_argument("--fps", type=float, default=0.5)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--video-max-pixels", type=int, default=151200)
    parser.add_argument("--max-new", type=int, default=25, help="Queries or adjacent video groups per invocation")
    parser.add_argument("--concurrent-processes", type=int, choices=(1, 2), default=1,
                        help="GPU scheduling only: exclusive GPU or at most two shared workers")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--grouped", dest="grouped", action="store_true", default=True,
                      help="Default: reuse every source video's prefixes across all its questions")
    mode.add_argument("--fresh", dest="grouped", action="store_false", help="Disable reuse for validation only")
    parser.add_argument("--ids", help="Comma-separated official query IDs, preserving official order")
    parser.add_argument("--reuse-root", type=Path, action="append", default=[],
                        help="Read-only prior official_full/forward3 control results; repeatable")
    args = parser.parse_args(argv)
    if (args.budget != 3000 or args.fps != 0.5 or args.chunk_size != 16
            or args.max_new_tokens != 16 or args.video_max_pixels != 151200 or args.max_new < 1):
        parser.error("The official comparison is pinned to K3000, FPS0.5, chunk16, max-new-tokens16, pixels151200")
    if args.arm == "dp_last32_k3000" and args.reuse_root:
        parser.error("The new arm cannot reuse older predictions")
    return args


def main(argv=None):
    args = parse_args(argv)
    expected = legacy.queries()
    selected = schedule_queries(select_queries(expected, args.ids), grouped=args.grouped)
    protocol = build_protocol(args, selected, expected)
    directory = args.output_root.expanduser().resolve() / args.arm
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "queries.jsonl"
    protocol_path = directory / "protocol.json"
    rows = read_progress(path, selected)
    if rows and not protocol_path.exists():
        raise ValueError("Existing progress has no pinned protocol")
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Protocol/code/assets changed; refusing to resume")
    from sparsemm_baselines.run_streamingbench import write_json
    from sparsemm_native.gpu_slots import gpu_lease
    from sparsemm_protokv.config import StreamingConfig

    write_json(protocol_path, protocol)
    write_json(directory / "query_manifest.json", selected)
    for name in SOURCE_FILES:
        destination = directory / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    reused = reusable_rows(args.reuse_root, args.arm, expected, protocol)
    if len(rows) == len(selected):
        return
    config = StreamingConfig(
        method="sparsemm", budget=3000, budget_scope="visual",
        sparsemm_budget_path=str(LAST32_BUDGET)
        if args.arm != "native" else "",
        sparsemm_cache_policy="two_bank",
        sparsemm_two_bank_gate="legacy_global")
    from sparsemm_native.streaming import load_model
    remaining = selected[len(rows):]
    batches = (legacy.grouped_batches(remaining, args.max_new) if args.grouped else
               [[query] for query in remaining[:args.max_new]])
    bridge = None
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    with gpu_lease(directory, concurrent_processes=args.concurrent_processes):
        for group in batches:
            missing = [query for query in group if query["query_id"] not in reused]
            if missing:
                if bridge is None:
                    bridge = load_model(legacy.MODEL, config, sample_fps=0.5,
                                        trace_path=directory / "cache_trace.jsonl", device="cuda",
                                        dtype="bfloat16", video_max_pixels=151200)
                if args.grouped:
                    results = run_grouped_video(bridge, missing)
                else:
                    results = {query["query_id"]: run_query(bridge, query) for query in missing}
            else:
                results = {}
            for query in group:
                row = reused.get(query["query_id"])
                if row is None:
                    response, frames, seconds, encoded_frames = results[query["query_id"]]
                    row = {key: query[key] for key in IDENTITY}
                    row.update(response=response, correct=legacy.official_correct(query["task"], response, query["target"]),
                               frames=frames, seconds=seconds, encoded_frames=encoded_frames,
                               execution_mode="grouped" if args.grouped else "fresh",
                               prompt_sha256=digest(query["prompt"]))
                append_row(path, rows, row, selected)
                print(f"{args.arm}: {len(rows)}/{len(selected)} {query['query_id']} {query['task']} "
                      f"correct={row['correct']} frames={row['frames']} reused={bool(row.get('reused_from'))}", flush=True)


if __name__ == "__main__":
    main()
