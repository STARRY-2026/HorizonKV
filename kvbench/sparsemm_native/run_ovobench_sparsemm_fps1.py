"""Resumable nine-task OVO control using the existing native SparseMM cache.

The 1000-image Qwen2.5-VL score asset, allocation formula, protect32 and
online attention selector are unchanged.  This entry point changes only the
experiment settings: fps=1, K=2576, dynamic processor resolution, nine tasks.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from sparsemm_native import run_ovobench as official
from sparsemm_native.run_ovobench_recent4 import digest, schedule_queries, select_queries, validate_row
from sparsemm_native.supervise_ovobench_dual import write_json


ROOT = official.ROOT
ARM = "sparsemm_native_fps1_k2576"
K = 2576
FPS = 1.0
MAX_PIXELS = None  # Keep the model processor's native dynamic resolution.
TASKS = official.BACKWARD | official.REALTIME
SOURCES = ("sparsemm_native/run_ovobench_sparsemm_fps1.py", "sparsemm_native/run_ovobench.py",
           "sparsemm_native/simplestream_resize.py",
           "sparsemm_native/streaming.py", "sparsemm_native/cache.py", "sparsemm_native/policy.py",
           "sparsemm_native/two_bank.py", "sparsemm_native/assets.py", "sparsemm_protokv/config.py",
           "sparsemm_protokv/cache.py", "sparsemm_protokv/streaming.py",
           "sparsemm_protokv/monkeypatch.py", "sparsemm_protokv/model_loader.py",
           "sparsemm_protokv/models/qwen2_5_vl_stream.py", "sparsemm_baselines/core.py",
           "sparsemm_baselines/adapter.py", "sparsemm_baselines/cache.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_queries(ids=None):
    expected = [q for q in official.queries() if q["task"] in TASKS]
    if len(expected) != 1468 or len({q["video_path"] for q in expected}) != 512:
        raise ValueError("The pinned nine-task source set changed")
    return expected, schedule_queries(select_queries(expected, ids))


def protocol(selected, expected):
    from sparsemm_native.policy import DEFAULT_SCORE_PATH, load_head_scores, allocate_head_budgets
    scores = load_head_scores(DEFAULT_SCORE_PATH)
    quotas, allocation = allocate_head_budgets(scores, budget=K, window_size=32)
    return dict(version="ovo_nine_sparsemm_native_fps1_k2576_v1", arm=ARM,
                model=str(official.MODEL.resolve()), model_config_sha256=sha(official.MODEL / "config.json"),
                annotation=str(official.ANNOTATIONS.resolve()), annotation_sha256=sha(official.ANNOTATIONS),
                query_count=len(selected), full_nine_query_count=len(expected),
                query_ids=[q["query_id"] for q in selected], query_sha256=digest(selected),
                full_nine_query_sha256=digest(expected), fps=FPS, mean_visual_budget=K,
                actual_configured_total_visual_slots=int(quotas.sum()),
                individual_quota_cap=None, window_size=32,
                selector="existing_SparseMM_static_scores_then_online_attention",
                sparsemm_score_path=str(DEFAULT_SCORE_PATH.resolve()), sparsemm_score_sha256=sha(DEFAULT_SCORE_PATH),
                sparsemm_allocation=allocation, dtype="bfloat16", video_max_pixels=MAX_PIXELS,
                max_new_tokens=16, greedy=True, repetition_penalty=1.1,
                score_mode="official_OVOBenchOfflineScore", grouped_by_source_video=True,
                same_video_prefix_reuse="when_question_resize_hw_matches",
                video_preprocessing="qwen_vl_utils_0.0.14_dynamic_video_resize_formula",
                trace_enabled=False,
                code_sha256={name: sha(ROOT / name) for name in SOURCES})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ids", help="Optional comma-separated smoke-test IDs")
    parser.add_argument("--max-groups", type=int, default=100000)
    args = parser.parse_args(argv)
    if args.max_groups < 1:
        parser.error("--max-groups must be positive")
    expected, selected = selected_queries(args.ids)
    directory = args.output_root.expanduser().resolve() / ARM
    directory.mkdir(parents=True, exist_ok=True)
    pin = protocol(selected, expected)
    protocol_path = directory / "protocol.json"
    progress = directory / "queries.jsonl"
    rows = [json.loads(line) for line in progress.read_text().splitlines() if line.strip()] if progress.exists() else []
    if rows and not protocol_path.exists():
        raise ValueError("Existing predictions lack a pinned protocol")
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != pin:
        raise ValueError("Code, model, score asset or query set changed; refusing to resume")
    if len(rows) > len(selected):
        raise ValueError("Excess predictions")
    for row, query in zip(rows, selected):
        validate_row(row, query)
    write_json(protocol_path, pin)
    write_json(directory / "query_manifest.json", selected)
    for name in SOURCES:
        snapshot = directory / "source_snapshot" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes((ROOT / name).read_bytes())
    if len(rows) == len(selected):
        return

    from sparsemm_native.gpu_slots import gpu_lease
    from sparsemm_native.streaming import load_model
    from sparsemm_protokv.config import StreamingConfig
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    config = StreamingConfig(method="sparsemm", budget=K, budget_scope="visual")
    with gpu_lease(directory, concurrent_processes=1) as admission:
        write_json(directory / "gpu_admission.json", admission)
        bridge = load_model(official.MODEL, config, sample_fps=FPS,
                            trace_path=None, device="cuda",
                            dtype="bfloat16", video_max_pixels=MAX_PIXELS)
        groups = official.grouped_batches(selected[len(rows):], args.max_groups)
        for group in groups:
            answers = official.run_grouped_video(
                bridge, group, FPS, 16, 16, resize_policy="simplestream_dynamic")
            for query in group:
                response, frames, seconds, encoded = answers[query["query_id"]]
                row = {key: query[key] for key in (
                    "query_id", "source_id", "probe_id", "task", "group", "timestamp", "clip_end", "target")}
                row.update(response=response, correct=official.official_correct(query["task"], response, query["target"]),
                           frames=frames, seconds=seconds, encoded_frames=encoded,
                           resize_hw=bridge._query_resize_hw[query["query_id"]],
                           encoded_visual_tokens=bridge._query_encoded_visual_tokens[query["query_id"]],
                           prompt_sha256=digest(query["prompt"]))
                validate_row(row, query)
                with progress.open("a") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                rows.append(row)
                write_json(directory / "summary.json", official.summary(rows, len(selected)))
                print(f"{ARM} {len(rows)}/{len(selected)} {query['query_id']} {query['task']} "
                      f"correct={row['correct']} frames={frames} elapsed={seconds:.2f}s", flush=True)


if __name__ == "__main__":
    main()
