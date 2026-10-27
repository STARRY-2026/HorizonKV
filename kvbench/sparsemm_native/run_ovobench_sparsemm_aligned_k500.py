"""Paired native OVO evaluation at fps=1 and mean visual K=500.

The two-bank arm is a *native streaming adaptation* of the 56-head 250/250
split. It does not claim the independent SimpleStream image-prefix prefill or
fixed pseudo-query selector used by the original history250 experiment.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

from sparsemm_native import run_ovobench_sparsemm_fps1 as base
from sparsemm_native import run_ovobench as official
from sparsemm_native.run_ovobench_recent4 import digest, validate_row
from sparsemm_native.simplestream_recent_share_attention import HEADS, SOURCE_TABLE
from sparsemm_native.supervise_ovobench_dual import write_json


K = 500
FPS = 1.0
ARMS = ("sparsemm_k500", "native_two_bank_250250_k500")
ROOT = official.ROOT
SOURCE = Path(__file__).resolve()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_budget_table(path):
    """Pin the same 56 low-recent heads as the SimpleStream history pilot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(HEADS) != 56 or len(set(HEADS)) != 56:
        raise ValueError("Expected exactly 56 distinct low-recent heads")
    rows = []
    for layer in range(28):
        for head in range(4):
            recent = 250 if (layer, head) in HEADS else 500
            rows.append((layer, head, K, recent, K - recent))
    content = "layer,kv_head,capacity_n,recent_r,history_q\n" + "".join(
        ",".join(str(value) for value in row) + "\n" for row in rows)
    if path.exists():
        if path.read_text() != content:
            raise ValueError("Existing two-bank allocation has changed")
    else:
        path.write_text(content)
    return path


def protocol(arm, selected, expected, budget_path):
    from sparsemm_native.policy import DEFAULT_SCORE_PATH, allocate_head_budgets, load_head_scores
    scores = load_head_scores(DEFAULT_SCORE_PATH)
    sparsemm_quotas, sparsemm_allocation = allocate_head_budgets(
        scores, budget=K, window_size=32)
    return dict(
        version="ovo_native_sparsemm_paired_k500_v1", arm=arm,
        method="sparsemm_original_allocation_and_selector" if arm == ARMS[0]
        else "fixed_250_recent_250_history_low56_native_online_attention",
        model=str(official.MODEL.resolve()),
        model_config_sha256=sha(official.MODEL / "config.json"),
        annotation=str(official.ANNOTATIONS.resolve()),
        annotation_sha256=sha(official.ANNOTATIONS),
        query_count=len(selected), full_nine_query_count=len(expected),
        query_ids=[q["query_id"] for q in selected],
        query_sha256=digest(selected), full_nine_query_sha256=digest(expected),
        fps=FPS, mean_visual_budget=K,
        actual_configured_total_visual_slots=(int(sparsemm_quotas.sum())
                                                if arm == ARMS[0] else 112 * K),
        individual_quota_cap=None if arm == ARMS[0] else K,
        window_size=32, dtype="bfloat16", video_max_pixels=None,
        max_new_tokens=16, greedy=True, repetition_penalty=1.1,
        score_mode="official_OVOBenchOfflineScore", grouped_by_source_video=True,
        same_video_prefix_reuse="when_question_resize_hw_matches",
        video_preprocessing="qwen_vl_utils_0.0.14_dynamic_video_resize_formula",
        video_mode="source_prefix_ceil_timestamp",
        sparsemm_score_path=str(DEFAULT_SCORE_PATH.resolve()),
        sparsemm_score_sha256=sha(DEFAULT_SCORE_PATH),
        sparsemm_allocation=sparsemm_allocation if arm == ARMS[0] else None,
        two_bank_budget_path=str(budget_path.resolve()) if arm == ARMS[1] else None,
        two_bank_budget_sha256=sha(budget_path) if arm == ARMS[1] else None,
        low_head_source_path=str(SOURCE_TABLE.resolve()),
        low_head_source_sha256=sha(SOURCE_TABLE),
        low_head_count=len(HEADS),
        baseline_mean_kv_budget_approximate=True,
        adaptation_difference=(None if arm == ARMS[0] else
            "online streaming attention and Qwen video temporal patches; "
            "not SimpleStream independent image-prefix prefill/pseudo-query selection"),
        code_sha256={name: sha(ROOT / name) for name in base.SOURCES},
        runner_sha256=sha(SOURCE),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--ids", help="Optional comma-separated smoke IDs")
    parser.add_argument("--max-groups", type=int, default=100000)
    parser.add_argument("--prepare-only", action="store_true",
                        help="Pin protocols and allocations without loading the model")
    args = parser.parse_args(argv)
    if args.max_groups < 1:
        parser.error("--max-groups must be positive")
    expected, selected = base.selected_queries(args.ids)
    root = args.output_root.expanduser().resolve()
    directory = root / args.arm
    directory.mkdir(parents=True, exist_ok=True)
    budget_path = write_budget_table(root / "native_two_bank_250250_k500.csv")
    pin = protocol(args.arm, selected, expected, budget_path)
    protocol_path = directory / "protocol.json"
    progress = directory / "queries.jsonl"
    rows = [json.loads(line) for line in progress.read_text().splitlines()
            if line.strip()] if progress.exists() else []
    if rows and not protocol_path.exists():
        raise ValueError("Existing predictions lack a pinned protocol")
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != pin:
        raise ValueError("Protocol changed; refusing to resume")
    if len(rows) > len(selected):
        raise ValueError("Excess predictions")
    for row, query in zip(rows, selected):
        validate_row(row, query)
    write_json(protocol_path, pin)
    write_json(directory / "query_manifest.json", selected)
    for name in (*base.SOURCES, "sparsemm_native/simplestream_recent_share_attention.py"):
        snapshot = directory / "source_snapshot" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes((ROOT / name).read_bytes())
    (directory / "source_snapshot" / SOURCE.name).write_bytes(SOURCE.read_bytes())
    if args.prepare_only or len(rows) == len(selected):
        return

    from sparsemm_native.gpu_slots import gpu_lease
    from sparsemm_native.streaming import load_model
    from sparsemm_protokv.config import StreamingConfig
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    settings = dict(method="sparsemm", budget=K, budget_scope="visual")
    if args.arm == ARMS[1]:
        settings.update(sparsemm_cache_policy="two_bank",
                        sparsemm_budget_path=str(budget_path),
                        sparsemm_two_bank_gate="per_head_capacity")
    config = StreamingConfig(**settings)
    with gpu_lease(directory, concurrent_processes=1) as admission:
        write_json(directory / "gpu_admission.json", admission)
        bridge = load_model(official.MODEL, config, sample_fps=FPS,
                            trace_path=None, device="cuda", dtype="bfloat16",
                            video_max_pixels=None)
        if args.arm == ARMS[1] and (
                bridge.recent_budgets is None or
                bridge.recent_budgets.tolist() != [
                    [250 if (layer, head) in HEADS else 500 for head in range(4)]
                    for layer in range(28)]):
            raise RuntimeError("Runtime two-bank quotas do not match pinned table")
        groups = official.grouped_batches(selected[len(rows):], args.max_groups)
        for group in groups:
            answers = official.run_grouped_video(
                bridge, group, FPS, 16, 16, resize_policy="simplestream_dynamic")
            for query in group:
                response, frames, seconds, encoded = answers[query["query_id"]]
                row = {key: query[key] for key in (
                    "query_id", "source_id", "probe_id", "task", "group",
                    "timestamp", "clip_end", "target")}
                row.update(response=response,
                           correct=official.official_correct(
                               query["task"], response, query["target"]),
                           frames=frames, seconds=seconds,
                           encoded_frames=encoded,
                           resize_hw=bridge._query_resize_hw[query["query_id"]],
                           encoded_visual_tokens=bridge._query_encoded_visual_tokens[
                               query["query_id"]],
                           prompt_sha256=digest(query["prompt"]))
                validate_row(row, query)
                with progress.open("a") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                rows.append(row)
                write_json(directory / "summary.json", official.summary(rows, len(selected)))
                print(f"{args.arm} {len(rows)}/{len(selected)} {query['query_id']} "
                      f"{query['task']} correct={row['correct']} frames={frames} "
                      f"elapsed={seconds:.2f}s", flush=True)


if __name__ == "__main__":
    main()
