"""Resumable one/four-frame recent + capped DP-history OVO nine-task run."""

import argparse
import json
import os
from pathlib import Path

from sparsemm_native import capped_history
from sparsemm_native import run_ovobench as official
from sparsemm_native.run_ovobench_recent4 import digest, validate_row
from sparsemm_native.run_ovobench_sparsemm_fps1 import selected_queries, sha
from sparsemm_native.run_ovobench_sparsemm_fps1 import MAX_PIXELS
from sparsemm_native.supervise_ovobench_dual import write_json


ARM = "recent1or4_history_cap4000"
ROOT = official.ROOT
SOURCES = ("sparsemm_native/run_ovobench_recent1or4_cap4000.py",
           "sparsemm_native/simplestream_resize.py",
           "sparsemm_native/recent1or4_history.py", "sparsemm_native/capped_history.py",
           "sparsemm_native/run_ovobench_sparsemm_fps1.py",
           "sparsemm_native/run_ovobench.py", "sparsemm_native/cache.py",
           "sparsemm_native/two_bank.py", "sparsemm_native/decoupled.py",
           "sparsemm_native/policy.py", "sparsemm_protokv/config.py",
           "sparsemm_protokv/cache.py", "sparsemm_protokv/streaming.py",
           "sparsemm_protokv/monkeypatch.py", "sparsemm_protokv/model_loader.py",
           "sparsemm_baselines/core.py", "sparsemm_baselines/adapter.py")


def protocol(selected, expected):
    from sparsemm_native.policy import DEFAULT_SCORE_PATH
    rows = capped_history.allocation()
    return dict(version="ovo_nine_recent1or4_history_cap4000_v1", arm=ARM,
                model=str(official.MODEL.resolve()), model_config_sha256=sha(official.MODEL / "config.json"),
                annotation=str(official.ANNOTATIONS.resolve()), annotation_sha256=sha(official.ANNOTATIONS),
                query_count=len(selected), full_nine_query_count=len(expected),
                query_ids=[q["query_id"] for q in selected], query_sha256=digest(selected),
                full_nine_query_sha256=digest(expected), fps=1.0,
                visual_budget_definition="dataset_average_recent_slots_plus_fixed_history_quotas",
                configured_history_visual_slots=sum(r["history_quota"] for r in rows),
                history_head_cap=4000, history_quota_source=str(capped_history.SOURCE),
                history_quota_source_sha256=sha(capped_history.SOURCE),
                one_frame_heads=[[r["layer"], r["kv_head"]] for r in rows if r["recent_group"] == "one_frame"],
                per_head_allocation=rows, protect_window=32,
                selector="existing_SparseMM_online_attention_for_history",
                sparsemm_score_path=str(DEFAULT_SCORE_PATH.resolve()), sparsemm_score_sha256=sha(DEFAULT_SCORE_PATH),
                dtype="bfloat16", video_max_pixels=MAX_PIXELS, max_new_tokens=16,
                greedy=True, repetition_penalty=1.1,
                score_mode="official_OVOBenchOfflineScore", grouped_by_source_video=True,
                same_video_prefix_reuse=True,
                video_preprocessing="qwen_vl_utils_0.0.14_formula_shared_longest_query_prefix_per_source_video",
                trace_enabled=False,
                image_unit="each_sampled_frame_as_independent_Qwen_image_features",
                recent_prefill="independent_one_image_or_four_image_prefix_from_common_text",
                code_sha256={name: sha(ROOT / name) for name in SOURCES})


def answer_grouped_dynamic(bridge, query, max_new_tokens, encoded_frames,
                           debug_explain=False, debug_max_new_tokens=96):
    """The legacy answer path with a dynamic-quota selector audit."""
    if debug_explain:
        raise ValueError("Explanation mode is disabled for the paired experiment")
    prompt = {"question": query.get("question", query["prompt"]),
              "formatted_question": query["prompt"],
              "prompt": bridge.get_prompt(query["prompt"], mc=query.get("mc", False))}
    response = str(bridge.question_answering(prompt, max_new_tokens=max_new_tokens))
    checks = bridge.validation_checks
    selectors, qa = checks["selector_checks"], checks["qa_checks"]
    if (len(qa) != 1 or qa[0].get("returned") is not True or qa[0].get("restored") is not True
            or checks["failures"] or (encoded_frames and not selectors)
            or any(check.get("validation_passed") is not True for check in selectors)):
        raise RuntimeError(f"Dynamic capped-history cache audit failed for {query['query_id']}")
    bridge._query_budget_audits[query["query_id"]] = dict(bridge.last_question_metrics)
    selectors.clear()
    qa.clear()
    return response


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
        raise ValueError("Code, model, calibration or query set changed; refusing to resume")
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
    from sparsemm_native.recent1or4_history import load_model
    from sparsemm_protokv.config import StreamingConfig
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    config = StreamingConfig(method="sparsemm", budget=2576, budget_scope="visual",
                             sparsemm_budget_path=str(capped_history.SOURCE),
                             sparsemm_cache_policy="two_bank",
                             sparsemm_two_bank_gate="per_head_capacity")
    official._answer_grouped = answer_grouped_dynamic
    with gpu_lease(directory, concurrent_processes=1) as admission:
        write_json(directory / "gpu_admission.json", admission)
        bridge = load_model(official.MODEL, config, sample_fps=1.0,
                            trace_path=None, device="cuda",
                            dtype="bfloat16", video_max_pixels=MAX_PIXELS)
        bridge._query_budget_audits = {}
        groups = official.grouped_batches(selected[len(rows):], args.max_groups)
        for group in groups:
            answers = official.run_grouped_video(
                bridge, group, 1.0, 16, 16, resize_policy="simplestream_video_constant")
            for query in group:
                response, frames, seconds, encoded = answers[query["query_id"]]
                row = {key: query[key] for key in (
                    "query_id", "source_id", "probe_id", "task", "group", "timestamp", "clip_end", "target")}
                row.update(response=response, correct=official.official_correct(query["task"], response, query["target"]),
                           frames=frames, seconds=seconds, encoded_frames=encoded,
                           resize_hw=bridge._query_resize_hw[query["query_id"]],
                           resize_reference_frames=bridge._query_resize_reference_frames[query["query_id"]],
                           encoded_visual_tokens=bridge._query_encoded_visual_tokens[query["query_id"]],
                           cache_budget_audit=bridge._query_budget_audits.pop(query["query_id"]),
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
