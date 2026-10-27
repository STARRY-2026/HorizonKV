"""Bounded GPU validation of recent-four quotas and multi-question prefix reuse."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    from sparsemm_native import run_ovobench as legacy
    from sparsemm_native import run_ovobench_recent4 as runner
    from sparsemm_native.recent4 import load_model
    from sparsemm_baselines.run_streamingbench import gpu_lease, write_json
    from sparsemm_protokv.config import StreamingConfig
    import torch

    ids = "9,10,104,105,34,433,530,774,966,1054,1117,1245,1454,1492_0,1492_1,1518_0,1518_1,1592_0,1592_1"
    queries = runner.select_queries(legacy.queries(), ids)
    scheduled = runner.schedule_queries(queries)
    config = StreamingConfig(
        method="sparsemm", budget=500, budget_scope="visual",
        sparsemm_budget_path=str(legacy.DP_DUAL_BUDGET),
        sparsemm_cache_policy="two_bank", sparsemm_two_bank_gate="legacy_global")
    allocations, counts = {}, Counter()
    with gpu_lease(args.output):
        bridge = load_model(legacy.MODEL, config, sample_fps=0.5, dtype="bfloat16",
                            video_max_pixels=151200, trace_path=args.output / "cache_trace.jsonl")
        original = bridge.forward_chunk

        def checked_forward(inputs_embeds, positions, compress=True, **kwargs):
            hidden = original(inputs_embeds, positions, compress=compress, **kwargs)
            if compress:
                cache = bridge.kv_cache
                snapshot = dict(cache.allocation)
                key = ",".join(map(str, snapshot["recent_unit_token_counts"]))
                allocations[key] = snapshot
                for layer in cache.last_metadata["layers"]:
                    counts["layer_updates"] += 1
                    counts["compressed_layer_updates"] += int(layer["compressed"])
                    counts["warmup_layer_updates"] += int(layer["reference_warmup_bypass"])
                    if not all(layer["position_checks"].values()):
                        raise AssertionError("Recent token identities/positions were not preserved")
                if int(cache.budgets.sum()) != 56000:
                    raise AssertionError("Global mean budget drifted")
            return hidden

        bridge.forward_chunk = checked_forward
        results = {}
        for mode in ("grouped", "fresh"):
            started = time.monotonic()
            rows = []
            batches = (legacy.grouped_batches(scheduled, len(scheduled)) if mode == "grouped"
                       else [[query] for query in queries])
            for group in batches:
                answers = runner.run_grouped_video(bridge, group)
                for query in group:
                    response, frames, seconds, encoded = answers[query["query_id"]]
                    row = dict(query_id=query["query_id"], task=query["task"], response=response,
                               target=query["target"], correct=legacy.official_correct(
                                   query["task"], response, query["target"]), frames=frames,
                               encoded_frames=encoded, seconds=seconds)
                    rows.append(row)
                    print(f"{mode} {row['query_id']} {row['task']} answer={response!r} "
                          f"frames={frames} encoded={encoded}", flush=True)
            results[mode] = dict(rows=rows, seconds=time.monotonic() - started,
                                 encoded_frames=sum(r["encoded_frames"] for r in rows),
                                 independent_prefix_frames=sum(r["frames"] for r in rows))
        grouped = {r["query_id"]: r for r in results["grouped"]["rows"]}
        fresh = {r["query_id"]: r for r in results["fresh"]["rows"]}
        differences = {qid: [grouped[qid]["response"], fresh[qid]["response"]]
                       for qid in grouped if grouped[qid]["response"] != fresh[qid]["response"]
                       or grouped[qid]["frames"] != fresh[qid]["frames"]}
        report = dict(
            query_count=len(queries), tasks=dict(Counter(q["task"] for q in queries)),
            all_answers_and_frame_counts_identical=not differences, differences=differences,
            allocation_examples=allocations, runtime_checks=dict(counts), results=results,
            peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
            code_sha256={name: legacy.sha256(runner.ROOT / name) for name in runner.SOURCE_FILES},
            history_weight_sha256=legacy.sha256(legacy.DP_DUAL_BUDGET),
            note="Integration validation only: selected short queries, not an accuracy estimate.",
        )
        write_json(args.output / "validation.json", report)
        if differences:
            raise AssertionError(f"Grouped/fresh predictions differ: {differences}")
        print(json.dumps({"queries": len(queries), "tasks": len(report["tasks"]),
                          "identical": True, "checks": dict(counts)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
