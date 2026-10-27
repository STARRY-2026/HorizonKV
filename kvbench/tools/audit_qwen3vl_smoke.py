"""Audit completed Qwen3-VL selector smoke matrices without loading a model."""

import argparse
import json
from pathlib import Path

from sparsemm_baselines.run_streamingbench import write_json


def audit(roots, native_probes=None):
    runs = []
    seen = set()
    for root in roots:
        for folder in sorted(root.glob("*_*bench")):
            result_path = folder / "results.json"
            if not result_path.is_file():
                continue
            result = json.loads(result_path.read_text())
            method, dataset = result["method"], result["dataset"]
            if result["status"] != "passed" or (method, dataset) in seen:
                raise ValueError(f"Invalid or duplicate run: {folder}")
            seen.add((method, dataset))
            if len(result["rows"]) != 1 or not result["rows"][0]["pred_choice"]:
                raise ValueError(f"Missing prediction: {folder}")
            entries = [json.loads(line) for line in (folder / "cache_trace.jsonl").read_text().splitlines()]
            prefills = [entry for entry in entries if entry["event"] == "video_prefill"]
            restored = [entry for entry in entries if entry["event"] == "answer_restored"]
            if len(restored) != 1 or not prefills or restored[0]["head_lengths"] != prefills[-1]["head_lengths"]:
                raise ValueError(f"Question did not restore cache: {folder}")
            if len(prefills) != result["video_prefills"] or sum(p["input_tokens"] for p in prefills) != result["visual_tokens"]:
                raise ValueError(f"Visual input count mismatch: {folder}")
            if len(prefills[-1]["head_lengths"]) != 36 or any(len(heads) != 8 for heads in prefills[-1]["head_lengths"]):
                raise ValueError(f"Wrong Qwen3-VL cache shape: {folder}")
            compressed = sum(any(layer["compressed"] for layer in prefill["layers"]) for prefill in prefills)
            if compressed != result["compression_events"] or (compressed == 0) != (method == "full"):
                raise ValueError(f"Compression event mismatch: {folder}")
            runs.append({"method": method, "dataset": dataset,
                         "visual_tokens": result["visual_tokens"],
                         "grids_thw": [prefill.get("grid_thw") for prefill in prefills],
                         "video_prefills": len(prefills),
                         "compression_events": compressed,
                         "answer_restored": True,
                         "calibration_status": result.get("calibration_status", "not_required")})
    methods = {row["method"] for row in runs}
    if len(runs) != 2 * len(methods) or {row["dataset"] for row in runs} != {"ovobench", "streamingbench"}:
        raise ValueError("Every method must cover both datasets")
    for dataset in ("ovobench", "streamingbench"):
        if len({row["visual_tokens"] for row in runs if row["dataset"] == dataset}) != 1:
            raise ValueError(f"Unequal visual token counts for {dataset}")
        grids = [row["grids_thw"] for row in runs if row["dataset"] == dataset]
        if any(grid != grids[0] for grid in grids) or not grids[0] or any(g is None for g in grids[0]):
            raise ValueError(f"Unequal or missing Qwen3 processor grids for {dataset}")
        if native_probes and dataset in native_probes:
            reference = json.loads(native_probes[dataset].read_text())["results"]
            if len(reference) != len(grids[0]) or [row["grid_thw"] for row in reference] != grids[0]:
                raise ValueError(f"Streaming bridge grid differs from native processor for {dataset}")
    return {"passed": True, "methods": sorted(methods), "runs": runs,
            "limitations": ["Synthetic allocations do not validate offline calibration.",
                            "The Qwen3 streaming bridge omits native timestamp text markers."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-ovo", type=Path)
    parser.add_argument("--native-streaming", type=Path)
    args = parser.parse_args()
    probes = {"ovobench": args.native_ovo, "streamingbench": args.native_streaming}
    report = audit(args.roots, {k: v for k, v in probes.items() if v is not None})
    write_json(args.output, report)
    print(json.dumps({"passed": report["passed"], "methods": report["methods"],
                      "runs": len(report["runs"])}, ensure_ascii=False))
