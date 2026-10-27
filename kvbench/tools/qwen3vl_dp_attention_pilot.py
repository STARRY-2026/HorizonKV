"""Exact K500 two-bank DP adaptation from real Qwen3 full-cache LU observations.

This deliberately changes the historical ranking to pure attention. It is a
Qwen3 adaptation pilot, not a replay of the old HERMES-ranked DP calibration.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from sparsemm_baselines.run_streamingbench import write_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def curves_for_observation(ranking, utility, floor, cap, recent_min):
    heads, tokens = utility.shape
    if ranking.shape != (heads, tokens - recent_min):
        raise ValueError("Rankings must cover all visual history before the protected window")
    if not np.isfinite(utility).all() or (utility < 0).any() or utility.sum() <= 0:
        raise ValueError("Projected utility must be finite, nonnegative and nonzero")
    values = np.full((heads, cap + 1), -np.inf)
    endpoints = np.full((heads, cap + 1), -1, dtype=np.int16)
    denominator = float(utility.sum())
    for head in range(heads):
        row = utility[head]
        order = ranking[head]
        if not np.array_equal(np.sort(order), np.arange(tokens - recent_min)):
            raise ValueError("Invalid pure-attention ranking permutation")
        recent_cumsum = np.r_[0., np.cumsum(row[::-1])]
        for recent in range(recent_min, cap + 1):
            eligible = order[order < tokens - recent]
            history_cumsum = np.r_[0., np.cumsum(row[eligible])]
            maximum_capacity = min(cap, recent + len(eligible))
            capacities = np.arange(max(floor, recent), maximum_capacity + 1)
            if not len(capacities):
                continue
            candidate = (recent_cumsum[recent] + history_cumsum[capacities - recent]) / denominator
            better = candidate > values[head, capacities]
            values[head, capacities[better]] = candidate[better]
            endpoints[head, capacities[better]] = recent
    if not np.isfinite(values[:, floor:cap + 1]).all():
        raise RuntimeError("DP pilot did not cover every feasible head capacity")
    return values, endpoints


def shared_split_curves(observations, floor, cap, recent_min):
    """Average real-video utilities before optimizing one deployed split."""
    heads = observations[0][2].shape[0]
    values = np.full((heads, cap + 1), -np.inf)
    endpoints = np.full((heads, cap + 1), -1, dtype=np.int16)
    denominators = [float(utility.sum()) for _, _, utility in observations]
    for head in range(heads):
        for recent in range(recent_min, cap + 1):
            capacities = np.arange(max(floor, recent), cap + 1)
            if not len(capacities):
                continue
            combined = np.zeros(len(capacities), dtype=np.float64)
            for (_, ranking, utility), denominator in zip(observations, denominators):
                row = utility[head]
                tokens = len(row)
                eligible = ranking[head][ranking[head] < tokens - recent]
                history = np.r_[0., np.cumsum(row[eligible])]
                if capacities[-1] - recent > len(eligible):
                    raise ValueError("The shared DP capacity exceeds one observed prefix")
                combined += (row[-recent:].sum() + history[capacities - recent]) / denominator
            combined /= len(observations)
            better = combined > values[head, capacities]
            values[head, capacities[better]] = combined[better]
            endpoints[head, capacities[better]] = recent
    return values, endpoints


def exact_allocation(values, average, floor, cap):
    heads = values.shape[0]
    total = heads * average
    extra_total = total - heads * floor
    max_extra = cap - floor
    if not 0 <= extra_total <= heads * max_extra:
        raise ValueError("Infeasible exact DP token budget")
    previous = np.full(extra_total + 1, -np.inf)
    previous[0] = 0.
    back = np.full((heads, extra_total + 1), -1, dtype=np.int16)
    previous_lo = previous_hi = 0
    for head in range(heads):
        processed = head + 1
        remaining = heads - processed
        lo = max(0, extra_total - remaining * max_extra)
        hi = min(extra_total, processed * max_extra)
        current = np.full_like(previous, -np.inf)
        for extra in range(max_extra + 1):
            lower = max(lo, previous_lo + extra)
            upper = min(hi, previous_hi + extra)
            if lower > upper:
                continue
            candidate = previous[lower - extra:upper - extra + 1] + values[head, floor + extra]
            section = current[lower:upper + 1]
            better = candidate > section
            section[better] = candidate[better]
            back[head, lower:upper + 1][better] = extra
        previous, previous_lo, previous_hi = current, lo, hi
        if processed % 32 == 0 or processed == heads:
            print(f"QWEN3_DP {processed}/{heads}", flush=True)
    if not np.isfinite(previous[extra_total]):
        raise RuntimeError("No exact DP solution")
    allocation = np.empty(heads, dtype=np.int16)
    state = extra_total
    for head in range(heads - 1, -1, -1):
        extra = int(back[head, state])
        if extra < 0:
            raise RuntimeError("Broken exact DP traceback")
        allocation[head] = floor + extra
        state -= extra
    if state or allocation.sum() != total or allocation.min() < floor or allocation.max() > cap:
        raise RuntimeError("Exact DP violated capacity constraints")
    return allocation, float(previous[extra_total])


def main(args):
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(out)
    source_manifest = json.loads(args.calibration_manifest.read_text())
    if source_manifest["status"] not in ("complete_pilot", "complete_formal") or not source_manifest["observations"]:
        raise ValueError("This allocation requires completed real Qwen3 LU observations")
    observations = []
    for record in source_manifest["observations"]:
        if record["visual_tokens"] < args.min_visual_tokens:
            continue
        source = Path(record["path"])
        if digest(source) != record["sha256"]:
            raise ValueError("LU observation SHA-256 differs from its manifest")
        with np.load(source, allow_pickle=False) as data:
            observations.append((record, data["ranking"].astype(np.int64),
                                 data["utility_projected"].astype(np.float64)))
    if not observations:
        raise ValueError("No Qwen3 observations satisfy the minimum visual prefix")
    heads = observations[0][2].shape[0]
    if heads != 36 * 8 or args.average != 500 or args.floor != 375:
        raise ValueError("Qwen3-VL-8B DP pilot expects 36x8 heads and the established K500/floor375")
    cap = min(args.cap, *(utility.shape[1] for _, _, utility in observations))
    out.mkdir(parents=True)
    started = time.perf_counter()
    if len(observations) > 1:
        values, endpoints = shared_split_curves(observations, args.floor, cap, args.recent_min)
    else:
        values, endpoints = curves_for_observation(observations[0][1], observations[0][2],
                                                    args.floor, cap, args.recent_min)
    allocation, objective = exact_allocation(values, args.average, args.floor, cap)
    selected_recent = endpoints[np.arange(heads), allocation]
    selected_history = allocation - selected_recent
    if (selected_recent < args.recent_min).any() or (selected_history < 0).any():
        raise RuntimeError("Invalid DP two-bank split")
    with (out / "per_head_allocation_attention_pilot_k500.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("head_id", "layer", "kv_head", "capacity_n", "recent_r", "history_q",
                         "history_share", "selected_proxy"))
        for head in range(heads):
            cap_n, recent = int(allocation[head]), int(selected_recent[head])
            writer.writerow((head, head // 8, head % 8, cap_n, recent, cap_n - recent,
                             (cap_n - recent) / cap_n, float(values[head, cap_n])))
    formal = source_manifest["status"] == "complete_formal"
    manifest = {"status": "complete_formal" if formal else "complete_pilot", "model": source_manifest["model"],
                "code_sha256": digest(__file__),
                "source_observations": [{"path": record["path"], "sha256": record["sha256"],
                                         "video_id": record["video_id"]} for record, _, _ in observations],
                "calibration_status": "formal_attention_ranked_DP_adaptation_not_legacy_HERMES_ranked" if formal else "bounded_attention_ranked_DP_adaptation_not_legacy_HERMES_ranked",
                "objective": "sum of recent projected utility plus top-attention-ranked history projected utility",
                "average": args.average, "head_floor": args.floor, "head_cap": cap,
                "minimum_visual_tokens": args.min_visual_tokens,
                "recent_min": args.recent_min, "total_slots": int(allocation.sum()),
                "capacity_min": int(allocation.min()), "capacity_max": int(allocation.max()),
                "recent_total": int(selected_recent.sum()), "history_total": int(selected_history.sum()),
                "objective_value": objective, "seconds": round(time.perf_counter() - started, 3),
                "finished_utc": datetime.now(timezone.utc).isoformat(), "full_benchmarks_paused": True}
    write_json(out / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "slots": manifest["total_slots"],
                      "objective": objective, "seconds": manifest["seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--average", type=int, default=500)
    parser.add_argument("--floor", type=int, default=375)
    parser.add_argument("--cap", type=int, default=750)
    parser.add_argument("--recent-min", type=int, default=32)
    parser.add_argument("--min-visual-tokens", type=int, default=33)
    main(parser.parse_args())
