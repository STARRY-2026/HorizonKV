#!/usr/bin/env python3
"""Export K800 runtime inputs from its already-frozen exact DP allocation.

This does not fit DP on evaluation data. It verifies the same deterministic
mapping against the previously exported K100/250/500/1000 runtime tables.
"""
import argparse
import csv
import json
from pathlib import Path

from run_experiment import ROOT, read_heads, sha256, sparsemm_budgets


SOURCES = {
    100: ("two_bank_grid_dp_k100_floor75_20260929", "two_bank_grid_dp_k100_floor75_20260929", "capacity_n"),
    250: ("two_bank_grid_dp_k250_floor188_20260923", "k250_cachecomp_score_pilot_20260923", "capacity_n"),
    500: ("two_bank_grid_dp_k500_floor375_20260925", "two_bank_grid_dp_k500_floor375_20260925", "capacity_n"),
    800: ("two_bank_grid_dp_k800_floor600_20260928", "two_bank_grid_dp_k800_floor600_20260928", "capacity_n"),
    1000: ("two_bank_grid_dp_floor750_20260923", "k1000_dp_cachecomp_score_20260924", "floor750_n"),
}


def derive(budget):
    source_dir, output_dir, capacity_column = SOURCES[budget]
    root = ROOT / "results/myobservations"
    source = root / source_dir / "per_head_allocation.csv"
    rows = read_heads(source, [capacity_column, "recent_r", "history_q"])
    minimum = int((budget - 32) * 0.2) + 32
    scores, profiles = [], []
    for row in rows:
        capacity, recent, history = (int(row[c]) for c in (capacity_column, "recent_r", "history_q"))
        if recent < 0 or history < 0 or recent + history != capacity:
            raise ValueError("Frozen allocation has inconsistent banks")
        identity = {"layer": int(row["layer"]), "kv_head": int(row["kv_head"])}
        scores.append({**identity, "score": capacity - minimum})
        profiles.append({**identity, "recent_ratio": recent / capacity,
                         "capacity_n": capacity, "recent_r": recent, "history_q": history})
    actual = sparsemm_budgets(scores, budget)
    if actual != [row["capacity_n"] for row in profiles] or sum(actual) != budget * 112:
        raise ValueError("Export does not reproduce exact frozen allocation")
    return source, root / output_dir, scores, profiles


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--write-k800", action="store_true", help="Write deterministically derived K800 runtime tables")
    args = p.parse_args()
    checks = []
    for budget in SOURCES:
        source, destination, scores, profiles = derive(budget)
        if budget != 800:
            old_scores = read_heads(destination / f"dp_k{budget}_exact_budget_scores.csv", ["score"])
            old_profiles = read_heads(destination / "recent_ratio_profile.csv", ["recent_ratio", "capacity_n", "recent_r", "history_q"])
            for old, new in zip(old_scores, scores):
                if float(old["score"]) != new["score"]:
                    raise ValueError(f"K{budget}: historical runtime score differs")
            for old, new in zip(old_profiles, profiles):
                for field in ("recent_ratio", "capacity_n", "recent_r", "history_q"):
                    if abs(float(old[field]) - new[field]) > 1e-12:
                        raise ValueError(f"K{budget}: historical recency profile differs")
        checks.append({"budget": budget, "source": str(source.relative_to(ROOT)),
                       "source_sha256": sha256(source), "exact_runtime_total": budget * 112,
                       "historical_export_matches": budget != 800})
        if budget == 800 and args.write_k800:
            for name, rows in (("dp_k800_exact_budget_scores.csv", scores), ("recent_ratio_profile.csv", profiles)):
                with (destination / name).open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
    provenance = {"generator": "tools/export_frozen_dp_profiles.py --write-k800",
                  "formula": "score = capacity_n - (int((K-32)*0.2)+32); recent_ratio=recent_r/capacity_n",
                  "note": "K800 derived from frozen allocation; no new DP fit or inference has been performed.",
                  "cross_checks": checks}
    if args.write_k800:
        destination = ROOT / "results/myobservations" / SOURCES[800][1]
        (destination / "runtime_export_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
