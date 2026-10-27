"""Fixed, capped historical KV quotas for the one/four-frame experiment.

The source is the four-frame split inferred from the saved LU-KV calibration.
The total historical quota is pinned to the already published 1.9x proposal;
only its distribution changes.  A monotone capped scaling conserves that
integer total, preserves zero heads, and never caps SparseMM controls.
"""

import csv
from pathlib import Path


SOURCE = (Path(__file__).resolve().parents[2] / "hermes/results/myobservations/"
          "lukv_samecalib_two_bank_k1000_20261022/per_head_allocations.csv")
TOTAL_HISTORY = 108069
CAP = 4000


def capped_apportion(weights, total=TOTAL_HISTORY, cap=CAP):
    """Allocate an exact total as min(cap, scale * weight), then round."""
    if not weights or any(type(w) is not int or w < 0 for w in weights):
        raise ValueError("Expected nonnegative integer historical weights")
    if type(total) is not int or type(cap) is not int or total < 0 or cap < 1:
        raise ValueError("Invalid total or per-head cap")
    if total > cap * sum(w > 0 for w in weights):
        raise ValueError("The cap cannot accommodate the requested total")
    if total == 0:
        return [0] * len(weights)
    lower, upper = 0.0, 1.0
    while sum(min(cap, upper * w) for w in weights) < total:
        upper *= 2
    for _ in range(80):
        mid = (lower + upper) / 2
        if sum(min(cap, mid * w) for w in weights) < total:
            lower = mid
        else:
            upper = mid
    raw = [min(cap, upper * w) for w in weights]
    quotas = [int(value) for value in raw]
    missing = total - sum(quotas)
    for index in sorted(range(len(weights)), key=lambda i: (-(raw[i] - quotas[i]), i)):
        if not missing:
            break
        if weights[index] and quotas[index] < cap:
            quotas[index] += 1
            missing -= 1
    if missing or sum(quotas) != total or any(q > cap or (not w and q) for w, q in zip(weights, quotas)):
        raise AssertionError("Capped apportionment failed to conserve the budget")
    return quotas


def load_source(path=SOURCE):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["frames"]) == 4]
    if len(rows) != 112 or {(int(r["layer"]), int(r["kv_head"])) for r in rows} != {
            (layer, head) for layer in range(28) for head in range(4)}:
        raise ValueError("The source must contain exactly 112 four-frame KV-head rows")
    rows.sort(key=lambda row: (int(row["layer"]), int(row["kv_head"])))
    return rows


def allocation(path=SOURCE, total=TOTAL_HISTORY, cap=CAP):
    rows = load_source(path)
    weights = [int(row["history_h"]) for row in rows]
    quotas = capped_apportion(weights, total, cap)
    ranked = sorted(range(len(rows)), key=lambda i: (
        int(rows[i]["recent_r"]) / int(rows[i]["capacity_n"]), i))
    one_frame = set(ranked[:56])
    return [dict(layer=int(row["layer"]), kv_head=int(row["kv_head"]),
                 recent_group="one_frame" if i in one_frame else "four_frames",
                 recent_share=int(row["recent_r"]) / int(row["capacity_n"]),
                 old_history_h=weights[i], history_quota=quotas[i])
            for i, row in enumerate(rows)]


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_csv(args.output, allocation())
