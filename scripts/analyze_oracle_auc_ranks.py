#!/usr/bin/env python3
"""Compare per-head oracle retention AUC with the saved selector AUCs."""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_uniform_selector_auc import COMPRESSION, POLICIES, _coverage


ROOT = Path("results/myobservations/uniform_selector_auc_50v_20260920")
SOURCE = Path("results/myobservations/lu_kv_offline_50v_20260913")
RECENT = 32


def oracle_curves():
    by_video = defaultdict(list)
    for observed in sorted(ROOT.glob("v*_q*.npz")):
        metadata = json.loads((ROOT / f"{observed.stem}.json").read_text())
        with np.load(SOURCE / f"{observed.stem}.npz") as data:
            utility = data["utility"].astype(np.float64)
        history = max(0, utility.shape[1] - RECENT)
        ranking = np.argsort(-utility[:, :history], axis=1, kind="stable")
        by_video[str(metadata["video_id"])].append(
            _coverage(utility, ranking, RECENT)
        )
    if len(by_video) != 50 or sum(map(len, by_video.values())) != 97:
        raise RuntimeError("Expected the completed 50-video, 97-observation experiment")
    video_curves = [np.nanmean(np.stack(parts), axis=0)
                    for parts in by_video.values()]
    return np.nanmean(np.stack(video_curves), axis=0)


def main():
    curves = oracle_curves()
    oracle_auc = np.trapezoid(np.nan_to_num(curves, nan=0.0), COMPRESSION, axis=1)
    original = pd.read_csv(ROOT / "uniform_selector_head_auc.csv")
    table = original.pivot(index=["layer", "kv_head"], columns="policy", values="auc")
    table = table[list(POLICIES)].copy()
    table.insert(0, "oracle_auc", oracle_auc)
    for name in ["oracle_auc", *POLICIES]:
        table[f"{name}_rank"] = table[name].rank(ascending=False, method="min").astype(int)
    for name in POLICIES:
        table[f"{name}_rank_gap"] = (
            table[f"{name}_rank"] - table["oracle_auc_rank"]
        ).abs()
        table[f"{name}_auc_gap"] = table["oracle_auc"] - table[name]
    attention = ["uniform_local", "uniform_global", "uniform_joint"]
    table["max_attention_rank_gap"] = table[
        [f"{name}_rank_gap" for name in attention]
    ].max(axis=1)
    table = table.reset_index()
    table.to_csv(ROOT / "oracle_auc_head_ranks.csv", index=False, float_format="%.10f")

    bins = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    counts, _ = np.histogram(oracle_auc, bins=bins)
    summary = {
        "videos": 50,
        "observations": 97,
        "heads": len(table),
        "recent": RECENT,
        "oracle_auc_mean": float(oracle_auc.mean()),
        "oracle_auc_std": float(oracle_auc.std()),
        "oracle_auc_quantiles": {
            str(q): float(np.quantile(oracle_auc, q))
            for q in (0, .1, .25, .5, .75, .9, 1)
        },
        "oracle_auc_histogram": [
            {"lower": lo, "upper": hi, "heads": int(count)}
            for lo, hi, count in zip(bins[:-1], bins[1:], counts)
        ],
        "rank_gap_counts": {
            name: {str(t): int((table[f"{name}_rank_gap"] <= t).sum())
                   for t in (5, 10, 20)}
            for name in POLICIES
        },
        "all_attention_within": {
            str(t): int((table["max_attention_rank_gap"] <= t).sum())
            for t in (5, 10, 20)
        },
    }
    (ROOT / "oracle_auc_rank_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    print("\nAll Local/Global/Joint rank gaps <= 10:")
    cols = ["layer", "kv_head", "oracle_auc", "oracle_auc_rank",
            "uniform_local_rank", "uniform_global_rank", "uniform_joint_rank"]
    print(table.loc[table.max_attention_rank_gap <= 10, cols]
          .sort_values("oracle_auc_rank").to_string(index=False))


if __name__ == "__main__":
    main()
