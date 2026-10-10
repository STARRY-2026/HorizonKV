#!/usr/bin/env python3
"""Test whether Local/Global top-k AUC predicts historical vs recent utility."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path("results/myobservations")
SOURCE = ROOT / "lu_kv_offline_50v_20260913"
RANKS = ROOT / "local_global_topk_50v_20260922/local_global_topk_head_ranks.csv"
OUTPUT = ROOT / "auc_temporal_utility_50v_20260922"
HELDOUT_SOURCE = ROOT / "lu_kv_balanced100v_20260915/traces"
HELDOUT_OUTPUT = ROOT / "auc_temporal_utility_heldout86_20260922"
RECENT = 32
M = 218  # K=250 historical slots


def rank_corr(x, y):
    def ranks(values):
        order = np.argsort(values, kind="stable")
        result = np.empty(len(values), dtype=np.int64)
        result[order] = np.arange(len(values))
        return result
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout86", action="store_true")
    args = parser.parse_args()
    source = HELDOUT_SOURCE if args.heldout86 else SOURCE
    output = HELDOUT_OUTPUT if args.heldout86 else OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    with RANKS.open(newline="") as handle:
        rank_rows = list(csv.DictReader(handle))
    local_auc = np.array([float(row["uniform_local_overlap_auc"]) for row in rank_rows])
    global_auc = np.array([float(row["uniform_global_overlap_auc"]) for row in rank_rows])
    consensus_auc = np.minimum(local_auc, global_auc)
    local_k250 = np.array([float(row["uniform_local_overlap_k250"]) for row in rank_rows])
    global_k250 = np.array([float(row["uniform_global_overlap_k250"]) for row in rank_rows])

    videos = json.loads((source / "manifest.json").read_text())["videos"]
    if args.heldout86:
        calibration_ids = {str(v["video_id"]) for v in
                           json.loads((SOURCE / "manifest.json").read_text())["videos"]}
        videos = [v for v in videos if str(v["video_id"]) not in calibration_ids]
        assert len(videos) == 86
    video_values = []
    for video in videos:
        observations = []
        for question in video["conversations"]:
            stem = f"v{video['video_id']}_q{question['source_question_index']}"
            with np.load(source / f"{stem}.npz") as data:
                utility = data["utility"].astype(np.float64)
                times = data["source_times"].astype(np.float64)
            heads, n = utility.shape
            history = n - RECENT
            if history < M:
                continue
            span = float(times.max() - times.min())
            if span <= 0:
                continue
            current25 = times >= times.max() - .25 * span
            current10 = times >= times.max() - .10 * span
            total = utility.sum(axis=1)
            mass25 = utility[:, current25].sum(axis=1) / np.maximum(total, 1e-12)
            mass10 = utility[:, current10].sum(axis=1) / np.maximum(total, 1e-12)
            recent32 = utility[:, -RECENT:].sum(axis=1) / np.maximum(total, 1e-12)
            oracle = np.argpartition(utility[:, :history], -M, axis=1)[:, -M:]
            top218_current25 = current25[oracle].mean(axis=1)
            top218_current10 = current10[oracle].mean(axis=1)
            recency_overlap = (oracle >= history - M).mean(axis=1)
            observations.append(np.stack([
                mass25, mass10, recent32, top218_current25,
                top218_current10, recency_overlap,
            ]))
        if observations:
            video_values.append(np.mean(observations, axis=0))
    values = np.mean(video_values, axis=0)
    names = ("utility_mass_current25", "utility_mass_current10", "utility_mass_recent32",
             "oracle_top218_current25", "oracle_top218_current10", "recency_top218_overlap_k250")
    metrics = dict(zip(names, values))
    top = np.argsort(-consensus_auc)[:28]
    bottom = np.argsort(consensus_auc)[:28]
    correlations = {}
    groups = {}
    bootstrap_rng = np.random.default_rng(20260922)
    stacked_videos = np.stack(video_values)
    for name, metric in metrics.items():
        correlations[name] = {
            "consensus_auc_spearman": rank_corr(consensus_auc, metric),
            "local_auc_spearman": rank_corr(local_auc, metric),
            "global_auc_spearman": rank_corr(global_auc, metric),
        }
        groups[name] = {
            "high_auc_mean": float(metric[top].mean()),
            "low_auc_mean": float(metric[bottom].mean()),
            "high_minus_low": float(metric[top].mean() - metric[bottom].mean()),
        }
        metric_index = names.index(name)
        per_video_metric = stacked_videos[:, metric_index, :]
        per_video_difference = (
            per_video_metric[:, top].mean(axis=1)
            - per_video_metric[:, bottom].mean(axis=1)
        )
        bootstrap = bootstrap_rng.choice(
            per_video_difference,
            size=(10000, len(per_video_difference)), replace=True,
        ).mean(axis=1)
        groups[name]["high_minus_low_bootstrap95"] = np.percentile(
            bootstrap, [2.5, 97.5]
        ).tolist()
    recency_comparison = {
        "mean_recency_overlap": float(metrics["recency_top218_overlap_k250"].mean()),
    }
    if not args.heldout86:
        recency_wins_local = np.where(metrics["recency_top218_overlap_k250"] > local_k250)[0]
        recency_wins_global = np.where(metrics["recency_top218_overlap_k250"] > global_k250)[0]
        recency_wins_both = np.intersect1d(recency_wins_local, recency_wins_global)
        recency_comparison.update({
            "mean_local_overlap": float(local_k250.mean()),
            "mean_global_overlap": float(global_k250.mean()),
            "heads_recency_beats_local": int(len(recency_wins_local)),
            "heads_recency_beats_global": int(len(recency_wins_global)),
            "heads_recency_beats_both": int(len(recency_wins_both)),
            "heads_recency_beats_both_ids": [[int(h // 4), int(h % 4)] for h in recency_wins_both],
        })
    report = {
        "cohort": "heldout86" if args.heldout86 else "calibration50",
        "videos": len(video_values), "observations": sum(len(v["conversations"]) for v in videos),
        "current_definition": "last 25% / 10% of observed video time, by source_times",
        "topk_definition": "top 218 historical utility tokens, excluding protected recent 32",
        "auc": "minimum of Local/Global top-k overlap AUC",
        "correlations": correlations, "top_bottom_quartile": groups,
        "recency_vs_attention_k250": recency_comparison,
    }
    rows = []
    for h in range(112):
        rows.append({
            "layer": h // 4, "kv_head": h % 4,
            "consensus_topk_auc": float(consensus_auc[h]),
            "local_topk_auc": float(local_auc[h]),
            "global_topk_auc": float(global_auc[h]),
            "local_overlap_k250": float(local_k250[h]),
            "global_overlap_k250": float(global_k250[h]),
            **{name: float(metric[h]) for name, metric in metrics.items()},
        })
    with (output / "head_temporal_utility.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
