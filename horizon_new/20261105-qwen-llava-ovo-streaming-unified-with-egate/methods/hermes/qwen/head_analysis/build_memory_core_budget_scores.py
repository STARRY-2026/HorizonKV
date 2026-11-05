"""Build KV-head budget score CSVs from memory-core profiling results.

Modes:
  --mode long_term_only     (default) existing behavior: use a single long-term metric
  --mode dual               combine long-term + short-term memory scores via max(),
                            so both "memory-readout" and "recent-sensitive" heads
                            receive high budget allocations.

Short-term metric: current_readout_share (attention on the most recently encoded chunk).
"""

import argparse
import os

import numpy as np
import pandas as pd


EPS = 1e-12


def normalize_positive(values, eps=EPS):
    values = np.asarray(values, dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.maximum(values, 0.0)
    if values.max() <= eps:
        return np.ones_like(values, dtype=np.float64)
    return values


def normalize_to_01(values, eps=EPS):
    """Normalize to [0, 1] range, preserving zero at zero."""
    values = np.asarray(values, dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.maximum(values, 0.0)
    vmax = values.max()
    if vmax <= eps:
        return np.zeros_like(values, dtype=np.float64)
    return values / vmax


def write_scores(df, out_path, scores, score_name):
    out = df[["layer", "kv_head"]].copy()
    out["score"] = np.asarray(scores, dtype=np.float64)
    out[score_name] = np.asarray(scores, dtype=np.float64)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path}")


def build_dual_scores(df, long_metric, short_metric):
    """Combine long-term and short-term scores via per-head max.

    Both scores are normalized to [0,1] first so that a head excelling
    in either dimension receives a high combined value.
    """
    long_raw = normalize_positive(df[long_metric].to_numpy())
    short_raw = normalize_positive(df[short_metric].to_numpy())

    long_norm = normalize_to_01(long_raw)
    short_norm = normalize_to_01(short_raw)

    # Max: a head gets boosted if it is strong in EITHER long OR short memory.
    combined = np.maximum(long_norm, short_norm)

    # Scale back to a positive range for proportional budget allocation.
    # Add a small floor so no head gets exactly zero.
    combined = combined + 0.01
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile_csv",
        default="results/observations/effective_memory_readout_core_top100_n4_o80/effective_readout_scores.csv",
    )
    parser.add_argument("--metric", default="internal_top100_mean_attention")
    parser.add_argument(
        "--mode",
        choices=["long_term_only", "dual"],
        default="long_term_only",
        help=(
            "long_term_only: use only --metric (existing behavior). "
            "dual: combine long-term (--metric) + short-term (current_readout_share) via max()"
        ),
    )
    parser.add_argument(
        "--short_metric",
        default="current_readout_share",
        help="Short-term metric column (used when --mode dual)",
    )
    parser.add_argument("--save_dir", default="results/observations/memory_core_budget_scores")
    parser.add_argument("--num_layers", type=int, default=28)
    parser.add_argument("--num_kv_heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df = pd.read_csv(args.profile_csv)
    required = {"layer", "kv_head", args.metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{args.profile_csv} missing columns: {sorted(missing)}")
    df = df.sort_values(["layer", "kv_head"]).reset_index(drop=True)

    expected = int(args.num_layers) * int(args.num_kv_heads)
    if len(df) != expected:
        raise ValueError(f"Expected {expected} layer-KV rows, got {len(df)}")

    os.makedirs(args.save_dir, exist_ok=True)

    if args.mode == "dual":
        # ----------- dual long + short ----------
        if args.short_metric not in df.columns:
            raise ValueError(
                f"--short_metric '{args.short_metric}' not found in {args.profile_csv}. "
                f"Available columns: {sorted(df.columns)}"
            )

        dual_scores = build_dual_scores(df, args.metric, args.short_metric)
        dual_label = f"ours_dual_{args.metric}_{args.short_metric}"

        write_scores(
            df,
            os.path.join(args.save_dir, "ours_dual_top100.csv"),
            dual_scores,
            dual_label,
        )

        # Also dump per-head long/short breakdown for inspection.
        long_raw = normalize_positive(df[args.metric].to_numpy())
        short_raw = normalize_positive(df[args.short_metric].to_numpy())
        debug = df[["layer", "kv_head"]].copy()
        debug["score"] = dual_scores
        debug["long_term_raw"] = long_raw
        debug["short_term_raw"] = short_raw
        debug["long_term_norm01"] = normalize_to_01(long_raw)
        debug["short_term_norm01"] = normalize_to_01(short_raw)
        debug_path = os.path.join(args.save_dir, "ours_dual_top100_debug.csv")
        debug.to_csv(debug_path, index=False)
        print(f"wrote {debug_path} (per-head long/short breakdown)")
        print(f"[dual] long_metric={args.metric}, short_metric={args.short_metric}")
        return

    # ----------- long_term_only (existing behavior) ----------
    scores = normalize_positive(df[args.metric].to_numpy())
    write_scores(df, os.path.join(args.save_dir, "ours_top100.csv"), scores, args.metric)
    write_scores(df, os.path.join(args.save_dir, "uniform.csv"), np.ones_like(scores), args.metric)

    # Invert within each layer so every layer keeps the same score multiset.
    inv = np.zeros_like(scores)
    for layer in range(int(args.num_layers)):
        idx = np.where(df["layer"].to_numpy() == layer)[0]
        vals = scores[idx]
        inv[idx] = vals.max() + vals.min() - vals
    inv = normalize_positive(inv)
    write_scores(df, os.path.join(args.save_dir, "inverted_top100.csv"), inv, args.metric)

    # Random layer-matched control: permute the four KV-head scores within each layer.
    rng = np.random.default_rng(int(args.seed))
    rnd = np.zeros_like(scores)
    for layer in range(int(args.num_layers)):
        idx = np.where(df["layer"].to_numpy() == layer)[0]
        rnd[idx] = rng.permutation(scores[idx])
    write_scores(
        df,
        os.path.join(args.save_dir, f"random_top100_seed{int(args.seed)}.csv"),
        rnd,
        args.metric,
    )


if __name__ == "__main__":
    main()
