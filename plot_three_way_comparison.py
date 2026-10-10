"""Plot 3-way comparison: Dual, Ours (n4_o80), and new profiling (n50_o5000).

All scores normalized via min-max for overlay comparison.
Top-10 heads are listed for each.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

base = "/opt/data/private/yujun/myprogram/hermes"

# ---- Load data ----
path_dual  = f"{base}/results/observations/memory_core_budget_scores_dual/ours_dual_top100.csv"
path_ours  = f"{base}/results/observations/memory_core_budget_scores/ours_top100.csv"
path_new   = f"{base}/results/myobservations/effective_memory_readout_core_top100_n50_o5000/effective_readout_scores.csv"

df_dual = pd.read_csv(path_dual)
df_ours = pd.read_csv(path_ours)
df_new  = pd.read_csv(path_new)

# The new file doesn't have a "score" column; use internal_top100_mean_attention (same as ours)
df_new["score"] = df_new["internal_top100_mean_attention"]

# Global head index
for df in [df_dual, df_ours, df_new]:
    df["head_idx"] = df["layer"] * 4 + df["kv_head"]

# Normalize
def normalize(series):
    s = series.astype(float)
    return (s - s.min()) / (s.max() - s.min())

df_dual["score_norm"]  = normalize(df_dual["score"])
df_ours["score_norm"]  = normalize(df_ours["score"])
df_new["score_norm"]   = normalize(df_new["score"])

# Top 10
top10_dual = df_dual.nlargest(10, "score")[["layer", "kv_head", "score", "head_idx"]].reset_index(drop=True)
top10_ours = df_ours.nlargest(10, "score")[["layer", "kv_head", "score", "head_idx"]].reset_index(drop=True)
top10_new  = df_new.nlargest(10, "score")[["layer", "kv_head", "score", "head_idx"]].reset_index(drop=True)

# Identify overlaps
dual_set = set(zip(top10_dual["layer"], top10_dual["kv_head"]))
ours_set = set(zip(top10_ours["layer"], top10_ours["kv_head"]))
new_set  = set(zip(top10_new["layer"],  top10_new["kv_head"]))

def print_top10(df, name, all_sets, my_set):
    print(f"\n{'='*70}")
    print(f"  {name} Top 10 Heads")
    print(f"{'-'*70}")
    for i, row in df.iterrows():
        tags = []
        for other_name, other_set in all_sets:
            if (row["layer"], row["kv_head"]) in other_set:
                tags.append(other_name)
        tag_str = f"  [also in: {', '.join(tags)}]" if tags else ""
        print(f"  Rank {i+1:2d}: L{int(row['layer']):2d} H{int(row['kv_head'])}  "
              f"score={row['score']:.6e}  idx={int(row['head_idx'])}{tag_str}")

all_sets = [("Dual", dual_set), ("Ours", ours_set)]
print_top10(top10_new,  "New  (n50_o5000)", all_sets, new_set)

all_sets = [("Dual", dual_set), ("New",  new_set)]
print_top10(top10_ours, "Ours (n4_o80)",    all_sets, ours_set)

all_sets = [("Ours", ours_set), ("New",  new_set)]
print_top10(top10_dual, "Dual",             all_sets, dual_set)

# ---- Summary ----
print(f"\n{'='*70}")
print("  Overlap Summary")
print(f"{'-'*70}")
for a_name, a_set in [("New", new_set), ("Ours", ours_set), ("Dual", dual_set)]:
    for b_name, b_set in [("New", new_set), ("Ours", ours_set), ("Dual", dual_set)]:
        if a_name >= b_name: continue
        both = a_set & b_set
        print(f"  {a_name} ∩ {b_name}: {len(both)} heads: {sorted(both)}")

# ---- Plot ----
fig, ax = plt.subplots(1, 1, figsize=(18, 8))

ax.plot(df_dual["head_idx"], df_dual["score_norm"],
        'o-', color='#2196F3', markersize=8, linewidth=1.5,
        label='Dual (long+short max)', alpha=0.8)
ax.plot(df_ours["head_idx"], df_ours["score_norm"],
        's--', color='#FF5722', markersize=8, linewidth=1.5,
        label='Ours n4_o80 (long-term only)', alpha=0.8)
ax.plot(df_new["head_idx"], df_new["score_norm"],
        '^-.', color='#4CAF50', markersize=8, linewidth=1.5,
        label='New n50_o5000 (long-term only)', alpha=0.8)

# Highlight Top 10
for df, top10, color, marker, label in [
    (df_dual, top10_dual, '#2196F3', 'D', 'Dual Top 10'),
    (df_ours, top10_ours, '#FF5722', 's', 'Ours Top 10'),
    (df_new,  top10_new,  '#4CAF50', '^', 'New Top 10'),
]:
    top_idxs = top10["head_idx"].values
    ax.scatter(top_idxs, df[df["head_idx"].isin(top_idxs)]["score_norm"],
               color=color, edgecolors='black', s=150, zorder=5, linewidths=2,
               marker=marker, label=label)

ax.set_xlabel('Attention Head Index (layer * 4 + kv_head)', fontsize=14)
ax.set_ylabel('Normalized Score (min-max)', fontsize=14)
ax.set_title('Memory Core Budget Scores: Dual vs Ours (n4_o80) vs New (n50_o5000)', fontsize=16)
ax.legend(fontsize=12, loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path = f"{base}/results/observations/three_way_comparison.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nFigure saved to: {output_path}")
plt.close()
