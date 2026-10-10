import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Read data
base = "/opt/data/private/yujun/myprogram/hermes"
dual_path = f"{base}/results/observations/memory_core_budget_scores_dual/ours_dual_top100.csv"
ours_path = f"{base}/results/observations/memory_core_budget_scores/ours_top100.csv"

df_dual = pd.read_csv(dual_path)
df_ours = pd.read_csv(ours_path)

# Compute global head index: layer * 4 + kv_head
df_dual["head_idx"] = df_dual["layer"] * 4 + df_dual["kv_head"]
df_ours["head_idx"] = df_ours["layer"] * 4 + df_ours["kv_head"]

# Normalize scores using min-max normalization
def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

df_dual["score_norm"] = normalize(df_dual["score"])
df_ours["score_norm"] = normalize(df_ours["score"])

# Top 10 heads by score
top10_dual = df_dual.nlargest(10, "score")[["layer", "kv_head", "score", "head_idx"]].reset_index(drop=True)
top10_ours = df_ours.nlargest(10, "score")[["layer", "kv_head", "score", "head_idx"]].reset_index(drop=True)

# Identify heads selected by both, only dual, only ours
dual_heads_set = set(zip(top10_dual["layer"], top10_dual["kv_head"]))
ours_heads_set = set(zip(top10_ours["layer"], top10_ours["kv_head"]))
both = dual_heads_set & ours_heads_set
only_dual = dual_heads_set - ours_heads_set
only_ours = ours_heads_set - dual_heads_set

print("=" * 70)
print("Dual Top 10 Heads (by raw score):")
print("-" * 70)
for i, row in top10_dual.iterrows():
    marker = ""
    if (row["layer"], row["kv_head"]) in both:
        marker = "  [BOTH]"
    print(f"  Rank {i+1:2d}: Layer {int(row['layer']):2d}, Head {int(row['kv_head'])}  "
          f"score={row['score']:.6f}  head_idx={int(row['head_idx'])}{marker}")

print()
print("=" * 70)
print("Ours Top 10 Heads (by raw score):")
print("-" * 70)
for i, row in top10_ours.iterrows():
    marker = ""
    if (row["layer"], row["kv_head"]) in both:
        marker = "  [BOTH]"
    print(f"  Rank {i+1:2d}: Layer {int(row['layer']):2d}, Head {int(row['kv_head'])}  "
          f"score={row['score']:.6f}  head_idx={int(row['head_idx'])}{marker}")

print()
print("=" * 70)
print("Comparison Summary:")
print("-" * 70)
print(f"  Heads in both top 10:               {len(both)}")
print(f"  Only in Dual top 10:                {len(only_dual)}")
print(f"  Only in Ours top 10:                {len(only_ours)}")

if only_dual:
    print(f"\n  Only in Dual:  {sorted(only_dual, key=lambda x: df_dual[(df_dual.layer==x[0])&(df_dual.kv_head==x[1])].score.values[0], reverse=True)}")
if only_ours:
    print(f"\n  Only in Ours:  {sorted(only_ours, key=lambda x: df_ours[(df_ours.layer==x[0])&(df_ours.kv_head==x[1])].score.values[0], reverse=True)}")

# ---- Plot ----
fig, ax1 = plt.subplots(1, 1, figsize=(18, 8))

# Normalized scores vs head index
ax1.plot(df_dual["head_idx"], df_dual["score_norm"], 'o-', color='#2196F3', markersize=8, linewidth=1.5, label='Dual', alpha=0.8)
ax1.plot(df_ours["head_idx"], df_ours["score_norm"], 's--', color='#FF5722', markersize=8, linewidth=1.5, label='Ours', alpha=0.8)

# Highlight top 10 for each
top10_dual_idxs = top10_dual["head_idx"].values
top10_ours_idxs = top10_ours["head_idx"].values

ax1.scatter(top10_dual_idxs, df_dual[df_dual["head_idx"].isin(top10_dual_idxs)]["score_norm"],
            color='#2196F3', edgecolors='black', s=150, zorder=5, linewidths=2, marker='D', label='Dual Top 10')
ax1.scatter(top10_ours_idxs, df_ours[df_ours["head_idx"].isin(top10_ours_idxs)]["score_norm"],
            color='#FF5722', edgecolors='black', s=150, zorder=5, linewidths=2, marker='s', label='Ours Top 10')

# Annotate top 10 head labels
for _, row in top10_dual.iterrows():
    ax1.annotate(f'L{int(row["layer"])}H{int(row["kv_head"])}',
                 (row["head_idx"], df_dual[df_dual["head_idx"]==row["head_idx"]]["score_norm"].values[0]),
                 textcoords="offset points", xytext=(10, 5), ha='left', fontsize=8, color='#1565C0')
for _, row in top10_ours.iterrows():
    if row["head_idx"] not in top10_dual_idxs:
        ax1.annotate(f'L{int(row["layer"])}H{int(row["kv_head"])}',
                     (row["head_idx"], df_ours[df_ours["head_idx"]==row["head_idx"]]["score_norm"].values[0]),
                     textcoords="offset points", xytext=(10, -10), ha='left', fontsize=8, color='#BF360C')

ax1.set_xlabel('Attention Head Index (layer * 4 + kv_head)', fontsize=14)
ax1.set_ylabel('Normalized Score (min-max)', fontsize=14)
ax1.set_title('Memory Core Budget Scores: Dual vs Ours', fontsize=16)
ax1.legend(fontsize=12, loc='upper right')
ax1.grid(True, alpha=0.3)

plt.tight_layout()
output_path = f"{base}/results/observations/memory_core_comparison.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nFigure saved to: {output_path}")
plt.close()
