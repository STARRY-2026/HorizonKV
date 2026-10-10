#!/usr/bin/env python3
"""对比 Standard HERMES vs KV Head Budget 实验结果，生成汇报用图表"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

# ─── 数据 ───────────────────────────────────────────────
tasks = [
    "Causal Reasoning", "Spatial Understanding", "Event Understanding",
    "Text-Rich Understanding", "Action Recognition", "Clips Summarize",
    "Prospective Reasoning", "Object Recognition", "Attribute Recognition",
    "Counting",
]
n_samples = [119, 142, 73, 139, 207, 39, 105, 157, 92, 182]

standard = [73.95, 73.94, 75.34, 82.01, 72.95, 94.87, 79.05, 77.07, 88.04, 51.10]
kvhead = [76.47, 75.35, 76.71, 83.45, 73.91, 94.87, 79.05, 76.43, 86.96, 49.45]
diffs = [kvhead[i] - standard[i] for i in range(len(tasks))]

overall_standard = 73.94
overall_kvhead = 74.34

OUT_DIR = "results/comparison"
os.makedirs(OUT_DIR, exist_ok=True)

# ─── 颜色 ───────────────────────────────────────────────
BLUE = "#4A90D9"
ORANGE = "#E8833A"
GREEN = "#2BA02C"
RED = "#D62728"
GRAY = "#999999"

# ─── 图1：总体准确率 + 任务级对比（分组柱状图）───────────
fig, (ax_overall, ax_task) = plt.subplots(1, 2, figsize=(16, 6),
    gridspec_kw={"width_ratios": [1, 3]})

# 1a. 总体准确率
x = [0, 1]
bars = ax_overall.bar(x, [overall_standard, overall_kvhead],
    color=[BLUE, ORANGE], width=0.5, edgecolor="white", linewidth=0.8)
ax_overall.set_xticks(x)
ax_overall.set_xticklabels(["Standard\nHERMES", "KV Head\nBudget"], fontsize=11)
ax_overall.set_ylabel("Accuracy (%)", fontsize=12)
ax_overall.set_title("Overall Accuracy", fontsize=14, fontweight="bold")
ax_overall.set_ylim(72, 76)
for bar, val in zip(bars, [overall_standard, overall_kvhead]):
    ax_overall.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
        f"{val:.2f}%", ha="center", fontsize=12, fontweight="bold",
        color=ORANGE if val == overall_kvhead else BLUE)
ax_overall.grid(axis="y", alpha=0.3)

# 1b. 任务级对比
x = np.arange(len(tasks))
width = 0.35
bars1 = ax_task.bar(x - width / 2, standard, width, label="Standard HERMES",
    color=BLUE, edgecolor="white", linewidth=0.5)
bars2 = ax_task.bar(x + width / 2, kvhead, width, label="KV Head Budget",
    color=ORANGE, edgecolor="white", linewidth=0.5)
ax_task.set_xticks(x)
ax_task.set_xticklabels(tasks, rotation=45, ha="right", fontsize=9)
ax_task.set_ylabel("Accuracy (%)", fontsize=12)
ax_task.set_title("Per-Task Accuracy", fontsize=14, fontweight="bold")
ax_task.legend(fontsize=10)
ax_task.grid(axis="y", alpha=0.3)

# 标注差值
for i in range(len(tasks)):
    if diffs[i] != 0:
        higher = max(standard[i], kvhead[i])
        color = GREEN if diffs[i] > 0 else RED
        symbol = "+" if diffs[i] > 0 else ""
        ax_task.annotate(f"{symbol}{diffs[i]:.2f}", xy=(x[i], higher + 1),
            ha="center", fontsize=7, fontweight="bold", color=color)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/accuracy_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"[OK] {OUT_DIR}/accuracy_comparison.png")

# ─── 图2：差值排序 ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
sorted_idx = np.argsort(diffs)
sorted_tasks = [tasks[i] for i in sorted_idx]
sorted_diffs = [diffs[i] for i in sorted_idx]
colors = [GREEN if d > 0 else RED if d < 0 else GRAY for d in sorted_diffs]

bars = ax.barh(sorted_tasks, sorted_diffs, color=colors, edgecolor="white",
    height=0.6)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Accuracy Change (%)", fontsize=12)
ax.set_title("KV Head Budget vs Standard HERMES — Per-Task Change\n(positive = improvement)", fontsize=14, fontweight="bold")
for bar, val in zip(bars, sorted_diffs):
    xpos = bar.get_width() + 0.08 if val >= 0 else bar.get_width() - 0.4
    ax.text(xpos, bar.get_y() + bar.get_height() / 2,
        f"{val:+.2f}%", va="center", fontsize=10, fontweight="bold",
        color=GREEN if val > 0 else RED)
ax.set_xlim(-3, 4)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/delta_sorted.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"[OK] {OUT_DIR}/delta_sorted.png")

# ─── 图3：按样本量加权的气泡图 ────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
sizes = [n * 2 for n in n_samples]
for i in range(len(tasks)):
    color = GREEN if diffs[i] > 0 else RED if diffs[i] < 0 else GRAY
    alpha = 0.7 if abs(diffs[i]) > 0.8 else 0.35
    ax.scatter(standard[i], kvhead[i], s=sizes[i], c=color, alpha=alpha,
        edgecolors="black", linewidth=0.5, zorder=5)
    ax.annotate(f"{tasks[i]}\n(n={n_samples[i]})",
        (standard[i], kvhead[i]), textcoords="offset points",
        xytext=(0, 14 if i % 2 == 0 else -18), ha="center", fontsize=7)

ax.plot([45, 100], [45, 100], "k--", alpha=0.3, linewidth=1)
ax.set_xlim(48, 97)
ax.set_ylim(48, 97)
ax.set_xlabel("Standard HERMES Accuracy (%)", fontsize=12)
ax.set_ylabel("KV Head Budget Accuracy (%)", fontsize=12)
ax.set_title("Bubble Chart: Per-Task Comparison\n(bubble size = sample count)", fontsize=14, fontweight="bold")
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/bubble_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"[OK] {OUT_DIR}/bubble_comparison.png")

# ─── 打印 Markdown 表格 ──────────────────────────────────
print("\n### 结果对比表\n")
print("| Task | Samples | Standard HERMES | KV Head Budget | Delta |")
print("|------|---------|-----------------|----------------|-------|")
for i in range(len(tasks)):
    delta_str = f"+{diffs[i]:.2f}%" if diffs[i] > 0 else f"{diffs[i]:.2f}%"
    print(f"| {tasks[i]} | {n_samples[i]} | {standard[i]:.2f}% | {kvhead[i]:.2f}% | {delta_str} |")
print(f"\n| **Overall** | **1255** | **{overall_standard:.2f}%** | **{overall_kvhead:.2f}%** | **+{overall_kvhead - overall_standard:.2f}%** |")

print(f"\n所有图表已保存到: {OUT_DIR}/")
