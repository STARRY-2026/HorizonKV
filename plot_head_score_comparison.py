import os

import matplotlib.pyplot as plt
import pandas as pd


BASE = "/opt/data/private/yujun/myprogram/hermes"
INPUTS = {
    "score_v1": f"{BASE}/results/observations/score_v1/score_v1.csv",
    "dual": f"{BASE}/results/observations/memory_core_budget_scores_dual/ours_dual_top100.csv",
    "ours_top100": f"{BASE}/results/observations/memory_core_budget_scores/ours_top100.csv",
}
OUTPUT_DIR = f"{BASE}/results/observations/head_score_comparison"


PLOT_STYLE = {
    "score_v1": {"color": "#2E7D32", "line": "-", "marker": "o"},
    "dual": {"color": "#1565C0", "line": "--", "marker": "D"},
    "ours_top100": {"color": "#EF6C00", "line": "-.", "marker": "s"},
}


def normalize(series):
    vmin = series.min()
    vmax = series.max()
    if vmax == vmin:
        return pd.Series([0.0] * len(series), index=series.index, dtype=float)
    return (series - vmin) / (vmax - vmin)


def load_table(name, path):
    df = pd.read_csv(path)
    required = {"layer", "kv_head", "score"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")

    df = df.sort_values(["layer", "kv_head"]).reset_index(drop=True)
    df["head_idx"] = df["layer"] * 4 + df["kv_head"]
    df["score_norm"] = normalize(df["score"])
    df["source"] = name
    return df


def build_top10(df):
    top10 = df.nlargest(10, "score")[["layer", "kv_head", "head_idx", "score", "score_norm"]].copy()
    top10 = top10.reset_index(drop=True)
    top10.insert(0, "rank", range(1, len(top10) + 1))
    top10["head_name"] = top10.apply(
        lambda row: f"L{int(row['layer'])}H{int(row['kv_head'])}", axis=1
    )
    return top10[
        ["rank", "head_name", "layer", "kv_head", "head_idx", "score", "score_norm"]
    ]


def compare_top10(name_a, top10_a, name_b, top10_b):
    set_a = set(zip(top10_a["layer"], top10_a["kv_head"]))
    set_b = set(zip(top10_b["layer"], top10_b["kv_head"]))
    both = set_a & set_b
    only_a = set_a - set_b
    only_b = set_b - set_a

    lines = [
        f"Comparison: {name_a} vs {name_b}",
        f"  overlap_count: {len(both)}",
        f"  only_{name_a}: {sorted(only_a)}",
        f"  only_{name_b}: {sorted(only_b)}",
        "",
    ]
    return "\n".join(lines)


def annotate_top10(ax, df, top10, color, y_offset):
    for _, row in top10.iterrows():
        ax.annotate(
            row["head_name"],
            (row["head_idx"], row["score_norm"]),
            textcoords="offset points",
            xytext=(6, y_offset),
            ha="left",
            fontsize=8,
            color=color,
        )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tables = {name: load_table(name, path) for name, path in INPUTS.items()}
    top10_tables = {name: build_top10(df) for name, df in tables.items()}

    fig, ax = plt.subplots(1, 1, figsize=(20, 9))

    for name, df in tables.items():
        style = PLOT_STYLE[name]
        ax.plot(
            df["head_idx"],
            df["score_norm"],
            linestyle=style["line"],
            marker=style["marker"],
            color=style["color"],
            linewidth=1.8,
            markersize=5,
            alpha=0.85,
            label=name,
        )

        top10 = top10_tables[name]
        ax.scatter(
            top10["head_idx"],
            top10["score_norm"],
            color=style["color"],
            edgecolors="black",
            linewidths=1.3,
            s=110,
            zorder=5,
            marker=style["marker"],
            label=f"{name} top10",
        )

    annotate_top10(ax, tables["score_v1"], top10_tables["score_v1"], PLOT_STYLE["score_v1"]["color"], 8)
    annotate_top10(ax, tables["dual"], top10_tables["dual"], PLOT_STYLE["dual"]["color"], -12)
    annotate_top10(
        ax,
        tables["ours_top100"],
        top10_tables["ours_top100"],
        PLOT_STYLE["ours_top100"]["color"],
        18,
    )

    ax.set_xlabel("Attention Head Index (layer * 4 + kv_head)", fontsize=14)
    ax.set_ylabel("Normalized Score", fontsize=14)
    ax.set_title("Head Score Comparison: score_v1 vs dual vs ours_top100", fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc="upper left", ncol=2)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "head_score_comparison.png")
    plt.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close()

    for name, top10 in top10_tables.items():
        out_path = os.path.join(OUTPUT_DIR, f"top10_{name}.csv")
        top10.to_csv(out_path, index=False)
        print(f"wrote {out_path}")

    summary_parts = []
    pairs = [
        ("score_v1", "dual"),
        ("score_v1", "ours_top100"),
        ("dual", "ours_top100"),
    ]
    for name_a, name_b in pairs:
        summary_parts.append(
            compare_top10(name_a, top10_tables[name_a], name_b, top10_tables[name_b])
        )

    summary_path = os.path.join(OUTPUT_DIR, "top10_overlap_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_parts))
    print(f"wrote {summary_path}")
    print(f"wrote {plot_path}")

    for name, top10 in top10_tables.items():
        print("=" * 80)
        print(f"Top 10 heads for {name}")
        print(top10.to_string(index=False))


if __name__ == "__main__":
    main()
