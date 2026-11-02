"""Plot a fixed-500-per-head two-bank allocation from saved offline curves."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GROUPS = ROOT / "assets/recent1or4/per_head_allocations.csv"
MIN_RECENT = 32
TOTAL_PER_HEAD = 500


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--curves', type=Path,
                        default=ROOT / 'assets/onefour_k500_offline/bank_curves.npz',
                        help='Frozen bank_curves.npz (defaults to the published K500 curves)')
    parser.add_argument('--groups', type=Path, default=GROUPS)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    curves = args.curves.resolve()
    groups = args.groups.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with np.load(curves) as data:
        recent_curve = data["recent"]
        history_curve = data["history"]
    if recent_curve.shape != (112, 513) or history_curve.shape[0] != 112:
        raise ValueError("Unexpected saved curve shape")
    source_rows = [row for row in csv.DictReader(groups.open(newline=""))
                   if int(row["frames"]) == 4]
    ranked = sorted(source_rows, key=lambda row: (
        int(row["recent_r"]) / int(row["capacity_n"]),
        int(row["layer"]), int(row["kv_head"])))
    if len(ranked) != 112 or len({int(row["head_id"]) for row in ranked}) != 112:
        raise ValueError("Expected 112 distinct heads in source table")
    one_frame_ids = {int(row["head_id"]) for row in ranked[:56]}
    source_by_id = {int(row["head_id"]): row for row in source_rows}

    choices = np.arange(MIN_RECENT, TOTAL_PER_HEAD + 1)
    objective = recent_curve[:, choices] + history_curve[:, TOTAL_PER_HEAD - choices]
    chosen_recent = choices[np.argmax(objective, axis=1)]
    chosen_history = TOTAL_PER_HEAD - chosen_recent
    chosen_utility = objective.max(axis=1)
    if not np.all(chosen_recent + chosen_history == TOTAL_PER_HEAD):
        raise AssertionError("Allocation violates fixed per-head total")

    rows = []
    for head_id in range(112):
        source = source_by_id[head_id]
        rows.append(dict(
            head_id=head_id, layer=head_id // 4, kv_head=head_id % 4,
            current_branch="one_frame" if head_id in one_frame_ids else "four_frames",
            recent_tokens=int(chosen_recent[head_id]),
            history_tokens=int(chosen_history[head_id]),
            total_tokens=TOTAL_PER_HEAD,
            recent_fraction=float(chosen_recent[head_id] / TOTAL_PER_HEAD),
            lukv_k1000_recent_share=(int(source["recent_r"]) /
                                     int(source["capacity_n"])),
            offline_proxy_utility=float(chosen_utility[head_id]),
        ))
    csv_path = output / "per_head_allocation.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)

    recent_map = chosen_recent.reshape(28, 4)
    history_map = chosen_history.reshape(28, 4)
    branch_map = np.array([1 if head_id in one_frame_ids else 4
                           for head_id in range(112)]).reshape(28, 4)
    fig, axes = plt.subplots(1, 3, figsize=(13, 17), sharey=True,
                             gridspec_kw={"width_ratios": [1, 1, .7]})
    for ax, values, title, cmap in (
            (axes[0], recent_map, "Recent tokens / head", "Blues"),
            (axes[1], history_map, "History tokens / head", "Oranges")):
        image = ax.imshow(values, aspect="auto", vmin=0, vmax=500, cmap=cmap)
        ax.set_title(title, fontsize=14, pad=14)
        fig.colorbar(image, ax=ax, fraction=.048, pad=.04)
        for layer in range(28):
            for head in range(4):
                value = int(values[layer, head])
                ax.text(head, layer, str(value), ha="center", va="center",
                        fontsize=8, color="white" if value >= 300 else "black")
    axes[2].imshow(branch_map, aspect="auto", vmin=1, vmax=4,
                   cmap=ListedColormap(["#8fc7b8", "#d5bd8c"]))
    axes[2].set_title("Current prefill branch", fontsize=14, pad=14)
    for layer in range(28):
        for head in range(4):
            axes[2].text(head, layer, str(branch_map[layer, head]),
                         ha="center", va="center", fontsize=8)
    for ax in axes:
        ax.set_xticks(range(4), [f"H{head}" for head in range(4)])
        ax.set_yticks(range(28), [f"L{layer}" for layer in range(28)])
        ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 28, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)
    axes[0].set_ylabel("Layer (0-based)")
    fig.suptitle("Offline two-bank allocation | 112 heads x 500 visual KV tokens",
                 fontsize=16, y=.995)
    fig.tight_layout(rect=(0, 0, 1, .984))
    fig.savefig(output / "allocation_heatmap.png", dpi=180)
    fig.savefig(output / "allocation_heatmap.pdf")
    plt.close(fig)

    summary = dict(
        protocol="offline full-cache projected-answer utility; not QA accuracy",
        choice="argmax R_h(r) + H_h(500-r), 32 <= r <= 500, each head independently",
        history_ranking="saved HERMES-scored candidates",
        one_four_split="lowest 56 K1000 recent-share heads use one-frame prefill",
        curve_sha256=sha256(curves), group_table_sha256=sha256(groups),
        heads=112, per_head_total=500, total_tokens=56000,
        recent_tokens=int(chosen_recent.sum()),
        history_tokens=int(chosen_history.sum()),
        mean_recent_per_head=float(chosen_recent.mean()),
        mean_history_per_head=float(chosen_history.mean()),
        offline_proxy_utility=float(chosen_utility.sum()),
        all_recent_500_proxy_utility=float(recent_curve[:, 500].sum()),
        heads_at_recent_floor=int(np.sum(chosen_recent == MIN_RECENT)),
        heads_recent_at_most_128=int(np.sum(chosen_recent <= 128)),
        group_summary={group: dict(
            count=len(subset), recent_tokens=sum(row["recent_tokens"] for row in subset),
            history_tokens=sum(row["history_tokens"] for row in subset),
            mean_recent=sum(row["recent_tokens"] for row in subset) / len(subset),
            mean_history=sum(row["history_tokens"] for row in subset) / len(subset),
        ) for group in ("one_frame", "four_frames")
            for subset in [[row for row in rows if row["current_branch"] == group]]},
    )
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
