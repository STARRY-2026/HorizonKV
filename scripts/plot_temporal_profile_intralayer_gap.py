"""Layer-vs-layer similarity heatmaps for the eight-bin temporal profiles.

Each cell (i, j) is the mean cosine similarity between the four physical KV
heads in layer i and the four heads in layer j. The diagonal is intralayer
similarity; off-diagonal cells are interlayer similarity. Panels compare the
raw/L1 profile (dominated by the shared recency baseline) with the within-bin
z-score profile (head-specific temporal preference).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze_temporal_profile_spatial_redundancy import (
    column_zscore,
    cosine_matrix,
    l1_normalize,
    load_profiles,
)


METRIC = "reuse_score"


def layer_similarity(values, index):
    layers = np.asarray([item[0] for item in index])
    sim = cosine_matrix(values)
    n_layers = int(np.max(layers)) + 1
    matrix = np.zeros((n_layers, n_layers), dtype=np.float64)
    for i in range(n_layers):
        idx_i = np.flatnonzero(layers == i)
        for j in range(n_layers):
            idx_j = np.flatnonzero(layers == j)
            block = sim[np.ix_(idx_i, idx_j)]
            if i == j:
                block = block[~np.eye(block.shape[0], dtype=bool)]
            matrix[i, j] = block.mean()
    return matrix


def plot_panel(ax, matrix, title, cmap, vmin, vmax):
    image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Layer", fontsize=8)
    ax.set_ylabel("Layer", fontsize=8)
    ax.set_xticks(range(0, matrix.shape[0], 4))
    ax.set_yticks(range(0, matrix.shape[0], 4))
    ax.tick_params(labelsize=7)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen_root", default="results/myobservations/temporal_topk_profile_50v_fixed448")
    parser.add_argument("--llava_root", default="results/myobservations/temporal_profile_llavaov7b_same50")
    parser.add_argument("--output_dir", default="results/myobservations/temporal_profile_spatial_redundancy")
    args = parser.parse_args()

    roots = {
        "Qwen2.5-VL-7B": Path(args.qwen_root),
        "LLaVA-OV-7B": Path(args.llava_root),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(8.6, 8.4))
    for row, (model, root) in enumerate(roots.items()):
        profiles = load_profiles(root / "raw_age_bin_attention.csv")
        raw = profiles[METRIC]["raw"]
        index = profiles[METRIC]["index"]

        raw_sim = layer_similarity(raw, index)
        z_sim = layer_similarity(column_zscore(raw), index)

        intra_raw = np.diag(raw_sim).mean()
        inter_raw = raw_sim[~np.eye(raw_sim.shape[0], dtype=bool)].mean()
        intra_z = np.diag(z_sim).mean()
        inter_z = z_sim[~np.eye(z_sim.shape[0], dtype=bool)].mean()

        im0 = plot_panel(
            axes[row, 0],
            raw_sim,
            f"{model} · raw/L1\nintra={intra_raw:.3f}, inter={inter_raw:.3f}",
            "YlOrRd",
            0.5,
            1.0,
        )
        im1 = plot_panel(
            axes[row, 1],
            z_sim,
            f"{model} · within-bin z-score\nintra={intra_z:.3f}, inter={inter_z:.3f}",
            "RdBu_r",
            -1.0,
            1.0,
        )
        fig.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)
        fig.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)

    fig.suptitle("Layer-vs-layer similarity of 8-bin temporal profiles (reuse_score)", fontsize=10)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.06, wspace=0.24, hspace=0.30)
    path = output_dir / "intralayer_gap_heatmaps.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print({"output": str(path)})


if __name__ == "__main__":
    main()
