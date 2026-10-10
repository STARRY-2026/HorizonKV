#!/usr/bin/env python3
"""Plot fixed-candidate recent/history utility surfaces from offline traces.

U_h(r,q) = utility of newest r tokens in the last two Qwen visual time units
         + utility of top-q HERMES-scored tokens strictly before those units.
This is an additive, full-cache answer-readout proxy, not compressed QA.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


ROOT = Path("results/myobservations/lu_kv_offline_50v_20260913")
OUT = Path("results/myobservations/dual_utility_surface_20260923")
STEP = 32
RECENTS = np.arange(0, 512 + STEP, STEP)
HISTORY = np.arange(0, 1472 + STEP, STEP)
HEADS = 112


def top_history_prefix(score, utility):
    limit = int(HISTORY[-1])
    top = np.argpartition(-score, limit - 1, axis=1)[:, :limit]
    top_score = np.take_along_axis(score, top, axis=1)
    order = np.argsort(-top_score, axis=1, kind="stable")
    top = np.take_along_axis(top, order, axis=1)
    rows = np.arange(HEADS)[:, None]
    return np.column_stack((np.zeros(HEADS), np.cumsum(utility[rows, top], axis=1)))


def collect():
    by_video = defaultdict(list)
    omitted = defaultdict(int)
    for path in sorted(ROOT.glob("v*_q*.npz")):
        with np.load(path) as z:
            u = z["utility"].astype(np.float64)
            scores = z["hermes_score"].astype(np.float64)
            times = z["source_times"].astype(np.float64)
        if u.shape[0] != HEADS or scores.shape != u.shape or len(times) != u.shape[1]:
            raise AssertionError(f"Bad trace dimensions: {path}")
        unique = np.unique(times)
        if len(unique) < 2:
            omitted["fewer_than_two_visual_units"] += 1
            continue
        recent_start = int(np.searchsorted(times, unique[-2], side="left"))
        if len(times) - recent_start < RECENTS[-1]:
            omitted["recent_fewer_than_512"] += 1
            continue
        if recent_start < HISTORY[-1]:
            omitted["history_fewer_than_1472"] += 1
            continue
        recent_u = u[:, recent_start:]
        newest = np.column_stack((np.zeros(HEADS), np.cumsum(recent_u[:, ::-1], axis=1)))
        hist = top_history_prefix(scores[:, :recent_start], u[:, :recent_start])
        r_curve = newest[:, RECENTS]
        h_curve = hist[:, HISTORY]
        full_head = u.sum(axis=1)
        if np.any(full_head <= 0):
            raise AssertionError(f"Head with nonpositive utility: {path}")
        vid = path.stem.split("_")[0]
        by_video[vid].append((r_curve, h_curve, full_head, unique[-1] - unique[-2]))
    if not by_video:
        raise ValueError("No observations satisfy the shared complete grid")
    return by_video, dict(omitted)


def surfaces(by_video):
    # Video-equal aggregation. Within a video, observations are averaged.
    global_surfaces = []
    head_surfaces = []
    recent_shares = []
    intervals = []
    for observations in by_video.values():
        global_obs = []
        head_obs = []
        share_obs = []
        for r, h, full, interval in observations:
            value = r[:, :, None] + h[:, None, :]
            global_obs.append(value.sum(axis=0) / full.sum())
            head_obs.append(value / full[:, None, None])
            share_obs.append(r[:, -1] / full)
            intervals.append(interval)
        global_surfaces.append(np.mean(global_obs, axis=0))
        head_surfaces.append(np.mean(head_obs, axis=0))
        recent_shares.append(np.mean(share_obs, axis=0))
    global_surface = np.mean(global_surfaces, axis=0)
    head_surface = np.mean(head_surfaces, axis=0)
    share = np.mean(recent_shares, axis=0)
    if not np.all(np.diff(global_surface, axis=0) >= -1e-10) or not np.all(np.diff(global_surface, axis=1) >= -1e-10):
        raise AssertionError("Surface must be nondecreasing along both capacities")
    if not np.allclose(global_surface, global_surface[:, :1] + global_surface[:1, :] - global_surface[0, 0]):
        raise AssertionError("Fixed-candidate surface should be additive")
    return global_surface, head_surface, share, intervals


def plot(surface, heads, share):
    pct = np.percentile(share, [10, 90])
    representative = [int(np.argmin(abs(share - target))) for target in pct]
    panels = [("All 112 heads · utility-weighted", surface)]
    for head in representative:
        panels.append((f"Head {head // 4}:{head % 4} · recent share {share[head]:.1%}", heads[head]))
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.9), layout="constrained")
    vmax = max(float(np.max(x)) for _, x in panels)
    for ax, (title, data) in zip(axes, panels):
        mesh = ax.pcolormesh(RECENTS, HISTORY, data.T, shading="nearest",
                             cmap="viridis", vmin=0, vmax=vmax, rasterized=True)
        ax.plot([0, 512], [1000, 488], color="white", linewidth=1.5,
                linestyle="--", label="r + q = 1000")
        ax.set(xlabel="Recent capacity r (tokens)", title=title, xlim=(0, 512), ylim=(0, 1472))
        ax.grid(color="white", alpha=.12, linewidth=.5)
    axes[0].set_ylabel("History capacity q (tokens)")
    axes[-1].legend(loc="lower right", framealpha=.85, fontsize=8)
    fig.colorbar(mesh, ax=axes, shrink=.83, label="Retained full-cache answer-utility proxy")
    fig.suptitle("Two-dimensional utility surface: recent versus history capacity", fontsize=13)
    fig.savefig(OUT / "two_dimensional_utility_surface.png", dpi=250)
    fig.savefig(OUT / "two_dimensional_utility_surface.pdf")
    plt.close(fig)
    return representative


def plot_marginal_choice(surface, heads, representative):
    """Visualize finite-difference advantage of the next 32 recent tokens."""
    panels = [("All 112 heads", surface)] + [
        (f"Head {head // 4}:{head % 4}", heads[head]) for head in representative
    ]
    diffs = [np.diff(data, axis=0)[:, :-1] - np.diff(data, axis=1)[:-1, :]
             for _, data in panels]
    bound = float(np.quantile(np.abs(np.concatenate([d.ravel() for d in diffs])), .98))
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.9), layout="constrained")
    for ax, (title, _), diff in zip(axes, panels, diffs):
        mesh = ax.pcolormesh(RECENTS[:-1], HISTORY[:-1], 100 * diff.T,
                             shading="nearest", cmap="PuOr_r",
                             norm=TwoSlopeNorm(vmin=-100 * bound, vcenter=0, vmax=100 * bound),
                             rasterized=True)
        ax.plot([0, 512], [1000, 488], color="black", alpha=.75,
                linewidth=1.1, linestyle="--")
        ax.set(xlabel="Current recent capacity r", title=title,
               xlim=(0, 480), ylim=(0, 1440))
    axes[0].set_ylabel("Current history capacity q")
    fig.colorbar(mesh, ax=axes, shrink=.83,
                 label="Next 32 recent − next 32 history (percentage points)")
    fig.suptitle("Which bank has higher local marginal utility?", fontsize=13)
    fig.savefig(OUT / "marginal_recent_vs_history.png", dpi=250)
    fig.savefig(OUT / "marginal_recent_vs_history.pdf")
    plt.close(fig)


def plot_3d_surface(surface, heads, representative):
    """Render the same numerical surface as height over two capacity axes."""
    recent_mesh, history_mesh = np.meshgrid(RECENTS, HISTORY, indexing="ij")
    panels = [("All 112 KV heads", surface)] + [
        (f"Head {head // 4}:{head % 4}", heads[head]) for head in representative
    ]
    fig = plt.figure(figsize=(17.6, 6.5), layout="constrained")
    for panel, (title, data) in enumerate(panels, 1):
        ax = fig.add_subplot(1, 3, panel, projection="3d")
        z = 100 * data
        ax.plot_surface(recent_mesh, history_mesh, z, cmap="viridis",
                        edgecolor=(1, 1, 1, .12), linewidth=.2,
                        rstride=1, cstride=1, antialiased=True, alpha=.95)
        budget_r = RECENTS
        budget_q = 1000 - budget_r
        budget_z = 100 * np.array([
            np.interp(q, HISTORY, data[i]) for i, q in enumerate(budget_q)
        ])
        lift = 1.0
        ax.plot(budget_r, budget_q, budget_z + lift, color="#e63946",
                linewidth=3.2, label="r + q = 1000", zorder=20)
        for r in (0, 256, 512):
            idx = int(r // STEP)
            ax.scatter([r], [1000-r], [budget_z[idx]+lift],
                       color="#e63946", edgecolor="white", s=32,
                       depthshade=False, zorder=21)
        ax.set(xlabel="Recent r", ylabel="History q", zlabel="Retained utility (%)",
               title=title, xlim=(0, 512), ylim=(0, 1472), zlim=(0, 65))
        ax.set_xticks((0, 256, 512))
        ax.set_yticks((0, 500, 1000, 1472))
        ax.view_init(elev=26, azim=-65)
        ax.set_box_aspect((1.15, 1.35, .8), zoom=.82)
        if panel == 1:
            ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Capacity–utility surfaces (height = retained answer-utility proxy)", fontsize=14)
    fig.savefig(OUT / "utility_surface_3d.png", dpi=250)
    fig.savefig(OUT / "utility_surface_3d.pdf")
    plt.close(fig)

    # A second perspective makes the recent/head-dependent slope easier to see.
    fig = plt.figure(figsize=(9, 7), layout="constrained")
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(recent_mesh, history_mesh, 100 * surface, cmap="viridis",
                    edgecolor=(1, 1, 1, .12), linewidth=.15, alpha=.96)
    budget_z = 100 * np.array([
        np.interp(1000-r, HISTORY, surface[i]) for i, r in enumerate(RECENTS)
    ])
    ax.plot(RECENTS, 1000-RECENTS, budget_z+1, color="#e63946",
            linewidth=3.5, label="r + q = 1000")
    ax.set(xlabel="Recent capacity r", ylabel="History capacity q",
           zlabel="Retained utility (%)", xlim=(0, 512), ylim=(0, 1472), zlim=(0, 65))
    ax.set_box_aspect((1.15, 1.35, .85), zoom=.88)
    ax.view_init(elev=26, azim=32)
    ax.legend(loc="upper right")
    fig.suptitle("All heads: opposite view of the same surface", fontsize=13)
    fig.savefig(OUT / "utility_surface_3d_opposite_view.png", dpi=250)
    plt.close(fig)


def write_csv(path, surface):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["recent_r", "history_q", "retained_proxy"])
        for ri, r in enumerate(RECENTS):
            for qi, q in enumerate(HISTORY):
                writer.writerow([int(r), int(q), float(surface[ri, qi])])


def main():
    by_video, omitted = collect()
    surface, head_surfaces, shares, intervals = surfaces(by_video)
    OUT.mkdir(parents=True, exist_ok=True)
    representative = plot(surface, head_surfaces, shares)
    plot_marginal_choice(surface, head_surfaces, representative)
    plot_3d_surface(surface, head_surfaces, representative)
    write_csv(OUT / "all_head_surface.csv", surface)
    for head in representative:
        write_csv(OUT / f"head_{head:03d}_surface.csv", head_surfaces[head])
    np.savez_compressed(OUT / "surfaces_and_marginals.npz", recent_capacity=RECENTS,
        history_capacity=HISTORY, all_head_surface=surface, head_surfaces=head_surfaces,
        next_32_recent=np.diff(head_surfaces, axis=1),
        next_32_history=np.diff(head_surfaces, axis=2),
        all_head_next_32_recent=np.diff(surface, axis=0),
        all_head_next_32_history=np.diff(surface, axis=1),
        head_recent_share=shares)
    report = {
        "videos": len(by_video), "observations": sum(len(x) for x in by_video.values()),
        "omitted": omitted, "visual_unit_interval_seconds": sorted(set(intervals)),
        "recent_candidates": "two latest distinct Qwen visual timestamps, fixed independently of r and q",
        "selector": "newest r recent tokens; top-q HERMES score among older visual tokens",
        "utility": "full-cache projected answer readout, not a causal or compressed-QA measurement",
        "aggregation": "observations equal within video, videos equal; full utility normalizes each observation",
        "additive_surface": True,
        "representative_heads": [{"id": h, "layer": h // 4, "kv_head": h % 4,
                                  "recent_share": float(shares[h])} for h in representative],
        "points": {f"r{r}_q{q}": float(surface[np.where(RECENTS == r)[0][0], np.where(HISTORY == q)[0][0]])
                   for r, q in ((0, 992), (256, 736), (512, 480))},
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
