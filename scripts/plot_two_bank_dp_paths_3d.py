#!/usr/bin/env python3
"""Overlay exact one-token DP tracebacks on matching two-bank utility surfaces.

The surfaces and paths both use the same 48-video/68-observation offline
full-cache answer-readout proxy. A plotted traceback is one optimal route to
the globally assigned endpoint; additive edge weights make routes non-unique.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from allocate_token_gradient import load_profiles
from allocate_two_bank_grid_dp import head_grid_dp, reconstruct_path, full_head_shares
from plot_two_dimensional_utility_surface import ROOT as TRACE_ROOT


OUT = Path("results/myobservations/two_bank_dp_paths_3d_20260928")
ALLOCATIONS = {
    250: Path("results/myobservations/two_bank_grid_dp_k250_floor188_20260923/per_head_allocation.csv"),
    500: Path("results/myobservations/two_bank_grid_dp_k500_floor375_20260925/per_head_allocation.csv"),
    800: Path("results/myobservations/two_bank_grid_dp_k800_floor600_20260928/per_head_allocation.csv"),
    1000: Path("results/myobservations/two_bank_grid_dp_floor750_20260923/per_head_allocation.csv"),
}
HEADS = (10, 57)  # The two head panels in the original utility-surface figure.
COLORS = {250: "#f16c45", 500: "#ffc247", 800: "#d377ed", 1000: "#24c6cf"}
MAX_HERMES_BUDGET = 1000


def read_allocations():
    tables = {}
    for budget, path in ALLOCATIONS.items():
        with path.open(newline="") as handle:
            tables[budget] = {int(row["head_id"]): row for row in csv.DictReader(handle)}
    return tables


def traceback_coordinates(back, recent, history):
    decisions = reconstruct_path(back, recent, history)
    r = np.zeros(len(decisions) + 1, dtype=np.int16)
    q = np.zeros(len(decisions) + 1, dtype=np.int16)
    for idx, decision in enumerate(decisions, 1):
        r[idx] = r[idx - 1] + int(decision == "R")
        q[idx] = q[idx - 1] + int(decision == "H")
    assert (int(r[-1]), int(q[-1])) == (recent, history)
    return r, q, decisions


def hermes_uniform_paths(videos, shares):
    """Actual 1D HERMES ranking, expressed in recent/history coordinates.

    Its height is the retained utility of the actual HERMES token subset, not
    the two-bank surface sampled at the same (possibly fractional) coordinates.
    """
    by_video = defaultdict(list)
    latest = MAX_HERMES_BUDGET - 32
    for path in sorted(TRACE_ROOT.glob("v*_q*.npz")):
        video = path.stem.split("_")[0]
        if video not in videos:
            continue
        with np.load(path) as data:
            utility = data["utility"]
            ranking = data["ranking"]
            times = data["source_times"]
            unique = np.unique(times)
            if len(unique) < 2:
                continue
            recent_start = int(np.searchsorted(times, unique[-2], side="left"))
            if len(times) - recent_start < 512 or recent_start < 1472:
                continue
            full = float(utility.sum())
            obs = {}
            for head in HEADS:
                selected = ranking[head, :latest]
                recent_count = 32 + np.r_[0, np.cumsum(selected >= recent_start)]
                retained = (utility[head, -32:].sum()
                            + np.r_[0, np.cumsum(utility[head, selected])]) / full
                obs[head] = (recent_count.astype(float), retained.astype(float))
            by_video[video].append(obs)
    if set(by_video) != set(videos):
        raise AssertionError("HERMES and 2D DP cohorts differ")
    paths = {}
    n = np.arange(32, MAX_HERMES_BUDGET + 1)
    for head in HEADS:
        r = np.mean([np.mean([obs[head][0] for obs in by_video[video]], axis=0)
                     for video in videos], axis=0)
        retained = np.mean([np.mean([obs[head][1] for obs in by_video[video]], axis=0)
                            for video in videos], axis=0)
        paths[head] = (n, r, n - r, 100.0 * retained / float(shares[head]))
    return paths


def main():
    profiles, _ = load_profiles()
    videos = sorted(profiles)
    recent_curves = np.mean([profiles[vid]["recent"] for vid in videos], axis=0)
    history_curves = np.mean([profiles[vid]["history"] for vid in videos], axis=0)
    full_shares = full_head_shares()
    tables = read_allocations()
    hermes_paths = hermes_uniform_paths(videos, full_shares)
    OUT.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(18.5, 8.3), layout="constrained")
    topfig, topaxes = plt.subplots(1, 2, figsize=(16.3, 7.2), layout="constrained")
    records = []
    path_rows = []
    hermes_rows = []

    for panel, head in enumerate(HEADS, 1):
        recent = recent_curves[head]
        history = history_curves[head]
        denominator = float(full_shares[head])
        value = lambda r, q: 100.0 * (recent[r] + history[q]) / denominator
        n_values, endpoint_recent, back = head_grid_dp(recent, history)

        sampled_r = np.unique(np.r_[np.arange(0, 513, 16), 512])
        sampled_q = np.unique(np.r_[np.arange(0, 1201, 20), 1200])
        rr, qq = np.meshgrid(sampled_r, sampled_q, indexing="ij")
        zz = value(rr, qq)
        zz = np.where(rr + qq <= 1500, zz, np.nan)

        ax = fig.add_subplot(1, 2, panel, projection="3d")
        ax.plot_surface(rr, qq, zz, cmap="viridis", alpha=.78,
                        linewidth=.18, edgecolor=(1, 1, 1, .14),
                        antialiased=True, shade=False)
        topax = topaxes[panel - 1]
        topax.contourf(rr, qq, zz, levels=24, cmap="viridis")
        topax.contour(rr, qq, zz, levels=8, colors="white", alpha=.36, linewidths=.55)

        for budget in (250, 500, 800, 1000):
            row = tables[budget][head]
            n = int(row["floor750_n"] if budget == 1000 else row["capacity_n"])
            r_end = int(row["recent_r"])
            q_end = int(row["history_q"])
            assert n == r_end + q_end
            assert int(endpoint_recent[n]) == r_end
            assert np.isclose(float(n_values[n]), float(row["selected_proxy"]), atol=1e-12)

            path_r, path_q, decisions = traceback_coordinates(back, r_end, q_end)
            path_z = value(path_r, path_q)
            color = COLORS[budget]
            label = f"DP K={budget} (head n={n})"
            # Small vertical lift keeps the path visible above the mesh.
            ax.plot(path_r, path_q, path_z + .7, color=color,
                    lw=2.6, alpha=.98, label=label, zorder=50)
            ax.scatter([r_end], [q_end], [path_z[-1] + .7],
                       c=color, edgecolors="white", linewidths=1.2,
                       s=62, depthshade=False, zorder=60)
            topax.plot(path_r, path_q, color=color, lw=2.1, label=label)
            topax.scatter([r_end], [q_end], c=color, edgecolors="white",
                          linewidths=1.0, s=68, zorder=10)
            topax.annotate(f"{budget}", (r_end, q_end),
                           xytext=(5, 5), textcoords="offset points",
                           color="white", fontsize=8, weight="bold",
                           bbox=dict(facecolor="#17232b", alpha=.78,
                                     edgecolor="none", pad=1.5))

            records.append({"head_id": head, "layer": head // 4, "kv_head": head % 4,
                            "global_K": budget, "head_capacity_n": n,
                            "recent_r": r_end, "history_q": q_end,
                            "retained_head_utility_pct": float(path_z[-1]),
                            "path_steps": len(decisions),
                            "recent_steps": decisions.count("R"),
                            "history_steps": decisions.count("H")})
            for step, (r, q, z) in enumerate(zip(path_r, path_q, path_z)):
                path_rows.append({"head_id": head, "global_K": budget,
                                  "step": step, "recent_r": int(r),
                                  "history_q": int(q),
                                  "retained_head_utility_pct": float(z)})

        hermes_n, hermes_r, hermes_q, hermes_z = hermes_paths[head]
        ax.plot(hermes_r[::4], hermes_q[::4], hermes_z[::4] + 1.1,
                color="#161922", lw=3.0, ls="--", label="HERMES uniform K=32–1000",
                zorder=70)
        topax.plot(hermes_r[::4], hermes_q[::4], color="#161922",
                   lw=2.4, ls="--", label="HERMES uniform K=32–1000")
        for budget in (250, 500, 800, 1000):
            idx = budget - 32
            ax.scatter([hermes_r[idx]], [hermes_q[idx]], [hermes_z[idx] + 1.1],
                       c="#161922", edgecolors="white", linewidths=.9,
                       s=37, depthshade=False, zorder=75)
            topax.scatter([hermes_r[idx]], [hermes_q[idx]],
                          c="#161922", edgecolors="white", linewidths=.8,
                          s=36, zorder=12)
        for i in range(len(hermes_n)):
            hermes_rows.append({"head_id": head, "uniform_budget_n": int(hermes_n[i]),
                                "mean_recent_r": float(hermes_r[i]),
                                "mean_history_q": float(hermes_q[i]),
                                "actual_retained_head_utility_pct": float(hermes_z[i])})

        ax.set(xlabel="Recent capacity R", ylabel="History capacity H",
               zlabel="Retained head utility (%)", xlim=(0, 512), ylim=(0, 1200),
               title=f"Head {head} (layer {head // 4}, KV {head % 4})")
        ax.set_zlim(0, max(44, float(np.nanmax(zz) * 1.06)))
        ax.set_box_aspect((1.0, 1.4, .75), zoom=.84)
        ax.view_init(elev=27, azim=-64)
        ax.legend(loc="upper left", fontsize=7.5, framealpha=.92)
        topax.set(xlabel="Recent capacity R (tokens)", ylabel="History capacity H (tokens)",
                  xlim=(0, 512), ylim=(0, 1200),
                  title=f"Head {head}: temporal capacity trajectories")
        topax.grid(color="white", linewidth=.45, alpha=.2)

    fig.suptitle("DP paths at four global budgets versus uniform-budget HERMES", fontsize=15)
    topfig.suptitle("Top view: numbers label DP budget K; black dashed line is HERMES", fontsize=13)
    for name, canvas in (("dp_paths_on_utility_surfaces_3d", fig),
                         ("dp_paths_on_utility_surfaces_top_view", topfig)):
        canvas.savefig(OUT / f"{name}.png", dpi=260)
        canvas.savefig(OUT / f"{name}.pdf")
        plt.close(canvas)

    with (OUT / "dp_path_points.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(path_rows[0]))
        writer.writeheader()
        writer.writerows(path_rows)
    with (OUT / "hermes_uniform_path.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(hermes_rows[0]))
        writer.writeheader()
        writer.writerows(hermes_rows)
    report = {
        "source": "same exact 1-token two-bank utility profiles used by the DP allocations",
        "cohort": {"videos": len(videos),
                   "observations": sum(profiles[vid]["observations"] for vid in videos)},
        "surface": "additive, full-cache projected answer-utility proxy; not compressed-QA accuracy",
        "budget_meaning": "Global K is average visual slots per head across 112 heads; endpoints show actual per-head allocations",
        "path_meaning": "one valid exact-DP traceback to each globally allocated endpoint; equivalent routes can exist",
        "hermes_line": "uniform per-head budget from 32 to 1000; (R,H) are video-equal means of actual HERMES survivors; height is actual HERMES offline utility proxy, not the two-bank surface value at mean (R,H)",
        "path_visualization": "DP lines lifted 0.7 percentage points and HERMES line 1.1 percentage points for visibility",
        "heads": records,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
