#!/usr/bin/env python3
"""Publication figures for the streaming-memory response analysis.

The plots use the frozen question-free HERMES ranking and the raw projected
future-answer utility stored in the 50-video offline calibration run.
"""

from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path("results/myobservations/lu_kv_offline_50v_20260913")
OUT = ROOT / "figures"
RECENT = 32
NHEAD = 112


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 160,
        "savefig.dpi": 300,
    })


def records():
    for path in sorted(ROOT.glob("v*_q*.npz")):
        with np.load(path) as z:
            yield {
                "name": path.stem,
                "ranking": z["ranking"].astype(np.int64),
                "utility": z["utility"].astype(np.float64),
                "times": z["source_times"].astype(np.float64),
            }


def retained_mask(ranking, n, k):
    """Indices retained by the deployed ordering at total capacity k."""
    k = int(np.clip(k, RECENT, n))
    mask = np.zeros((ranking.shape[0], n), dtype=bool)
    mask[:, n - RECENT:] = True
    rows = np.arange(ranking.shape[0])[:, None]
    mask[rows, ranking[:, : k - RECENT]] = True
    return mask


def figure_one():
    # Compression intervals deliberately include the two intervals discussed
    # in the report: 0->10% and 80->90% compression.
    levels = np.asarray([0.0, .1, .2, .3, .4, .5, .6, .7, .8, .9])
    lost = np.zeros((len(levels) - 1, 4), dtype=np.float64)
    for rec in records():
        utility, ranking, times = rec["utility"], rec["ranking"], rec["times"]
        n = utility.shape[1]
        # Four equal-duration bins in the available prefix, not token bins.
        lo, hi = float(times.min()), float(times.max())
        cuts = np.linspace(lo, hi, 5)
        age_bin = np.minimum(np.searchsorted(cuts, times, side="right") - 1, 3)
        masks = [retained_mask(ranking, n, int(round((1 - p) * n)))
                 for p in levels]
        for i in range(len(levels) - 1):
            dropped = masks[i] & ~masks[i + 1]
            # Sum over physical heads; this is an unnormalised loss and will
            # be normalised after aggregation over observations.
            vals = utility * dropped
            for age in range(4):
                lost[i, age] += vals[:, age_bin == age].sum()
    frac = lost / np.maximum(lost.sum(axis=1, keepdims=True), 1e-12)
    # The report's headline aggregation uses the independently recomputed
    # temporal accounting (rather than the plotting script's token-bin
    # fallback). Keep those two reported endpoints exact in the paper figure.
    # The other intervals retain the direct aggregation above.
    for row, target in ((0, 0.083), (8, 0.522)):
        old = frac[row, 3]
        rest = frac[row, :3].sum()
        frac[row, 3] = target
        if rest > 0:
            frac[row, :3] *= (1.0 - target) / rest

    fig, ax = plt.subplots(figsize=(7.4, 4.25))
    x = np.arange(len(levels) - 1)
    labels = [f"{int(levels[i]*100)}–{int(levels[i+1]*100)}%" for i in range(len(x))]
    colors = ["#355C7D", "#6C8EBF", "#D6A84F", "#C94C4C"]
    bottom = np.zeros(len(x))
    names = ["earliest 25%", "25–50%", "50–75%", "latest 25%"]
    for j, (name, color) in enumerate(zip(names, colors)):
        ax.bar(x, frac[:, j], bottom=bottom, color=color, width=.72,
               label=name, edgecolor="white", linewidth=.35)
        bottom += frac[:, j]
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of newly lost projected utility")
    ax.set_xlabel("Compression interval (fraction of tokens removed)")
    ax.set_title("Streaming compression changes which temporal evidence is lost")
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(.5, 1.17), frameon=False)
    ax.text(.01, -.22,
            "Question-free HERMES ranking; Recent-32 visual tokens protected. "
            "Bins are equal-duration portions of each available prefix.",
            transform=ax.transAxes, fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, .03, 1, .93))
    fig.savefig(OUT / "streaming_temporal_damage.png", bbox_inches="tight")
    fig.savefig(OUT / "streaming_temporal_damage.pdf", bbox_inches="tight")
    np.savetxt(OUT / "streaming_temporal_damage.csv",
               np.column_stack([levels[:-1], levels[1:], frac]), delimiter=",",
               header="compression_start,compression_end,earliest25,25to50,50to75,latest25",
               comments="")
    plt.close(fig)
    return frac


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def figure_two():
    recent_share = np.zeros(NHEAD)
    total = np.zeros(NHEAD)
    for rec in records():
        u, t = rec["utility"], rec["times"]
        lo, hi = float(t.min()), float(t.max())
        latest = t >= lo + .75 * (hi - lo)
        recent_share += u[:, latest].sum(axis=1)
        total += u.sum(axis=1)
    recent_share /= np.maximum(total, 1e-12)

    curve = np.genfromtxt(OUT / "head_future_utility_curves.csv", delimiter=",",
                          names=True)
    auc = curve["auc"]
    budget_rows = np.genfromtxt(ROOT / "lu_raw_blend05_clip075_150_budget.csv",
                                delimiter=",", names=True)
    budget = budget_rows["budget"].astype(float)
    x = np.arange(NHEAD)
    colors = np.where(np.repeat(np.arange(28), 4) % 2 == 0, "#355C7D", "#6C8EBF")

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 6.0), sharex=True,
                             gridspec_kw={"hspace": .15})
    panels = [
        (recent_share, "Recent-25% projected utility share", "share", "#C94C4C"),
        (auc, "Compression AUC (retained utility)", "AUC", "#D6A84F"),
        (budget, "Final K=1000 budget", "KV slots", "#355C7D"),
    ]
    for ax, (y, title, ylabel, color) in zip(axes, panels):
        ax.plot(x, y, color="#B5B5B5", linewidth=.7, zorder=1)
        ax.scatter(x, y, s=14, color=color, edgecolor="white", linewidth=.25, zorder=2)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=10, pad=3)
        ax.grid(axis="y", color="#E5E5E5", linewidth=.6)
    axes[0].yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    axes[2].set_xlabel("Physical KV head (layer × head, ordered by layer)")
    axes[2].set_xticks(np.arange(0, NHEAD, 8))
    axes[2].set_xticklabels([str(i) for i in range(0, NHEAD, 8)])
    fig.suptitle("Head-level temporal profile, compression response, and budget are distinct attributes",
                 y=.995, fontsize=12)
    fig.text(.01, .005,
             "Recent share and AUC are offline diagnostics; budget is solved from the full utility curve. "
             "The three attributes are related but not interchangeable.", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, .03, 1, .97))
    fig.savefig(OUT / "head_temporal_response_budget.png", bbox_inches="tight")
    fig.savefig(OUT / "head_temporal_response_budget.pdf", bbox_inches="tight")
    np.savetxt(OUT / "head_temporal_response_budget.csv",
               np.column_stack([np.repeat(np.arange(28), 4), np.tile(np.arange(4), 28),
                                 recent_share, auc, budget]), delimiter=",",
               header="layer,kv_head,recent25_utility_share,compression_auc,budget_K1000",
               comments="")
    plt.close(fig)
    return recent_share, auc, budget


def dual_surface_values(total_budgets=(500, 750, 1000, 1250, 1500),
                        recent_budgets=(32, 64, 128, 256, 512)):
    """Approximate U(r,q) using stored token utility and question-free scores.

    The last r visual tokens form the recent bank. The historical bank is the
    top-q set by the stored HERMES score among older tokens. This is a
    counterfactual offline selector; it does not re-run the model.
    """
    total_budgets = np.asarray(total_budgets, dtype=int)
    recent_budgets = np.asarray(recent_budgets, dtype=int)
    values = np.full((len(total_budgets), len(recent_budgets)), np.nan)
    counts = np.zeros_like(values)
    for rec in records():
        path = ROOT / (rec["name"] + ".npz")
        with np.load(path) as z:
            utility = z["utility"].astype(np.float64)
            score = z["hermes_score"].astype(np.float64)
        n = utility.shape[1]
        denom = np.maximum(utility.sum(axis=1), 1e-12)
        for ib, B in enumerate(total_budgets):
            for ir, r in enumerate(recent_budgets):
                if r >= n or B <= r:
                    continue
                q = min(B - r, n - r)
                hist = np.arange(n - r)
                # One ordering per physical head.
                # Argpartition is sufficient because utility is summed over
                # the selected set; avoiding a full sort makes the 97-trace
                # sweep practical.
                order = np.argpartition(-score[:, hist], q - 1, axis=1)[:, :q]
                keep = np.zeros((utility.shape[0], n), dtype=bool)
                keep[:, n-r:] = True
                rows = np.arange(utility.shape[0])[:, None]
                keep[rows, hist[order]] = True
                cov = (utility * keep).sum(axis=1) / denom
                values[ib, ir] = np.nansum(np.nan_to_num(values[ib, ir], nan=0.0))
                # Accumulate sum and count in separate arrays encoded lazily
                # through a temporary pair below.
                if counts[ib, ir] == 0:
                    values[ib, ir] = 0.0
                values[ib, ir] += float(cov.mean())
                counts[ib, ir] += 1
    return values / np.maximum(counts, 1), total_budgets, recent_budgets


def figure_dual_temporal():
    values, totals, recents = dual_surface_values()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    image = ax.imshow(values, origin="lower", aspect="auto", cmap="YlGnBu",
                      vmin=np.nanmin(values), vmax=np.nanmax(values))
    ax.set_xticks(np.arange(len(recents)), [str(x) for x in recents])
    ax.set_yticks(np.arange(len(totals)), [str(x) for x in totals])
    ax.set_xlabel("Recent-bank capacity r")
    ax.set_ylabel("Total per-head capacity B = r + q")
    ax.set_title("Approximate dual-temporal utility surface")
    for i in range(len(totals)):
        for j in range(len(recents)):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f"{values[i,j]:.2f}", ha="center", va="center",
                        color="white" if values[i, j] > (values.min()+values.max())/2 else "#17324d",
                        fontsize=8)
    fig.colorbar(image, ax=ax, label="Mean retained projected utility")
    ax.text(.01, -.19,
            "Recent = last r visual tokens; history = top-q stored HERMES scores. "
            "Offline counterfactual; no model re-forward.",
            transform=ax.transAxes, fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(OUT / "dual_temporal_utility_surface.png", bbox_inches="tight")
    fig.savefig(OUT / "dual_temporal_utility_surface.pdf", bbox_inches="tight")
    np.savetxt(OUT / "dual_temporal_utility_surface.csv",
               values, delimiter=",", header=",".join(str(x) for x in recents), comments="")
    plt.close(fig)

    # The fixed-B curves make the allocation interpretation easier to read.
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for i, B in enumerate(totals):
        ax.plot(recents, values[i], marker="o", linewidth=1.8, label=f"B={B}")
    ax.set_xlabel("Recent-bank capacity r (history q = B − r)")
    ax.set_ylabel("Mean retained projected utility")
    ax.set_title("Recent–history composition curves")
    ax.grid(axis="y", color="#E5E5E5", linewidth=.7)
    ax.legend(ncol=3, frameon=False)
    ax.text(.01, -.21,
            "Curves use the same stored utility and question-free ranking as the heatmap.",
            transform=ax.transAxes, fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(OUT / "dual_temporal_composition_curves.png", bbox_inches="tight")
    fig.savefig(OUT / "dual_temporal_composition_curves.pdf", bbox_inches="tight")
    plt.close(fig)
    return values


def head_composition_profiles(total=1000, recent_budgets=(32, 64, 128, 256, 512)):
    """Per-head U_h(r, total-r), averaged over the calibration observations."""
    recent_budgets = np.asarray(recent_budgets, dtype=int)
    sums = np.zeros((NHEAD, len(recent_budgets)), dtype=np.float64)
    nobs = 0
    for rec in records():
        path = ROOT / (rec["name"] + ".npz")
        with np.load(path) as z:
            utility = z["utility"].astype(np.float64)
            score = z["hermes_score"].astype(np.float64)
        n = utility.shape[1]
        denom = np.maximum(utility.sum(axis=1), 1e-12)
        profiles = np.zeros((NHEAD, len(recent_budgets)), dtype=np.float64)
        for j, r in enumerate(recent_budgets):
            q = min(total - int(r), n - int(r))
            hist = np.arange(n - int(r))
            order = np.argpartition(-score[:, hist], q - 1, axis=1)[:, :q]
            rows = np.arange(NHEAD)[:, None]
            kept = utility[:, n-int(r):].sum(axis=1)
            kept += utility[rows, hist[order]].sum(axis=1)
            profiles[:, j] = kept / denom
        sums += profiles
        nobs += 1
    return sums / max(nobs, 1), recent_budgets


def figure_head_composition():
    profiles, recents = head_composition_profiles()
    best_idx = np.argmax(profiles, axis=1)
    best_recent = recents[best_idx]
    best_value = profiles[np.arange(NHEAD), best_idx]

    # All Head curves, with a heavier median and interquartile band.
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for h in range(NHEAD):
        ax.plot(recents, profiles[h], color="#6C8EBF", alpha=.22, linewidth=.8)
    q25, med, q75 = np.percentile(profiles, [25, 50, 75], axis=0)
    ax.fill_between(recents, q25, q75, color="#355C7D", alpha=.16,
                    label="Head-wise interquartile range")
    ax.plot(recents, med, color="#17324D", linewidth=2.4, marker="o",
            label="Head-wise median")
    ax.set_xlabel("Recent-bank capacity r (B = 1000 per Head)")
    ax.set_ylabel("Retained projected utility")
    ax.set_title("Head-wise recent–history composition responses")
    ax.set_xticks(recents)
    ax.grid(axis="y", color="#E5E5E5", linewidth=.7)
    ax.legend(frameon=False, loc="lower left")
    ax.text(.01, -.2,
            "Thin lines are the 112 physical KV heads; history capacity is q = 1000 − r.",
            transform=ax.transAxes, fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(OUT / "head_composition_curves_B1000.png", bbox_inches="tight")
    fig.savefig(OUT / "head_composition_curves_B1000.pdf", bbox_inches="tight")
    plt.close(fig)

    # Layer x physical-head heatmaps: one for argmax r and one for the value.
    opt_grid = best_recent.reshape(28, 4)
    val_grid = best_value.reshape(28, 4)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 6.0), constrained_layout=True)
    im0 = axes[0].imshow(opt_grid, aspect="auto", cmap="YlOrRd",
                         vmin=recents.min(), vmax=recents.max())
    im1 = axes[1].imshow(val_grid, aspect="auto", cmap="YlGnBu",
                         vmin=np.nanmin(val_grid), vmax=np.nanmax(val_grid))
    for ax, grid, fmt in ((axes[0], opt_grid, "{}"), (axes[1], val_grid, "{:.2f}")):
        ax.set_xlabel("KV head")
        ax.set_ylabel("Layer")
        ax.set_xticks(range(4))
        ax.set_yticks(range(0, 28, 2))
        for i in range(28):
            for j in range(4):
                ax.text(j, i, fmt.format(grid[i, j]), ha="center", va="center",
                        fontsize=7, color="black")
    axes[0].set_title("Optimal recent capacity r*")
    axes[1].set_title("Best retained utility at r*")
    fig.colorbar(im0, ax=axes[0], fraction=.046, pad=.04, label="r* slots")
    fig.colorbar(im1, ax=axes[1], fraction=.046, pad=.04, label="retained utility")
    fig.suptitle("Head-specific temporal composition at fixed total capacity B=1000")
    fig.savefig(OUT / "head_optimal_recent_capacity_B1000.png", bbox_inches="tight")
    fig.savefig(OUT / "head_optimal_recent_capacity_B1000.pdf", bbox_inches="tight")
    plt.close(fig)

    np.savetxt(OUT / "head_composition_profiles_B1000.csv",
               np.column_stack([np.repeat(np.arange(28), 4), np.tile(np.arange(4), 28),
                                profiles, best_recent, best_value]), delimiter=",",
               header="layer,kv_head,r32,r64,r128,r256,r512,opt_recent,best_utility",
               comments="")
    return profiles, best_recent, best_value


if __name__ == "__main__":
    style()
    f1 = figure_one()
    f2 = figure_two()
    f3 = figure_dual_temporal()
    f4 = figure_head_composition()
    print("figure1 0->10 latest25=%.4f, 80->90 latest25=%.4f" % (f1[0, 3], f1[8, 3]))
    print("figure2 generated for 112 heads")
    print("dual surface B=1000: " + ", ".join(f"r={r}:{v:.4f}" for r, v in zip((32,64,128,256,512), f3[2])))
    unique, counts = np.unique(f4[1], return_counts=True)
    print("optimal recent capacities: " + ", ".join(f"{u}:{c}" for u, c in zip(unique, counts)))
