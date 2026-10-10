#!/usr/bin/env python3
"""Render the head-specific temporal lifecycle overview figure."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "cvpr" / "figure1"
FRAMES = OUT / "intro_assets"

NAVY = "#17324D"
TEXT = "#344B5F"
MUTED = "#667C8E"
BLUE = "#4F7EB5"
TEAL = "#168F88"
TEAL_LIGHT = "#D8EFEC"
ORANGE = "#E97934"
ORANGE_LIGHT = "#F7DFD0"
RED = "#C94F45"
PAST = "#DFE8EE"
MIXED = "#E6E9F1"
LINE = "#B9C8D2"
PANEL = "#F7F9FB"


def rect(ax, x, y, w, h, fc, ec=LINE, lw=1.0, z=1):
    patch = Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                      linewidth=lw, zorder=z)
    ax.add_patch(patch)
    return patch


def label(ax, x, y, value, size=8, weight="normal", color=TEXT,
          ha="left", va="center", z=10):
    ax.text(x, y, value, fontsize=size, fontweight=weight, color=color,
            ha=ha, va=va, family="DejaVu Sans", zorder=z)


def token_row(ax, x, y, count, color, spacing=1.35, radius=0.28):
    for i in range(count):
        ax.add_patch(Circle((x + i * spacing, y), radius,
                            facecolor=color, edgecolor="none", zorder=5))


def arrow(ax, x1, y1, x2, y2, color=MUTED, lw=1.4):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=10, linewidth=lw,
                                 color=color, zorder=6))


def add_frame(ax, path, x, y, w, h, evidence=False):
    image = Image.open(path).convert("RGB")
    ax.imshow(np.asarray(image), extent=(x, x + w, y, y + h),
              interpolation="lanczos", aspect="auto", zorder=2)
    rect(ax, x, y, w, h, "none", ORANGE if evidence else LINE,
         2.3 if evidence else 0.8, z=4)


def draw():
    fig, ax = plt.subplots(figsize=(24, 14), dpi=160)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    label(ax, 3, 58.2, "Streaming KV Cache Needs Head-Specific Temporal Management",
          20, "bold", NAVY)
    label(ax, 3, 56.7,
          "A token evolves from fresh input to distant evidence, but physical KV heads do not share the same temporal demand.",
          9.5, color=MUTED)
    ax.plot([3, 97], [55.8, 55.8], color=NAVY, lw=1.3)

    label(ax, 3, 54.7, "Streaming video", 10, "bold", NAVY)
    label(ax, 10.2, 54.7,
          "unknown future query  ·  irreversible eviction  ·  fixed KV budget",
          8, color=MUTED)

    frame_names = [
        "sample17_278s.png", "sample17_294s.png", "sample17_310s.png",
        "sample17_342s.png", "sample17_350s.png", "sample17_356s.png",
        "sample17_362s.png",
    ]
    x0, y0, fw, fh, gap = 3, 49.7, 7.2, 4.2, 0.7
    for i, name in enumerate(frame_names):
        add_frame(ax, FRAMES / name, x0 + i * (fw + gap), y0, fw, fh,
                  evidence=(i == 3))
    evidence_x = x0 + 3 * (fw + gap) + fw / 2
    label(ax, evidence_x, 49.0, "historical evidence", 7.5, "bold",
          ORANGE, ha="center")
    ax.plot([3, 59.6], [47.9, 47.9], color=MUTED, lw=1.0)
    arrow(ax, 59.6, 47.9, 62, 47.9)
    label(ax, 3, 47.1, "past", 7, color=MUTED)
    label(ax, 56.2, 47.1, "current", 7, color=MUTED)

    ax.add_patch(Circle((64.1, 51.7), 1.25, facecolor=NAVY,
                        edgecolor="none", zorder=5))
    label(ax, 64.1, 51.7, "Q", 11, "bold", "white", ha="center")
    rect(ax, 66, 49.7, 31, 4.2, "#F5F8FA", LINE, 0.8)
    label(ax, 67.2, 52.4,
          "What is the man using to cook food in the pan?",
          8.2, "bold", NAVY)
    label(ax, 67.2, 50.8,
          "The relevant evidence appeared much earlier in the stream.",
          7.3, color=MUTED)

    # Existing methods
    rect(ax, 2.3, 30.3, 95.4, 15.8, PANEL, LINE, 0.8)
    label(ax, 3.3, 44.7,
          "(a) Existing methods impose a global temporal organization",
          12.5, "bold", NAVY)
    ax.plot([50, 50], [31.2, 43.6], color=LINE, lw=0.8)

    label(ax, 3.8, 42.7, "Content-aware compression (e.g., HERMES)",
          9.5, "bold", NAVY)
    label(ax, 3.8, 41.3,
          "Historical and newly arrived KV states enter the same selection process.",
          7.3, color=MUTED)
    label(ax, 4.2, 39.4, "HISTORICAL KV", 6.5, "bold", MUTED)
    label(ax, 20.5, 39.4, "NEW KV", 6.5, "bold", MUTED)
    rect(ax, 4.2, 36.8, 15, 2.1, PAST, "#8DA2B2", 0.8)
    token_row(ax, 5.2, 37.85, 10, "#8EA5B6", spacing=1.25)
    ax.add_patch(Circle((9.0, 37.85), 0.43, facecolor=ORANGE,
                        edgecolor="white", linewidth=1.2, zorder=6))
    rect(ax, 20.5, 36.8, 7.4, 2.1, TEAL_LIGHT, "#62AAA5", 0.8)
    token_row(ax, 21.4, 37.85, 5, TEAL, spacing=1.25)
    arrow(ax, 29.0, 37.85, 33.1, 37.85)
    label(ax, 31.1, 39.0, "UNIFIED", 6.2, "bold", MUTED, ha="center")
    label(ax, 31.1, 36.7, "TOP-B", 6.2, "bold", MUTED, ha="center")
    rect(ax, 34, 36.8, 12.7, 2.1, "white", BLUE, 1.0)
    token_row(ax, 35.1, 37.85, 6, BLUE, spacing=1.25)
    token_row(ax, 42.6, 37.85, 3, TEAL, spacing=1.25)
    ax.add_patch(Circle((32.8, 37.85), 0.42, facecolor="none",
                        edgecolor=ORANGE, linewidth=1.4, linestyle="--", zorder=7))
    ax.plot([32.35, 33.25], [37.4, 38.3], color=RED, lw=2.0, zorder=8)
    ax.plot([33.25, 32.35], [37.4, 38.3], color=RED, lw=2.0, zorder=8)
    label(ax, 40.4, 35.6, "fresh and old tokens compete for the same slots",
          7.1, color=MUTED, ha="center")
    rect(ax, 3.9, 31.6, 43.0, 2.6, "#FFF8F3", "#E8B38E", 0.7)
    label(ax, 4.8, 33.1, "Cross-age competition", 8, "bold", NAVY)
    label(ax, 4.8, 32.2,
          "Fresh details may be compressed early; sparse history may be displaced later.",
          6.8, color=TEXT)

    label(ax, 52, 42.7, "Global temporal decoupling (e.g., DSCache)",
          9.5, "bold", NAVY)
    label(ax, 52, 41.3,
          "Past and instant caches are separated, but all heads share one split.",
          7.3, color=MUTED)
    label(ax, 52.3, 39.5, "PHYSICAL HEAD", 6.4, "bold", MUTED)
    label(ax, 66.0, 39.5, "PAST CACHE", 6.4, "bold", MUTED)
    label(ax, 84.5, 39.5, "INSTANT", 6.4, "bold", MUTED)
    for i, head in enumerate(["H1", "H2", "H3"]):
        y = 38.0 - i * 2.3
        label(ax, 53, y, head, 8, "bold", NAVY)
        rect(ax, 56, y - 0.7, 25, 1.4, PAST, "#8DA2B2", 0.7)
        rect(ax, 81, y - 0.7, 11.5, 1.4, TEAL_LIGHT, "#62AAA5", 0.7)
    ax.plot([81, 81], [32.8, 39.0], color=NAVY, lw=1.0, ls="--")
    ax.plot([94, 94, 95.2, 95.2, 94], [38.7, 32.7, 32.7, 38.7, 38.7],
            color=ORANGE, lw=1.5)
    label(ax, 95.7, 35.7, "same split\nfor every head", 7.2, "bold", ORANGE)
    label(ax, 74.4, 31.4, "Different heads, identical temporal organization",
          8, "bold", NAVY, ha="center")

    # Ours
    label(ax, 3, 28.4,
          "(b) Ours: learn how each physical KV head uses information over time",
          12.5, "bold", NAVY)
    label(ax, 3, 27.0,
          "Offline temporal profiling turns heterogeneous readout into head-specific cache lifecycles.",
          8.5, color=MUTED)

    label(ax, 3.5, 24.8, "Offline temporal profile", 9.5, "bold", NAVY)
    label(ax, 3.5, 23.5,
          r"Continuous demand over token age, summarized by $\mathbf{P}_g=(H_g,N_g)$",
          7.4, color=MUTED)
    ax.plot([5, 37], [10.6, 10.6], color=MUTED, lw=1.0)
    arrow(ax, 37, 10.6, 38.2, 10.6)
    ax.plot([5, 5], [10.6, 21.8], color=MUTED, lw=1.0)
    arrow(ax, 5, 21.8, 5, 22.5)
    label(ax, 5, 9.5, "new", 7, color=MUTED)
    label(ax, 36, 9.5, "old", 7, color=MUTED)
    label(ax, 21, 8.4, "token age", 7.2, color=MUTED, ha="center")
    label(ax, 2.5, 16.5, "readout demand", 7.2, color=MUTED,
          ha="center", va="center")
    x = np.linspace(5.2, 37.2, 180)
    t = (x - x.min()) / (x.max() - x.min())
    history = 12.3 + 7.4 / (1 + np.exp(-8 * (t - 0.62))) + 0.45 * np.sin(8 * np.pi * t)
    mixed = 15.8 + 0.85 * np.sin(3.2 * np.pi * t) + 0.35 * np.sin(9 * np.pi * t)
    fresh = 12.3 + 7.4 / (1 + np.exp(8 * (t - 0.35))) + 0.35 * np.sin(7 * np.pi * t)
    ax.plot(x, history, color=ORANGE, lw=2.7)
    ax.plot(x, mixed, color=BLUE, lw=2.2)
    ax.plot(x, fresh, color=TEAL, lw=2.7)
    label(ax, 39.0, history[-1], "historical reuse", 7.3, "bold", ORANGE)
    label(ax, 39.0, mixed[-1], "mixed horizon", 7.3, "bold", BLUE)
    label(ax, 39.0, fresh[-1], "fresh persistence", 7.3, "bold", TEAL)
    label(ax, 5, 6.7, "Observed historical readout is stable on unseen videos",
          7.2, color=MUTED)
    label(ax, 5, 5.3, r"split-half Spearman $\rho=0.938$",
          8.3, "bold", NAVY)

    arrow(ax, 45, 16.1, 52, 16.1, NAVY, 1.8)
    label(ax, 48.5, 17.7, "profile → policy", 8, "bold", NAVY, ha="center")
    label(ax, 48.5, 14.5, "frozen at deployment", 7, color=MUTED, ha="center")

    label(ax, 54, 24.8, "Head-specific temporal lifecycle", 9.5, "bold", NAVY)
    label(ax, 54, 23.5,
          "Different heads receive different protection, promotion, and retention policies.",
          7.4, color=MUTED)
    label(ax, 56.5, 21.8, "NEW / PROTECTED", 6.4, "bold", MUTED)
    label(ax, 78.5, 21.8, "PROMOTED HISTORY", 6.4, "bold", MUTED)

    policy_rows = [
        ("H1", 20.1, 8.0, 28.0, ORANGE_LIGHT, "long-history head", ORANGE),
        ("H2", 16.5, 14.0, 19.0, MIXED, "mixed head", BLUE),
        ("H3", 12.9, 21.0, 8.0, MIXED, "fresh-information head", TEAL),
    ]
    for head, y, new_w, hist_w, hist_color, kind, kind_color in policy_rows:
        label(ax, 54.2, y, head, 8.5, "bold", NAVY)
        rect(ax, 56.5, y - 0.75, new_w, 1.5, TEAL_LIGHT, "#62AAA5", 0.8)
        rect(ax, 56.5 + new_w, y - 0.75, hist_w, 1.5,
             hist_color, "#8396B5" if kind != "long-history head" else "#DC8B59", 0.8)
        label(ax, min(56.5 + new_w + hist_w + 1.2, 91.5), y,
              kind, 7.2, "bold", kind_color)
    ax.add_patch(Circle((88.0, 20.1), 0.40, facecolor=ORANGE,
                        edgecolor="white", linewidth=1.2, zorder=6))

    arrow(ax, 58, 8.1, 91, 8.1, TEAL, 1.8)
    stages = [(58, "ARRIVAL"), (69, "PROTECTION"),
              (80, "PROMOTION"), (91, "RETENTION")]
    for sx, name in stages:
        ax.add_patch(Circle((sx, 8.1), 0.32,
                            facecolor=TEAL if sx < 80 else ORANGE,
                            edgecolor="white", linewidth=0.6, zorder=7))
        label(ax, sx, 6.9, name, 6.2, "bold", MUTED, ha="center")

    rect(ax, 3, 1.25, 94, 3.1, "#EEF6F5", "#85BAB6", 0.8)
    label(ax, 5, 3.05, "Head-specific temporal management", 8.8, "bold", NAVY)
    label(ax, 5, 2.05,
          "Allocate cache by how each head uses information as it ages—not one global past/current rule.",
          7.1, color=TEXT)
    ax.plot([64, 64], [1.7, 3.9], color="#9ABDB9", lw=0.8)
    label(ax, 66, 2.8,
          r"$\sum_g(B_g^{hist}+B_g^{new})=B_{total}$",
          11, "bold", NAVY)
    label(ax, 87.2, 2.8, "same total KV budget", 8, "bold", TEAL)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "head_temporal_lifecycle_overview.png",
                dpi=160, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT / "head_temporal_lifecycle_overview.pdf",
                bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT / "head_temporal_lifecycle_overview_mpl.svg",
                bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


if __name__ == "__main__":
    draw()
