"""Render a paper-style prototype for the HE-Head readout timeline."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from decord import VideoReader
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "cvpr" / "figure1"
OUT_PATH = OUT_DIR / "head_readout_progressbar_prototype.png"

TOP_CSV = (
    ROOT
    / "results/observations/effective_readout_core_top100_attention_top4_n2_o8"
    / "true_question_readout_selected_kv_heads.csv"
)
BOTTOM_CSV = (
    ROOT
    / "results/observations/effective_readout_last4_spiky_core_attention_bottom4_n2_o8"
    / "true_question_readout_selected_kv_heads.csv"
)
VIDEO_PATH = "/data/streamingbench/videos/sample_17_real.mp4"

NAVY = "#18324A"
TEAL = "#168C86"
TEAL_DARK = "#0B6864"
TEAL_LIGHT = "#DDF3F0"
CORAL = "#DE6655"
CORAL_LIGHT = "#FBE8E3"
GOLD = "#E5A832"
INK = "#273640"
MUTED = "#687985"
GRID = "#DCE4E8"
LIGHT = "#F4F7F8"


def load_curve(path, name):
    frame = pd.read_csv(path)
    part = frame[frame["name"] == name].sort_values("x")
    # The prior-memory region ends near x=0.2226. Remove both boundary windows.
    part = part[(part["x"] >= 0.0225) & (part["x"] <= 0.1875)].copy()
    x = part["x"].to_numpy(float)
    y = part["mean_density"].to_numpy(float)
    x = (x - x.min()) / (x.max() - x.min())
    y = np.convolve(y, np.asarray([0.2, 0.6, 0.2]), mode="same") * 1e4
    return x, y


def extract_frames(times):
    reader = VideoReader(VIDEO_PATH, num_threads=1)
    fps = float(reader.get_avg_fps())
    frames = []
    for timestamp in times:
        index = min(len(reader) - 1, int(timestamp * fps))
        frames.append(reader[index].asnumpy())
    return frames


def style_axis(axis):
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color(GRID)
    axis.tick_params(axis="both", colors=MUTED, labelsize=8, length=0)
    axis.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.65)
    axis.set_xlim(0, 1)


def add_frame_strip(axis, frames, times, evidence_index):
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    frame_width = 0.112
    lefts = np.linspace(0.005, 0.995 - frame_width, len(frames))
    centers = lefts + frame_width / 2
    for index, (left, center, frame, timestamp) in enumerate(zip(lefts, centers, frames, times)):
        image = Image.fromarray(frame)
        width, height = image.size
        target_ratio = 16 / 9
        if width / height > target_ratio:
            crop_width = int(height * target_ratio)
            left = (width - crop_width) // 2
            image = image.crop((left, 0, left + crop_width, height))
        else:
            crop_height = int(width / target_ratio)
            top = (height - crop_height) // 2
            image = image.crop((0, top, width, top + crop_height))
        inset = axis.inset_axes([left, 0.20, frame_width, 0.69], transform=axis.transAxes)
        inset.imshow(np.asarray(image))
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_color(TEAL if index == evidence_index else "white")
            spine.set_linewidth(3.0 if index == evidence_index else 1.2)
        axis.text(center, 0.08, f"{timestamp}s", ha="center", va="center", fontsize=8, color=MUTED)
    axis.annotate(
        "video history",
        xy=(0.97, 0.01),
        xytext=(0.03, 0.01),
        arrowprops={"arrowstyle": "->", "color": MUTED, "linewidth": 1.2},
        ha="left",
        va="center",
        fontsize=8,
        color=MUTED,
    )
    return centers


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    he_x, he_y = load_curve(TOP_CSV, "L22-KV0")
    weak_x, weak_y = load_curve(BOTTOM_CSV, "L0-KV3")
    shared_max = max(float(he_y.max()), float(weak_y.max())) * 1.13

    times = [300, 310, 320, 330, 340, 350, 356, 362]
    evidence_index = 4
    frames = extract_frames(times)

    fig = plt.figure(figsize=(16, 8.2), dpi=200, facecolor="white")
    grid = fig.add_gridspec(
        6,
        12,
        height_ratios=[0.74, 1.18, 1.18, 1.52, 0.56, 0.16],
        left=0.045,
        right=0.965,
        top=0.94,
        bottom=0.07,
        hspace=0.42,
        wspace=0.20,
    )

    title_ax = fig.add_subplot(grid[0, :])
    title_ax.axis("off")
    title_ax.text(
        0,
        0.78,
        "Different Heads Read Different Parts of Video History",
        fontsize=22,
        fontweight="bold",
        color=NAVY,
        va="top",
    )
    title_ax.text(
        0,
        0.22,
        "True-question readout over boundary-cleaned internal memory",
        fontsize=11,
        color=MUTED,
        va="top",
    )
    title_ax.text(
        0.985,
        0.58,
        'Question at 363s\n"What is the man using to cook food in the pan?"',
        fontsize=11,
        fontweight="bold",
        color=NAVY,
        ha="right",
        va="center",
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "white", "edgecolor": "#3568A8", "linewidth": 1.4},
    )

    he_ax = fig.add_subplot(grid[1, 1:11])
    weak_ax = fig.add_subplot(grid[2, 1:11], sharex=he_ax)
    for axis in [he_ax, weak_ax]:
        style_axis(axis)
        axis.set_ylim(0, shared_max)
        axis.set_yticks([0, round(shared_max * 0.5, 1)])
        axis.tick_params(axis="x", labelbottom=False)

    he_ax.plot(he_x, he_y, color=TEAL, linewidth=2.1)
    he_ax.fill_between(he_x, 0, he_y, color=TEAL, alpha=0.23)
    weak_ax.plot(weak_x, weak_y, color=CORAL, linewidth=2.1)
    weak_ax.fill_between(weak_x, 0, weak_y, color=CORAL, alpha=0.18)

    he_ax.text(-0.105, 0.73, "HE-Head", transform=he_ax.transAxes, fontsize=12, fontweight="bold", color=TEAL_DARK)
    he_ax.text(-0.105, 0.45, "L22-KV0", transform=he_ax.transAxes, fontsize=9, color=MUTED)
    he_ax.text(-0.105, 0.22, "high $M_g$", transform=he_ax.transAxes, fontsize=8.5, color=TEAL_DARK)
    weak_ax.text(-0.105, 0.73, "Weak Head", transform=weak_ax.transAxes, fontsize=12, fontweight="bold", color=CORAL)
    weak_ax.text(-0.105, 0.45, "L0-KV3", transform=weak_ax.transAxes, fontsize=9, color=MUTED)
    weak_ax.text(-0.105, 0.22, "low $M_g$", transform=weak_ax.transAxes, fontsize=8.5, color=CORAL)
    he_ax.text(-0.018, 1.03, "readout density (x 1e-4)", transform=he_ax.transAxes, fontsize=7.8, color=MUTED, ha="right")

    peak_index = int(np.argmax(he_y))
    he_ax.scatter([he_x[peak_index]], [he_y[peak_index]], s=55, color=TEAL, edgecolor="white", linewidth=1.2, zorder=5)
    he_ax.annotate(
        "selective internal-memory peak",
        xy=(he_x[peak_index], he_y[peak_index]),
        xytext=(min(he_x[peak_index] + 0.09, 0.72), shared_max * 0.87),
        fontsize=9,
        color=TEAL_DARK,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": TEAL, "linewidth": 1.2},
    )
    weak_ax.text(
        0.985,
        0.72,
        "same observations  |  same cache protocol  |  same y-scale",
        transform=weak_ax.transAxes,
        ha="right",
        fontsize=8.5,
        color=MUTED,
    )

    frame_ax = fig.add_subplot(grid[3, 1:11])
    centers = add_frame_strip(frame_ax, frames, times, evidence_index)
    frame_ax.annotate(
        "answer-supporting frame\n(red kitchen tongs)",
        xy=(centers[evidence_index], 0.90),
        xytext=(centers[evidence_index] - 0.10, 1.12),
        ha="center",
        va="bottom",
        fontsize=9,
        color=TEAL_DARK,
        fontweight="bold",
        arrowprops={"arrowstyle": "-|>", "color": TEAL, "linewidth": 1.4},
    )
    frame_ax.axvline(0.995, color="#3568A8", linewidth=2.2, ymin=0.10, ymax=0.95)
    frame_ax.text(0.995, 0.98, "$t_q$", ha="center", va="bottom", fontsize=10, color="#3568A8", fontweight="bold")

    takeaway_ax = fig.add_subplot(grid[4, 1:11])
    takeaway_ax.axis("off")
    takeaway_ax.add_patch(
        plt.Rectangle((0.0, 0.10), 1.0, 0.80, transform=takeaway_ax.transAxes, facecolor=LIGHT, edgecolor=GRID, linewidth=0.8)
    )
    takeaway_ax.text(0.02, 0.50, "Observation", fontsize=10, color=NAVY, fontweight="bold", va="center")
    takeaway_ax.text(
        0.13,
        0.50,
        "Physical KV Heads exhibit different readout over clean video history under real questions.",
        fontsize=10,
        color=INK,
        fontweight="bold",
        va="center",
    )
    takeaway_ax.text(
        0.975,
        0.50,
        "Seeing is not remembering.",
        fontsize=10.5,
        color=TEAL_DARK,
        fontweight="bold",
        ha="right",
        va="center",
    )

    foot_ax = fig.add_subplot(grid[5, :])
    foot_ax.axis("off")
    foot_ax.text(
        0,
        0.5,
        "Layout prototype. Curves are real true-question readout densities averaged over 8 profile observations and shown on normalized internal-memory position; the video/question pair illustrates the intended single-case alignment and must be replaced by a matched token-to-frame capture in the final paper figure.",
        fontsize=7.7,
        color=MUTED,
        va="center",
    )

    fig.savefig(OUT_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUT_PATH)


if __name__ == "__main__":
    main()
