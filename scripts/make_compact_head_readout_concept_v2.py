"""Create a dense conceptual head-readout panel with MCQ and frame strip."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from decord import VideoReader
from matplotlib.patches import FancyBboxPatch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "cvpr" / "figure1"
OUT_PATH = OUT_DIR / "head_readout_compact_concept_v2.png"
VIDEO_PATH = "/data/streamingbench/videos/sample_17_real.mp4"

NAVY = "#19334C"
INK = "#2B3943"
MUTED = "#71808A"
BLUE = "#4A78B4"
TEAL = "#14918A"
TEAL_DARK = "#086B65"
TEAL_PALE = "#DEF3F0"
ORANGE = "#E98247"
ORANGE_DARK = "#CB6431"
ORANGE_PALE = "#FBE7D8"
PANEL = "#F3F8FA"
LINE = "#BDD0D7"
WHITE = "#FFFFFF"


def rounded(axis, xy, width, height, face, edge, radius=0.015, linewidth=1.0):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.005,rounding_size={radius}",
        transform=axis.transAxes,
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        clip_on=False,
    )
    axis.add_patch(patch)
    return patch


def crop_16_9(image):
    width, height = image.size
    ratio = 16 / 9
    if width / height > ratio:
        crop_width = int(height * ratio)
        left = (width - crop_width) // 2
        return image.crop((left, 0, left + crop_width, height))
    crop_height = int(width / ratio)
    top = (height - crop_height) // 2
    return image.crop((0, top, width, top + crop_height))


def extract_frames(times):
    reader = VideoReader(VIDEO_PATH, num_threads=1)
    fps = float(reader.get_avg_fps())
    frames = []
    for timestamp in times:
        index = min(len(reader) - 1, int(timestamp * fps))
        frames.append(np.asarray(crop_16_9(Image.fromarray(reader[index].asnumpy()))))
    return frames


def gaussian(x, center, width, height):
    return height * np.exp(-0.5 * ((x - center) / width) ** 2)


def conceptual_curves(x, evidence_center):
    he_specs = [
        (0.035, 0.016, 0.22),
        (0.105, 0.023, 0.39),
        (0.175, 0.018, 0.26),
        (evidence_center, 0.027, 0.86),
        (0.365, 0.020, 0.31),
        (0.445, 0.028, 0.45),
        (0.545, 0.018, 0.20),
        (0.625, 0.024, 0.34),
        (0.715, 0.022, 0.25),
        (0.805, 0.028, 0.42),
        (0.920, 0.022, 0.24),
    ]
    other_specs = [
        (0.025, 0.018, 0.31),
        (0.090, 0.027, 0.48),
        (0.185, 0.022, 0.25),
        (0.345, 0.025, 0.18),
        (0.435, 0.018, 0.28),
        (0.555, 0.024, 0.22),
        (0.675, 0.025, 0.34),
        (0.790, 0.030, 0.51),
        (0.885, 0.026, 0.76),
        (0.955, 0.018, 0.45),
    ]
    he = 0.055 + 0.018 * (np.sin(17 * np.pi * x) + 1)
    other = 0.055 + 0.016 * (np.cos(15 * np.pi * x) + 1)
    for center, width, height in he_specs:
        he += gaussian(x, center, width, height)
    for center, width, height in other_specs:
        other += gaussian(x, center, width, height)
    return he, other


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    times = [278, 286, 294, 302, 310, 318, 326, 334, 342, 350, 356, 362]
    evidence_index = 3
    frames = extract_frames(times)

    fig = plt.figure(figsize=(7.2, 5.0), dpi=240, facecolor=WHITE)
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.set_xlim(0, 1)
    canvas.set_ylim(0, 1)
    canvas.axis("off")
    rounded(canvas, (0.012, 0.018), 0.976, 0.964, PANEL, LINE, radius=0.022, linewidth=1.15)

    canvas.text(0.035, 0.950, "(a) Head-wise historical readout", fontsize=12.8, fontweight="bold", color=NAVY, va="center")

    rounded(canvas, (0.035, 0.746), 0.930, 0.155, WHITE, BLUE, radius=0.015, linewidth=1.1)
    canvas.text(0.056, 0.854, "Q", fontsize=10.5, fontweight="bold", color=WHITE, ha="center", va="center", bbox={"boxstyle": "circle,pad=0.25", "facecolor": BLUE, "edgecolor": BLUE})
    canvas.text(0.089, 0.854, "What is the man using to cook food in the pan?", fontsize=9.8, fontweight="bold", color=INK, va="center")
    canvas.text(0.942, 0.854, "$t_q=363s$", fontsize=8.8, fontweight="bold", color=BLUE, ha="right", va="center")
    options = [
        ("A", "wooden spoon"),
        ("B", "metal spatula"),
        ("C", "red kitchen tongs"),
        ("D", "fork"),
    ]
    option_x = [0.090, 0.315, 0.550, 0.805]
    option_w = [0.190, 0.200, 0.220, 0.125]
    for (letter, text), left, width in zip(options, option_x, option_w):
        fill = TEAL_PALE if letter == "C" else "#F5F7F8"
        edge = TEAL if letter == "C" else "#E0E7EA"
        rounded(canvas, (left, 0.775), width, 0.042, fill, edge, radius=0.008, linewidth=0.65)
        canvas.text(left + 0.012, 0.796, f"{letter}.", fontsize=7.7, fontweight="bold", color=TEAL_DARK if letter == "C" else MUTED, va="center")
        canvas.text(left + 0.036, 0.796, text, fontsize=7.4, color=INK, va="center")

    label_left = 0.035
    track_left = 0.160
    track_right = 0.965
    track_width = track_right - track_left
    track_h = 0.103
    he_y0 = 0.603
    other_y0 = 0.485
    for y0 in [he_y0, other_y0]:
        rounded(canvas, (track_left, y0), track_width, track_h, WHITE, "#D8E4E8", radius=0.009, linewidth=0.70)
    rounded(canvas, (label_left, he_y0 + 0.017), 0.108, 0.068, TEAL_PALE, TEAL_PALE, radius=0.010)
    rounded(canvas, (label_left, other_y0 + 0.017), 0.108, 0.068, ORANGE_PALE, ORANGE_PALE, radius=0.010)
    canvas.text(label_left + 0.054, he_y0 + 0.051, "HE-Head", fontsize=9.2, fontweight="bold", color=TEAL_DARK, ha="center", va="center")
    canvas.text(label_left + 0.054, other_y0 + 0.051, "Other Head", fontsize=9.0, fontweight="bold", color=ORANGE_DARK, ha="center", va="center")

    x = np.linspace(0, 1, 900)
    evidence_center = (evidence_index + 0.5) / len(times)
    he, other = conceptual_curves(x, evidence_center)
    shared_max = max(float(he.max()), float(other.max())) * 1.04

    def draw_track(values, y0, color):
        px = track_left + x * track_width
        py = y0 + 0.010 + values / shared_max * (track_h - 0.020)
        canvas.fill_between(px, y0 + 0.010, py, color=color, alpha=0.18, transform=canvas.transAxes, linewidth=0)
        canvas.plot(px, py, color=color, linewidth=1.65, transform=canvas.transAxes, solid_capstyle="round")

    draw_track(he, he_y0, TEAL)
    draw_track(other, other_y0, ORANGE)

    strip_left = track_left
    strip_right = track_right
    frame_gap = 0.004
    frame_width = (strip_right - strip_left - frame_gap * (len(frames) - 1)) / len(frames)
    frame_y = 0.260
    frame_h = 0.190
    centers = []
    for index, (frame, timestamp) in enumerate(zip(frames, times)):
        left = strip_left + index * (frame_width + frame_gap)
        center = left + frame_width / 2
        centers.append(center)
        frame_axis = fig.add_axes([left, frame_y, frame_width, frame_h])
        frame_axis.imshow(frame)
        frame_axis.set_xticks([])
        frame_axis.set_yticks([])
        for spine in frame_axis.spines.values():
            spine.set_linewidth(2.4 if index == evidence_index else 0.6)
            spine.set_color(TEAL if index == evidence_index else WHITE)
        canvas.text(center, frame_y - 0.012, f"{timestamp}", fontsize=6.3, color=MUTED, ha="center", va="top")

    evidence_x = centers[evidence_index]
    canvas.plot([evidence_x, evidence_x], [frame_y + frame_h, he_y0 + track_h], color=TEAL, linewidth=1.05, linestyle=(0, (3, 3)), alpha=0.85)
    canvas.text(evidence_x, he_y0 + track_h + 0.006, "evidence", fontsize=7.5, fontweight="bold", color=TEAL_DARK, ha="center", va="bottom")

    canvas.text(0.090, frame_y + frame_h / 2, "Video\nhistory", fontsize=8.9, fontweight="bold", color=NAVY, ha="center", va="center")
    canvas.annotate("", xy=(track_right, 0.218), xytext=(track_left, 0.218), arrowprops={"arrowstyle": "->", "color": MUTED, "linewidth": 0.9})
    canvas.plot([track_right, track_right], [0.203, frame_y + frame_h], color=BLUE, linewidth=1.8)
    canvas.text(track_right, 0.185, "$t_q$", fontsize=8.5, fontweight="bold", color=BLUE, ha="center")

    canvas.text(0.040, 0.115, "HE-Head", fontsize=9.0, fontweight="bold", color=TEAL_DARK, va="center")
    canvas.text(0.145, 0.115, "retrieves C", fontsize=9.0, fontweight="bold", color=INK, va="center")
    canvas.text(0.535, 0.115, "Other Head", fontsize=9.0, fontweight="bold", color=ORANGE_DARK, va="center")
    canvas.text(0.665, 0.115, "misses historical evidence", fontsize=9.0, fontweight="bold", color=INK, va="center")
    canvas.text(0.965, 0.045, "Conceptual illustration", fontsize=6.5, color=MUTED, ha="right", va="center")

    fig.savefig(OUT_PATH, dpi=240, facecolor=WHITE)
    plt.close(fig)
    print(OUT_PATH)


if __name__ == "__main__":
    main()
