"""Create a compact conceptual HE-Head panel inspired by dense CVPR figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from decord import VideoReader
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "cvpr" / "figure1"
OUT_PATH = OUT_DIR / "head_readout_compact_concept.png"
VIDEO_PATH = "/data/streamingbench/videos/sample_17_real.mp4"

NAVY = "#18324A"
INK = "#293842"
MUTED = "#6E7C86"
BLUE = "#4B78B3"
TEAL = "#15928B"
TEAL_DARK = "#086862"
TEAL_PALE = "#DDF3EF"
ORANGE = "#E8884F"
ORANGE_PALE = "#FBE8D9"
RED = "#D95F54"
PANEL = "#F3F8FA"
LINE = "#B9CDD5"
WHITE = "#FFFFFF"


def rounded(axis, xy, width, height, face, edge, radius=0.015, linewidth=1.0):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        transform=axis.transAxes,
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        clip_on=False,
    )
    axis.add_patch(patch)
    return patch


def extract_frames(times):
    reader = VideoReader(VIDEO_PATH, num_threads=1)
    fps = float(reader.get_avg_fps())
    frames = []
    for timestamp in times:
        index = min(len(reader) - 1, int(timestamp * fps))
        image = Image.fromarray(reader[index].asnumpy())
        width, height = image.size
        ratio = 16 / 9
        if width / height > ratio:
            crop_width = int(height * ratio)
            left = (width - crop_width) // 2
            image = image.crop((left, 0, left + crop_width, height))
        else:
            crop_height = int(width / ratio)
            top = (height - crop_height) // 2
            image = image.crop((0, top, width, top + crop_height))
        frames.append(np.asarray(image))
    return frames


def gaussian(x, center, width, height):
    return height * np.exp(-0.5 * ((x - center) / width) ** 2)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    times = [300, 310, 320, 330, 340, 350, 362]
    evidence_index = 4
    frames = extract_frames(times)

    fig = plt.figure(figsize=(7.2, 5.0), dpi=240, facecolor=WHITE)
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.set_xlim(0, 1)
    canvas.set_ylim(0, 1)
    canvas.axis("off")

    rounded(canvas, (0.012, 0.018), 0.976, 0.964, PANEL, LINE, radius=0.022, linewidth=1.2)

    canvas.text(
        0.035,
        0.945,
        "Same video, same question - different historical readout",
        fontsize=13.5,
        fontweight="bold",
        color=NAVY,
        va="center",
    )
    rounded(canvas, (0.035, 0.833), 0.93, 0.075, WHITE, BLUE, radius=0.016, linewidth=1.15)
    canvas.text(0.055, 0.870, "Q", fontsize=12, fontweight="bold", color=WHITE, ha="center", va="center", bbox={"boxstyle": "circle,pad=0.28", "facecolor": BLUE, "edgecolor": BLUE})
    canvas.text(
        0.092,
        0.870,
        'What is the man using to cook food in the pan?',
        fontsize=10.2,
        fontweight="bold",
        color=INK,
        va="center",
    )
    canvas.text(0.943, 0.870, "$t_q=363s$", fontsize=9.2, fontweight="bold", color=BLUE, ha="right", va="center")

    track_left = 0.180
    track_right = 0.958
    track_width = track_right - track_left
    he_y0, weak_y0 = 0.681, 0.566
    track_h = 0.090
    for y0 in [he_y0, weak_y0]:
        rounded(canvas, (track_left, y0), track_width, track_h, WHITE, "#D8E4E8", radius=0.010, linewidth=0.75)

    rounded(canvas, (0.035, he_y0 + 0.012), 0.125, 0.065, TEAL_PALE, TEAL_PALE, radius=0.012)
    canvas.text(0.0975, he_y0 + 0.045, "HE-Head", fontsize=10, fontweight="bold", color=TEAL_DARK, ha="center", va="center")
    rounded(canvas, (0.035, weak_y0 + 0.012), 0.125, 0.065, ORANGE_PALE, ORANGE_PALE, radius=0.012)
    canvas.text(0.0975, weak_y0 + 0.045, "Weak Head", fontsize=10, fontweight="bold", color=RED, ha="center", va="center")

    x = np.linspace(0, 1, 500)
    evidence_center = (evidence_index + 0.5) / len(times)
    he = 0.06 + gaussian(x, 0.23, 0.035, 0.24) + gaussian(x, 0.47, 0.045, 0.18) + gaussian(x, evidence_center, 0.035, 0.82)
    weak = 0.055 + gaussian(x, 0.08, 0.030, 0.19) + gaussian(x, 0.91, 0.040, 0.61)

    def draw_track(values, y0, color):
        px = track_left + x * track_width
        py = y0 + 0.012 + values / 1.05 * (track_h - 0.024)
        canvas.fill_between(px, y0 + 0.012, py, color=color, alpha=0.18, transform=canvas.transAxes, linewidth=0)
        canvas.plot(px, py, color=color, linewidth=2.0, transform=canvas.transAxes, solid_capstyle="round")

    draw_track(he, he_y0, TEAL)
    draw_track(weak, weak_y0, ORANGE)

    evidence_x = track_left + evidence_center * track_width
    latest_x = track_left + 0.91 * track_width
    canvas.plot([evidence_x, evidence_x], [0.272, he_y0 + track_h], color=TEAL, linewidth=1.0, linestyle=(0, (3, 3)), alpha=0.8)
    canvas.plot([latest_x, latest_x], [weak_y0, weak_y0 + track_h], color=ORANGE, linewidth=1.0, linestyle=(0, (3, 3)), alpha=0.8)
    canvas.text(evidence_x, he_y0 + track_h + 0.012, "evidence peak", fontsize=8.2, fontweight="bold", color=TEAL_DARK, ha="center")
    canvas.text(latest_x, weak_y0 + track_h + 0.012, "recent peak", fontsize=8.2, fontweight="bold", color=RED, ha="center")

    strip_left = track_left
    strip_right = track_right
    gap = 0.009
    frame_width = (strip_right - strip_left - gap * (len(frames) - 1)) / len(frames)
    frame_y, frame_h = 0.300, 0.205
    centers = []
    for index, (frame, timestamp) in enumerate(zip(frames, times)):
        left = strip_left + index * (frame_width + gap)
        centers.append(left + frame_width / 2)
        frame_axis = fig.add_axes([left, frame_y, frame_width, frame_h])
        frame_axis.imshow(frame)
        frame_axis.set_xticks([])
        frame_axis.set_yticks([])
        for spine in frame_axis.spines.values():
            spine.set_linewidth(2.5 if index == evidence_index else 0.8)
            spine.set_color(TEAL if index == evidence_index else WHITE)
        canvas.text(left + frame_width / 2, frame_y - 0.022, f"{timestamp}s", fontsize=7.5, color=MUTED, ha="center", va="top")

    canvas.text(0.090, frame_y + frame_h / 2, "Video\nhistory", fontsize=9.5, fontweight="bold", color=NAVY, ha="center", va="center")
    canvas.annotate("", xy=(track_right, 0.255), xytext=(track_left, 0.255), arrowprops={"arrowstyle": "->", "color": MUTED, "linewidth": 1.0})
    canvas.plot([track_right, track_right], [0.242, 0.515], color=BLUE, linewidth=2.0)
    canvas.text(track_right, 0.225, "$t_q$", fontsize=9, fontweight="bold", color=BLUE, ha="center")

    rounded(canvas, (0.035, 0.105), 0.445, 0.095, TEAL_PALE, TEAL_PALE, radius=0.015)
    canvas.text(0.057, 0.164, "HE-Head", fontsize=9.2, fontweight="bold", color=TEAL_DARK, va="center")
    canvas.text(0.057, 0.128, "retrieves answer-supporting history", fontsize=8.3, color=INK, va="center")
    canvas.text(0.455, 0.152, "finds: red tongs", fontsize=9, fontweight="bold", color=TEAL_DARK, ha="right", va="center")

    rounded(canvas, (0.495, 0.105), 0.470, 0.095, ORANGE_PALE, ORANGE_PALE, radius=0.015)
    canvas.text(0.518, 0.164, "Weak Head", fontsize=9.2, fontweight="bold", color=RED, va="center")
    canvas.text(0.518, 0.128, "focuses on recent / structural positions", fontsize=8.3, color=INK, va="center")
    canvas.text(0.942, 0.152, "misses evidence", fontsize=9, fontweight="bold", color=RED, ha="right", va="center")

    canvas.text(0.035, 0.053, "Historical readout is head-specific.", fontsize=10.2, fontweight="bold", color=INK, va="center")
    canvas.text(0.963, 0.053, "Seeing is not remembering.", fontsize=10.2, fontweight="bold", color=TEAL_DARK, ha="right", va="center")
    canvas.text(0.963, 0.023, "Conceptual illustration", fontsize=6.8, color=MUTED, ha="right", va="bottom")

    fig.savefig(OUT_PATH, dpi=240, facecolor=WHITE)
    plt.close(fig)
    print(OUT_PATH)


if __name__ == "__main__":
    main()
