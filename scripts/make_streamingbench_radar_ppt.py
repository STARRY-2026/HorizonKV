"""Create an editable StreamingBench task radar and average-score summary."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.chart.data import ChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_TICK_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results/cvpr/figure1"
PPT_PATH = OUT_DIR / "streamingbench_method_radar.pptx"
PREVIEW_PATH = OUT_DIR / "streamingbench_method_radar_preview.png"

TASKS = ["Action", "Attribute", "Object", "Spatial", "Event", "Text-rich", "Counting", "Causal", "Clips", "Prospective"]
METHODS = {
    "HERMES (local)": {
        "avg": 76.95,
        "values": [68.84, 85.81, 80.11, 71.95, 74.84, 84.11, 52.85, 75.00, 86.44, 78.70],
        "color": "14918A",
        "source": "local",
    },
    "StreamingVLM (local)": {
        "avg": 76.95,
        "values": [70.54, 88.12, 79.84, 72.76, 76.10, 83.49, 41.97, 72.66, 88.33, 82.41],
        "color": "4A78B4",
        "source": "local",
    },
    "StreamForest-7B": {
        "avg": 77.26,
        "values": [75.64, 84.26, 83.11, 69.11, 77.50, 78.19, 54.40, 82.81, 82.65, 76.85],
        "color": "D95F54",
        "source": "reported",
    },
    "TimeChat-Online-7B": {
        "avg": 75.36,
        "values": [69.60, 83.33, 80.22, 64.63, 76.10, 78.50, 57.98, 82.03, 79.50, 78.70],
        "color": "E6A52E",
        "source": "reported",
    },
    "Dispider-7B": {
        "avg": 67.63,
        "values": [62.16, 73.08, 74.92, 62.91, 74.44, 59.92, 45.80, 75.53, 74.10, 76.14],
        "color": "8C79B5",
        "source": "reported",
    },
}

NAVY, INK, MUTED = "19334C", "2B3943", "71808A"
PANEL, LINE, WHITE = "F3F8FA", "BDD0D7", "FFFFFF"
OURS_PALE, LIGHT = "DEF3F0", "E8EEF1"


def rgb(value):
    return RGBColor.from_string(value)


def add_text(slide, value, x, y, w, h, size=10, color=INK, bold=False,
             align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.MIDDLE):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = shape.text_frame
    frame.clear()
    frame.margin_left = frame.margin_right = Inches(0.02)
    frame.margin_top = frame.margin_bottom = Inches(0.01)
    frame.vertical_anchor = valign
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    run = paragraph.add_run()
    run.text = value
    run.font.name = "Arial"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = rgb(color)
    return shape


def add_rect(slide, x, y, w, h, fill=WHITE, edge=LINE, rounded=False, width=0.8):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(fill)
    shape.line.color.rgb = rgb(edge)
    shape.line.width = Pt(width)
    return shape


def add_line(slide, x1, y1, x2, y2, color=LINE, width=0.8):
    line = slide.shapes.add_connector(1, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    line.line.color.rgb = rgb(color)
    line.line.width = Pt(width)
    return line


def add_native_radar(slide):
    data = ChartData()
    data.categories = TASKS
    for method, info in METHODS.items():
        data.add_series(method, info["values"])
    chart = slide.shapes.add_chart(
        XL_CHART_TYPE.RADAR_MARKERS,
        Inches(0.50), Inches(1.20), Inches(8.20), Inches(5.45), data,
    ).chart
    chart.has_title = False
    chart.has_legend = True
    chart.legend.position = XL_LEGEND_POSITION.BOTTOM
    chart.legend.include_in_layout = False
    chart.legend.font.name = "Arial"
    chart.legend.font.size = Pt(8.5)
    chart.value_axis.minimum_scale = 0
    chart.value_axis.maximum_scale = 100
    chart.value_axis.major_unit = 20
    chart.value_axis.tick_labels.font.name = "Arial"
    chart.value_axis.tick_labels.font.size = Pt(7)
    chart.value_axis.tick_label_position = XL_TICK_LABEL_POSITION.NEXT_TO_AXIS
    chart.category_axis.tick_labels.font.name = "Arial"
    chart.category_axis.tick_labels.font.size = Pt(8.5)
    chart.category_axis.tick_labels.font.bold = True
    chart.category_axis.tick_labels.font.color.rgb = rgb(INK)
    for series, (_, info) in zip(chart.series, METHODS.items()):
        series.format.line.color.rgb = rgb(info["color"])
        series.format.line.width = Pt(2.0 if info["source"] == "local" else 1.25)
        series.marker.format.fill.solid()
        series.marker.format.fill.fore_color.rgb = rgb(info["color"])
        series.marker.format.line.color.rgb = rgb(WHITE)
        series.marker.size = 5 if info["source"] == "local" else 4
    return chart


def add_average_panel(slide):
    x, y, w, h = 8.92, 1.20, 3.96, 5.45
    add_rect(slide, x, y, w, h, PANEL, LINE, rounded=True)
    add_text(slide, "Average accuracy (%)", x + 0.22, y + 0.18, 3.48, 0.33, size=12, color=NAVY, bold=True)
    add_text(slide, "StreamingBench, 10-task mean", x + 0.22, y + 0.50, 3.48, 0.23, size=8, color=MUTED)
    add_line(slide, x + 0.22, y + 0.82, x + 3.72, y + 0.82)

    row_y = y + 1.02
    for method, info in METHODS.items():
        add_rect(slide, x + 0.22, row_y + 0.11, 0.18, 0.18, info["color"], info["color"], rounded=True)
        add_text(slide, method, x + 0.53, row_y, 2.25, 0.40, size=8.8, color=INK, bold=info["source"] == "local")
        add_text(slide, f'{info["avg"]:.2f}', x + 2.90, row_y, 0.70, 0.40, size=11, color=info["color"], bold=True, align=PP_ALIGN.RIGHT)
        row_y += 0.55

    add_line(slide, x + 0.22, row_y + 0.02, x + 3.72, row_y + 0.02)
    add_rect(slide, x + 0.18, row_y + 0.24, 3.58, 0.66, OURS_PALE, OURS_PALE, rounded=True)
    add_text(slide, "MemHeadKV (ours)", x + 0.42, row_y + 0.35, 2.08, 0.34, size=10, color="086B65", bold=True)
    # Intentionally leave the result cell empty until the full evaluation finishes.
    add_rect(slide, x + 2.90, row_y + 0.38, 0.64, 0.24, WHITE, "8FC9C4", rounded=True)

    add_text(slide, "Local reruns", x + 0.25, y + 4.79, 0.92, 0.24, size=7.3, color=NAVY, bold=True)
    add_text(slide, "HERMES, StreamingVLM", x + 1.14, y + 4.79, 2.48, 0.24, size=7.3, color=MUTED)
    add_text(slide, "Reported", x + 0.25, y + 5.07, 0.92, 0.24, size=7.3, color=NAVY, bold=True)
    add_text(slide, "HERMES Table 11", x + 1.14, y + 5.07, 2.48, 0.24, size=7.3, color=MUTED)


def make_ppt():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.2)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = rgb(WHITE)
    add_text(slide, "StreamingBench Performance Profile", 0.38, 0.16, 7.10, 0.43, size=20, color=NAVY, bold=True)
    add_text(slide, "Task-wise accuracy; average score shown at right", 7.22, 0.23, 5.72, 0.28, size=9.5, color=MUTED, align=PP_ALIGN.RIGHT)
    add_line(slide, 0.38, 0.67, 12.95, 0.67)
    add_native_radar(slide)
    add_average_panel(slide)
    add_text(
        slide,
        "External online methods use their reported settings (1 FPS); local reruns use 0.5 FPS and 6K KV. Use the final controlled table for claims.",
        0.48, 6.78, 12.35, 0.23, size=7.2, color=MUTED, align=PP_ALIGN.CENTER,
    )
    prs.save(PPT_PATH)


def make_preview():
    import matplotlib.pyplot as plt
    import numpy as np

    angles = np.linspace(0, 2 * np.pi, len(TASKS), endpoint=False).tolist()
    closed_angles = angles + angles[:1]
    fig = plt.figure(figsize=(13.333, 7.2), dpi=180, facecolor="white")
    ax = fig.add_axes([0.055, 0.14, 0.60, 0.73], polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(TASKS, fontsize=8.5, fontweight="bold", color="#" + INK)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=7, color="#" + MUTED)
    ax.grid(color="#dbe5e9", linewidth=0.7)
    ax.spines["polar"].set_color("#" + LINE)
    for method, info in METHODS.items():
        values = info["values"] + info["values"][:1]
        linewidth = 2.1 if info["source"] == "local" else 1.3
        ax.plot(closed_angles, values, color="#" + info["color"], linewidth=linewidth, marker="o", markersize=3.2, label=method)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.20), ncol=3, frameon=False, fontsize=7.5)

    fig.text(0.035, 0.935, "StreamingBench Performance Profile", fontsize=20, fontweight="bold", color="#" + NAVY)
    fig.text(0.96, 0.94, "Task-wise accuracy; average score shown at right", ha="right", fontsize=9, color="#" + MUTED)
    panel = fig.add_axes([0.69, 0.15, 0.275, 0.72])
    panel.axis("off")
    panel.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor="#" + PANEL, edgecolor="#" + LINE, linewidth=1))
    panel.text(0.08, 0.93, "Average accuracy (%)", fontsize=12, fontweight="bold", color="#" + NAVY)
    y = 0.81
    for method, info in METHODS.items():
        panel.add_patch(plt.Rectangle((0.08, y - 0.012), 0.035, 0.024, color="#" + info["color"]))
        panel.text(0.15, y, method, va="center", fontsize=8.5, fontweight="bold" if info["source"] == "local" else "normal", color="#" + INK)
        panel.text(0.91, y, f'{info["avg"]:.2f}', va="center", ha="right", fontsize=10.5, fontweight="bold", color="#" + info["color"])
        y -= 0.105
    panel.add_patch(plt.Rectangle((0.06, y - 0.035), 0.88, 0.105, facecolor="#" + OURS_PALE, edgecolor="none"))
    panel.text(0.14, y + 0.017, "MemHeadKV (ours)", va="center", fontsize=10, fontweight="bold", color="#086b65")
    panel.add_patch(plt.Rectangle((0.78, y - 0.002), 0.13, 0.04, facecolor="white", edgecolor="#8fc9c4"))
    panel.text(0.08, 0.09, "External values: HERMES Table 11", fontsize=7.2, color="#" + MUTED)
    panel.text(0.08, 0.05, "Local: 0.5 FPS, 6K KV", fontsize=7.2, color="#" + MUTED)
    fig.text(0.5, 0.025, "External methods use reported settings; use the final controlled table for claims.", ha="center", fontsize=7.2, color="#" + MUTED)
    fig.savefig(PREVIEW_PATH, facecolor="white")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_ppt()
    make_preview()
    print(PPT_PATH)
    print(PREVIEW_PATH)


if __name__ == "__main__":
    main()
