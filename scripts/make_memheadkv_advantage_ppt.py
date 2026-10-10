"""Create an editable mechanism-advantage figure for MemHeadKV."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results/cvpr/figure1"
PPT_PATH = OUT_DIR / "memheadkv_mechanism_advantage.pptx"
PREVIEW_PATH = OUT_DIR / "memheadkv_mechanism_advantage_preview.png"

NAVY, INK, MUTED = "19334C", "2B3943", "71808A"
BLUE, TEAL, TEAL_DARK = "4A78B4", "14918A", "086B65"
ORANGE, ORANGE_DARK, CORAL = "E98247", "CB6431", "D95F54"
PANEL, LINE, WHITE = "F3F8FA", "BDD0D7", "FFFFFF"
TEAL_PALE, ORANGE_PALE, RED_PALE = "DEF3F0", "FBE7D8", "FBE3E1"
GRAY, GRAY_PALE = "AAB8BE", "E7ECEF"


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


def add_circle(slide, cx, cy, diameter, fill, edge=None, width=0.5):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.OVAL,
        Inches(cx - diameter / 2), Inches(cy - diameter / 2),
        Inches(diameter), Inches(diameter),
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(fill)
    shape.line.color.rgb = rgb(edge or fill)
    shape.line.width = Pt(width)
    return shape


def add_line(slide, x1, y1, x2, y2, color=LINE, width=0.8):
    line = slide.shapes.add_connector(1, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    line.line.color.rgb = rgb(color)
    line.line.width = Pt(width)
    return line


def add_arrow(slide, x, y, w, h, fill=TEAL):
    shape = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(fill)
    shape.line.fill.background()
    return shape


def add_cache_row(slide, x, y, units, fill, evidence=False, max_units=9):
    unit_w, gap = 0.25, 0.045
    for i in range(max_units):
        active = i < units
        cell = add_rect(
            slide, x + i * (unit_w + gap), y, unit_w, 0.31,
            fill if active else WHITE,
            fill if active else "D8E2E6",
            rounded=True,
            width=0.45,
        )
        if evidence and i == min(units - 1, 7):
            cell.fill.fore_color.rgb = rgb(ORANGE)
            cell.line.color.rgb = rgb(ORANGE_DARK)
            add_circle(slide, x + i * (unit_w + gap) + unit_w / 2, y + 0.155, 0.075, WHITE, WHITE)


def panel_title(slide, label, title, subtitle, x, y, accent):
    add_circle(slide, x + 0.15, y + 0.15, 0.30, accent)
    add_text(slide, label, x, y, 0.30, 0.30, size=9, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, title, x + 0.40, y - 0.02, 4.70, 0.32, size=12, color=NAVY, bold=True)
    add_text(slide, subtitle, x + 0.40, y + 0.28, 4.70, 0.23, size=7.6, color=MUTED)


def draw_uniform_panel(slide, x0):
    panel_title(slide, "a", "Uniform head budgets", "HERMES cache allocation", x0 + 0.24, 1.07, BLUE)
    add_rect(slide, x0 + 0.34, 1.82, 4.95, 0.66, WHITE, BLUE, rounded=True, width=1.0)
    add_text(slide, "Historical evidence appears early", x0 + 0.52, 1.93, 2.35, 0.24, size=9, color=INK, bold=True)
    for i in range(11):
        fill = ORANGE if i == 2 else GRAY_PALE
        edge = ORANGE_DARK if i == 2 else LINE
        add_rect(slide, x0 + 2.97 + i * 0.19, 1.98, 0.15, 0.22, fill, edge, rounded=True, width=0.5)
    add_text(slide, "evidence", x0 + 3.25, 2.20, 0.80, 0.19, size=6.5, color=ORANGE_DARK, bold=True, align=PP_ALIGN.CENTER)

    labels = [("HE-Head", TEAL_DARK, TEAL_PALE), ("Head 2", MUTED, GRAY_PALE), ("Head 3", MUTED, GRAY_PALE), ("Head 4", MUTED, GRAY_PALE)]
    y0 = 2.95
    for idx, (label, text_color, badge_fill) in enumerate(labels):
        y = y0 + idx * 0.62
        add_rect(slide, x0 + 0.45, y, 0.86, 0.34, badge_fill, badge_fill, rounded=True)
        add_text(slide, label, x0 + 0.47, y, 0.82, 0.34, size=7.4, color=text_color, bold=idx == 0, align=PP_ALIGN.CENTER)
        add_cache_row(slide, x0 + 1.58, y + 0.01, 6, BLUE, evidence=False)
        add_text(slide, "6", x0 + 4.42, y, 0.34, 0.34, size=8, color=BLUE, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, "Same capacity for every physical KV Head", x0 + 1.26, 5.49, 3.48, 0.27, size=8.5, color=MUTED, bold=True, align=PP_ALIGN.CENTER)
    add_rect(slide, x0 + 0.66, 5.96, 4.45, 0.64, RED_PALE, RED_PALE, rounded=True)
    add_text(slide, "Historical evidence can be evicted first", x0 + 0.91, 6.08, 3.95, 0.26, size=10, color=CORAL, bold=True, align=PP_ALIGN.CENTER)


def draw_ours_panel(slide, x0):
    panel_title(slide, "b", "Evidence-aligned budgets", "MemHeadKV cache allocation", x0 + 0.24, 1.07, TEAL)
    add_rect(slide, x0 + 0.34, 1.82, 4.95, 0.66, WHITE, TEAL, rounded=True, width=1.0)
    add_text(slide, "Offline HE profile reallocates capacity", x0 + 0.52, 1.93, 2.55, 0.24, size=9, color=INK, bold=True)
    add_text(slide, "same total = 24 units", x0 + 3.32, 1.93, 1.66, 0.24, size=8.2, color=TEAL_DARK, bold=True, align=PP_ALIGN.RIGHT)

    rows = [("HE-Head", 9, TEAL, True), ("Head 2", 3, ORANGE, False), ("Head 3", 8, TEAL, False), ("Head 4", 4, ORANGE, False)]
    y0 = 2.95
    for idx, (label, units, fill, evidence) in enumerate(rows):
        y = y0 + idx * 0.62
        badge_fill = TEAL_PALE if fill == TEAL else ORANGE_PALE
        badge_text = TEAL_DARK if fill == TEAL else ORANGE_DARK
        add_rect(slide, x0 + 0.45, y, 0.86, 0.34, badge_fill, badge_fill, rounded=True)
        add_text(slide, label, x0 + 0.47, y, 0.82, 0.34, size=7.4, color=badge_text, bold=idx == 0, align=PP_ALIGN.CENTER)
        add_cache_row(slide, x0 + 1.58, y + 0.01, units, fill, evidence=evidence)
        add_text(slide, str(units), x0 + 4.42, y, 0.34, 0.34, size=8, color=badge_text, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, "More capacity where historical readout is stable", x0 + 1.17, 5.49, 3.66, 0.27, size=8.5, color=MUTED, bold=True, align=PP_ALIGN.CENTER)
    add_rect(slide, x0 + 0.66, 5.96, 4.45, 0.64, TEAL_PALE, TEAL_PALE, rounded=True)
    add_text(slide, "Historical evidence remains retrievable", x0 + 0.91, 6.08, 3.95, 0.26, size=10, color=TEAL_DARK, bold=True, align=PP_ALIGN.CENTER)


def make_ppt():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.2)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = rgb(WHITE)
    add_text(slide, "Same Budget, Better Aligned with Historical Readout", 0.38, 0.14, 8.85, 0.43, size=20, color=NAVY, bold=True)
    add_text(slide, "Mechanism advantage; downstream gain pending", 8.75, 0.23, 4.20, 0.27, size=9, color=MUTED, align=PP_ALIGN.RIGHT)
    add_line(slide, 0.38, 0.66, 12.95, 0.66)
    add_rect(slide, 0.28, 0.82, 5.78, 6.02, PANEL, LINE, rounded=True)
    add_rect(slide, 7.27, 0.82, 5.78, 6.02, PANEL, LINE, rounded=True)
    draw_uniform_panel(slide, 0.28)
    draw_ours_panel(slide, 7.27)
    add_arrow(slide, 6.25, 3.20, 0.82, 0.42, TEAL)
    add_text(slide, "HE-guided\nreallocation", 6.16, 3.68, 1.00, 0.56, size=7.7, color=TEAL_DARK, bold=True, align=PP_ALIGN.CENTER)
    badges = [("HE-Heads", "10 / 112 (8.9%)"), ("Split stability", "rho = 0.938"), ("Top-10 overlap", "8 / 10")]
    bx = 5.94
    for title, value in badges:
        add_rect(slide, bx, 4.63, 1.42, 0.65, WHITE, LINE, rounded=True)
        add_text(slide, title, bx + 0.08, 4.69, 1.26, 0.20, size=6.5, color=MUTED, bold=True, align=PP_ALIGN.CENTER)
        add_text(slide, value, bx + 0.08, 4.92, 1.26, 0.24, size=8.0, color=TEAL_DARK, bold=True, align=PP_ALIGN.CENTER)
        bx += 0.0  # badges intentionally overlap the center column vertically, not horizontally
        break
    # Stack the remaining evidence badges compactly in the center column.
    for idx, (title, value) in enumerate(badges[1:]):
        yy = 5.40 + idx * 0.66
        add_rect(slide, 5.94, yy, 1.42, 0.57, WHITE, LINE, rounded=True)
        add_text(slide, title, 6.00, yy + 0.03, 1.30, 0.18, size=6.0, color=MUTED, bold=True, align=PP_ALIGN.CENTER)
        add_text(slide, value, 6.00, yy + 0.23, 1.30, 0.23, size=7.7, color=TEAL_DARK, bold=True, align=PP_ALIGN.CENTER)
    prs.save(PPT_PATH)


def make_preview():
    # Render a compact visual twin for quick inspection; the PPT remains fully editable.
    width, height = 2400, 1296
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font_dir = Path(__import__("matplotlib").get_data_path()) / "fonts" / "ttf"

    def f(size, bold=False):
        name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        return ImageFont.truetype(str(font_dir / name), size)

    draw.text((65, 40), "Same Budget, Better Aligned with Historical Readout", font=f(38, True), fill="#" + NAVY)
    draw.text((2330, 55), "Mechanism advantage; downstream gain pending", font=f(16), fill="#" + MUTED, anchor="ra")
    panels = [(50, 150, 1065, 1215), (1335, 150, 2350, 1215)]
    for box in panels:
        draw.rounded_rectangle(box, 24, fill="#" + PANEL, outline="#" + LINE, width=2)
    draw.text((105, 205), "(a) Uniform head budgets", font=f(25, True), fill="#" + NAVY)
    draw.text((1390, 205), "(b) Evidence-aligned budgets", font=f(25, True), fill="#" + NAVY)
    draw.text((105, 250), "HERMES cache allocation", font=f(16), fill="#" + MUTED)
    draw.text((1390, 250), "MemHeadKV cache allocation", font=f(16), fill="#" + MUTED)

    def rows(x0, units, colors, evidence=False):
        labels = ["HE-Head", "Head 2", "Head 3", "Head 4"]
        for j, (label, n, color) in enumerate(zip(labels, units, colors)):
            y = 445 + j * 105
            draw.text((x0, y + 24), label, font=f(17, j == 0), fill="#" + (TEAL_DARK if j == 0 else MUTED), anchor="rm")
            for i in range(9):
                xx = x0 + 35 + i * 53
                fill = "#" + color if i < n else "white"
                outline = "#" + color if i < n else "#d8e2e6"
                draw.rounded_rectangle((xx, y, xx + 40, y + 48), 9, fill=fill, outline=outline, width=2)
                if evidence and j == 0 and i == 7:
                    draw.rounded_rectangle((xx, y, xx + 40, y + 48), 9, fill="#" + ORANGE, outline="#" + ORANGE_DARK, width=2)

    rows(235, [6, 6, 6, 6], [BLUE] * 4)
    rows(1520, [9, 3, 8, 4], [TEAL, ORANGE, TEAL, ORANGE], True)
    draw.text((557, 360), "Equal capacity per head", font=f(19, True), fill="#" + INK, anchor="mm")
    draw.text((1842, 360), "Equal total = 24 units", font=f(19, True), fill="#" + TEAL_DARK, anchor="mm")
    draw.rounded_rectangle((185, 930, 930, 1035), 18, fill="#" + RED_PALE)
    draw.text((558, 982), "Historical evidence can be evicted first", font=f(20, True), fill="#" + CORAL, anchor="mm")
    draw.rounded_rectangle((1470, 930, 2215, 1035), 18, fill="#" + TEAL_PALE)
    draw.text((1842, 982), "Historical evidence remains retrievable", font=f(20, True), fill="#" + TEAL_DARK, anchor="mm")
    draw.polygon([(1125, 560), (1240, 560), (1240, 530), (1310, 605), (1240, 680), (1240, 650), (1125, 650)], fill="#" + TEAL)
    draw.text((1215, 720), "HE-guided\nreallocation", font=f(16, True), fill="#" + TEAL_DARK, anchor="mm", align="center")
    draw.rounded_rectangle((1080, 820, 1325, 900), 15, fill="white", outline="#" + LINE, width=2)
    draw.text((1202, 860), "10 / 112 HE-Heads", font=f(16, True), fill="#" + TEAL_DARK, anchor="mm")
    image.save(PREVIEW_PATH, dpi=(300, 300), optimize=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_ppt()
    make_preview()
    print(PPT_PATH)
    print(PREVIEW_PATH)


if __name__ == "__main__":
    main()
