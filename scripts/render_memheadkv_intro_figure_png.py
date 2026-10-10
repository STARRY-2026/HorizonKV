"""Render the compact MemHeadKV overview as a paper-ready raster image."""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/cvpr/figure1/figure1_memheadkv_intro_compact.png"
ASSETS = ROOT / "results/cvpr/figure1/intro_assets"
SCORES = ROOT / "results/myobservations/effective_memory_readout_core_top100_n50_o5000/effective_readout_scores.csv"

W, H = 3200, 1728
NAVY, INK, MUTED = "#19334c", "#2b3943", "#71808a"
BLUE, TEAL, TEAL_DARK = "#4a78b4", "#14918a", "#086b65"
ORANGE, ORANGE_DARK, CORAL = "#e98247", "#cb6431", "#d95f54"
PANEL, LINE, WHITE = "#f3f8fa", "#bdd0d7", "#ffffff"
TEAL_PALE, ORANGE_PALE = "#def3f0", "#fbe7d8"


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    font_dir = Path(__import__("matplotlib").get_data_path()) / "fonts" / "ttf"
    return ImageFont.truetype(str(font_dir / name), size)


def rounded(draw, box, radius=16, fill=WHITE, outline=LINE, width=2):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def text(draw, xy, value, size, fill=INK, bold=False, anchor="la"):
    draw.text(xy, value, font=font(size, bold), fill=fill, anchor=anchor)


def panel_title(draw, x, label, title, accent):
    draw.ellipse((x, 198, x + 60, 258), fill=accent)
    text(draw, (x + 30, 228), label, 24, WHITE, True, "mm")
    text(draw, (x + 82, 228), title, 29, NAVY, True, "lm")


def gaussian(x, center, width, height):
    return height * np.exp(-0.5 * ((x - center) / width) ** 2)


def curves(count=120):
    x = np.linspace(0, 1, count)
    he = 0.06 + 0.035 * (np.sin(15 * np.pi * x) + 1)
    other = 0.06 + 0.03 * (np.cos(13 * np.pi * x) + 1)
    for c, w, a in [(0.04,.02,.25),(.11,.025,.42),(.18,.02,.28),(.292,.027,.92),(.38,.022,.34),(.47,.027,.48),(.58,.022,.26),(.68,.025,.39),(.81,.028,.46),(.93,.02,.28)]:
        he += gaussian(x, c, w, a)
    for c, w, a in [(0.04,.02,.34),(.12,.028,.52),(.20,.023,.29),(.36,.025,.23),(.46,.02,.32),(.60,.025,.28),(.72,.028,.39),(.83,.03,.56),(.91,.028,.86),(.97,.02,.50)]:
        other += gaussian(x, c, w, a)
    scale = max(he.max(), other.max())
    return he / scale, other / scale


def draw_track(draw, box, values, color, pale):
    x0, y0, x1, y1 = box
    rounded(draw, box, 12, WHITE, "#d8e4e8", 2)
    xs = np.linspace(x0 + 5, x1 - 5, len(values))
    ys = y1 - 8 - values * (y1 - y0 - 18)
    polygon = [(x0 + 5, y1 - 7)] + list(zip(xs, ys)) + [(x1 - 5, y1 - 7)]
    draw.polygon(polygon, fill=pale)
    draw.line(list(zip(xs, ys)), fill=color, width=5, joint="curve")


def draw_panel_a(canvas, draw, x0):
    panel_title(draw, x0 + 24, "a", "Head-wise historical readout", BLUE)
    rounded(draw, (x0 + 24, 290, x0 + 942, 510), 20, WHITE, BLUE, 3)
    draw.ellipse((x0 + 48, 318, x0 + 106, 376), fill=BLUE)
    text(draw, (x0 + 77, 347), "Q", 27, WHITE, True, "mm")
    text(draw, (x0 + 130, 340), "What is the man using to cook food in the pan?", 21, INK, True, "lm")
    options = [("A. wooden spoon", x0+130, 386, 350, False), ("B. metal spatula", x0+500, 386, 330, False),
               ("C. red kitchen tongs", x0+130, 438, 410, True), ("D. fork", x0+560, 438, 270, False)]
    for label, x, y, width, selected in options:
        rounded(draw, (x, y, x+width, y+42), 12, TEAL_PALE if selected else "#f5f7f8", TEAL if selected else "#e0e7ea", 2)
        text(draw, (x+width/2, y+21), label, 16, TEAL_DARK if selected else MUTED, selected, "mm")

    track_x0, track_x1 = x0 + 190, x0 + 942
    he_y, other_y = (558, 690), (720, 852)
    rounded(draw, (x0+24, 598, x0+174, 644), 12, TEAL_PALE, TEAL_PALE)
    rounded(draw, (x0+24, 760, x0+174, 806), 12, ORANGE_PALE, ORANGE_PALE)
    text(draw, (x0+99, 621), "HE-Head", 18, TEAL_DARK, True, "mm")
    text(draw, (x0+99, 783), "Other Head", 17, ORANGE_DARK, True, "mm")
    he, other = curves()
    draw_track(draw, (track_x0, he_y[0], track_x1, he_y[1]), he, TEAL, TEAL_PALE)
    draw_track(draw, (track_x0, other_y[0], track_x1, other_y[1]), other, ORANGE, ORANGE_PALE)

    paths = sorted(ASSETS.glob("sample17_*s.png"), key=lambda p: int(p.stem.split("_")[-1][:-1]))
    gap, fy = 6, 900
    fw = int((track_x1-track_x0-gap*11)/12)
    evidence = 3
    for i, path in enumerate(paths[:12]):
        frame = Image.open(path).convert("RGB")
        ratio = fw / 120
        target_w = int(frame.height * ratio)
        left = max(0, (frame.width-target_w)//2)
        frame = frame.crop((left, 0, left+target_w, frame.height)).resize((fw, 120))
        x = track_x0 + i*(fw+gap)
        canvas.paste(frame, (x, fy))
        draw.rectangle((x, fy, x+fw, fy+120), outline=TEAL if i==evidence else WHITE, width=7 if i==evidence else 2)
    ex = track_x0 + evidence*(fw+gap) + fw//2
    draw.line((ex, 532, ex, fy+120), fill=TEAL, width=4)
    text(draw, (ex, 530), "evidence", 18, TEAL_DARK, True, "mb")
    text(draw, (x0+45, 1090), "HE-Head -> C", 21, TEAL_DARK, True)
    text(draw, (x0+920, 1090), "Other Head -> misses evidence", 20, ORANGE_DARK, True, "ra")


def mix_color(lo, hi, t):
    a = np.array([int(lo[i:i+2],16) for i in (1,3,5)])
    b = np.array([int(hi[i:i+2],16) for i in (1,3,5)])
    c = np.rint(a + np.clip(t,0,1)*(b-a)).astype(int)
    return tuple(c.tolist())


def draw_panel_b(draw, x0):
    panel_title(draw, x0 + 24, "b", "Discovering HE-Heads", TEAL)
    text(draw, (x0+58, 310), "True-question attention", 20, NAVY, True)
    segments = [(x0+58, 370, 120, ORANGE_PALE, "start", ORANGE_DARK), (x0+178,370,330,TEAL_PALE,"clean internal history",TEAL_DARK),
                (x0+508,370,120,ORANGE_PALE,"boundary",ORANGE_DARK), (x0+628,370,210,"#e4eef6","current",BLUE)]
    for x,y,w,c,label,tc in segments:
        draw.rectangle((x,y,x+w,y+58), fill=c)
        text(draw,(x+w/2,y+29),label,15,tc,True,"mm")
    text(draw,(x0+118,458),"exclude",15,CORAL,True,"ma")
    text(draw,(x0+568,458),"exclude",15,CORAL,True,"ma")
    rounded(draw,(x0+94,520,x0+820,650),18,WHITE,TEAL,3)
    text(draw,(x0+457,570),r"M = E_q [ Top100Mean( A_q(H_internal) ) ]",21,NAVY,True,"mm")
    text(draw,(x0+457,620),"boundary-cleaned historical readout",16,MUTED,False,"mm")

    df = pd.read_csv(SCORES)
    matrix = np.full((28,4),np.nan)
    for row in df.itertuples():
        matrix[int(row.layer),int(row.kv_head)] = float(row.internal_top100_mean_attention)
    lo, hi = np.nanquantile(matrix,[.05,.95]); threshold=np.nanquantile(matrix,.90)
    gx, gy, cw, ch = x0+110, 730, 74, 25
    for h in range(4): text(draw,(gx+h*cw+cw/2,gy-22),f"K{h}",15,MUTED,False,"mm")
    for l in range(28):
        for h in range(4):
            v=matrix[l,h]; c=mix_color("#fff0cf","#c94455",(v-lo)/max(hi-lo,1e-12))
            draw.rectangle((gx+h*cw,gy+l*ch,gx+(h+1)*cw,gy+(l+1)*ch),fill=c,outline=TEAL_DARK if v>=threshold else WHITE,width=3 if v>=threshold else 1)
    text(draw,(gx-12,gy+4),"L0",14,MUTED,False,"ra")
    text(draw,(gx-12,gy+28*ch),"L27",14,MUTED,False,"ra")
    text(draw,(x0+650,800),"28 layers x 4\nphysical KV Heads",19,NAVY,True,"mm")
    rounded(draw,(x0+570,900,x0+860,1005),16,TEAL_PALE,TEAL_PALE)
    text(draw,(x0+715,952),"Historical\nEvidence Heads",20,TEAL_DARK,True,"mm")
    rounded(draw,(x0+560,1120,x0+870,1260),16,WHITE,LINE,2)
    text(draw,(x0+715,1160),"Stable on unseen videos",16,MUTED,True,"mm")
    text(draw,(x0+715,1210),"Spearman rho = 0.938",21,TEAL_DARK,True,"mm")


def cache_rows(draw,x,y,factors,colors):
    for i,(factor,color) in enumerate(zip(factors,colors)):
        yy=y+i*92
        text(draw,(x,yy+24),f"H{i+1}",15,MUTED,False,"ra")
        rounded(draw,(x+20,yy,x+20+270*factor,yy+48),13,color,color)
        n=max(2,round(8*factor))
        for k in range(n):
            px=x+42+(220*factor)*k/max(n-1,1)
            draw.ellipse((px-7,yy+17,px+7,yy+31),fill=WHITE)


def draw_panel_c(draw,x0):
    panel_title(draw,x0+24,"c","Evidence-aware KV allocation",ORANGE)
    text(draw,(x0+230,330),"Uniform HERMES",21,NAVY,True,"mm")
    text(draw,(x0+725,330),"MemHeadKV",21,ORANGE_DARK,True,"mm")
    cache_rows(draw,x0+55,395,[.72]*6,[BLUE]*6)
    cache_rows(draw,x0+550,395,[1,.42,.88,.50,.94,.58],[TEAL,ORANGE,TEAL,ORANGE,TEAL,ORANGE])
    text(draw,(x0+500,650),"same total B",17,TEAL_DARK,True,"mm")
    rounded(draw,(x0+80,1020,x0+900,1140),18,WHITE,LINE,2)
    labels=[("Offline profile",x0+190),("Head-wise budget",x0+475),("Ragged cache",x0+770)]
    for label,x in labels: text(draw,(x,1080),label,17,NAVY,True,"mm")
    draw.line((x0+290,1080,x0+360,1080),fill=LINE,width=5)
    draw.line((x0+605,1080,x0+670,1080),fill=LINE,width=5)
    badges=[("Training-free",x0+85,240,TEAL_PALE,TEAL_DARK),("Physical KV Heads",x0+345,300,ORANGE_PALE,ORANGE_DARK),("Same Budget",x0+665,230,"#e4eef6",BLUE)]
    for label,x,w,c,tc in badges:
        rounded(draw,(x,1220,x+w,1280),16,c,c)
        text(draw,(x+w/2,1250),label,16,tc,True,"mm")
    text(draw,(x0+490,1370),"Protect the Heads that retrieve historical evidence.",21,ORANGE_DARK,True,"mm")


def main():
    canvas=Image.new("RGB",(W,H),WHITE); draw=ImageDraw.Draw(canvas)
    text(draw,(70,65),"Not All Visual Heads Remember",47,NAVY,True)
    text(draw,(3130,83),"Discovering Historical Evidence Heads for Streaming Video Understanding",25,MUTED,False,"ra")
    draw.line((70,150,3130,150),fill=LINE,width=3)
    panels=[(55,175,1040,1540),(1080,175,2120,1540),(2160,175,3145,1540)]
    for box in panels: rounded(draw,box,22,PANEL,LINE,2)
    draw_panel_a(canvas,draw,55); draw_panel_b(draw,1080); draw_panel_c(draw,2160)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(OUT,dpi=(300,300),optimize=True)
    print(OUT)


if __name__ == "__main__":
    main()
