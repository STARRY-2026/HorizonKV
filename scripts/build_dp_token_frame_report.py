#!/usr/bin/env python3
"""Build a local scientific evidence viewer from measured DP/Hermes traces."""
from __future__ import annotations

import bisect
import csv
import html
import json
from pathlib import Path

import numpy as np
from decord import VideoReader
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/myobservations/dp_token_frame_study_20261010"
BANDS = {"early": (0, 8), "middle": (8, 19), "late": (19, 28)}


def jsonl(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def main():
    review = json.loads((OUT / "manual_review.json").read_text())
    comparison = json.loads((OUT / "experiment_comparison.json").read_text())
    manifests = jsonl(OUT / "dp/source_manifest.jsonl")
    videos = {str(r["video_id"]): r for r in manifests if r["record_type"] == "video"}
    encodes = {}
    for r in manifests:
        if r["record_type"] == "encode":
            encodes.setdefault(str(r["video_id"]), []).append(r)
    traces = {policy: {(str(r["video_id"]), int(r["question_index"])): r
                      for r in jsonl(OUT / policy / "tokens.jsonl")}
              for policy in ("dp", "hermes")}
    image_dir = OUT / "input_frames"
    image_dir.mkdir(exist_ok=True)
    payload = []
    roi_records = []
    for case in review["cases"]:
        if not case["is_diagnostic_target"]:
            continue
        vid, qi = str(case["video_id"]), int(case["question_index"])
        calls = [r for r in encodes[vid] if int(r["question_index"]) <= qi]
        starts = [r["uid_start"] for r in calls]
        groups = []
        by_call = []
        for call in calls:
            assert call["fixed_frame_canvas_size"] == 0, "Only direct native resize is supported"
            t_count, gh, gw = call["llm_grid_thw"]
            cells = gh * gw
            call_groups = []
            for t in range(t_count):
                samples = [call["samples"][min(t * 2 + j, len(call["samples"]) - 1)]
                           for j in (0, 1)]
                group = {"uid_start": call["uid_start"] + t * cells, "h": gh, "w": gw,
                         "samples": samples, "resized_hw": call["processor_resized_hw"],
                         "dp": [0] * cells, "hermes": [0] * cells,
                         "dp_mask": [0] * cells, "hermes_mask": [0] * cells,
                         "dp_attention": [0.] * cells, "hermes_attention": [0.] * cells}
                groups.append(group)
                call_groups.append(group)
            by_call.append(call_groups)
        for policy in ("dp", "hermes"):
            for head in traces[policy][vid, qi]["heads"]:
                bit = 1 << (int(head["layer"]) * 4 + int(head["kv_head"]))
                for uid, attention in zip(head["visual_token_uids"], head["visual_attention"]):
                    ci = bisect.bisect_right(starts, uid) - 1
                    call = calls[ci]
                    _, gh, gw = call["llm_grid_thw"]
                    t, cell = divmod(uid - call["uid_start"], gh * gw)
                    group = by_call[ci][t]
                    group[policy][cell] += 1
                    group[policy + "_mask"][cell] |= bit
                    group[policy + "_attention"][cell] += attention
        # Save display copies from precisely the decoder indices used by the loader.
        sample_map = {s["sample_ordinal"]: s for g in groups for s in g["samples"]}
        reader = VideoReader(videos[vid]["video_path"], num_threads=1)
        ordinals = sorted(sample_map)
        for offset in range(0, len(ordinals), 12):
            chunk = ordinals[offset:offset + 12]
            missing = [ordinal for ordinal in chunk if not (image_dir / f"v{vid}_s{ordinal}.jpg").exists()]
            if missing:
                frames = reader.get_batch([sample_map[s]["source_frame_index"] for s in missing]).asnumpy()
                for ordinal, frame in zip(missing, frames):
                    image = Image.fromarray(frame)
                    image.thumbnail((960, 960))
                    image.save(image_dir / f"v{vid}_s{ordinal}.jpg", quality=92)
        manual = case["human_review"]
        recommended = manual.get("existing_offline_input_sample_ordinals")
        if not recommended:
            recommended = [min(ordinals, key=lambda s: abs(sample_map[s]["decoder_pts"][0] - seconds))
                           for seconds in manual["recommended_frame_times_s"]]
        focus = sorted({i for i, g in enumerate(groups)
                        if any(s["sample_ordinal"] in recommended for s in g["samples"])})
        for index in focus:
            g = groups[index]
            rows, cols = np.indices((g["h"], g["w"]))
            x0, x1 = cols / g["w"], (cols + 1) / g["w"]
            y0, y1 = rows / g["h"], (rows + 1) / g["h"]
            for roi in manual["manual_rois"]:
                rx0, ry0, rx1, ry1 = roi["xyxy"]
                intersects = ((x1 > rx0) & (x0 < rx1) & (y1 > ry0) & (y0 < ry1)).ravel()
                record = {"video_id": vid, "question_index": qi, "group_index": index,
                          "sample_ordinals": [s["sample_ordinal"] for s in g["samples"]],
                          "roi": roi["label"], "xyxy": roi["xyxy"],
                          "intersecting_source_cells": int(intersects.sum())}
                for policy in ("dp", "hermes"):
                    counts = np.asarray(g[policy])[intersects]
                    record[policy + "_unique_cells_kept"] = int((counts > 0).sum())
                    record[policy + "_physical_copies"] = int(counts.sum())
                    record[policy + "_visual_question_attention_sum"] = float(np.asarray(g[policy + "_attention"])[intersects].sum())
                    for name, (lo, hi) in BANDS.items():
                        bandmask = ((1 << (hi * 4)) - 1) ^ ((1 << (lo * 4)) - 1)
                        masks = np.asarray(g[policy + "_mask"], dtype=object)[intersects]
                        record[f"{policy}_{name}_physical_copies"] = sum((int(v) & bandmask).bit_count() for v in masks)
                roi_records.append(record)
        case_data = {"video_id": vid, "question_index": qi, "question": case["question"],
                     "choices": case["choices"], "gold": case["gold_answer"],
                     "existing": case["existing_K1000"], "human_review": manual,
                     "focus": focus, "roi_sample_ordinals": recommended, "groups": groups,
                     "source_hw": videos[vid]["loaded_shape"][1:3],
                     "roi_rows": [r for r in roi_records if r["video_id"] == vid]}
        for g in groups:
            for s in g["samples"]:
                s["image"] = f"input_frames/v{vid}_s{s['sample_ordinal']}.jpg"
            for policy in ("dp", "hermes"):
                g[policy + "_mask"] = [format(m, "x") for m in g[policy + "_mask"]]
        payload.append(case_data)
        print(f"Built v{vid} q{qi}: {len(groups)} temporal groups", flush=True)
    (OUT / "roi_retention.json").write_text(json.dumps({"definition": "Cells with nonzero area intersection with approximate manual boxes. Spatial anchor coverage, not semantic evidence or a causal test. Counts include all 112 heads unless named band. One token can cover both images.", "rows": roi_records}, ensure_ascii=False, indent=2) + "\n")
    write_overview(payload)
    write_timelines(payload)
    write_token_example(payload)
    write_html(payload)


def write_token_example(cases):
    """Show a single physical-copy difference on both actual source frames."""
    case = next(c for c in cases if c["video_id"] == "398")
    group = case["groups"][0]
    example = json.loads((OUT / "representative_token_examples.json").read_text())["examples"][0]
    uid, layer, head = example["uid"], example["layer"], example["kv_head"]
    bit = 1 << (layer * 4 + head)
    roi = next(r["xyxy"] for r in case["human_review"]["manual_rois"] if r["label"] == "name-number pair")
    fig, axes = plt.subplots(2, 3, figsize=(9, 8), squeeze=False)
    for row, sample in enumerate(group["samples"]):
        image = np.asarray(Image.open(OUT / sample["image"]))
        ih, iw = image.shape[:2]
        x0, y0, x1, y1 = roi
        crop = (int(x0 * iw), int(y0 * ih), int(np.ceil(x1 * iw)), int(np.ceil(y1 * ih)))
        panels = [image]
        for policy in ("hermes", "dp"):
            kept = np.asarray([bool(int(m, 16) & bit) for m in group[policy + "_mask"]], dtype=np.uint8)
            mask = np.asarray(Image.fromarray(kept.reshape(group["h"], group["w"]) * 255).resize((iw, ih), Image.Resampling.NEAREST)) > 0
            panels.append(np.where(mask[..., None], image, np.full_like(image, 220)))
        for col, panel in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(panel[crop[1]:crop[3], crop[0]:crop[2]])
            bx0, by0, bx1, by1 = example["source_normalized_xyxy"]
            ax.add_patch(plt.Rectangle((bx0 * iw - crop[0], by0 * ih - crop[1]),
                                      (bx1 - bx0) * iw, (by1 - by0) * ih,
                                      fill=False, edgecolor="#e88a00", linewidth=2))
            ax.set_axis_off()
            policy_label = ("Source crop", f"HERMES L{layer}H{head}", f"DP L{layer}H{head}")[col]
            ax.set_title(f"{policy_label}\ns{sample['sample_ordinal']} / f{sample['source_frame_index']} @ {sample['decoder_pts'][0]:.3f}s", fontsize=10)
    fig.suptitle(f"UID {uid}: one source token, two sampled frames\nHERMES keeps this copy at L{layer}H{head}; DP keeps the same UID in {example['global_uid_copy_count']['dp']} other heads", fontsize=12)
    fig.text(.5, .015, "Grey = this head did not retain that source position. Orange = selected UID anchor.\nPixel reconstruction shows source anchors, not decoded KV content or exclusive receptive fields.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .93))
    fig.savefig(OUT / "token_1230_frame_pair.png", dpi=160)
    plt.close(fig)


def write_timelines(cases):
    fig, axes = plt.subplots(len(cases), 2, figsize=(14, 2.8 * len(cases)), squeeze=False)
    for axrow, case in zip(axes, cases):
        x = [g["samples"][-1]["decoder_pts"][0] for g in case["groups"]]
        for policy, color in (("hermes", "#34465b"), ("dp", "#247fcb")):
            copies = [sum(g[policy]) for g in case["groups"]]
            mass = [sum(g[policy + "_attention"]) for g in case["groups"]]
            axrow[0].plot(x, np.asarray(copies) / sum(copies), "o-", ms=2, color=color, label=policy.upper())
            axrow[1].plot(x, np.asarray(mass) / sum(mass), "o-", ms=2, color=color, label=policy.upper())
        for ax in axrow:
            for i in case["focus"]:
                ax.axvline(x[i], color="#edab35", linewidth=1, alpha=.7)
            ax.set_xlabel("Source decoder PTS (pair right edge), seconds")
            ax.grid(alpha=.18)
            ax.legend(fontsize=8)
        axrow[0].set_ylabel("Share of retained visual KV copies")
        axrow[1].set_ylabel("Share of visual question attention")
        axrow[0].set_title(f"v{case['video_id']} q{case['question_index']} | physical retention")
        axrow[1].set_title("Question-prompt readout (not causal attribution)")
    fig.tight_layout()
    fig.savefig(OUT / "retention_timelines.png", dpi=150)
    plt.close(fig)


def write_overview(cases):
    selected = []
    for case in cases:
        if case["video_id"] in ("398", "269", "124"):
            for index in case["focus"]:
                selected.append((case, case["groups"][index]))
    fig, axes = plt.subplots(len(selected), 4, figsize=(16, 3.6 * len(selected)), squeeze=False)
    for axrow, (case, group) in zip(axes, selected):
        sample = next((s for s in group["samples"] if s["sample_ordinal"] in case["roi_sample_ordinals"]), group["samples"][-1])
        image = np.asarray(Image.open(OUT / sample["image"]))
        ih, iw = image.shape[:2]
        for ax in axrow:
            ax.imshow(image)
            ax.set_axis_off()
        pair_label = " + ".join(f"s{s['sample_ordinal']} @{s['decoder_pts'][0]:.3f}s" for s in group["samples"])
        axrow[0].set_title(f"v{case['video_id']} q{case['question_index']} | {pair_label}\nbackground: s{sample['sample_ordinal']} / source frame {sample['source_frame_index']}", fontsize=9)
        for ax, policy in zip(axrow[1:3], ("hermes", "dp")):
            heat = np.asarray(group[policy]).reshape(group["h"], group["w"])
            ax.imshow(heat, extent=(0, iw, ih, 0), cmap="magma", vmin=0, vmax=112,
                      alpha=np.where(heat > 0, .65, .12), interpolation="nearest")
            ax.set_title(f"{policy.upper()} source-token copies / 112 heads", fontsize=10)
        delta = (np.asarray(group["dp"]) - group["hermes"]).reshape(group["h"], group["w"])
        axrow[3].imshow(delta, extent=(0, iw, ih, 0), cmap="coolwarm", vmin=-112, vmax=112,
                       alpha=.7, interpolation="nearest")
        axrow[3].set_title("DP - HERMES copies (red: more DP)", fontsize=10)
        for ax in axrow:
            for roi in case["human_review"]["manual_rois"]:
                x0, y0, x1, y1 = roi["xyxy"]
                ax.add_patch(plt.Rectangle((x0 * iw, y0 * ih), (x1 - x0) * iw, (y1 - y0) * ih,
                                          fill=False, edgecolor="#00ffbf", linewidth=.9))
    fig.suptitle("Measured KV source-token retention at question boundary\nA temporal token covers a sampled frame pair; maps show anchors, not isolated semantic evidence", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(OUT / "retained_token_overview.png", dpi=150)
    plt.close(fig)


def write_html(cases):
    template = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DP 保留 token → 实际帧证据</title>
<style>body{font:16px/1.65 system-ui,sans-serif;color:#202b37;background:#f5f7fa;margin:auto;max-width:1440px;padding:26px}h1{font-size:28px}h2{font-size:21px;margin-top:28px}p{max-width:1100px}select,button{font:inherit;padding:6px;margin:4px 12px 4px 0;max-width:100%}.card{background:white;border:1px solid #dce2ea;padding:18px;border-radius:10px;margin:16px 0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}canvas{width:100%;display:block;border:1px solid #bdc8d4;cursor:crosshair}.small{font-size:13px;color:#536474}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid #e2e6ec;text-align:left;padding:7px;vertical-align:top}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}.caption{font-size:13px}.notice{border-left:4px solid #4389c5;padding-left:14px}a{color:#155998}.tags{color:#485a73}#inspect{background:#eff4fb;min-height:60px;padding:12px}img{max-width:100%}@media(max-width:800px){.grid{grid-template-columns:1fr}body{padding:12px}}</style>
<h1>当前 DP 方法保留了哪些视觉 token？</h1><p>本报告使用最新主方法 <b>DP capacity + headwise α/k，K=1000</b> 的新运行缓存，与同设置 Hermes 逐 token 对照。4 个诊断视频、6 次问答（含所有前序问答），原始分辨率、0.5 FPS、每次编码最多 16 帧；没有使用旧 LU trace 代替 DP。样本为定向案例选择，不能代表总体性能。</p>
<p>当前实现把离线DP结果用于逐头容量，并将recent比例映射成在线α/k打分参数；本次日志中 memory_current_split=False。离线配置中的recent_r不是运行时独立近期bank的硬配额，实际保留内容需要由UID trace核对。K1000是平均每个层/KV头的视觉容量；本次两方法每题均保留112,000份视觉KV，另有每头14个初始化文本位置。</p><div class="card"><b>相同题目上的实验差距</b><table><tr><th>预算 / 测试范围</th><th>Hermes</th><th>DP</th><th>DP − Hermes</th></tr><tr><td>K1000 / 250 视频 1255 题</td><td>69.72%</td><td>70.60%</td><td>+0.88 pp；36 胜 / 25 负</td></tr><tr><td>K1000 / 前150视频 754题</td><td>67.11%</td><td>68.57%</td><td>+1.46 pp</td></tr><tr><td>K100 / 前150视频</td><td>55.97%</td><td>55.57%</td><td>−0.40 pp</td></tr><tr><td>K250 / 前150视频</td><td>58.36%</td><td>60.21%</td><td>+1.86 pp</td></tr><tr><td>K500 / 前150视频</td><td>63.26%</td><td>64.85%</td><td>+1.59 pp</td></tr></table><p class="small">K1000 旧 LU 为71.31%，当前DP低0.72pp。更晚的9月30日 recent32-unbounded-noak 是预算消融，不是主方法。完整数据为498视频，当前主结果覆盖250视频；398是视频ID。详见 <a href="experiment_comparison.json">原始结果复算与来源哈希</a>。</p></div>
<p class="notice"><b>如何读这张图：</b>UID 是原始视觉输出 token 的不可变编号，不是文字词表编号。每个 token 锚定处理后 28×28 像素网格，通常融合两张采样帧；单帧尾部会重复。下面同时展示两帧。视觉编码器和后续语言层已混合上下文，因此格子表示 <b>token 的来源位置</b>，不能等同于其唯一感受野或语义内容。保留、问题 attention 和答对是三个不同事实。</p>
<div class="card"><label>案例 <select id="case"></select></label><div id="question"></div><p id="answers"></p><p id="review"></p><label>时间组 <select id="group"></select></label><button id="focus">跳到人工核验的证据组</button><br><label>层 / KV head <select id="head"><option value="frequency">全部112头：保留复制数</option><option value="union">全部112头：任意头保留</option><option value="early">L0–7：保留复制数 /32</option><option value="middle">L8–18：保留复制数 /44</option><option value="late">L19–27：保留复制数 /36</option></select></label><label>显示 <select id="mode"><option value="retention">保留来源位置</option><option value="retained_only">只拼回保留位置的像素（所选头）</option><option value="original">原始输入帧（显示副本）</option></select></label><p class="small">蓝色网格为被保留的来源 token，透明度随保留复制数增加（双方在同一模式下共用尺度：全部头0–112，浅/中/深层分别0–32/44/36，union及单头为二值）；橙框只画在人工核验的对应采样帧上，是近似ROI。单头模式为二值保留。点击任意格子查看UID、真实帧索引/PTS和具体保留头。未着色只表示所选头未保留该来源token，不代表该图像信息在其它token中不存在。</p><div class="grid"><div><b>HERMES</b><div id="hcanvases"></div></div><div><b>DP</b><div id="dcanvases"></div></div></div><div id="inspect">点击图中格子查看精确对应关系。</div><p id="groupstats" class="small"></p></div>
<div class="card"><h2>人工ROI的空间保留量</h2><p class="small">只计算与手工框有非零面积相交的来源网格；框可能跨越字边缘/背景。单个网格通常覆盖两帧，以下数据不证明该token已编码或使用了对应文字/物体。复制数按各层各头累计，不能当作独立证据条数。</p><div id="roi"></div><p id="limits" class="notice"></p></div>
<div class="card"><h2>可复查文件</h2><ul><li><a href="token_correspondence.csv.gz">逐来源 token 完整对应表（gzip CSV，含两边保留头bitmask）</a></li><li><a href="frame_groups.csv">所有时空组的保留量与问题读出量</a> · <a href="retention_summary.json">题级/分层统计</a></li><li><a href="validation.json">一致性验证和原答案复现检查</a> · <a href="annotation.json">完整因果前缀</a> · <a href="roi_retention.json">ROI统计</a> · <a href="input_frame_previews_manifest.json">帧预览来源及哈希</a></li><li><a href="dp/tokens.jsonl">DP原始逐头trace</a> · <a href="hermes/tokens.jsonl">Hermes原始逐头trace</a> · <a href="dp/source_manifest.jsonl">DP实际输入及编码grid</a> · <a href="hermes/source_manifest.jsonl">Hermes实际输入及编码grid</a> · <a href="dp/command.json">DP完整命令及代码哈希</a> · <a href="hermes/command.json">Hermes完整命令及代码哈希</a></li><li><a href="case_findings.json">案例结论与限制</a> · <a href="retained_token_overview.png">静态总览图</a> · <a href="retention_timelines.png">时序保留与读出分布</a></li><li><a href="all_dp_regressions_vs_hermes.csv">全部25个Hermes正确、DP错误案例</a>（本次精确token追踪覆盖上述4个诊断视频）</li></ul><p>映射：在某次encode中 j=UID−uid_start，S=H×W，t=⌊j/S⌋，row=⌊(j mod S)/W⌋，col=j mod W。时间对应该encode的第2t与2t+1张采样帧（奇数尾部重复），不是用重编号的mRoPE位置倒推。所有question_index / sample_ordinal / layer / head均从0开始。真实时间使用decoder PTS；采样使用原loader的round(source_fps)，因此实际时间不必严格等于sample_ordinal/0.5。</p><p>question attention是实际多选问题及选项全部prompt位置、每KV头对应7个query heads的平均读出，各模型各自上下文归一化；不是生成答案token的贡献，也不能跨模型解释为因果价值。当前DP与Hermes同时改变预算和α/k，必须做预算×α/k的2×2消融或同状态token干预，才能归因具体算法组件。</p></div>
<script>const D=__DATA__;const $=id=>document.getElementById(id);let active=0,group=0,focusidx=0;const imgs={};function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}function pop(v){let n=0;while(v){v&=v-1n;n++}return n}function headnames(m){let n=BigInt('0x'+m),a=[];for(let i=0;i<112;i++)if(n&(1n<<BigInt(i)))a.push('L'+Math.floor(i/4)+'H'+i%4);return a.join(', ')||'无'}function value(g,p,i){let h=$('head').value;if(h==='frequency')return [g[p][i],112];if(h==='union')return [+!!g[p][i],1];let m=BigInt('0x'+g[p+'_mask'][i]);if(['early','middle','late'].includes(h)){let [a,b]=h==='early'?[0,32]:h==='middle'?[32,76]:[76,112];return [pop((m>>BigInt(a))&((1n<<BigInt(b-a))-1n)),b-a]}return [Number((m>>BigInt(h))&1n),1]}
function setcase(){active=+$('case').value;const c=D[active];$('question').textContent=c.question;$('answers').textContent='Gold: '+c.gold+' | Hermes: '+c.existing.hermes.answer+' | DP: '+c.existing.dp.answer;$('review').textContent=c.human_review.visible_evidence;$('limits').textContent=c.human_review.limits;$('group').innerHTML=c.groups.map((g,i)=>'<option value="'+i+'">'+i+' | '+g.samples.map(s=>'s'+s.sample_ordinal+' / '+s.decoder_pts[0].toFixed(3)+'s').join(' + ')+'</option>').join('');group=c.focus[0]||0;$('group').value=group;focusidx=0;let rows=c.roi_rows.map(r=>'<tr><td>'+r.sample_ordinals.join('+')+'</td><td>'+esc(r.roi)+'</td><td>'+r.intersecting_source_cells+'</td><td>'+r.hermes_unique_cells_kept+' / '+r.hermes_physical_copies+'</td><td>'+r.dp_unique_cells_kept+' / '+r.dp_physical_copies+'</td></tr>').join('');$('roi').innerHTML='<table><tr><th>采样帧</th><th>人工ROI</th><th>相交格子数</th><th>Hermes: unique / copies</th><th>DP: unique / copies</th></tr>'+rows+'</table>';render()}
function render(){const c=D[active],g=c.groups[group];for(const [p,target] of [['hermes','hcanvases'],['dp','dcanvases']]){$(target).innerHTML='';for(let n=0;n<2;n++){const s=g.samples[n],canvas=document.createElement('canvas'),caption=document.createElement('div');canvas.width=900;canvas.height=Math.round(900*c.source_hw[0]/c.source_hw[1]);caption.className='caption';caption.textContent='s'+s.sample_ordinal+' | 原视频帧 '+s.source_frame_index+' | PTS '+s.decoder_pts[0].toFixed(6)+'s'+(n&&s.sample_ordinal===g.samples[0].sample_ordinal?'（重复尾帧）':'');$(target).append(canvas,caption);const draw=()=>{const ctx=canvas.getContext('2d');ctx.drawImage(imgs[s.image],0,0,canvas.width,canvas.height);if($('mode').value==='retained_only'){ctx.fillStyle='#dfe4ea';ctx.fillRect(0,0,canvas.width,canvas.height);const im=imgs[s.image],cw=canvas.width/g.w,ch=canvas.height/g.h;for(let i=0;i<g.h*g.w;i++){if(value(g,p,i)[0]){let x=i%g.w,y=Math.floor(i/g.w);ctx.drawImage(im,x*im.naturalWidth/g.w,y*im.naturalHeight/g.h,im.naturalWidth/g.w,im.naturalHeight/g.h,x*cw,y*ch,cw,ch)}}}if($('mode').value==='retention'){let cw=canvas.width/g.w,ch=canvas.height/g.h;for(let i=0;i<g.h*g.w;i++){let [v,max]=value(g,p,i);if(v){ctx.fillStyle='rgba(15,135,255,'+(0.18+0.7*Math.sqrt(v/max))+')';ctx.fillRect(i%g.w*cw,Math.floor(i/g.w)*ch,cw,ch)}}}ctx.strokeStyle='#ffab20';ctx.lineWidth=2;for(const r of (c.roi_sample_ordinals.includes(s.sample_ordinal)?c.human_review.manual_rois:[])){let[x0,y0,x1,y1]=r.xyxy;ctx.strokeRect(x0*canvas.width,y0*canvas.height,(x1-x0)*canvas.width,(y1-y0)*canvas.height)}};let im=imgs[s.image];if(!im){im=new Image();imgs[s.image]=im;im.src=s.image}if(im.complete&&im.naturalWidth)draw();else im.addEventListener('load',draw,{once:true});canvas.onclick=e=>{let b=canvas.getBoundingClientRect(),col=Math.min(g.w-1,Math.floor((e.clientX-b.left)/b.width*g.w)),row=Math.min(g.h-1,Math.floor((e.clientY-b.top)/b.height*g.h)),i=row*g.w+col;let [sh,sw]=c.source_hw;$('inspect').innerHTML='<b>UID '+(g.uid_start+i)+' · row '+row+' / col '+col+'</b><br>两帧: '+g.samples.map(x=>'s'+x.sample_ordinal+' → 原帧f'+x.source_frame_index+' @ '+x.decoder_pts[0].toFixed(6)+'s').join(' + ')+'<br>原图来源几何框xyxy: '+[col/g.w*sw,row/g.h*sh,(col+1)/g.w*sw,(row+1)/g.h*sh].map(x=>x.toFixed(1)).join(', ')+'<br>HERMES: '+g.hermes[i]+'/112 · '+esc(headnames(g.hermes_mask[i]))+'<br>DP: '+g.dp[i]+'/112 · '+esc(headnames(g.dp_mask[i]))+'<br>问题attention合计 H='+g.hermes_attention[i].toExponential(3)+' / DP='+g.dp_attention[i].toExponential(3)}}}$('groupstats').textContent='本组UID '+g.uid_start+'–'+(g.uid_start+g.h*g.w-1)+'；grid '+g.h+'×'+g.w+'，processor图像 '+g.resized_hw.join('×')+'；H copies='+g.hermes.reduce((a,b)=>a+b,0)+' / DP copies='+g.dp.reduce((a,b)=>a+b,0)}
$('case').innerHTML=D.map((c,i)=>'<option value="'+i+'">v'+c.video_id+' q'+c.question_index+' | '+esc(c.question)+'</option>').join('');for(let l=0;l<28;l++)for(let h=0;h<4;h++)$('head').add(new Option('L'+l+' / KV head '+h,l*4+h));$('case').onchange=setcase;$('group').onchange=()=>{group=+$('group').value;render()};$('head').onchange=render;$('mode').onchange=render;$('focus').onclick=()=>{const f=D[active].focus;group=f[(++focusidx)%f.length];$('group').value=group;render()};setcase();</script></html>'''
    findings_path = OUT / "case_findings.json"
    if findings_path.exists():
        findings = json.loads(findings_path.read_text())
        paragraphs = "".join(f"<p><b>v{html.escape(str(c['video_id']))} q{c['question_index']}：</b>"
                             f"{html.escape(c['finding_zh'])}</p>" for c in findings["cases"])
        card = '<div class="card"><h2>这次实际观察到什么</h2>' + paragraphs + '</div>'
        template = template.replace('<div class="card"><h2>可复查文件</h2>', card + '<div class="card"><h2>可复查文件</h2>')
    example_path = OUT / "representative_token_examples.json"
    if example_path.exists():
        examples = json.loads(example_path.read_text())["examples"]
        rows = []
        for e in examples:
            if not e["available"]:
                continue
            source = " + ".join(f"f{s['source_frame_index']} @{s['decoder_pts'][0]:.3f}s" for s in e["source_samples"])
            rows.append(f"<tr><td>{e['uid']} / ({e['grid_row']},{e['grid_col']})</td>"
                        f"<td>L{e['layer']} H{e['kv_head']}</td>"
                        f"<td>{'保留' if e['hermes_retained_this_copy'] else '无此复制'}</td>"
                        f"<td>{'保留' if e['dp_retained_this_copy'] else '无此复制'}</td>"
                        f"<td>{source}</td><td>{e['global_uid_copy_count']['hermes']} / {e['global_uid_copy_count']['dp']}</td></tr>")
        card = '<div class="card"><h2>三个可直接定位的 token 样例（v398 q0，时间组0）</h2>'
        card += '<table><tr><th>UID / (row,col)</th><th>具体层/头</th><th>Hermes</th><th>DP</th><th>两张原帧</th><th>全112头复制数 H / DP</th></tr>' + ''.join(rows) + '</table>'
        card += '<p class="small">“无此复制”只指表中具体层/头。全局仍可能有同UID在其它头中保留，不能解释为来源网格彻底消失。<a href="representative_token_examples.json">精确坐标、注意力与所有保留头</a> · <a href="token_1230_frame_pair.png">UID1230在两张原帧中的逐头像素位置重建</a>。</p></div>'
        template = template.replace('<div class="card"><h2>可复查文件</h2>', card + '<div class="card"><h2>可复查文件</h2>')
    serialized = json.dumps(cases, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    (OUT / "report.html").write_text(template.replace("__DATA__", serialized))


if __name__ == "__main__":
    main()
