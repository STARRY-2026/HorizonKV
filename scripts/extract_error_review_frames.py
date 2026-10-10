#!/usr/bin/env python3
"""CPU-only evidence extraction; reproduce sampled frame ordinals and PTS.

Run with the hermes-qwen environment (decord/Pillow/numpy). Images labelled
INPUT are source frames selected by the experiment, before VL preprocessing
and KV compression. SOURCE frames are legal prefix evidence, not model input.
Neither label establishes final KV retention. No model is loaded.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from decord import VideoReader, cpu
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-sampled-task", action="append", default=[])
    parser.add_argument("--overview-task", action="append", default=[])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(args.cases.open()))
    font = ImageFont.load_default(size=17)
    reports = []
    for case in rows:
        vid, qi = case["video_id"], case["question_index"]
        destination = args.output / f"v{vid}_q{qi}"
        destination.mkdir(exist_ok=True)
        manifest_path = destination / "manifest.json"
        if manifest_path.exists():
            reports.append(json.loads(manifest_path.read_text()))
            print(f"existing v{vid}/q{qi}", flush=True)
            continue
        vr = VideoReader(case["video_path"], ctx=cpu(0), num_threads=1)
        fps = vr.get_avg_fps()
        step = int(round(fps) / 0.5)
        n_requested = math.ceil(float(case["end_time"]) * 0.5)
        sampled = list(range(0, len(vr), step))[:n_requested]
        assert sampled
        pts = np.asarray(vr.get_frame_timestamp(list(range(len(vr)))))[:, 0]
        assert np.all(np.diff(pts) >= 0)
        cut = float(case["end_time"])

        def source_at(seconds):
            return max(0, min(len(vr) - 1, int(np.searchsorted(pts, seconds + 1e-6, side="right") - 1)))

        # Local chronological tail and legal source frames near the cut.
        tail = sampled[max(0, len(sampled) - 8):]
        tail += [source_at(max(0, cut - delta)) for delta in (1.0, 0.5, 0.0)]
        tail = sorted(set(tail))
        anchors = sorted(set(sampled[min(len(sampled) - 1, int((len(sampled) - 1) * fraction))]
                             for fraction in (0, .2, .4, .6, .8)))
        groups = [("anchors", anchors), ("tail", tail)]
        if case["task"] in args.full_sampled_task:
            groups.append(("full", sampled))
        elif case["task"] in args.overview_task:
            groups.append(("overview", sampled[::4]))
        needed = sorted(set(i for _, group in groups for i in group))
        sampled_lookup = {idx: j for j, idx in enumerate(sampled)}
        records, thumbnails = [], {}
        source_shape = None

        def frame_data():
            for start in range(0, len(needed), 16):
                chunk = needed[start:start + 16]
                batch = vr.get_batch(chunk).asnumpy()
                yield from zip(chunk, batch)

        for idx, data in frame_data():
            if source_shape is None:
                source_shape = list(data.shape)
            im = Image.fromarray(data)
            file_path = destination / f"frame_{idx}.jpg"
            im.save(file_path, quality=89)
            record = {"frame_index": idx, "pts_seconds": float(pts[idx]),
                      "sample_ordinal": sampled_lookup.get(idx),
                      "experiment_input_frame": idx in sampled_lookup,
                      "after_question_cut": bool(pts[idx] > cut + 1e-5),
                      "image_path": str(file_path.resolve())}
            records.append(record)
            tile = Image.new("RGB", (480, 294), "#131a22")
            im.thumbnail((476, 264))
            tile.paste(im, ((480 - im.width) // 2, 28 + (264 - im.height) // 2))
            kind = f"INPUT s{sampled_lookup[idx]}" if idx in sampled_lookup else "SOURCE (not sampled)"
            if record["after_question_cut"]:
                kind += " AFTER CUT"
            ImageDraw.Draw(tile).text((6, 4), f"{kind} | f{idx} | {pts[idx]:.3f}s", font=font, fill="white")
            thumbnails[idx] = tile
        sheets = []
        for name, indices in groups:
            for start in range(0, len(indices), 20):
                chunk = indices[start:start + 20]
                sheet = Image.new("RGB", (1920, 294 * math.ceil(len(chunk) / 4) + 32), "#131a22")
                ImageDraw.Draw(sheet).text((8, 5), f"v{vid}/q{qi} cut={cut:g}s | {name}", font=font, fill="white")
                for j, idx in enumerate(chunk):
                    sheet.paste(thumbnails[idx], ((j % 4) * 480, 32 + (j // 4) * 294))
                sheet_path = destination / f"{name}_{start // 20 + 1}.jpg"
                sheet.save(sheet_path, quality=91)
                sheets.append({"group": name, "path": str(sheet_path.resolve()), "frame_indices": chunk})
        result = {"video_id": vid, "question_index": qi, "task": case["task"],
                  "question": case["question"], "choices": json.loads(case["choices"]),
                  "gold": case["answer"], "dp_answer": case["dp1000_answer"],
                  "hermes_answer": case["hermes1000_answer"], "lu_answer": case["lu1000_answer"],
                  "video_path": case["video_path"], "end_time": cut,
                  "avg_fps": fps, "rounded_fps": round(fps), "stride_frames": step,
                  "requested_sample_count": n_requested, "actual_sample_count": len(sampled),
                  "last_input_frame_index": sampled[-1], "last_input_pts": float(pts[sampled[-1]]),
                  "sampled_frames_after_cut": [i for i in sampled if pts[i] > cut + 1e-5],
                  "source_shape": source_shape, "records": records, "sheets": sheets,
                  "scope": "CPU source-frame evidence before VL preprocessing and KV compression; no inference"}
        manifest_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        reports.append(result)
        print(f"decoded v{vid}/q{qi}: {len(needed)} images, {len(sheets)} sheets, last={pts[sampled[-1]]:.3f}, cut={cut:g}", flush=True)
        del vr, thumbnails
    (args.output / "index.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n")
    print(f"completed {len(reports)} cases", flush=True)


if __name__ == "__main__":
    main()
