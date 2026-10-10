#!/usr/bin/env python3
"""Extract the top-attended source frame for each temporal regression."""

import argparse
import json
from pathlib import Path

from decord import VideoReader, cpu
from PIL import Image, ImageDraw, ImageFont


TOP_TIMES = {
    (12, 3): 94.0,
    (27, 2): 414.0,
    (29, 2): 324.0,
    (35, 0): 120.0,
    (51, 2): 258.0,
    (55, 4): 490.0,
    (92, 2): 246.0,
    (92, 3): 366.0,
    (93, 0): 34.0,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    annotation = {int(v["video_id"]): v for v in json.load(open(args.anno))}

    images = []
    for (video_id, question_index), time_sec in TOP_TIMES.items():
        video = annotation[video_id]
        reader = VideoReader(video["video_path"], ctx=cpu(0), num_threads=1)
        frame_index = min(len(reader) - 1, round(time_sec * reader.get_avg_fps()))
        image = Image.fromarray(reader[frame_index].asnumpy()).convert("RGB")
        path = output / f"v{video_id}_q{question_index}_t{time_sec:g}s.jpg"
        image.save(path, quality=95)
        images.append((path.name, image))

    thumb_size = (360, 210)
    label_height = 28
    sheet = Image.new("RGB", (3 * 376, 3 * 246), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for idx, (label, image) in enumerate(images):
        image.thumbnail(thumb_size, Image.Resampling.LANCZOS)
        x = (idx % 3) * 376 + (360 - image.width) // 2
        y = (idx // 3) * 246 + label_height
        sheet.paste(image, (x, y))
        draw.text(((idx % 3) * 376 + 8, (idx // 3) * 246 + 7), label, fill="black", font=font)
    sheet.save(output / "contact_sheet.jpg", quality=95)
    print(output / "contact_sheet.jpg")


if __name__ == "__main__":
    main()
