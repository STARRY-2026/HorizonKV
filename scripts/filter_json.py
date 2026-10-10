#!/usr/bin/env python3
"""从 streamingbench_realtime.json 中筛选 sample_201 到 sample_250"""
import json, re

SRC = "data/streamingbench/streamingbench_realtime.json"
DST = "data/streamingbench/streamingbench_realtime_half.json"

with open(SRC) as f:
    data = json.load(f)

filtered = []
for item in data:
    m = re.search(r"sample_(\d+)_real", item["video_path"])
    if m:
        sid = int(m.group(1))
        if 1 <= sid <= 250:
            filtered.append(item)

with open(DST, "w") as f:
    json.dump(filtered, f, indent=4, ensure_ascii=False)

print(f"原文件: {len(data)} 条")
print(f"筛选后: {len(filtered)} 条 (sample_1 ~ sample_250)")
print(f"输出:   {DST}")
