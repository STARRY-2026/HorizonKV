#!/bin/bash
# ============================================================
# 全量下载：Real-Time Visual Understanding 1-500（10 个 ZIP）
# 每个 ZIP 下载后立即解压并删除，最后统一创建软链接
# 用法: bash scripts/download_full.sh
# ============================================================
set -euo pipefail

# 使用国内镜像（关键！）
export HF_ENDPOINT="https://hf-mirror.com"

REPO="mjuicem/StreamingBench"
DATA_ROOT="${DATA_ROOT:-/data}"
EXTRACT_TO="$DATA_ROOT/StreamingBench/data/real"
DOWNLOAD_DIR="$DATA_ROOT/.downloads/streamingbench"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PID_FILE="$DATA_ROOT/download_full.pid"
export DATA_ROOT

mkdir -p "$EXTRACT_TO"
mkdir -p "$DOWNLOAD_DIR"
echo "$$" > "$PID_FILE"
trap 'rm -f "$PID_FILE"' EXIT

echo "============================================================="
echo "   HERMES 数据集下载脚本"
echo "   HF 镜像: $HF_ENDPOINT"
echo "   目标: Real-Time Visual Understanding 1-500"
echo "   解压位置: $EXTRACT_TO"
echo "============================================================="
echo ""

STARTS=(1 51 101 151 201 251 301 351 401 451)
TOTAL=${#STARTS[@]}
PARALLEL_DOWNLOADS="${PARALLEL_DOWNLOADS:-3}"

download_range() {
    local start="$1"
    local index="$2"
    local end=$((start + 49))
    local zip_name="Real-Time Visual Understanding_${start}-${end}.zip"
    local done_marker="$EXTRACT_TO/.extracted_${start}-${end}"
    local zip_path="$DOWNLOAD_DIR/$zip_name"
    local part_path="$zip_path.part"
    local encoded_zip_name="${zip_name// /%20}"
    local download_url="${HF_ENDPOINT}/datasets/${REPO}/resolve/main/${encoded_zip_name}?download=true"

    if [ -f "$done_marker" ]; then
        echo "  [$index/$TOTAL] $start-$end 已完成，跳过下载"
        return
    fi
    if [ -f "$zip_path" ]; then
        echo "  [$index/$TOTAL] $start-$end 已下载，等待解压"
        return
    fi

    echo "  [$index/$TOTAL] 开始下载 $start-$end"
    curl --location --fail --silent --show-error --continue-at - \
        --retry 10 --retry-all-errors --retry-delay 5 \
        --connect-timeout 30 --speed-time 120 --speed-limit 1024 \
        --output "$part_path" \
        "$download_url"
    mv "$part_path" "$zip_path"
    echo "  [$index/$TOTAL] 下载完成 $start-$end ($(du -h "$zip_path" | cut -f1))"
}

echo "  并行下载数: $PARALLEL_DOWNLOADS"
for INDEX in "${!STARTS[@]}"; do
    download_range "${STARTS[$INDEX]}" "$((INDEX + 1))" &
    while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL_DOWNLOADS" ]; do
        wait -n
    done
done
wait

echo "  所有分包下载完成，开始串行解压"
for START in "${STARTS[@]}"; do
    END=$((START + 49))
    ZIP_NAME="Real-Time Visual Understanding_${START}-${END}.zip"
    DONE_MARKER="$EXTRACT_TO/.extracted_${START}-${END}"
    ZIP_PATH="$DOWNLOAD_DIR/$ZIP_NAME"
    if [ -f "$DONE_MARKER" ]; then
        continue
    fi
    echo "  [解压] $START-$END"
    if command -v unzip >/dev/null 2>&1; then
        unzip -q -o "$ZIP_PATH" -d "$EXTRACT_TO"
    else
        python3 -m zipfile -e "$ZIP_PATH" "$EXTRACT_TO"
    fi
    touch "$DONE_MARKER"
    rm "$ZIP_PATH"
done

echo ""
echo "============================================================="
echo "  所有 ZIP 处理完毕，创建软链接..."
echo "============================================================="
cd "$PROJECT_DIR"
bash scripts/link_data.sh

echo ""
echo "============================================================="
echo "  最终统计"
echo "============================================================="
VIDEO_COUNT=$(ls "$DATA_ROOT"/streamingbench/videos/*.mp4 2>/dev/null | wc -l)
echo "  软链接成功: $VIDEO_COUNT 个视频"

python3 - <<'PY'
import json
import os

with open("data/streamingbench/streamingbench_realtime.json", encoding="utf-8") as f:
    annotations = json.load(f)

data_root = os.environ.get("DATA_ROOT", "/data")
expected_paths = [
    os.path.join(data_root, os.path.relpath(item["video_path"], "/data"))
    for item in annotations
]
missing = [path for path in expected_paths if not os.path.isfile(path)]
question_count = sum(len(item.get("conversations", [])) for item in annotations)
print(f"  完整标注: {len(annotations)} 个视频, {question_count} 道题")
print(f"  缺失视频: {len(missing)}")
if missing:
    print("  前 10 个缺失路径:")
    for path in missing[:10]:
        print(f"    {path}")
    raise SystemExit(1)
PY
