#!/bin/bash
# ============================================================
# 快速试跑：下载 1 个 ZIP（401-450），解压，软链接，验证
# 用法: bash scripts/download_test.sh
# ============================================================
set -euo pipefail

# 使用国内镜像（关键！）
export HF_ENDPOINT="https://hf-mirror.com"

REPO="mjuicem/StreamingBench"
ZIP_NAME="Real-Time Visual Understanding_151-200.zip"
EXTRACT_TO="/data/StreamingBench/data/real"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== HF 镜像: $HF_ENDPOINT ==="
echo "=== 1. 创建目录 ==="
mkdir -p "$EXTRACT_TO"

echo "=== 2. 下载 $ZIP_NAME ==="
# 下载到临时目录（避免 local-dir 缓存问题）
TMP_DIR=$(mktemp -d)
hf download "$REPO" \
    --repo-type dataset \
    --include "$ZIP_NAME" \
    --local-dir "$TMP_DIR"

# 验证下载
if [ ! -f "$TMP_DIR/$ZIP_NAME" ]; then
    echo "错误: 下载失败，文件不存在: $TMP_DIR/$ZIP_NAME"
    echo "请检查 HF_ENDPOINT 或网络连接"
    rm -rf "$TMP_DIR"
    exit 1
fi
echo "  下载成功: $(ls -lh "$TMP_DIR/$ZIP_NAME" | awk '{print $5}')"

echo "=== 3. 解压到 $EXTRACT_TO ==="
unzip -o "$TMP_DIR/$ZIP_NAME" -d "$EXTRACT_TO"

echo "=== 4. 删除 ZIP ==="
rm -rf "$TMP_DIR"

echo "=== 5. 创建软链接 ==="
cd "$PROJECT_DIR"
bash scripts/link_data.sh

echo ""
echo "=== 6. 验证 ==="
if [ -f /data/streamingbench/videos/sample_151_real.mp4 ]; then
    echo "试跑成功！sample_151_real.mp4 已就绪"
else
    echo "错误：软链接未创建，请检查"
    exit 1
fi
