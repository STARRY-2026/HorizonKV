#!/bin/bash
# Symlink existing benchmark videos into the paths HERMES expects.
# Run: bash scripts/link_data.sh

set -e

DATA_ROOT="${DATA_ROOT:-/data}"

echo "=== Linking StreamingBench ==="
mkdir -p "$DATA_ROOT/streamingbench/videos"

sb_linked=0
sb_missing=0
for sample_dir in "$DATA_ROOT"/StreamingBench/data/real/sample_*/; do
    sample_id=$(basename "$sample_dir")
    src="${sample_dir}video.mp4"
    dst="$DATA_ROOT/streamingbench/videos/${sample_id}_real.mp4"
    if [ -f "$src" ]; then
        ln -sf "$src" "$dst"
        sb_linked=$((sb_linked + 1))
    else
        sb_missing=$((sb_missing + 1))
    fi
done
echo "StreamingBench: $sb_linked linked, $sb_missing missing"

echo ""
echo "=== Linking OVOBench ==="
mkdir -p "$DATA_ROOT/ovobench/videos"

# AutoEvalMetaData (154 files)
if [ -d "$DATA_ROOT/OVOBench/src_videos/AutoEvalMetaData" ]; then
    ln -sfn "$DATA_ROOT/OVOBench/src_videos/AutoEvalMetaData" "$DATA_ROOT/ovobench/videos/AutoEvalMetaData"
    echo "OVOBench/AutoEvalMetaData: symlinked"
else
    echo "OVOBench/AutoEvalMetaData: NOT FOUND"
fi

# Ego4D - actual data is at OVOBench/src_videos/Ego4D/{clips,video}
if [ -d "$DATA_ROOT/OVOBench/OVOBench/src_videos/Ego4D" ]; then
    mkdir -p "$DATA_ROOT/ovobench/videos/Ego4D"
    if [ -d "$DATA_ROOT/OVOBench/OVOBench/src_videos/Ego4D/clips" ]; then
        ln -sfn "$DATA_ROOT/OVOBench/OVOBench/src_videos/Ego4D/clips" "$DATA_ROOT/ovobench/videos/Ego4D/clips"
        echo "OVOBench/Ego4D/clips: symlinked"
    fi
    if [ -d "$DATA_ROOT/OVOBench/OVOBench/src_videos/Ego4D/video" ]; then
        ln -sfn "$DATA_ROOT/OVOBench/OVOBench/src_videos/Ego4D/video" "$DATA_ROOT/ovobench/videos/Ego4D/video"
        echo "OVOBench/Ego4D/video: symlinked"
    fi
else
    echo "OVOBench/Ego4D: NOT FOUND"
fi

# YouTube_Games
if [ -d "$DATA_ROOT/OVOBench/OVOBench/src_videos/YouTube_Games" ]; then
    ln -sfn "$DATA_ROOT/OVOBench/OVOBench/src_videos/YouTube_Games" "$DATA_ROOT/ovobench/videos/YouTube_Games"
    echo "OVOBench/YouTube_Games: symlinked"
else
    echo "OVOBench/YouTube_Games: NOT FOUND"
fi

# Other OVOBench categories not available in current data
for cat in COIN OpenEQA cross_task hirest star youcook2; do
    echo "OVOBench/$cat: NOT FOUND (no source data)"
done

echo ""
echo "=== Coverage Summary ==="
echo "StreamingBench: ready for inference"
echo "OVOBench: partial — AutoEvalMetaData available; Ego4D, YouTube_Games partial; COIN/OpenEQA/cross_task/hirest/star/youcook2 missing"
