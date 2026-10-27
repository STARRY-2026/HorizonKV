"""Qwen-vl-utils 0.0.14 video resize rule used by released SimpleStream.

The sampled frame count is supplied by the OVO runner so that each question's
source prefix receives the same duration-dependent pixel limit. Sampling and
cache policies remain those of the two kvbench arms.
"""

import math


FACTOR = 28
MIN_TOKENS = 128
MAX_TOKENS = 768
MODEL_SEQ_LEN = 128000
FRAME_FACTOR = 2


def _round(value):
    return round(value / FACTOR) * FACTOR


def _floor(value):
    return math.floor(value / FACTOR) * FACTOR


def _ceil(value):
    return math.ceil(value / FACTOR) * FACTOR


def resized_hw(height, width, sampled_frames):
    """Match fetch_video's max-pixel formula and smart_resize quantization."""
    if min(height, width, sampled_frames) < 1 or max(height, width) / min(height, width) > 200:
        raise ValueError("Invalid source frame dimensions or sampled frame count")
    nframes = _ceil_even(sampled_frames)
    min_pixels = MIN_TOKENS * FACTOR**2
    max_pixels = max(min(MAX_TOKENS * FACTOR**2,
                         MODEL_SEQ_LEN * FACTOR**2 * 0.9 / nframes * FRAME_FACTOR),
                     int(min_pixels * 1.05))
    h, w = max(FACTOR, _round(height)), max(FACTOR, _round(width))
    if h * w > max_pixels:
        scale = math.sqrt(height * width / max_pixels)
        h, w = _floor(height / scale), _floor(width / scale)
    elif h * w < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        h, w = _ceil(height * scale), _ceil(width * scale)
    if min(h, w) < FACTOR:
        raise ValueError("Resize produced an invalid visual grid")
    return h, w


def _ceil_even(value):
    return value + value % 2


def resize_hwc_batch(frames, height, width):
    """Resize decoded uint8 HWC frames with SimpleStream's bicubic antialiasing."""
    from torchvision.transforms import InterpolationMode, functional

    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("Expected a decoded N,H,W,3 frame batch")
    if tuple(frames.shape[1:3]) == (height, width):
        return frames
    chw = frames.permute(0, 3, 1, 2)
    return functional.resize(chw, [height, width],
                             interpolation=InterpolationMode.BICUBIC,
                             antialias=True).permute(0, 2, 3, 1).contiguous()
