"""Shared, opt-in spatial inputs for Qwen image and video experiments.

This module changes pixels only. It never samples, drops, duplicates or reorders
frames. Qwen video encoding still combines pairs of frames; image encoding does
not combine different real frames.
"""

from __future__ import annotations

import hashlib
from numbers import Integral


PATCH_SIZE = 14
SPATIAL_MERGE_SIZE = 2
TEMPORAL_PATCH_SIZE = 2


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def fixed_input_metadata(canvas_size=896, pad_value=127):
    """Return the JSON-compatible input contract; ``None`` disables the option."""
    if canvas_size is None:
        return {"enabled": False}
    size = _integer(canvas_size, "canvas_size", PATCH_SIZE * SPATIAL_MERGE_SIZE)
    if size % (PATCH_SIZE * SPATIAL_MERGE_SIZE):
        raise ValueError("canvas_size must be divisible by 28")
    pad = _integer(pad_value, "pad_value")
    if pad > 255:
        raise ValueError("pad_value must be <= 255")
    spatial = size // PATCH_SIZE
    tokens = (spatial // SPATIAL_MERGE_SIZE) ** 2
    return {
        "enabled": True,
        "version": "fixed_square_rgb_v1",
        "canvas_size": size,
        "resized_hw": [size, size],
        "resize": "aspect_preserving_bicubic_antialias",
        "rounding": "nearest_integer_half_up",
        "padding": "centered_constant_rgb",
        "pad_value": pad,
        "odd_padding_extra": "bottom_and_right",
        "patch_size": PATCH_SIZE,
        "spatial_merge_size": SPATIAL_MERGE_SIZE,
        "spatial_grid": [spatial, spatial],
        "temporal_patch_size": TEMPORAL_PATCH_SIZE,
        "tokens_per_temporal_unit": tokens,
        "image_tokens_per_real_frame": tokens,
        "video_tokens_for_real_frames": f"ceil(T/{TEMPORAL_PATCH_SIZE})*{tokens}",
        "sampling_changed": False,
    }


def _frame_layout(frames):
    import torch

    if not isinstance(frames, torch.Tensor) or frames.dtype != torch.uint8:
        raise TypeError("frames must be a torch.uint8 RGB tensor")
    if frames.device.type != "cpu":
        raise ValueError("fixed input preparation expects CPU frames")
    if frames.ndim != 4 or frames.shape[0] < 1:
        raise ValueError("frames must be a nonempty NHWC or NCHW batch")
    # Decoded video is normally HWC. For an ambiguous (...,3,...,3) shape,
    # require an explicit conversion by the caller rather than changing axes.
    is_hwc, is_chw = frames.shape[-1] == 3, frames.shape[1] == 3
    if is_hwc == is_chw:
        raise ValueError("frames must have one unambiguous RGB channel axis")
    chw = frames.permute(0, 3, 1, 2) if is_hwc else frames
    if min(chw.shape[-2:]) < 1:
        raise ValueError("frame height and width must be positive")
    return chw, is_hwc


def prepare_fixed_frames(frames, canvas_size=896, pad_value=127):
    """Letterbox a CPU uint8 NHWC/NCHW batch, preserving its layout and order.

    Source frames are never changed. ``canvas_size=None`` is an explicit no-op.
    Resize the original uint8 tensor using torchvision's bicubic antialiasing,
    then center it on a gray square; odd padding goes to the bottom/right.
    """
    metadata = fixed_input_metadata(canvas_size, pad_value)
    if not metadata["enabled"]:
        return frames
    import torch
    from torchvision.transforms import InterpolationMode, functional

    chw, was_hwc = _frame_layout(frames)
    height, width = map(int, chw.shape[-2:])
    size = metadata["canvas_size"]
    longest = max(height, width)
    # Integer half-up rounding avoids platform/float-dependent half ties.
    target_height = max(1, (2 * height * size + longest) // (2 * longest))
    target_width = max(1, (2 * width * size + longest) // (2 * longest))
    resized = functional.resize(
        chw, [target_height, target_width],
        interpolation=InterpolationMode.BICUBIC, antialias=True,
    ) if (height, width) != (target_height, target_width) else chw
    output = torch.full((int(chw.shape[0]), 3, size, size), metadata["pad_value"],
                        dtype=torch.uint8, device="cpu")
    top, left = (size - target_height) // 2, (size - target_width) // 2
    output[:, :, top:top + target_height, left:left + target_width] = resized
    return output.permute(0, 2, 3, 1).contiguous() if was_hwc else output


def frame_rgb_sha256(frames):
    """Hash canonical NHWC bytes so HWC/CHW callers can compare actual pixels."""
    chw, _ = _frame_layout(frames)
    canonical = chw.permute(0, 2, 3, 1).contiguous()
    return hashlib.sha256(canonical.numpy().tobytes()).hexdigest()


def assert_qwen_grid(grid, tokens, raw_frames=None, *, canvas_size=896, kind="video"):
    """Assert Qwen's actual post-processor grid and total encoded token count.

    ``grid`` is one ``[T,H,W]`` row or a matrix of such rows. For a single
    video and known raw frame count, require T=ceil(raw_frames/2). Image rows
    each require T=1. ``tokens`` is the combined count, not a per-unit budget.
    The returned audit is JSON-compatible. This function does not resize data.
    """
    metadata = fixed_input_metadata(canvas_size)
    if not metadata["enabled"]:
        raise ValueError("grid validation requires an enabled fixed input size")
    if kind not in {"video", "image"}:
        raise ValueError("kind must be 'video' or 'image'")
    values = grid.tolist() if hasattr(grid, "tolist") else list(grid)
    if not values:
        raise AssertionError("Qwen grid must not be empty")
    if not isinstance(values[0], (list, tuple)):
        values = [values]
    rows = []
    for row in values:
        if len(row) != 3:
            raise AssertionError("Each Qwen grid row must contain T,H,W")
        try:
            row = [_integer(value, "grid dimension", 1) for value in row]
        except ValueError as exc:
            raise AssertionError(str(exc)) from exc
        if row[1:] != metadata["spatial_grid"]:
            raise AssertionError(f"Unexpected Qwen spatial grid: {row[1:]}")
        if kind == "image" and row[0] != 1:
            raise AssertionError("Each independently encoded image must have grid T=1")
        rows.append(row)
    actual_tokens = _integer(tokens, "tokens", 1)
    expected_tokens = sum(row[0] for row in rows) * metadata["tokens_per_temporal_unit"]
    if actual_tokens != expected_tokens:
        raise AssertionError(f"Expected {expected_tokens} encoded visual tokens, got {actual_tokens}")
    if raw_frames is not None:
        raw_frames = _integer(raw_frames, "raw_frames", 1)
        if kind == "video":
            if len(rows) != 1 or rows[0][0] != (raw_frames + 1) // TEMPORAL_PATCH_SIZE:
                raise AssertionError("Video grid T must equal ceil(raw_frames/2) for one encoded chunk")
        elif len(rows) != raw_frames:
            raise AssertionError("Independent image grid rows must match real frame count")
    return {
        "kind": kind, "grid_thw": rows, "raw_frames": raw_frames,
        "visual_tokens": actual_tokens,
        "tokens_per_temporal_unit": metadata["tokens_per_temporal_unit"],
    }
