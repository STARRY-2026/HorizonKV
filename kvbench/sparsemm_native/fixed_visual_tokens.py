"""Optional exact per-frame visual-token control for Qwen2.5-VL ablations.

The released SimpleStream source is intentionally left untouched. The caller
may apply ``install`` before starting a new evaluation process.
"""

import math

from PIL import Image

GRID_PIXELS = 28  # 14-pixel ViT patch times 2x2 spatial merge.


def canvas_shape(tokens: int, width: int, height: int) -> tuple[int, int]:
    """Pick an exact-token canvas closest to the frame's aspect ratio."""
    if tokens < 64 or width <= 0 or height <= 0:
        raise ValueError('tokens must be >= 64 and frame dimensions positive')
    candidates = []
    for rows in range(8, math.isqrt(tokens) + 1):
        if tokens % rows:
            continue
        columns = tokens // rows
        for grid_h, grid_w in ((rows, columns), (columns, rows)):
            if grid_h >= 8 and grid_w >= 8 and max(grid_h, grid_w) / min(grid_h, grid_w) <= 4:
                error = abs(math.log((grid_w / grid_h) / (width / height)))
                candidates.append((error, grid_w * GRID_PIXELS, grid_h * GRID_PIXELS))
    if not candidates:
        raise ValueError(f'{tokens} tokens has no supported 28-pixel canvas')
    _, target_width, target_height = min(candidates)
    return target_width, target_height


def letterbox(frame: Image.Image, tokens: int) -> Image.Image:
    """Preserve aspect ratio and pad to a canvas yielding exactly N tokens."""
    width, height = canvas_shape(tokens, *frame.size)
    scale = min(width / frame.width, height / frame.height)
    resized = frame.convert('RGB').resize(
        (max(1, round(frame.width * scale)), max(1, round(frame.height * scale))),
        Image.Resampling.BICUBIC,
    )
    canvas = Image.new('RGB', (width, height), color=(114, 114, 114))
    canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return canvas


def install(tokens: int) -> None:
    """Patch only the current worker's input frames, leaving upstream files intact."""
    canvas_shape(tokens, 16, 9)
    from lib import recent_window_eval_qwen25 as upstream

    original = upstream.RecentWindowQAModel.encode_vision

    def encode_fixed(self, frames):
        prepared = [letterbox(frame, tokens) for frame in frames]
        embeds, grid = original(self, prepared)
        counts = (grid.prod(dim=-1) // (self.merge_size ** 2)).tolist()
        if counts != [tokens] * len(frames):
            raise RuntimeError(f'Expected {tokens} visual tokens per frame; got {counts}')
        return embeds, grid

    upstream.RecentWindowQAModel.encode_vision = encode_fixed
