"""Check the optional HERMES-sized Qwen video forward without loading weights."""

from contextlib import nullcontext
from types import SimpleNamespace
import unittest

import torch

from sparsemm_protokv.streaming import StreamingQwen


class _Inputs(dict):
    def to(self, *_args):
        return self


class _Processor:
    def __init__(self):
        self.frame_counts = []

    def __call__(self, *, text, videos, return_tensors, **_kwargs):
        assert text == [""] and return_tensors == "pt"
        self.frame_counts.append(int(videos.shape[0]))
        grid = torch.tensor([[int((videos.shape[0] + 1) // 2), 2, 2]])
        return _Inputs(video_grid_thw=grid, pixel_values_videos=torch.zeros(1))


class _Model:
    device = torch.device("cpu")

    def parameters(self):
        yield torch.zeros(1)

    def get_video_features(self, _pixels, grid):
        return [torch.zeros((int(grid[0, 0]), 4))]


class ChunkModeTest(unittest.TestCase):
    def bridge(self, mode):
        bridge = StreamingQwen.__new__(StreamingQwen)
        bridge.is_llava = False
        bridge.config = SimpleNamespace(video_chunk_mode=mode)
        bridge.model_config = SimpleNamespace(vision_config=SimpleNamespace(
            temporal_patch_size=2, patch_size=14))
        bridge.processor = _Processor()
        bridge.model = _Model()
        bridge.video_max_pixels = None
        bridge._measure = lambda _field: nullcontext()
        bridge._video_positions = lambda _grid, count: torch.arange(count).view(1, 1, -1).expand(3, 1, -1)
        bridge.forward_calls = []
        bridge.forward_chunk = lambda features, positions, **_kw: bridge.forward_calls.append(
            (int(features.shape[1]), dict(bridge._pending_visual_input)))
        bridge._pending_visual_input = None
        bridge._seen_visual_tokens = bridge._next_position = bridge.total_processed_frames = 0
        bridge.timings = {name: 0 for name in ("vision_seconds", "video_prefills", "raw_frames", "visual_tokens")}
        return bridge

    def test_five_frames_commit_once_in_hermes_mode(self):
        frames = torch.zeros((5, 28, 28, 3), dtype=torch.uint8)
        chunk = self.bridge("hermes_chunk")
        legacy = self.bridge("temporal_patch")
        StreamingQwen.encode_video_chunk(chunk, frames)
        StreamingQwen.encode_video_chunk(legacy, frames)
        self.assertEqual(chunk.processor.frame_counts, [5])
        self.assertEqual(legacy.processor.frame_counts, [2, 2, 1])
        self.assertEqual(chunk.timings["visual_tokens"], legacy.timings["visual_tokens"])
        self.assertEqual(chunk.forward_calls[0][1], {
            "raw_frames": 5, "source_hw": [28, 28], "grid_thw": [3, 2, 2],
            "resized_hw": [28, 28], "visual_tokens": 3})


if __name__ == "__main__":
    unittest.main()
