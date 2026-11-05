"""Ordinary 448² paired-video input for Qwen HERMES and Full KV."""
import types
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
from qwen_vl_utils.vision_process import smart_resize


def _normal(frames):
    if frames.shape[-1] == 3:
        frames = frames.permute(0, 3, 1, 2)
    height, width = smart_resize(int(frames.shape[-2]), int(frames.shape[-1]),
                                 factor=28, min_pixels=448 * 448, max_pixels=448 * 448)
    return resize(frames, [height, width], interpolation=InterpolationMode.BICUBIC,
                  antialias=True).float()


def install_dense(model, method):
    if method not in ('hermes', 'fullkv'):
        raise ValueError('Unexpected method')
    original_encode = model.encode_video_chunk

    @torch.inference_mode()
    def encode(self, frames, is_last_chunk=False):
        if frames is None or len(frames) == 0:
            return
        if len(frames) > 2:
            raise ValueError('Qwen frame batch must contain <=2 frames')
        normal = _normal(frames)
        # The ordinary fused unit is the only visual prefill for this input.
        original_encode(normal, is_last_chunk=False)
        print('QWEN_448_VIDEO_DENSE', 'frames', len(frames),
              'size', list(normal.shape[-2:]), flush=True)

    model.encode_video_chunk = types.MethodType(encode, model)
