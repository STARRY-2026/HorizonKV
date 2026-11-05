"""DSCache Qwen frame resizing helpers; not wired into EGate inference."""
from qwen_vl_utils.vision_process import smart_resize
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

def regular_frame_size(height, width):
    return smart_resize(height, width, factor=28,
                        min_pixels=448 * 448, max_pixels=448 * 448)

def resize_regular_frames(frames):
    height, width = regular_frame_size(frames.shape[-2], frames.shape[-1])
    return resize(frames, [height, width], interpolation=InterpolationMode.BICUBIC,
                  antialias=True).float()

def qwen_pixel_budgets(stride=2.0):
    return dict(regular=448 * 448, latest=int(448 * 448 * stride * stride))
