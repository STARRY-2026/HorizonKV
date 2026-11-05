"""Qwen 1-fps paired-video input at ordinary 448² target pixel area."""
from contextlib import nullcontext
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
from qwen_vl_utils.vision_process import smart_resize

def install(method):
    from sparsemm_protokv.streaming import StreamingQwen
    from sparsemm_native.streaming import SparseMMStreamingQwen
    from sparsemm_baselines.adapter import SparseMMQwen
    cls = SparseMMQwen if method in ("adakv","pyramidkv") else (StreamingQwen if method == "lukv" else SparseMMStreamingQwen)
    @torch.inference_mode()
    def process(self, frames):
        if frames.shape[-1] == 3: frames = frames.permute(0, 3, 1, 2)
        h,w = smart_resize(int(frames.shape[-2]), int(frames.shape[-1]), factor=28,
                           min_pixels=448*448, max_pixels=448*448)
        frames = resize(frames, [h,w], interpolation=InterpolationMode.BICUBIC,
                        antialias=True).float()
        measure = self._measure("vision_seconds") if hasattr(self,"_measure") else nullcontext()
        with measure:
            inputs=self.processor(text=[""],videos=frames,return_tensors="pt").to(self.device,self.dtype)
            grid=inputs["video_grid_thw"]
            features=self.model.get_video_features(inputs["pixel_values_videos"],grid)[0].unsqueeze(0)
        positions=self._video_positions(grid[0].tolist(),features.shape[1])
        if method in ("adakv","pyramidkv"):
            self.forward_chunk(features,positions,compress=True)
        else:
            self.forward_chunk(features,positions,compress=True,time=self.total_processed_frames+len(frames))
        self._seen_visual_tokens+=features.shape[1]
        self._next_position=max(self._next_position,int(positions.max())+1)
        self.total_processed_frames+=len(frames)
        self.last_encoded_frames=len(frames)
        print("QWEN_448_VIDEO", "grid", grid[0].tolist(),
              "visual_tokens",features.shape[1],"input_frames",len(frames),flush=True)

    @torch.inference_mode()
    def encode(self, video_chunk, is_last_chunk=False):
        if video_chunk is None or len(video_chunk)==0:return
        if len(video_chunk)>2: raise ValueError("Qwen input supports <=2 frames")
        process(self,video_chunk)

    cls.encode_video_chunk=encode
