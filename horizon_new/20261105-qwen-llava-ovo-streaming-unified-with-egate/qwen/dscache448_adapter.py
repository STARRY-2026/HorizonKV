"""Experimental Qwen DSCache-style latest-frame replay for SparseMM/LU-KV."""
import copy
from contextlib import nullcontext
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
from qwen_vl_utils.vision_process import smart_resize

_COUNTERS = ("_next_position", "_seen_visual_tokens", "total_processed_frames",
             "last_encoded_frames", "_prefills", "_compressions", "last_metadata")

def set_size(processor, pixels):
    # transformers versions differ on whether video resize reads min/max or size.
    for name in ("image_processor", "video_processor"):
        component = getattr(processor, name, None)
        if component is None: continue
        component.min_pixels = pixels
        component.max_pixels = pixels
        if hasattr(component, "size"):
            component.size = {"shortest_edge": pixels, "longest_edge": pixels}

def install(method):
    from sparsemm_protokv.streaming import StreamingQwen
    from sparsemm_native.streaming import SparseMMStreamingQwen
    from sparsemm_baselines.adapter import SparseMMQwen
    cls = SparseMMQwen if method in ("adakv","pyramidkv") else (StreamingQwen if method == "lukv" else SparseMMStreamingQwen)
    original_clear = cls.clear_cache
    original_answer = cls.question_answering

    def clear(self):
        original_clear(self)
        self._dscache_latest = None

    @torch.inference_mode()
    def process(self, frames, high=False):
        if frames.shape[-1] == 3: frames = frames.permute(0, 3, 1, 2)
        if not high:
            h,w = smart_resize(int(frames.shape[-2]), int(frames.shape[-1]), factor=28,
                               min_pixels=448*448, max_pixels=448*448)
            frames = resize(frames, [h,w], interpolation=InterpolationMode.BICUBIC,
                            antialias=True).float()
            processor = self.processor
        else:
            if len(frames)!=1: raise ValueError("Latest frame must be a single frame")
            frames=frames.float()
            if not hasattr(self, "_dscache_high_processor"):
                self._dscache_high_processor=copy.deepcopy(self.processor)
                set_size(self._dscache_high_processor,896*896)
            processor=self._dscache_high_processor
        measure = self._measure("vision_seconds") if hasattr(self,"_measure") else nullcontext()
        with measure:
            inputs=processor(text=[""],videos=frames,return_tensors="pt").to(self.device,self.dtype)
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
        print("DSCACHE_QWEN", "latest" if high else "regular", "grid", grid[0].tolist(),
              "visual_tokens",features.shape[1],"input_frames",len(frames),flush=True)

    @torch.inference_mode()
    def encode(self, video_chunk, is_last_chunk=False):
        if video_chunk is None or len(video_chunk)==0:return
        if len(video_chunk)>2: raise ValueError("Qwen input supports <=2 frames")
        # Keep only this bounded chunk, not the earlier video's frame features.
        self._dscache_latest=(copy.deepcopy(self.kv_cache),
                              {k:copy.deepcopy(getattr(self,k)) for k in _COUNTERS},
                              video_chunk.detach().clone())
        process(self,video_chunk,high=False)

    @torch.inference_mode()
    def answer(self,*args,**kwargs):
        if self._dscache_latest is None: raise RuntimeError("No latest frame")
        prior,counters,frames=self._dscache_latest
        current=self.kv_cache
        current_counters={k:copy.deepcopy(getattr(self,k)) for k in _COUNTERS}
        if hasattr(current,"state_tensors"):
            state=current.state_tensors()
        else:
            state={f"{i}:{j}":v for i,layer in enumerate(current.snapshot()) for j,v in enumerate(layer)}
        saved_state={k:(v,v.clone()) for k,v in state.items()}
        try:
            self.kv_cache=copy.deepcopy(prior)
            for k,v in counters.items():setattr(self,k,copy.deepcopy(v))
            # Replay a paired chunk as previous normal frame + latest high frame.
            # Its temporal pairing differs from original DSCache; experimental.
            if len(frames)==2:process(self,frames[:1],high=False)
            process(self,frames[-1:],high=True)
            return original_answer(self,*args,**kwargs)
        finally:
            self.kv_cache=current
            for k,v in current_counters.items():setattr(self,k,v)
            if hasattr(current,"state_tensors"):
                restored=current.state_tensors()
            else:
                restored={f"{i}:{j}":v for i,layer in enumerate(current.snapshot()) for j,v in enumerate(layer)}
            if (self.kv_cache is not current or restored.keys()!=saved_state.keys() or
                any(restored[k] is not ref or not torch.equal(restored[k],value)
                    for k,(ref,value) in saved_state.items())):
                raise RuntimeError("Persistent cache changed during high-resolution answer")
            print("DSCACHE_QWEN_RESTORED",method,flush=True)
    cls.clear_cache=clear
    cls.encode_video_chunk=encode
    cls.question_answering=answer
