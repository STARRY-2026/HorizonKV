"""Experimental temporary latest-frame input for dense Qwen HERMES/Full KV."""
import copy
import types
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
from qwen_vl_utils.vision_process import smart_resize
from dscache448_adapter import set_size

_FIELDS=("kv_cache","_position_ids_cache","_layer_position_ids",
         "total_processed_frames","last_encoded_frames","token_activity_cache")

def _capture(model):
    return {k:copy.deepcopy(getattr(model,k)) for k in _FIELDS if hasattr(model,k)}

def _restore(model, values):
    for k,v in values.items():setattr(model,k,v)

def _normal(frames):
    if frames.shape[-1]==3:frames=frames.permute(0,3,1,2)
    h,w=smart_resize(int(frames.shape[-2]),int(frames.shape[-1]),factor=28,
                     min_pixels=448*448,max_pixels=448*448)
    return resize(frames,[h,w],interpolation=InterpolationMode.BICUBIC,antialias=True).float()

def install_dense(model,method):
    if method not in ("hermes","fullkv"):raise ValueError("Unexpected method")
    original_encode=model.encode_video_chunk
    original_answer=model.question_answering
    high_processor=copy.deepcopy(model.processor)
    set_size(high_processor,896*896)
    model._dscache_replay_active=False

    @torch.inference_mode()
    def encode(self,frames,is_last_chunk=False):
        if frames is None or len(frames)==0:return
        if len(frames)>2:raise ValueError("Qwen frame batch must contain <=2 frames")
        self._dscache_latest=(_capture(self),frames.detach().clone())
        normal=_normal(frames)
        original_encode(normal,is_last_chunk=False)
        print("DSCACHE_QWEN_DENSE regular", "frames",len(frames), "size",list(normal.shape[-2:]),flush=True)

    @torch.inference_mode()
    def answer(self,*args,**kwargs):
        if self._dscache_replay_active or kwargs.get("pseudo_forward",False):
            return original_answer(*args,**kwargs)
        if not hasattr(self,"_dscache_latest"):raise RuntimeError("Missing latest input")
        prior,frames=self._dscache_latest
        # All current tensors are retained separately; replay cannot mutate them.
        current={k:getattr(self,k) for k in _FIELDS if hasattr(self,k)}
        before=[(k.detach().clone(),v.detach().clone()) for k,v in current["kv_cache"]]
        positions_before=[None if v is None else v.detach().clone()
                          for v in current["_position_ids_cache"]]
        try:
            _restore(self,copy.deepcopy(prior))
            self._dscache_replay_active=True
            if len(frames)==2:
                self._encode_video_frames_internal(_normal(frames[:1]),self.processor)
            last=frames[-1:]
            if last.shape[-1]==3:last=last.permute(0,3,1,2)
            last=last.float()
            inputs=high_processor(text=[""],videos=last,return_tensors="pt")
            grid=inputs["video_grid_thw"][0].tolist()
            self._encode_video_frames_internal(last,high_processor)
            self.predict_and_compress()
            print("DSCACHE_QWEN_DENSE latest","grid",grid,
                  "visual_tokens",int(grid[0]*grid[1]*grid[2]//4),flush=True)
            return original_answer(*args,**kwargs)
        finally:
            _restore(self,current)
            self._dscache_replay_active=False
            if len(self.kv_cache)!=len(before) or any(
                not torch.equal(k,bk) or not torch.equal(v,bv)
                for (k,v),(bk,bv) in zip(self.kv_cache,before)):
                raise RuntimeError("Persistent dense KV changed during answer replay")
            positions_after=self._position_ids_cache
            if len(positions_after)!=len(positions_before) or any(
                (a is None)!=(b is None) or (a is not None and not torch.equal(a,b))
                for a,b in zip(positions_after,positions_before)):
                raise RuntimeError("Persistent position state changed during answer replay")
            if any(getattr(self,k)!=current[k] for k in ("total_processed_frames","last_encoded_frames")):
                raise RuntimeError("Persistent frame counters changed during answer replay")
            print("DSCACHE_QWEN_DENSE_RESTORED",method,flush=True)
    model.encode_video_chunk=types.MethodType(encode,model)
    model.question_answering=types.MethodType(answer,model)
