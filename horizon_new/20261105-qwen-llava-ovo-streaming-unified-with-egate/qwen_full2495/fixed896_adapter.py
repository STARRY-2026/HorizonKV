"""Experimental per-image Qwen input for streaming baselines."""
from pathlib import Path
import sys
import torch
from PIL import Image
root=Path('/mnt/aifordata2026-p/common/aifordata/541707')
import importlib.util
from methods.registry import METHODS_ROOT
path=METHODS_ROOT/'_shared/fixed_inputs.py'
spec=importlib.util.spec_from_file_location('egate_fixed_inputs',path)
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
prepare_fixed_frames,assert_qwen_grid=mod.prepare_fixed_frames,mod.assert_qwen_grid

def install(method):
    from sparsemm_protokv.streaming import StreamingQwen
    from sparsemm_native.streaming import SparseMMStreamingQwen
    cls=StreamingQwen if method=='lukv' else SparseMMStreamingQwen
    @torch.inference_mode()
    def encode(self, video_chunk, is_last_chunk=False):
        if video_chunk is None or video_chunk.shape[0] == 0: return
        frames=prepare_fixed_frames(video_chunk.cpu())
        if frames.shape[-1]!=3: frames=frames.permute(0,2,3,1).contiguous()
        self.processor.image_processor.do_resize=False
        size=len(frames)  # Update once after encoding the full input batch.
        for start in range(0,len(frames),size):
            group=frames[start:start+size]
            with self._measure('vision_seconds'):
                inp=self.processor.image_processor(images=[Image.fromarray(f.numpy()) for f in group],return_tensors='pt',do_resize=False)
                grid=inp['image_grid_thw'].to(self.device)
                raw=self.model.get_image_features(inp['pixel_values'].to(self.device,dtype=self.dtype),grid)
                if isinstance(raw,(tuple,list)): features=torch.cat(list(raw),dim=0) if len(raw)>1 else raw[0]
                else: features=raw
                if features.ndim==3: features=features.reshape(-1,features.shape[-1])
                assert_qwen_grid(grid,features.shape[0],kind='image')
                features=features.unsqueeze(0)
            positions=[]
            for i,row in enumerate(grid):
                old=self.total_processed_frames
                self.total_processed_frames=old+i
                positions.append(self._video_positions(row.tolist(),1024))
                self.total_processed_frames=old
            positions=torch.cat(positions,dim=2)
            print('FIXED896_BATCH', 'frames', len(group), 'tokens', features.shape[1], flush=True)
            self.forward_chunk(features,positions,compress=True,time=self.total_processed_frames+len(group))
            self._seen_visual_tokens+=features.shape[1]
            self._next_position=max(self._next_position,int(positions.max())+1)
            self.total_processed_frames+=len(group)
            self.timings['video_prefills']+=1
            self.timings['raw_frames']+=len(group)
            self.timings['visual_tokens']+=features.shape[1]
        self.last_encoded_frames=video_chunk.shape[0]
    cls.encode_video_chunk=encode
