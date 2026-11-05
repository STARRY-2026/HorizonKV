"""LLaVA Qwen2 bridge around the frozen HERMES scoring and pruning methods."""
import ast, contextlib, io, logging, re, types
from pathlib import Path
import torch
import torch.nn.functional as F
from inference.reindex_1d import (_get_rotary_module, get_cache_seq_len, compute_cos_sin_for_positions,
                                  rotary_delta, apply_rotary_delta_to_keys_only, apply_rotary_pos_emb_1d)

from methods.registry import qwen_impl
SOURCE=qwen_impl('hermes')/'inference/llavaov_hermes.py'
METHODS={'_set_rotary_required_len','_prune_activity_cache','_get_cache_seq_len_per_layer',
         '_get_next_start_pos_per_layer','_sanitize_keep_indices','_build_position_ids',
         '_shrink_positions_and_rerotate_keys','predict_next_question','apply_kv_cache_pruning_strict',
         'allocate_budget_by_depth','_compute_attention_scores_manually','prune_kv_cache_by_attention','pseudo_forward'}
tree=ast.parse(SOURCE.read_text())
original=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LlavaOneVision_Hermes')
methods=[n for n in original.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in METHODS]
assert {n.name for n in methods}==METHODS
classnode=ast.ClassDef(name='FrozenMethods',bases=[],keywords=[],body=methods,decorator_list=[])
module=ast.fix_missing_locations(ast.Module(body=[classnode],type_ignores=[]))
logger=logging.getLogger('hermes')
exec(compile(module,str(SOURCE),'exec'),globals())

class HermesBridge(FrozenMethods):
    def __init__(self,model,tokenizer,stream):
        self.model=model; self.stream=stream; self.processor=types.SimpleNamespace(tokenizer=tokenizer)
        self.language_model=types.SimpleNamespace(model=model.get_model(),config=model.config)
        self.device=model.device; self.num_layers=len(model.get_model().layers); self.kv_size=500
        self.streaming=True; self.short_term_threshold=int(self.num_layers*0.3); self.long_term_threshold=int(self.num_layers*0.7)
        self._layer_position_ids={}; self.token_activity_cache=[None]*self.num_layers
        self._position_ids_cache=stream.positions; self.visual_start_idx=stream.prefix
        self.conv_history=[]; self.kv_cache=None
    def get_input_embeddings(self): return self.model.get_input_embeddings()
    def get_gpu_memory_usage_gb(self): return torch.cuda.memory_allocated()/1024**3
    def sync_from_stream(self):
        self._position_ids_cache=self.stream.positions
        self.visual_start_idx=self.stream.prefix
        self.kv_cache=[]
        for layer in self.stream.cache.layers:
            k=torch.stack([layer.get_segment(h)[0] for h in range(self.stream.nk)])[None]
            v=torch.stack([layer.get_segment(h)[1] for h in range(self.stream.nk)])[None]
            self.kv_cache.append((k,v))
    def sync_to_stream(self):
        self.stream.positions=self._position_ids_cache
        for i,(k,v) in enumerate(self.kv_cache):
            layer=self.stream.cache.layers[i]; length=k.shape[2]
            layer.flat_k=k[0].reshape(-1,self.stream.d).contiguous()
            layer.flat_v=v[0].reshape(-1,self.stream.d).contiguous()
            layer.flat_pos=self._position_ids_cache[i].to(dtype=torch.float32).view(1,-1).expand(3,-1).repeat(1,self.stream.nk).contiguous()
            layer.head_lens=layer.head_lens.new_full((self.stream.nk,),length); layer._refresh()
    @torch.inference_mode()
    def compress(self):
        self.sync_from_stream(); before=[x[0].shape[2] for x in self.kv_cache]
        with contextlib.redirect_stdout(io.StringIO()):
            local,global_=self.predict_next_question(); self.pseudo_forward(local,global_)
        self.sync_to_stream(); after=[x[0].shape[2] for x in self.kv_cache]
        assert all(len(pos)==n for pos,n in zip(self.stream.positions,after))
        if before[0]>self.kv_size:
            assert all(n<=self.visual_start_idx+self.kv_size for n in after)
        print('HERMES_COMPRESSION','before',min(before),max(before),'after',min(after),max(after),flush=True)
