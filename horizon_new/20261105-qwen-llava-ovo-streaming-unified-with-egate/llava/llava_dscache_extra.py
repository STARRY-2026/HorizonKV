"""DSCache-inspired LLaVA inputs, applied without replacing the baseline policy.

Temporary latest-frame update is replayed from the saved pre-frame cache.
The baseline cache is restored after answering.
"""
import copy
import contextlib
import io
import os
import torch

def _capture(adapter, method):
    s=adapter.stream
    state={'cache':s.cache.snapshot(), 'cursor':s.cursor, 'seen':s.seen,
           'metadata':list(s.metadata)}
    if hasattr(s,'positions'):
        state['positions']=[x.clone() for x in s.positions]
    if method=='hermes':
        h=adapter.hermes
        for name in ('_layer_position_ids','token_activity_cache','_position_ids_cache','kv_cache'):
            state['hermes_'+name]=copy.deepcopy(getattr(h,name))
    return state

def _restore(adapter, method, state):
    s=adapter.stream
    s.cache.restore(state['cache']);s.cursor=state['cursor'];s.seen=state['seen']
    s.metadata=list(state['metadata'])
    if 'positions' in state:s.positions=[x.clone() for x in state['positions']]
    if method=='hermes':
        for name in ('_layer_position_ids','token_activity_cache','_position_ids_cache','kv_cache'):
            setattr(adapter.hermes,name,copy.deepcopy(state['hermes_'+name]))

def install(Adapter, method):
    original_clear=Adapter.clear_cache
    original_answer=Adapter.question_answering
    original_compress=Adapter.predict_and_compress
    def clear(self):
        original_clear(self)
        self.latest_pre_state=None
        self.latest_high_features=None
    @torch.inference_mode()
    def encode(self,frames,is_last_chunk=False):
        if len(frames)!=1:raise RuntimeError('DSCache input requires per-frame updates')
        pix=self.processor.preprocess(frames.numpy(),return_tensors='pt')['pixel_values'].to(device=self.model.device,dtype=next(self.model.parameters()).dtype)
        if tuple(pix.shape[-2:])!=(384,384):raise RuntimeError('Expected LLaVA 384x384 image')
        features=self.model.encode_images(pix)
        low=self.model.get_2dPool(features,stride=2)
        high=self.model.get_2dPool(features,stride=1)
        if low.shape[1]!=196 or high.shape[1]!=729:
            raise RuntimeError(f'Unexpected vision tokens: low={low.shape[1]} high={high.shape[1]}')
        self.latest_pre_state=_capture(self,method)
        self.latest_high_features=high.reshape(1,-1,high.shape[-1])
        s=self.stream
        if method in ('sparsemm','lukv'):
            s.metadata=[];s.forward(low.reshape(1,-1,low.shape[-1]),compress=True)
        else:
            s.forward(low.reshape(1,-1,low.shape[-1]),compress=False)
            s.seen+=low.shape[0]*low.shape[1]
            if method=='fullkv':
                expected=s.prefix+s.seen
                if any(any(n!=expected for n in layer) for layer in s.cache.head_lengths()):
                    raise RuntimeError('Full KV lost visual tokens')
        self.chunks+=1
        if self.chunks==1:print('DSCACHE_INPUT ordinary_tokens=196 latest_tokens=729',flush=True)
    @torch.inference_mode()
    def answer(self,*args,**kwargs):
        if self.latest_pre_state is None:raise RuntimeError('Missing latest frame state')
        current=_capture(self,method)
        try:
            _restore(self,method,self.latest_pre_state)
            high=self.latest_high_features
            if method in ('sparsemm','lukv'):
                self.stream.metadata=[]
                self.stream.forward(high,compress=True)
            else:
                self.stream.forward(high,compress=False);self.stream.seen+=high.shape[1]
                if method=='hermes':
                    with contextlib.redirect_stdout(io.StringIO()):self.hermes.compress()
                if method=='fullkv':
                    expected=self.stream.prefix+self.stream.seen
                    if any(any(n!=expected for n in layer) for layer in self.stream.cache.head_lengths()):
                        raise RuntimeError('Full KV lost high-resolution tokens')
            return original_answer(self,*args,**kwargs)
        finally:
            _restore(self,method,current)
            if os.environ.get('HORIZON_DSCACHE_VERIFY_STATE')=='1':
                restored=self.stream.cache.snapshot()
                if any(any(a is not b for a,b in zip(x,y)) for x,y in zip(restored,current['cache'])):
                    raise RuntimeError('Cache tensors differ after answer restoration')
                if self.stream.cursor!=current['cursor'] or self.stream.seen!=current['seen']:
                    raise RuntimeError('Cache counters differ after answer restoration')
                if 'positions' in current and any(not torch.equal(a,b) for a,b in zip(self.stream.positions,current['positions'])):
                    raise RuntimeError('HERMES position state differs after answer restoration')
                print('DSCACHE_STATE_RESTORED',method,flush=True)
    if method=='hermes':
        def quiet_compress(self):
            with contextlib.redirect_stdout(io.StringIO()):return original_compress(self)
        Adapter.predict_and_compress=quiet_compress
    Adapter.clear_cache=clear
    Adapter.encode_video_chunk=encode
    Adapter.question_answering=answer
