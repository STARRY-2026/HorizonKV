"""Experimental KV-only batching and visual-query SnapKV-style selection."""
import math
import torch
import torch.nn.functional as F
from transformers import DynamicCache
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb
from pae_variable_recent import compact_positions, TOKENS_PER_FRAME
from sparsemm_native.dscache_position_agnostic import remove_mrope_from_key, apply_mrope_to_key

class RecentBatch:
    def __init__(self, core, prefix):
        self.core=core; self.prefix=prefix; self.n=prefix.shape[1]; self.frames=0
        self.full=DynamicCache(); self.one=None; self.full_q=None; self.one_q=None
        p=compact_positions(list(range(-self.n,0)),self.n,device=core.device)
        core(inputs_embeds=prefix,position_ids=p,past_key_values=self.full,use_cache=True,return_dict=True)
    def _forward(self, embed, positions, cache):
        snapshots=[None]*len(self.core.language_model.layers); hooks=[]
        for i,layer in enumerate(self.core.language_model.layers):
            def capture(module, args, output, i=i): snapshots[i]=(output[:,-32:].detach().clone(),args[0][:,-32:].detach().clone())
            hooks.append(layer.self_attn.q_proj.register_forward_hook(capture))
        try: self.core(inputs_embeds=embed,position_ids=positions,past_key_values=cache,use_cache=True,return_dict=True)
        finally:
            for h in hooks:h.remove()
        if any(x is None for x in snapshots):raise RuntimeError('Missing visual queries')
        return snapshots
    def append_many(self, embeds):
        count=embeds.shape[0] // TOKENS_PER_FRAME
        if count < 1 or embeds.shape[0] != count*TOKENS_PER_FRAME or self.frames+count > 4:
            raise RuntimeError("Invalid recent visual block")
        offset=self.frames*TOKENS_PER_FRAME
        uids=list(range(offset, offset+count*TOKENS_PER_FRAME))
        positions=compact_positions(uids,self.n,device=self.core.device,frame_order=list(range(self.frames+count)))
        self.full_q=self._forward(embeds[None],positions,self.full)
        p=compact_positions(list(range(-self.n,0)),self.n,device=self.core.device)
        self.one=DynamicCache();self.core(inputs_embeds=self.prefix,position_ids=p,past_key_values=self.one,use_cache=True,return_dict=True)
        self.one_q=self._forward(embeds[-TOKENS_PER_FRAME:][None],positions[:,:,-TOKENS_PER_FRAME:],self.one)
        self.frames+=count
    def positions(self):
        return compact_positions(list(range(-self.n,0))+list(range(self.frames*TOKENS_PER_FRAME)),self.n,device=self.core.device)
    def probes(self, positions=None):
        p=(self.positions()[:,:,-32:] if positions is None else positions)
        return [(rotate_query(self.core,i,self.full_q[i],p),rotate_query(self.core,i,self.one_q[i],p)) for i in range(len(self.full_q))]

def rotate_query(core,layer_idx,snapshot,positions):
    qflat,normed=snapshot;att=core.language_model.layers[layer_idx].self_attn
    length=qflat.shape[1]; q=qflat.view(1,length,att.num_heads,att.head_dim).transpose(1,2)
    dummy=q.new_zeros((1,att.num_key_value_heads,length,att.head_dim))
    cos,sin=core.language_model.rotary_emb(normed,positions)
    q,_=apply_multimodal_rotary_pos_emb(q,dummy,cos,sin,att.rope_scaling['mrope_section'])
    return q[0]

def snap_select(key,queries,_unused,visual_start,budget,protect=32):
    n=key.shape[0]; visual=n-visual_start;budget=min(int(budget),visual)
    if visual<=budget:return torch.arange(visual_start,n,device=key.device)
    if budget<=0:return torch.empty(0,dtype=torch.long,device=key.device)
    recent=min(protect,visual,budget); end=n-recent; count=budget-recent
    scores=queries.float()@key.float().T/math.sqrt(key.shape[-1]);qlen=queries.shape[1]
    # The observation queries are the newest query tokens. Mask future keys.
    mask=torch.arange(n,device=key.device)[None,:] > (n-qlen+torch.arange(qlen,device=key.device))[:,None]
    scores.masked_fill_(mask.unsqueeze(0),torch.finfo(scores.dtype).min)
    weights=torch.softmax(scores,dim=-1).mean(dim=(0,1))[visual_start:end]
    if weights.numel():weights=F.avg_pool1d(weights[None,None],kernel_size=5,padding=2,stride=1)[0,0]
    chosen=torch.topk(weights,count).indices.sort().values+visual_start if count else torch.empty(0,dtype=torch.long,device=key.device)
    return torch.cat((chosen,torch.arange(end,n,device=key.device)))

def transfer_batch(core,recent,allocation,previous):
    n=recent.n;start=previous['seen_frames'] if previous else 0;prior=previous['heads'] if previous else {}
    prefix=list(range(-n,0));uids=list(range(start*TOKENS_PER_FRAME,(start+4)*TOKENS_PER_FRAME))
    frames=sorted({x//TOKENS_PER_FRAME for e in prior.values() for x in e['uids']}|set(range(start,start+4)))
    recent_pos=recent.positions();one_pos=torch.cat((recent_pos[:,:,:n],recent_pos[:,:,-TOKENS_PER_FRAME:]),dim=2)
    local=recent.probes(); qpos=compact_positions(uids[-32:],n,device=core.device,frame_order=frames)
    history_q=recent.probes(qpos);heads={}
    for li,layer in enumerate(core.language_model.layers):
        sections=layer.self_attn.rope_scaling['mrope_section'];fr,fv=recent.full[li];orr,ov=recent.one[li]
        raw=remove_mrope_from_key(core,fr,recent_pos,sections).to(fr.dtype);one_raw=remove_mrope_from_key(core,orr,one_pos,sections).to(orr.dtype)
        for h in range(4):
            cfg=allocation[li,h]; old=prior.get((li,h)); old_count=len(old['uids']) if old else 0
            recent_budget=int(cfg['recent_quota'])+int(cfg['history_quota'])-old_count;single=cfg['recent_group']=='one_frame'
            rot,k,v,ids,q=(orr,one_raw,ov,uids[-TOKENS_PER_FRAME:],local[li][1]) if single else (fr,raw,fv,uids,local[li][0])
            group=q.shape[0]//4;idx=snap_select(rot[0,h],q[h*group:(h+1)*group],None,n,recent_budget)
            keys=torch.cat(([old['keys']] if old else [])+[k[0,h].index_select(0,idx)],dim=0)
            vals=torch.cat(([old['values']] if old else [])+[v[0,h].index_select(0,idx)],dim=0)
            visual_ids=(old['uids'] if old else [])+[ids[i-n] for i in idx.tolist()]
            candidate=torch.cat((k[0,h,:n],keys),dim=0);pos=compact_positions(prefix+visual_ids,n,device=core.device,frame_order=frames)
            rotated=apply_mrope_to_key(core,candidate[None,None],pos,sections).to(candidate.dtype)[0,0]
            hq=history_q[li][1 if single else 0]
            keep=snap_select(rotated,hq[h*group:(h+1)*group],None,n,int(cfg['history_quota']),protect=0)-n
            newids=[visual_ids[i] for i in keep.tolist()]
            if len(newids)>int(cfg['history_quota']) or len(set(newids))!=len(newids):raise RuntimeError('History invariant failed')
            heads[li,h]={'keys':keys.index_select(0,keep).contiguous(),'values':vals.index_select(0,keep).contiguous(),'uids':newids}
    return {'heads':heads,'seen_frames':start+4}
