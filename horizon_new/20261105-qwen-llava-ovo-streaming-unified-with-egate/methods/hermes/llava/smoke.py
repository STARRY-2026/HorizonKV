import argparse, json, os, time
import torch
from decord import VideoReader, cpu
import llava.model.llava_arch as arch
from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower
arch.build_vision_tower = lambda cfg, **kw: SigLipVisionTower(cfg.mm_vision_tower, vision_tower_cfg=cfg, **kw)
from llava.model.builder import load_pretrained_model
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from sparsemm_baselines.cache import RaggedCache
from sparsemm_baselines.core import BaselineConfig, select_layer
try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None

ROOT = '/mnt/aifordata2026-p/common/aifordata/541707'

def attention(q, k, v, cache):
    _, nq, n, d = q.shape
    nk = k.shape[1]; groups = nq // nk
    lengths = cache.head_lens.tolist()
    ks, vs = [], []
    for h in range(nk):
        oldk, oldv = cache.get_segment(h)
        ks.append(torch.cat((oldk, k[0, h])))
        vs.append(torch.cat((oldv, v[0, h])))
    if flash_attn_varlen_func is not None and q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
        qq = q[0].reshape(nk, groups, n, d).permute(0, 2, 1, 3).reshape(-1, groups, d).contiguous()
        kk = torch.cat(ks)[:, None].contiguous(); vv = torch.cat(vs)[:, None].contiguous()
        lens = torch.tensor([x+n for x in lengths], dtype=torch.int32, device=q.device)
        cu_k = torch.cat((lens.new_zeros(1), lens.cumsum(0, dtype=torch.int32)))
        cu_q = torch.arange(nk+1, device=q.device, dtype=torch.int32)*n
        out = flash_attn_varlen_func(qq, kk, vv, cu_q, cu_k, n, int(lens.max()), softmax_scale=d**-0.5, causal=True)
        return out.reshape(nk,n,groups,d).permute(0,2,1,3).reshape(nq,n,d).transpose(0,1).reshape(1,n,-1)
    outs = []
    for h in range(nk):
        logits = q[0,h*groups:(h+1)*groups].float() @ ks[h].float().T * d**-0.5
        mask = torch.arange(ks[h].shape[0],device=q.device)[None,:] > (lengths[h]+torch.arange(n,device=q.device))[:,None]
        logits.masked_fill_(mask, -torch.inf)
        outs.append(torch.softmax(logits,dim=-1).to(q.dtype) @ vs[h])
    return torch.cat(outs).transpose(0,1).reshape(1,n,-1)

class Streaming:
    def __init__(self, model, method):
        self.model=model; self.lm=model.get_model(); self.cfg=BaselineConfig(method=method,budget=500,budget_scope='visual')
        c=model.config; self.nq=c.num_attention_heads; self.nk=c.num_key_value_heads; self.d=c.hidden_size//self.nq
        self.cache=RaggedCache(len(self.lm.layers),self.nk,self.d,model.device,next(model.parameters()).dtype)
        self.cursor=0; self.prefix=0; self.seen=0; self.metadata=[]
        self.positions=[torch.empty(0,dtype=torch.long,device=model.device) for _ in self.lm.layers]

    @torch.inference_mode()
    def forward(self, x, compress=False):
        n=x.shape[1]
        hidden=x
        for i, layer in enumerate(self.lm.layers):
            start=int(self.positions[i][-1])+1 if self.positions[i].numel() else 0
            pos=torch.arange(start,start+n,device=x.device).view(1,-1)
            residual=hidden; z=layer.input_layernorm(hidden); att=layer.self_attn
            q=att.q_proj(z).view(1,n,self.nq,self.d).transpose(1,2)
            k=att.k_proj(z).view(1,n,self.nk,self.d).transpose(1,2)
            v=att.v_proj(z).view(1,n,self.nk,self.d).transpose(1,2)
            cos,sin=att.rotary_emb(v,seq_len=start+n)
            q,k=apply_rotary_pos_emb(q,k,cos,sin,pos)
            cache=self.cache.layers[i]
            out=attention(q,k,v,cache)
            cache.append_chunk(k,v,pos.expand(3,-1))
            self.positions[i]=torch.cat((self.positions[i],pos[0]))
            if compress:
                sel,meta=select_layer([cache.get_segment(h)[0] for h in range(self.nk)],q[0],self.cfg,layer_idx=i,num_layers=len(self.lm.layers),prefix_tokens=self.prefix,seen_tokens=self.prefix+self.seen+n)
                cache.prune(sel); self.metadata.append(meta)
            hidden=residual+att.o_proj(out.contiguous())
            hidden=hidden+layer.mlp(layer.post_attention_layernorm(hidden))
        self.cursor+=n
        if compress: self.seen+=n
        return self.lm.norm(hidden)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--method',choices=['adakv','pyramidkv'],required=True); p.add_argument('--sample',type=int,default=2); p.add_argument('--out',required=True); args=p.parse_args()
    started=time.time()
    ckpt=ROOT+'/model/llava-onevision-qwen2-7b-ov'; sig='/tmp/siglip-so400m-patch14-384'
    tok,model,processor,_=load_pretrained_model(ckpt,None,'llava_qwen',device_map='cuda:0',attn_implementation='sdpa',multimodal=True,overwrite_config={'mm_vision_tower':sig})
    model.eval()
    if any(p.is_meta for p in model.parameters()): raise RuntimeError('unmaterialized model parameters')
    print('MODEL_READY',round(time.time()-started,1),'flash',bool(flash_attn_varlen_func),flush=True)
    with torch.inference_mode():
        probe=tok('A short decoder parity check.',return_tensors='pt',add_special_tokens=False).input_ids.to(model.device)
        expected=model.get_model()(input_ids=probe,use_cache=False,return_dict=True).last_hidden_state
        checker=Streaming(model,args.method)
        actual=checker.forward(model.get_input_embeddings()(probe))
        parity=(actual.float()-expected.float()).abs()
        print('DECODER_PARITY_MAX',float(parity.max()),'MEAN',float(parity.mean()),flush=True)
        if not torch.allclose(actual.float(),expected.float(),atol=0.1,rtol=0.05):
            raise RuntimeError('manual decoder parity check failed')
    ann=ROOT+'/code/horizonkv/releases/20261022-ovobench-adapter-v1/kvbench/configs/ovobench_smoke_warehouse.json'
    item=json.load(open(ann))[args.sample]; question=item['conversations'][0]
    reader=VideoReader(item['video_path'],ctx=cpu(0),num_threads=1); fps=reader.get_avg_fps()
    end=min(float(question['end_time']),len(reader)/fps)
    indices=[min(int(t*fps),len(reader)-1) for t in range(0,max(1,int(end)),2)]
    frames=reader.get_batch(indices).asnumpy()
    print('VIDEO',item['video_id'],'frames',len(frames),'fps',fps,'end',end,flush=True)
    with torch.inference_mode():
        pix=processor.preprocess(frames,return_tensors='pt')['pixel_values'].to(device=model.device,dtype=next(model.parameters()).dtype)
        features=model.encode_images(pix)
        features=model.get_2dPool(features)
        print('FEATURES',list(features.shape),flush=True)
        s=Streaming(model,args.method)
        prefix='<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
        ids=tok(prefix,return_tensors='pt',add_special_tokens=False).input_ids.to(model.device)
        s.forward(model.get_input_embeddings()(ids)); s.prefix=ids.shape[1]
        s.forward(features.reshape(1,-1,features.shape[-1]),compress=True)
        print('COMPRESSED',sum(bool(m.get('compressed')) for m in s.metadata),'headlens',s.cache.head_lengths(),flush=True)
        choices='\n'.join(f'{chr(65+i)}. {x}' for i,x in enumerate(question['choices']))
        prompt='\n'+question['question']+'\n'+choices+'\nAnswer with the option letter.<|im_end|>\n<|im_start|>assistant\n'
        ids=tok(prompt,return_tensors='pt',add_special_tokens=False).input_ids.to(model.device)
        out=s.forward(model.get_input_embeddings()(ids)); generated=[]
        for j in range(16):
            token=int(model.lm_head(out[:,-1]).argmax(-1).item()); generated.append(token)
            if token==tok.eos_token_id: break
            out=s.forward(model.get_input_embeddings()(torch.tensor([[token]],device=model.device)))
        answer=tok.decode(generated,skip_special_tokens=True)
    result={'method':args.method,'video_id':item['video_id'],'question':question['question'],'answer':answer,'correct_choice':question['correct_choice'],'frames':len(frames),'fps':0.5,'chunk_frames':16,'feature_shape':list(features.shape),'compressed_layers':sum(bool(m.get('compressed')) for m in s.metadata),'visual_budget_per_head_average':500,'elapsed_seconds':round(time.time()-started,2),'flash_attention_available':bool(flash_attn_varlen_func),'metadata':s.metadata}
    os.makedirs(os.path.dirname(args.out),exist_ok=True)
    with open(args.out,'w') as f: json.dump(result,f,indent=2,default=str)
    print('RESULT',json.dumps({k:v for k,v in result.items() if k!='metadata'}),flush=True)

if __name__=='__main__': main()
