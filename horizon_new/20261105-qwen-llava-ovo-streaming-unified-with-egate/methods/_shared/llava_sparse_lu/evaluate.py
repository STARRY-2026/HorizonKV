import argparse, csv, hashlib, json, os, time, traceback
from pathlib import Path
import torch
from smoke import ROOT, Streaming
from llava.model.builder import load_pretrained_model
from sparsemm_baselines.run_streamingbench import prepare_annotations, analyze_video, load_resume, write_results

class Adapter:
    def __init__(self, model, tokenizer, processor, method):
        self.model=model.eval(); self.tokenizer=tokenizer; self.processor=processor; self.method=method; self.clear_cache()
    def clear_cache(self):
        self.stream=Streaming(self.model,self.method); self.chunks=0
    @torch.inference_mode()
    def encode_init_prompt(self):
        prefix='<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
        ids=self.tokenizer(prefix,return_tensors='pt',add_special_tokens=False).input_ids.to(self.model.device)
        self.stream.forward(self.model.get_input_embeddings()(ids)); self.stream.prefix=ids.shape[1]
    @torch.inference_mode()
    def encode_video_chunk(self,frames,is_last_chunk=False):
        pix=self.processor.preprocess(frames.numpy(),return_tensors='pt')['pixel_values'].to(device=self.model.device,dtype=next(self.model.parameters()).dtype)
        feats=self.model.get_2dPool(self.model.encode_images(pix),stride=2)
        self.stream.metadata=[]
        self.stream.forward(feats.reshape(1,-1,feats.shape[-1]),compress=True)
        self.chunks+=1
        print('CHUNK',self.chunks,'frames',len(frames),'visual_tokens',feats.shape[0]*feats.shape[1], 'compressed_layers',sum(bool(x.get('compressed')) for x in self.stream.metadata),flush=True)
    def predict_and_compress(self): pass
    def get_prompt(self,text,mc=False):
        return '\n'+text+'<|im_end|><|im_start|>assistant\n'+('Best option: (' if mc else '')
    @torch.inference_mode()
    def question_answering(self, input_text,max_new_tokens=16,temperature=0,repetition_penalty=1.1):
        saved=self.stream.cache.snapshot(); cursor=self.stream.cursor
        try:
            ids=self.tokenizer(input_text['prompt'],return_tensors='pt',add_special_tokens=False).input_ids.to(self.model.device)
            hidden=self.stream.forward(self.model.get_input_embeddings()(ids)); tokens=[]
            for step in range(max_new_tokens):
                scores=self.model.lm_head(hidden[:,-1])[0].clone()
                for token in set(tokens): scores[token]=scores[token]*repetition_penalty if scores[token]<0 else scores[token]/repetition_penalty
                token=int(scores.argmax()); tokens.append(token)
                if token==self.tokenizer.eos_token_id or step+1==max_new_tokens: break
                hidden=self.stream.forward(self.model.get_input_embeddings()(torch.tensor([[token]],device=self.model.device)))
            return self.tokenizer.decode(tokens,skip_special_tokens=True)
        finally:
            self.stream.cache.restore(saved); self.stream.cursor=cursor

def main():
    p=argparse.ArgumentParser(); p.add_argument('--method',required=True,choices=['sparsemm','lukv']); p.add_argument('--annotation',required=True); p.add_argument('--output',required=True); args=p.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    raw=Path(args.annotation).read_bytes(); asset=(ROOT+'/code/horizonkv/releases/20261022-ovobench-adapter-v1/kvbench/assets/sparsemm/head_scores.csv' if args.method=='sparsemm' else ROOT+'/code/horizonkv/releases/20261026-lukv-cal50-ovobench-v1/kvbench/sparsemm_lukv/assets/attention_native50v_20261023/profile.json'); config={'method':args.method,'annotation_sha256':hashlib.sha256(raw).hexdigest(),'fps':0.5,'chunk_frames':16,'budget':500,'budget_scope':'visual','window':32,'model':ROOT+'/model/llava-onevision-qwen2-7b-ov','vision_encoder':'siglip-so400m-patch14-384','pool':'bilinear stride 2','dtype':'float16','calibration':'Qwen2.5-VL cross-model transfer; not calibrated on LLaVA','asset':asset,'asset_sha256':hashlib.sha256(Path(asset).read_bytes()).hexdigest(),'version':1}
    manifest=out/'config.json'
    if manifest.exists() and json.loads(manifest.read_text())!=config: raise ValueError('Output config mismatch')
    if not manifest.exists(): manifest.write_text(json.dumps(config,indent=2)+'\n')
    data=prepare_annotations(args.annotation,max_videos=None,max_questions=1)
    rows,done=load_resume(out/'results.csv',data)
    print('RESUME',done,'/',len(data),flush=True)
    if done==len(data): return
    tokenizer,model,processor,_=load_pretrained_model(config['model'],None,'llava_qwen',device_map='cuda:0',attn_implementation='sdpa',multimodal=True,overwrite_config={'mm_vision_tower':'/tmp/siglip-so400m-patch14-384'})
    if any(x.is_meta for x in model.parameters()): raise RuntimeError('unmaterialized parameters')
    adapter=Adapter(model,tokenizer,processor,args.method)
    for item in data[done:]:
        start=time.monotonic()
        try:
            new=analyze_video(adapter,item,0.5,16,torch)
            rows.extend(new); write_results(out/'results.csv',rows)
            print('DONE',len(rows),'/',len(data),'seconds',round(time.monotonic()-start,1),'id',item['video_id'],flush=True)
        except Exception:
            print('FAILED',item['video_id'],flush=True); traceback.print_exc(); raise

if __name__=='__main__': main()
