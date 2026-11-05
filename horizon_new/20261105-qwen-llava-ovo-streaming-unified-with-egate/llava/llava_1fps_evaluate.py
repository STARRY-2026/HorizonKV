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
        if tuple(pix.shape[-2:]) != (384,384): raise RuntimeError('DSCache reference expects 384x384 LLaVA input')
        encoded=self.model.encode_images(pix)
        if os.environ.get('HORIZON_INPUT_PROFILE')=='dscache':
            if len(frames)!=1: raise RuntimeError('dscache profile requires one frame per update')
            if self.chunks == 0: print('INPUT_PROFILE ordinary_tokens=196 latest_replay=false',flush=True)
        feats=self.model.get_2dPool(encoded,stride=2)
        if feats.shape[1] != 196: raise RuntimeError('Unexpected ordinary-frame LLaVA token count')
        self.stream.metadata=[]
        self.stream.forward(feats.reshape(1,-1,feats.shape[-1]),compress=True)
        self.chunks+=1

    def predict_and_compress(self): pass
    def get_prompt(self,text,mc=False):
        return '\n'+text+'<|im_end|><|im_start|>assistant\n'+('Best option: (' if mc else '')
    @torch.inference_mode()
    def question_answering(self, input_text,max_new_tokens=16,temperature=0,repetition_penalty=1.1):
        saved=self.stream.cache.snapshot(); cursor=self.stream.cursor; seen=self.stream.seen
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
            self.stream.cache.restore(saved); self.stream.cursor=cursor; self.stream.seen=seen

def main():
    p=argparse.ArgumentParser(); p.add_argument('--method',required=True,choices=['adakv','pyramidkv']); p.add_argument('--annotation',required=True); p.add_argument('--output',required=True); args=p.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    raw=Path(args.annotation).read_bytes(); config={'method':args.method,'annotation_sha256':hashlib.sha256(raw).hexdigest(),'fps':1.0,'chunk_frames':1,'budget':500,'budget_scope':'visual','window':32,'model':ROOT+'/model/llava-onevision-qwen2-7b-ov','vision_encoder':'siglip-so400m-patch14-384','pool':'ordinary stride 2; no answer-time frame replay','input_profile':os.environ.get('HORIZON_INPUT_PROFILE','original'),'dtype':'float16','version':2}
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
            new=analyze_video(adapter,item,1.0,1,torch)
            rows.extend(new); write_results(out/'results.csv',rows)
            print('DONE',len(rows),'/',len(data),'seconds',round(time.monotonic()-start,1),'id',item['video_id'],flush=True)
        except Exception:
            print('FAILED',item['video_id'],flush=True); traceback.print_exc(); raise

if __name__=='__main__': main()
