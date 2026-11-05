#!/usr/bin/env python3
"""Run one StreamingBench shard with all questions in chronological order."""
import argparse, importlib.util, os, re, runpy, sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from methods.registry import (PROJECT_ROOT, METHODS_ROOT, SHARED_KVBENCH, LLAVA_BACKBONE, SPARSEMM_SCORES, LUKV_PROFILE, llava_impl, qwen_impl)
from run import ROOT, REL, PROFILE, LLAVA
p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--method',required=True); p.add_argument('--annotation',required=True); p.add_argument('--output',required=True); p.add_argument('--gpu',required=True); p.add_argument('--chunk-frames',type=int,default=16); p.add_argument('--visual-input',choices=['native','fixed896-image','dscache448-video'],default='native'); p.add_argument('--sample-fps',type=float,default=0.5); a=p.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=a.gpu; os.environ['SPARSEMM_GPU_INDEX']=a.gpu
if a.model=='llavaov':
    folder=llava_impl(a.method); sys.path.insert(0,str(folder))
    spec=importlib.util.spec_from_file_location('streaming_llava_eval',folder/'evaluate.py'); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    prepare=mod.prepare_annotations
    def all_questions(path,max_videos=None,max_questions=None,**kw): return prepare(path,max_videos=max_videos,max_questions=None,**kw)
    mod.prepare_annotations=all_questions
    if a.method=='hermes':
        original=mod.Adapter.question_answering
        def answer(self,input_text,*args,**kwargs):
            result=original(self,input_text,*args,**kwargs)
            # Match HERMES question history for later question-prediction probes.
            text=input_text.get('formatted_question','')
            options=re.findall(r'\([A-Z]\)\s*(.+?)(?=\n\([A-Z]\)|\nThe best answer|\n*$)',text,re.DOTALL)
            self.hermes.conv_history.append((input_text['question'],result,[x.strip() for x in options] or None))
            return result
        mod.Adapter.question_answering=answer
    sys.argv=[str(folder/'evaluate.py'),'--method',a.method,'--annotation',a.annotation,'--output',a.output]
    mod.main(); sys.exit(0)
if a.method in ('adakv','pyramidkv') and a.visual_input!='dscache448-video':
    from sparsemm_baselines.adapter import SparseMMQwen
    SparseMMQwen._trace=lambda self,entry: None
    sys.argv=['sparsemm_baselines.run_streamingbench','--method',a.method,'--anno-path',a.annotation,'--model-path',str(ROOT/'model/Qwen2.5-VL-7B-Instruct'),'--output-dir',a.output,'--max-videos',str(len(__import__('json').loads(Path(a.annotation).read_text()))),'--budget','500','--budget-scope','visual','--dtype','float16','--sample-fps','0.5','--encode-chunk-size',str(a.chunk_frames),'--resume-experiments']
    runpy.run_module('sparsemm_baselines.run_streamingbench',run_name='__main__',alter_sys=True); sys.exit(0)
from sparsemm_baselines.adapter import SparseMMQwen
SparseMMQwen._trace=lambda self,entry: None
from sparsemm_baselines.run_streamingbench import analyze_video, gpu_lease, load_resume, prepare_annotations, write_results
out=Path(a.output); data=prepare_annotations(a.annotation,max_videos=None,max_questions=None); result=out/'results.csv'
rows,done=load_resume(result,data)
print('RESUME',done,'/',len(data),flush=True)
if done==len(data): sys.exit(0)
if a.visual_input=='fixed896-image':
    from fixed896_adapter import install
    install(a.method)
elif a.visual_input=='dscache448-video' and a.method in ('adakv','pyramidkv','sparsemm','lukv'):
    from dscache448_adapter import install
    install(a.method)
with gpu_lease(out):
    import torch
    if a.method in ('sparsemm','lukv','adakv','pyramidkv'):
        from transformers import Qwen2_5_VLForConditionalGeneration,Qwen2_5_VLProcessor
        from sparsemm_protokv.config import StreamingConfig
        modelpath=ROOT/'model/Qwen2.5-VL-7B-Instruct'
        processor=Qwen2_5_VLProcessor.from_pretrained(modelpath,local_files_only=True,**({'min_pixels':448*448,'max_pixels':448*448} if a.visual_input=='dscache448-video' else {}))
        base=Qwen2_5_VLForConditionalGeneration.from_pretrained(modelpath,torch_dtype=torch.float16,attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
        if a.method in ('adakv','pyramidkv'):
            from sparsemm_baselines.core import BaselineConfig
            config=BaselineConfig(method=a.method,budget=500,budget_scope='visual',window_size=32,kernel_size=5,pooling='avgpool',gqa_reduce='mean',score_precision='float32')
            bridge=SparseMMQwen(base,processor,config,a.sample_fps,None)
        elif a.method=='lukv':
            from sparsemm_lukv.streaming import LUStreamingQwen
            config=StreamingConfig(method='lukv',budget=500,budget_scope='visual',window_size=32,kernel_size=5,pooling='avgpool',gqa_reduce='mean',score_precision='float32',lukv_profile_path=str(PROFILE))
            bridge=LUStreamingQwen(base,processor,config,a.sample_fps,None,True)
        else:
            from sparsemm_native.streaming import SparseMMStreamingQwen
            score=SPARSEMM_SCORES
            config=StreamingConfig(method='sparsemm',budget=500,budget_scope='visual',window_size=32,kernel_size=5,pooling='avgpool',score_precision='float32',sparsemm_score_path=str(score),sparsemm_ratio=.1)
            import hashlib
            digest=hashlib.sha256(score.read_bytes()).hexdigest()
            if digest!='9cc0f47f24a0af4306ce7367806516d119dc4f57df77d064a3eac59c05b69dab': raise ValueError('SparseMM score fingerprint changed')
            bridge=SparseMMStreamingQwen(base,processor,config,a.sample_fps,None,True,asset_metadata={'path':str(score),'sha256':digest,'shape':[28,4]})
    else:
        os.environ['HORIZONKV_ATTENTION_BACKEND']='flash_attention_2'
        frozen=qwen_impl(a.method)
        sys.path.insert(0,str(frozen))
        from inference.qwenvl_hermes import load_model
        bridge,_=load_model(model_path=str(ROOT/'model/Qwen2.5-VL-7B-Instruct'),kv_size=500,streaming=True,device='cuda',sample_fps=a.sample_fps,compress_mode='full' if a.method=='fullkv' else 'hermes',fixed_base_resolution=(a.visual_input=='dscache448-video'))
        if a.visual_input=='dscache448-video':
            from qwen_dense_dscache_adapter import install_dense
            install_dense(bridge,a.method)
    with torch.inference_mode():
        for i,video in enumerate(data[done:],start=done):
            rows.extend(analyze_video(bridge,video,a.sample_fps,a.chunk_frames,torch)); write_results(result,rows)
            print('COMPLETED',i+1,'/',len(data),flush=True)
load_resume(result,data,require_complete=True)
print('COMPLETE',len(rows),'questions',flush=True)
