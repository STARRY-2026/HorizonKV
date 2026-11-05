#!/usr/bin/env python3
"""Unified OVO-Bench / StreamingBench launcher using pinned model adapters."""
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from methods.registry import (PROJECT_ROOT, METHODS_ROOT, SHARED_KVBENCH, LLAVA_BACKBONE, SPARSEMM_SCORES, LUKV_PROFILE, llava_impl, qwen_impl)
ROOT=Path('/mnt/aifordata2026-p/common/aifordata/541707')
REL=ROOT/'code/horizonkv/releases'
OLD=Path(__file__).resolve().parent/'ovo_backend'
SOURCE=REL/'20261022-ovobench-adapter-v1/data/streamingbench/streamingbench_realtime_local.json'
PROFILE=LUKV_PROFILE
VIDEO=ROOT/'dataset/StreamingBench/videos'
METHODS=('adakv','pyramidkv','sparsemm','lukv','hermes','fullkv','uniform')
LLAVA={name:str(llava_impl(name)) for name in ('adakv','pyramidkv','sparsemm','lukv','hermes','fullkv')}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_exact(p,s):
    if p.exists() and p.read_text()!=s: raise RuntimeError('Resume configuration changed: '+str(p))
    if not p.exists(): p.write_text(s)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark',required=True,choices=['ovobench','streamingbench'])
    p.add_argument('--model',required=True,choices=['qwen25vl','llavaov'])
    p.add_argument('--method',required=True,choices=METHODS)
    p.add_argument('--gpus',default='0'); p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--annotation',type=Path); p.add_argument('--max-videos',type=int)
    p.add_argument('--cohort',choices=['heldout','full'],default='heldout')
    p.add_argument('--allow-calibration-overlap',action='store_true')
    p.add_argument('--video-root',type=Path,default=VIDEO); p.add_argument('--dry-run',action='store_true')
    p.add_argument('--visual-input',choices=['native','fixed896-image','dscache448-video'],default='native')
    p.add_argument('--sample-fps',type=float,default=0.5)
    p.add_argument('--chunk-frames',type=int,default=16,help='StreamingBench Qwen outer frame batch; internal update granularity remains method-specific')
    a=p.parse_args(); gs=a.gpus.split(',')
    if a.sample_fps <= 0: p.error('--sample-fps must be positive')
    if a.sample_fps != 0.5 and (a.benchmark not in ('streamingbench','ovobench') or a.model != 'qwen25vl' or a.method not in ('lukv','sparsemm','adakv','pyramidkv','hermes','fullkv')): p.error('Custom fps is supported only for StreamingBench Qwen LU-KV/SparseMM')
    if a.visual_input=='dscache448-video' and (a.chunk_frames != 2 or a.sample_fps != 1.0): p.error('dscache448-video requires --chunk-frames 2 --sample-fps 1')
    if a.visual_input != 'native' and (a.benchmark not in ('streamingbench','ovobench') or a.model != 'qwen25vl' or a.method not in ('lukv','sparsemm','adakv','pyramidkv','hermes','fullkv')): p.error('fixed896-image currently supports StreamingBench Qwen LU-KV and SparseMM only')
    if a.chunk_frames < 1: p.error('--chunk-frames must be positive')
    if a.chunk_frames != 16 and (a.benchmark not in ('streamingbench','ovobench') or a.model != 'qwen25vl'): p.error('--chunk-frames is currently supported only for StreamingBench Qwen')
    if any(not x.isdigit() for x in gs) or len(gs)!=len(set(gs)): p.error('gpus must be unique IDs, e.g. 0,1')
    if a.max_videos is not None and a.max_videos<1: p.error('max-videos must be positive')
    out=a.output_dir.resolve()
    if a.method=='uniform' and a.model!='llavaov': p.error('Uniform SnapKV adapter currently supports llavaov only')
    if a.benchmark=='ovobench' and a.model=='qwen25vl' and a.visual_input=='dscache448-video':
        cmd=[sys.executable,str(Path(__file__).with_name('ovo_qwen_dscache_run.py')),'--method',a.method,'--gpus',a.gpus,'--output-dir',str(out)]
        if a.annotation:cmd+=['--annotation',str(a.annotation.resolve())]
        if a.max_videos is not None:cmd+=['--max-videos',str(a.max_videos)]
        if a.dry_run:cmd+=['--dry-run']
        print('Qwen OVO DSCache-input delegate:',json.dumps(cmd),flush=True)
        return subprocess.call(cmd)
    if a.benchmark=='ovobench':
        cmd=[sys.executable,str(OLD/'run.py'),'--model',a.model,'--method',a.method,'--gpus',a.gpus,'--output-dir',str(out)]
        if a.annotation: cmd+=['--annotation',str(a.annotation.resolve())]
        if a.max_videos: cmd+=['--max-videos',str(a.max_videos)]
        if a.dry_run: cmd+=['--dry-run']
        print('OVO delegate:',json.dumps(cmd),flush=True); return subprocess.call(cmd)
    anno=(a.annotation or SOURCE).resolve(); raw=json.loads(anno.read_text()); profile=json.loads(PROFILE.read_text())
    cal=profile['calibration']['selected_videos']
    def name(x): return Path(x if isinstance(x,str) else x.get('video_path',x.get('path',''))).name
    calibration={name(x) for x in cal}
    if len(calibration)!=50 or '' in calibration: raise RuntimeError('Unexpected calibration media list')
    if len({str(x['video_id']) for x in raw})!=len(raw): raise ValueError('Duplicate video IDs')
    overlap=[name(x['video_path']) for x in raw if name(x['video_path']) in calibration]
    if a.cohort=='heldout': data=[x for x in raw if name(x['video_path']) not in calibration]
    else:
        data=raw
        if a.method=='lukv' and overlap and not a.allow_calibration_overlap: raise ValueError('LU-KV full cohort overlaps calibration; use heldout or explicitly pass --allow-calibration-overlap')
    # Keep duplicate correct options, scoring by selected option text.
    excluded=[]; cleaned=[]
    for v in data:
        item=dict(v); questions=[]
        for original in v['conversations']:
            q=dict(original); choices=q['choices']; answer=q['answer']
            if answer not in choices: raise ValueError('Answer missing from choices')
            if len(set(choices))!=len(choices) and not q.get('correct_choice'):
                q['correct_choice']='ABCDEFGH'[choices.index(answer)]
            questions.append(q)
        if questions:
            item['conversations']=questions; cleaned.append(item)
    data=cleaned
    if a.max_videos is not None:
        if a.max_videos>len(data): raise ValueError('max-videos exceeds cohort')
        data=data[:a.max_videos]
    if not data: raise ValueError('No videos selected')
    for video in data:
        orig=Path(video['video_path']); new=a.video_root/orig.name
        if not orig.is_file() and not new.is_file(): raise FileNotFoundError(f'Media missing: {orig} or {new}')
        video['video_path']=str(orig if orig.is_file() else new)
        if not video.get('conversations'): raise ValueError('Video without questions')
    gs=gs[:min(len(gs),len(data))]
    config=dict(version=1,benchmark=a.benchmark,model=a.model,method=a.method,source=str(anno),source_sha256=sha(anno),cohort=a.cohort,excluded_ambiguous_questions=excluded,profile_sha256=sha(PROFILE),calibration_overlap_selected=sum(name(x['video_path']) in calibration for x in data),videos=len(data),questions=sum(len(x['conversations']) for x in data),gpus=gs,fps=a.sample_fps,visual_input=a.visual_input,chunk_frames=a.chunk_frames,visual_budget=None if a.method=='fullkv' else 500,dtype='float16',detailed_cache_trace=False,calibration='Qwen2.5-VL cross-model transfer' if a.model=='llavaov' and a.method in ('sparsemm','lukv') else None,code_sha256={x:sha(Path(__file__).parent/x) for x in ('run.py','worker.py','fixed896_adapter.py','dscache448_adapter.py','qwen_dense_dscache_adapter.py','qwen_vl_utils/vision_process.py')})
    if a.visual_input=='dscache448-video':
        config.update(latest_frame_pixels=896*896, latest_frame_replay=True,
                      input_parity_note='Experimental: paired latest chunk is replayed as separate ordinary and high-frame units; not numerically equivalent to DSCache temporal pairing.')
    print(json.dumps(config,indent=2,ensure_ascii=False),flush=True)
    if a.dry_run: return 0
    if out.exists() and not (out/'manifest.json').exists() and any(out.iterdir()): raise RuntimeError('Nonempty output directory without manifest')
    out.mkdir(parents=True,exist_ok=True); (out/'annotations').mkdir(exist_ok=True)
    write_exact(out/'manifest.json',json.dumps(config,indent=2,ensure_ascii=False)+'\n')
    env=os.environ.copy(); env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',DECORD_EOF_RETRY_MAX='40960')
    paths=[Path('/tmp/llava-ovo-transformers440')] if a.model=='llavaov' else []
    if a.visual_input in ('fixed896-image','dscache448-video'): paths.append(ROOT/'code/horizonkv/runtime/qwen-py310-deps-20261016')
    paths += [LLAVA_BACKBONE,SHARED_KVBENCH,METHODS_ROOT/'lukv',METHODS_ROOT/'sparsemm',qwen_impl('hermes'),ROOT/'code/horizonkv/runtime/ppu-py310-overlay-20261015']
    paths.insert(0, METHODS_ROOT/'_shared/equivalent_choices')
    env['PYTHONPATH']=os.pathsep.join(map(str,[PROJECT_ROOT]+paths)); procs=[]
    for i,gpu in enumerate(gs):
        shard=data[i*len(data)//len(gs):(i+1)*len(data)//len(gs)]
        annot=out/'annotations'/f'shard_{i:02}.json'; write_exact(annot,json.dumps(shard,ensure_ascii=False))
        dest=out/f'shard_{i:02}'; dest.mkdir(exist_ok=True)
        cmd=[sys.executable,str(Path(__file__).with_name('worker.py')),'--model',a.model,'--method',a.method,'--annotation',str(annot),'--output',str(dest),'--gpu',gpu,'--chunk-frames',str(a.chunk_frames),'--sample-fps',str(a.sample_fps),'--visual-input',a.visual_input]
        childenv=env.copy(); childenv.update(CUDA_VISIBLE_DEVICES=gpu,SPARSEMM_GPU_INDEX=gpu)
        with (dest/'run.log').open('a') as log: proc=subprocess.Popen(cmd,env=childenv,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        procs.append(proc); print('START',i,'GPU',gpu,'PID',proc.pid,'VIDEOS',len(shard),flush=True)
    try:
        codes=[proc.wait() for proc in procs]
    except BaseException:
        for proc in procs:
            if proc.poll() is None: proc.terminate()
        for proc in procs: proc.wait()
        raise
    print('EXIT CODES',codes,flush=True); return int(any(codes))
if __name__=='__main__': sys.exit(main())
