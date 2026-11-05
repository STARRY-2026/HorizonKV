#!/usr/bin/env python3
"""Experimental OVO-Bench Qwen baseline runner with DSCache-style input."""
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.append(str(Path(__file__).resolve().parents[1]))
from methods.registry import (PROJECT_ROOT, METHODS_ROOT, SHARED_KVBENCH, LLAVA_BACKBONE, SPARSEMM_SCORES, LUKV_PROFILE, llava_impl, qwen_impl)
ROOT=Path('/mnt/aifordata2026-p/common/aifordata/541707')
REL=ROOT/'code/horizonkv/releases'
ANNO=REL/'20261022-ovobench-adapter-v1/data/ovobench/ovobench_realtime_backward_warehouse.json'
METHODS=('adakv','pyramidkv','sparsemm','lukv','hermes','fullkv')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save_exact(p,s):
    if p.exists() and p.read_text()!=s: raise RuntimeError('Resume configuration differs: '+str(p))
    if not p.exists():p.write_text(s)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',choices=METHODS,required=True)
    parser.add_argument('--gpus',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--annotation',type=Path,default=ANNO)
    parser.add_argument('--max-videos',type=int)
    parser.add_argument('--dry-run',action='store_true')
    a=parser.parse_args()
    gpus=a.gpus.split(',')
    if not gpus or any(not x.isdigit() for x in gpus) or len(set(gpus))!=len(gpus):
        parser.error('gpus must be distinct numeric IDs')
    anno=a.annotation.resolve()
    data=json.loads(anno.read_text())
    if not isinstance(data,list) or not data:parser.error('annotation must be a nonempty list')
    n=len(data) if a.max_videos is None else a.max_videos
    if n<1 or n>len(data):parser.error('invalid max-videos')
    data=data[:n]
    if len({str(v['video_id']) for v in data})!=n:parser.error('duplicate video IDs')
    if any(len(v.get('conversations',[]))!=1 for v in data):
        parser.error('OVO runner requires one question per annotation record')
    if any(not Path(v['video_path']).is_file() for v in data):
        raise FileNotFoundError('At least one annotation video path is inaccessible')
    gpus=gpus[:min(len(gpus),n)]
    codes=('run.py','worker.py','dscache448_adapter.py','qwen_dense_dscache_adapter.py','ovo_qwen_dscache_run.py','qwen_vl_utils/vision_process.py')
    profile=LUKV_PROFILE
    cfg=dict(version=1,benchmark='ovobench',model='qwen25vl',method=a.method,
             annotation=str(anno),annotation_sha256=sha(anno),max_videos=n,gpus=gpus,
             fps=1.0,chunk_frames=2,visual_input='dscache448-video',
             ordinary_pixel_area=448*448,latest_pixel_area=896*896,
             budget_visual_mean=None if a.method=='fullkv' else 500,
             dtype='float16',trace=False,profile_sha256=sha(profile) if a.method=='lukv' else None,
             code_sha256={x:sha(HERE/x) for x in codes},
             note='Experimental Qwen OVO input adaptation. Latest-frame replay changes pairing; not DSCache cache reproduction. Long-video performance unvalidated.')
    print(json.dumps(cfg,indent=2),flush=True)
    if a.dry_run:return 0
    out=a.output_dir.resolve()
    if out.exists() and not (out/'unified_manifest.json').exists() and any(out.iterdir()):
        raise RuntimeError('Nonempty output without manifest')
    out.mkdir(parents=True,exist_ok=True);(out/'annotations').mkdir(exist_ok=True)
    save_exact(out/'unified_manifest.json',json.dumps(cfg,indent=2)+chr(10))
    env=os.environ.copy()
    env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',DECORD_EOF_RETRY_MAX='40960')
    paths=[ROOT/'code/horizonkv/runtime/qwen-py310-deps-20261016',
           SHARED_KVBENCH,
           METHODS_ROOT/'lukv',
           METHODS_ROOT/'sparsemm',
           qwen_impl('hermes'),
           ROOT/'code/horizonkv/runtime/ppu-py310-overlay-20261015']
    env['PYTHONPATH']=os.pathsep.join(map(str,[PROJECT_ROOT]+paths))
    procs=[]
    for i,gpu in enumerate(gpus):
        shard=data[i*n//len(gpus):(i+1)*n//len(gpus)]
        annotation=out/'annotations'/f'shard_{i:02}.json'
        save_exact(annotation,json.dumps(shard,ensure_ascii=False))
        dest=out/f'shard_{i:02}';dest.mkdir(exist_ok=True)
        cmd=[sys.executable,str(HERE/'worker.py'),'--model','qwen25vl','--method',a.method,
             '--annotation',str(annotation),'--output',str(dest),'--gpu',gpu,
             '--chunk-frames','2','--sample-fps','1','--visual-input','dscache448-video']
        child=env.copy();child.update(CUDA_VISIBLE_DEVICES=gpu,SPARSEMM_GPU_INDEX=gpu)
        with (dest/'run.log').open('a') as log:
            proc=subprocess.Popen(cmd,env=child,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        procs.append((i,proc));print('START',i,'GPU',gpu,'PID',proc.pid,'RECORDS',len(shard),flush=True)
    failures=[]
    try:
        for i,proc in procs:
            rc=proc.wait();print('EXIT',i,rc,flush=True)
            if rc:failures.append((i,rc))
    except BaseException:
        for _,proc in procs:
            if proc.poll() is None:proc.terminate()
        for _,proc in procs:proc.wait()
        raise
    return int(bool(failures))

if __name__=='__main__':sys.exit(main())
