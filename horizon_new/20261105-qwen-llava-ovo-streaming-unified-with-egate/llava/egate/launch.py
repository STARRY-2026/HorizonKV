"""Launch 8 durable EGate StreamingBench workers with pinned heldout shards."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path("/mnt/aifordata2026-p/common/aifordata/541707")
here=Path(__file__).resolve().parent
reference=root/"experiments/horizonkv/llava-streamingbench-6methods-heldout-k500-fps05-20261102/hermes/annotations"
out=root/"experiments/horizonkv/streaming-egate-qwen-heldout-k500-fps05-20261103"
overlay=Path("/tmp/egate-transformers4524");other=root/"code/horizonkv/runtime/ppu-py310-overlay-20261015"
if not (overlay/"transformers/__init__.py").exists():raise RuntimeError("Pinned ephemeral runtime missing")
shards=[json.loads((reference/f"shard_{i:02}.json").read_text()) for i in range(8)]
videos=sum(map(len,shards));questions=sum(len(x["conversations"]) for s in shards for x in s)
if (videos,questions)!=(448,2244):raise RuntimeError("Unexpected held-out cohort")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest={"benchmark":"streamingbench","model":"qwen25vl","method":"horizonkv-egate-experimental","cohort":"heldout","videos":videos,"questions":questions,"gpus":[str(i) for i in range(8)],"fps":.5,"frame_encoding":"per-image fixed896 padded","recent_window":"last 1 to 4 sampled images","history":"reconstructed per question","visual_budget_per_head":500,"dtype":"fp16","code_sha256":{str(x):sha(x) for x in [here/"worker.py",here/"pae_variable_recent.py",here/"egate_protocol.py",here/"assets/per_head_allocation.csv"]},"limitations":"experimental prompt and compact positions; not parity verified with original; preprocessing differs from prior Qwen and LLaVA baselines"}
out.mkdir(parents=True,exist_ok=True);(out/"annotations").mkdir(exist_ok=True);mp=out/"manifest.json"; encoded=json.dumps(manifest,indent=2,ensure_ascii=False)+"\n"
if mp.exists() and mp.read_text()!=encoded:raise RuntimeError("Output manifest differs; choose a new output directory")
if not mp.exists():mp.write_text(encoded)
for i,shard in enumerate(shards):
 path=out/"annotations"/f"shard_{i:02}.json";raw=json.dumps(shard,ensure_ascii=False)
 if path.exists() and path.read_text()!=raw:raise RuntimeError("Existing annotation differs")
 if not path.exists():path.write_text(raw)
for i in range(8):
 shard=out/f"shard_{i:02}";shard.mkdir(exist_ok=True);env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(i),HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",PYTHONUNBUFFERED="1",OMP_NUM_THREADS="2",PYTHONPATH=str(overlay)+os.pathsep+str(other));cmd=[sys.executable,str(here/"worker.py"),"--annotation",str(out/"annotations"/f"shard_{i:02}.json"),"--output",str(shard)]
 with (shard/"run.log").open("a") as f:proc=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 print("GPU",i,"PID",proc.pid,"VIDEOS",len(shards[i]),flush=True)
print("OUTPUT",out,flush=True)
