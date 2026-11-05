"""Experimental StreamingBench Qwen EGate worker, per-question durable results."""
import argparse,csv,gc,json,math,os,sys,time
from collections import deque
from pathlib import Path
import torch
from PIL import Image
from decord import VideoReader
from transformers import Qwen2_5_VLForConditionalGeneration,Qwen2_5_VLProcessor,DynamicCache
from egate_protocol import load_allocation,format_mc_question,route_answer
root=Path("/mnt/aifordata2026-p/common/aifordata/541707")
egate=root/"code/horizonkv/releases/20261103-horizonkv-egate-v1/HorizonKV-feature-ovo-dp-recent4-publication/kvbench"
sys.path.insert(0,str(egate))
from kvbench.fixed_inputs import prepare_fixed_frames
from pae_variable_recent import compact_positions,combine_answer_cache
from snap_stream import RecentBatch,transfer_batch,snap_select
from sparsemm_native.simplestream_dscache_history64 import _pseudo_queries_and_keys,_attention_visual_indices,_prefill_question_suffix,_choice_from_first_logits
from sparsemm_native.hermes_kv_head_ragged import HeadRaggedLayerCache,HeadRaggedCache

p=argparse.ArgumentParser();p.add_argument("--annotation",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--max-questions",type=int);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True);dest=a.output/"results.csv"; dataset=json.loads(a.annotation.read_text()); expected=[(str(v["video_id"]),str(i),q["question"]) for v in dataset for i,q in enumerate(v["conversations"])]
rows=list(csv.DictReader(dest.open(newline=""))) if dest.exists() else []
if [(r["video_id"],r["question_index"],r["question"]) for r in rows] != expected[:len(rows)]:raise RuntimeError("Resume output does not match annotations")
if a.max_questions is not None and len(rows)>=a.max_questions:print("ALREADY_DONE",len(rows),flush=True);sys.exit(0)
modelpath=root/"model/Qwen2.5-VL-7B-Instruct"; processor=Qwen2_5_VLProcessor.from_pretrained(modelpath,local_files_only=True,use_fast=False);processor.image_processor.do_resize=False; tokenizer=processor.tokenizer
model=Qwen2_5_VLForConditionalGeneration.from_pretrained(modelpath,torch_dtype=torch.float16,attn_implementation="flash_attention_2",local_files_only=True).to("cuda").eval();core=model.model;alloc,digest=load_allocation();print("MODEL_READY","done",len(rows),flush=True)
columns=["video_id","question_index","question","task","correct_choice","pred_choice","qa_acc","recent_answer","history_answer","gate_source","sampled_frames","seconds"]
completed=len(rows); seen=0
with torch.inference_mode():
 for item in dataset:
    qcount=len(item["conversations"])
    if seen+qcount<=completed:seen+=qcount;continue
    reader=VideoReader(item["video_path"],num_threads=1)
    try:
        fps=round(reader.get_avg_fps())
        if fps<=0:raise ValueError("Invalid source fps")
        step=int(fps/1.0);clip=item.get("clip"); begin=max(0,int(clip[0]*fps)) if clip else 0; end=min(len(reader),int(clip[1]*fps)+1) if clip else len(reader);indices=list(range(begin,end,step)); processed=0; bank=None; recent_batch=None
        prefix_ids=tokenizer("<|im_start|>user\n<|vision_start|>",return_tensors="pt",add_special_tokens=False).input_ids.to("cuda");prefix=core.get_input_embeddings()(prefix_ids);n=prefix.shape[1]
        recent_batch=RecentBatch(core,prefix)
        for qi,q in enumerate(item["conversations"]):
            if seen<completed:seen+=1;continue
            t0=time.monotonic();target=min(len(indices),math.ceil(float(q["end_time"])*1.0))
            if target<1:raise ValueError("Question has no sampled video frames")
            if len(q["choices"]) not in (4,5):raise ValueError("Expected 4 or 5 answer choices")
            if target<processed:raise RuntimeError("Question timestamps must be nondecreasing")
            while processed<target:
                if recent_batch.frames==4:
                    bank=transfer_batch(core,recent_batch,alloc,bank)
                    del recent_batch
                    recent_batch=RecentBatch(core,prefix)
                count=min(4-recent_batch.frames,target-processed)
                raw=torch.from_numpy(reader.get_batch(indices[processed:processed+count]).asnumpy());frames=prepare_fixed_frames(raw)
                inputs=processor.image_processor(images=[Image.fromarray(f.numpy()) for f in frames],return_tensors="pt",do_resize=False)
                grid=inputs["image_grid_thw"].to("cuda");feat=core.get_image_features(inputs["pixel_values"].to("cuda",dtype=torch.float16),grid)
                if isinstance(feat,(tuple,list)): feat=torch.cat(list(feat),dim=0) if len(feat)>1 else feat[0]
                if feat.ndim==3: feat=feat.reshape(-1,feat.shape[-1])
                if feat.shape[0]!=count*1024:raise RuntimeError("Unexpected visual tokens")
                recent_batch.append_many(feat)
                print("VISUAL_BLOCK",count,"tokens",feat.shape[0],flush=True)
                del raw,frames,inputs,grid,feat
                processed+=count
            nr=recent_batch.frames;positions=recent_batch.positions();all_cache=recent_batch.full;one_cache=recent_batch.one
            one_idx=list(range(n))+list(range(n+(nr-1)*1024,n+nr*1024));probes=recent_batch.probes();audits=[];answers=[]
            for use_history in (False,True):
                audit={};layers,hs,rs,start,_=combine_answer_cache(core,bank,all_cache,one_cache,positions,one_idx,[1024]*nr,n,alloc,probes,snap_select,HeadRaggedLayerCache,include_history=use_history,position_limit=int(model.config.text_config.max_position_embeddings),reserve_history_for_budget=True,selection_audit=audit)
                text,gate=format_mc_question(q["question"],q["choices"],recent_gate=not use_history);suffix=tokenizer("<|vision_end|>\n"+text+"<|im_end|>\n<|im_start|>assistant\n",return_tensors="pt",add_special_tokens=False).input_ids.to("cuda");pos=torch.arange(start,start+suffix.shape[1],device="cuda").view(1,1,-1).expand(3,1,-1)
                hidden=_prefill_question_suffix(core,core.get_input_embeddings()(suffix),pos,HeadRaggedCache(layers));logits=model.lm_head(hidden)[0,-1];letters="ABCDEF"[:len(q["choices"])+(0 if use_history else 1)];_,answer,_=_choice_from_first_logits(logits,tokenizer,letters); answers.append(answer);audits.append(audit)
            if audits[0]["recent_selection_sha256"]!=audits[1]["recent_selection_sha256"] or audits[0]["suffix_start"]!=audits[1]["suffix_start"]:raise RuntimeError("Recent caches differ across passes")
            result,source=route_answer(answers[0],answers[1],len(q["choices"]));correct=q.get("correct_choice") or "ABCDEFGH"[q["choices"].index(q["answer"])]
            row=dict(video_id=item["video_id"],question_index=qi,question=q["question"],task=q.get("task",""),correct_choice=correct,pred_choice=result,qa_acc=100.0*(correct==result),recent_answer=answers[0],history_answer=answers[1],gate_source=source,sampled_frames=target,seconds=round(time.monotonic()-t0,3))
            exists=dest.exists();
            with dest.open("a",newline="",encoding="utf-8") as out:
                writer=csv.DictWriter(out,fieldnames=columns)
                if not exists:writer.writeheader()
                writer.writerow(row);out.flush();os.fsync(out.fileno())
            seen+=1;print("DONE",seen,"id",item["video_id"],"q",qi,"frames",target,"answers",answers,"final",result,"correct",correct,"sec",row["seconds"],flush=True)
            if a.max_questions is not None and seen>=a.max_questions:sys.exit(0)
    finally:
        del reader
        gc.collect()
print("COMPLETE",seen,flush=True)
