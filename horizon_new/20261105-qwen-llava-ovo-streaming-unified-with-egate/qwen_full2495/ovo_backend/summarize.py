#!/usr/bin/env python3
"""Verify complete sharded results and summarize the nine OVO task categories."""
import argparse,csv,json,os
from pathlib import Path
RT=('OJR','ATR','ACR','OCR','STU','FPD'); BACK=('EPM','ASI','HLD')
def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('output_dir',type=Path); args=parser.parse_args()
    root=args.output_dir.resolve(); manifest=json.loads((root/'unified_manifest.json').read_text())
    count=manifest['max_videos']; n=len(manifest['gpus']); expected={}; rows=[]; seen=set(); header=None
    for i in range(n):
        anno=json.loads((root/'annotations'/f'shard_{i:02}.json').read_text())
        for x in anno:
            q=x['conversations'][0]; k=(str(x['video_id']),q['question']); expected[k]=(q.get('task',q.get('question_type',x.get('task',''))) or '',q.get('correct_choice'))
        with (root/f'shard_{i:02}'/'results.csv').open(newline='',encoding='utf-8-sig') as f:
            reader=csv.DictReader(f)
            if header is None: header=reader.fieldnames
            if header!=reader.fieldnames: raise RuntimeError('Different CSV fields across shards')
            for r in reader:
                k=(r['video_id'],r['question'])
                if k in seen: raise RuntimeError(f'Duplicate question: {k}')
                seen.add(k); rows.append(r)
    if len(rows)!=count or len(expected)!=count or seen!=set(expected): raise RuntimeError('Results incomplete or do not match annotation')
    scores={}
    for r in rows:
        k=(r['video_id'],r['question']); task,correct=expected[k]
        if r.get('task')!=task or correct and r.get('correct_choice')!=correct: raise RuntimeError(f'Incorrect task or answer metadata: {k}')
        if float(r['qa_acc'])!=100.*(r['pred_choice']==r['correct_choice']): raise RuntimeError(f'Inconsistent accuracy: {k}')
    for task in RT+BACK:
        values=[float(r['qa_acc']) for r in rows if r.get('task')==task]
        if values: scores[task]={'count':len(values),'correct':sum(v==100 for v in values),'accuracy':sum(values)/len(values)}
    result={'model':manifest.get('model'),'method':manifest['method'],'count':len(rows),'micro_accuracy':sum(float(r['qa_acc']) for r in rows)/len(rows),'tasks':scores,'calibration':manifest.get('calibration')}
    if all(t in scores for t in RT+BACK):
        a=sum(scores[t]['accuracy'] for t in RT)/len(RT); b=sum(scores[t]['accuracy'] for t in BACK)/len(BACK)
        result.update(RT=a,Back=b,Avg=(a+b)/2)
    tmp=root/'results.csv.tmp'
    with tmp.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=header); writer.writeheader(); writer.writerows(rows)
    os.replace(tmp,root/'results.csv')
    (root/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
