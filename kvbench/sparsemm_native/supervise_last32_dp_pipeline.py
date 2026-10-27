"""Detached sequential last32 calibration, exact K500 DP and official OVO test.

Wait for the active two-process SimpleStream run to release GPU memory. Every
stage is checkpointed. A failed validation stops the pipeline rather than
silently falling back to HERMES or pseudo-query attention.
"""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from sparsemm_native import run_ovobench_last32_dp as runner
from sparsemm_native.supervise_ovobench_recent4_only import split_videos

ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT.parent
HERMES_PYTHON = Path('/home/yujun/.conda/envs/hermes-qwen/bin/python')
OUTPUT = ROOT/'results/ovobench_last32_dp_k500_fps05_20261022'
BLOCKER = ROOT/'results/ovobench_simplestream_dynamic_headmix11_full_20261022/status.json'
CALIBRATION = PROGRAM/'hermes/results/myobservations/last32_visual_attention_calibration_20261022'
ALLOCATION = PROGRAM/'hermes/results/myobservations/last32_two_bank_k500_floor375_recent32_20261022'


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    os.replace(temporary,path)


def count(path):
    if not path.exists():return 0
    with path.open() as f:return sum(1 for line in f if line.strip())


def command(argv,log_path,cwd=PROGRAM):
    with log_path.open('ab',buffering=0) as log:
        proc=subprocess.Popen([str(v) for v in argv],cwd=cwd,stdin=subprocess.DEVNULL,
                              stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while proc.poll() is None:
            update(stage_pid=proc.pid)
            time.sleep(15)
        if proc.returncode:
            raise RuntimeError(f'Stage exited {proc.returncode}: {log_path}')


state={'state':'initializing','pid':os.getpid(),'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}


def update(**fields):
    state.update(fields,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    save(OUTPUT/'status.json',state)


def wait_for_gpu():
    update(state='waiting_for_current_ovo_run',blocking_experiment=str(BLOCKER))
    while True:
        status=json.loads(BLOCKER.read_text())
        if status.get('state')=='complete':return
        if status.get('state')=='paused':
            old=BLOCKER.parent
            retained=0
            for shard in ('shard0','shard1'):
                path=old/'shards'/shard/'results_incremental.jsonl'
                for line in path.open():
                    if json.loads(line)['task'] not in runner.legacy.FORWARD:
                        retained+=1
            if retained!=1468:
                raise RuntimeError(f'Prior nine-task run has only {retained}/1468 saved results')
            free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free',
                                          '--format=csv,noheader,nounits'],text=True)
            if int(free.strip())>=28000:
                update(previous_experiment_paused_with_nine_tasks=True)
                return
        if status.get('state')=='failed':
            raise RuntimeError('Existing OVO run failed before GPU was released')
        time.sleep(30)


def run_calibration():
    collector=PROGRAM/'hermes/scripts/collect_last32_visual_attention.py'
    allocation=PROGRAM/'hermes/scripts/allocate_last32_two_bank_k500.py'
    update(state='calibration_smoke')
    command([HERMES_PYTHON,'-u',collector,'--output',CALIBRATION,'--max-observations','1'],
            OUTPUT/'calibration_smoke.log')
    checkpoint=json.loads((CALIBRATION/'status.json').read_text())
    if checkpoint.get('completed',0)<1:
        raise RuntimeError('No true last32 calibration checkpoint committed')
    update(state='calibration_full',completed_observations=checkpoint['completed'])
    command([HERMES_PYTHON,'-u',collector,'--output',CALIBRATION],OUTPUT/'calibration_full.log')
    if json.loads((CALIBRATION/'status.json').read_text()).get('state')!='complete':
        raise RuntimeError('68 last32 calibration observations did not complete')
    update(state='exact_dp')
    command([HERMES_PYTHON,'-u',allocation],OUTPUT/'allocation.log')
    if json.loads((ALLOCATION/'report.json').read_text()).get('status')!='complete':
        raise RuntimeError('Exact K500 dual-cache DP did not complete')


def worker_args(name, ids, root):
    return [HERMES_PYTHON,'-u','-m','sparsemm_native.run_ovobench_last32_dp',
            '--arm','dp_last32','--output-root',root,'--ids',','.join(ids),
            '--budget','500','--fps','0.5','--chunk-size','16',
            '--max-new-tokens','16','--video-max-pixels','151200',
            '--grouped','--concurrent-processes','2','--max-new','100000']


def run_pilot():
    expected=runner.legacy.queries()
    chosen=[];counts={}
    for query in expected:
        if query['task'] in runner.legacy.FORWARD:continue
        if counts.get(query['task'],0)<2:
            chosen.append(query['query_id'])
            counts[query['task']]=counts.get(query['task'],0)+1
    if len(chosen)!=18:raise AssertionError('Pilot lacks two official queries per retained task')
    update(state='official_pilot',pilot_query_ids=chosen)
    args=worker_args('pilot',chosen,OUTPUT/'pilot')
    args[args.index('--concurrent-processes')+1]='1'
    command(args,OUTPUT/'pilot.log',cwd=ROOT)
    if count(OUTPUT/'pilot/dp_last32/queries.jsonl')!=18:
        raise RuntimeError('Official pilot did not finish 18 queries')


def run_full():
    expected=runner.legacy.queries()
    selected=runner.schedule_queries([q for q in expected if q['task'] not in runner.legacy.FORWARD])
    jobs=split_videos(selected)
    if sum(len(job['query_ids']) for job in jobs.values())!=1468:
        raise AssertionError('Nine-task official query sharding changed')
    update(state='official_full',full_shards={name:len(job['query_ids']) for name,job in jobs.items()})
    attempts={name:0 for name in jobs};active={}
    while True:
        completed={name:count(OUTPUT/'full/shards'/name/'dp_last32/queries.jsonl') for name in jobs}
        update(state='official_full',completed_queries=sum(completed.values()),shard_queries=completed,
               worker_pids={name:proc.pid for name,proc in active.items() if proc.poll() is None})
        if all(completed[name]==len(jobs[name]['query_ids']) for name in jobs):break
        for name,job in jobs.items():
            if completed[name]==len(job['query_ids']):continue
            proc=active.get(name)
            if proc is not None and proc.poll() is None:continue
            if attempts[name]>=3:raise RuntimeError(f'{name} failed three times')
            log_path=OUTPUT/(name+'.log')
            with log_path.open('ab',buffering=0) as log:
                proc=subprocess.Popen([str(v) for v in worker_args(name,job['query_ids'],OUTPUT/'full/shards'/name)],
                    cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active[name]=proc;attempts[name]+=1
        time.sleep(20)
    for proc in active.values():
        if proc.poll() is None:proc.wait(timeout=60)
    rows={}
    for name in jobs:
        for line in (OUTPUT/'full/shards'/name/'dp_last32/queries.jsonl').open():
            row=json.loads(line)
            if row['query_id'] in rows:raise RuntimeError('Duplicate official query')
            rows[row['query_id']]=row
    if set(rows)!={q['query_id'] for q in selected}:
        raise RuntimeError('Nine-task official query set incomplete')
    ordered=[rows[q['query_id']] for q in selected]
    for row,query in zip(ordered,selected):runner.validate_row(row,query)
    destination=OUTPUT/'full/dp_last32';destination.mkdir(parents=True,exist_ok=True)
    (destination/'queries.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in ordered))
    summary=runner.legacy.summary(ordered,1468)
    save(destination/'summary.json',summary)
    update(state='complete',completed_queries=1468,
           nine_task_group_average=summary['hermes_category_average'],
           question_accuracy=summary['question_accuracy'],worker_pids={})


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    try:
        wait_for_gpu()
        run_calibration()
        run_pilot()
        run_full()
    except BaseException as error:
        update(state='failed',error=f'{type(error).__name__}: {error}')
        raise


if __name__=='__main__':main()
