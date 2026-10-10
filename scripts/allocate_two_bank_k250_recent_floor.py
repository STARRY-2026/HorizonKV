#!/usr/bin/env python3
"""Sensitivity of K=250 two-bank allocation to a per-head recent floor."""

import csv
import json
from pathlib import Path

import numpy as np

from allocate_token_gradient import load_profiles
from allocate_two_bank_grid_dp_floor750 import exact_floor_allocation


BASE = Path("results/myobservations/two_bank_grid_dp_k250_floor188_20260923")
OUT = Path("results/myobservations/two_bank_k250_recent_floor_20260923")
HEADS = 112
TOTAL = 28000
FLOOR = 188
CAP = 375
RECENT_FLOORS = (32, 64, 128)


def restricted_n_v(recent, history, recent_floor):
    values = np.full((HEADS, CAP+1), -np.inf)
    endpoint = np.full((HEADS, CAP+1), -1, dtype=np.int16)
    rows = np.arange(HEADS)[:, None]
    for n in range(max(FLOOR,recent_floor), CAP+1):
        rr = np.arange(recent_floor, min(n, recent.shape[1]-1)+1)
        qq = n-rr
        feasible = qq < history.shape[1]
        rr, qq = rr[feasible], qq[feasible]
        candidate = recent[rows,rr[None,:]] + history[rows,qq[None,:]]
        best = np.argmax(candidate, axis=1)
        values[:,n] = candidate[np.arange(HEADS),best]
        endpoint[:,n] = rr[best]
    return values,endpoint


def main():
    profiles,_ = load_profiles()
    videos = sorted(profiles)
    recent = np.mean([profiles[v]["recent"] for v in videos],axis=0)
    history = np.mean([profiles[v]["history"] for v in videos],axis=0)
    old = list(csv.DictReader((BASE/"per_head_allocation.csv").open()))
    old_n = np.array([int(row["capacity_n"]) for row in old])
    old_r = np.array([int(row["recent_r"]) for row in old])
    baseline = float(sum(float(row["selected_proxy"]) for row in old))
    OUT.mkdir(parents=True,exist_ok=True)
    report = {"protocol":{"videos":len(videos),
                         "observations":sum(profiles[v]["observations"] for v in videos),
                         "average_per_head":250,"global_budget":TOTAL,
                         "head_capacity_bounds":[FLOOR,CAP],
                         "selector":"newest-r recent prefix + score-ranked older history prefix",
                         "caveat":"In-sample full-cache answer-utility proxy; not sequential eviction or compressed QA"},
              "baseline":{"recent_floor":0,"objective":baseline,
                          "recent_total":int(old_r.sum()),
                          "heads_below_recent32":int(np.sum(old_r<32))},
              "recent_floor_variants":{}}
    for recent_floor in RECENT_FLOORS:
        curves,endpoints = restricted_n_v(recent,history,recent_floor)
        alloc,objective = exact_floor_allocation(curves,total=TOTAL,floor=FLOOR,cap=CAP)
        idx=np.arange(HEADS)
        rr=endpoints[idx,alloc]
        qq=alloc-rr
        assert int(alloc.sum())==TOTAL and int(rr.min())>=recent_floor
        assert np.isclose(objective,(recent[idx,rr]+history[idx,qq]).sum(),atol=1e-9)
        rows=[{"head_id":int(h),"layer":int(h//4),"kv_head":int(h%4),
               "capacity_n":int(alloc[h]),"recent_r":int(rr[h]),
               "history_q":int(qq[h]),
               "delta_n_vs_no_recent_floor":int(alloc[h]-old_n[h]),
               "delta_r_vs_no_recent_floor":int(rr[h]-old_r[h])} for h in idx]
        with (OUT/f"per_head_recent{recent_floor}.csv").open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        per_video=[]
        for video in videos:
            p=profiles[video]
            before=float((p["recent"][idx,old_r]+p["history"][idx,old_n-old_r]).sum())
            after=float((p["recent"][idx,rr]+p["history"][idx,qq]).sum())
            per_video.append(100*(after-before))
        report["recent_floor_variants"][str(recent_floor)]={
            "objective":objective,"delta_vs_baseline_pp":100*(objective-baseline),
            "recent_total":int(rr.sum()),"history_total":int(qq.sum()),
            "heads_at_recent_floor":int(np.sum(rr==recent_floor)),
            "heads_at_head_floor":int(np.sum(alloc==FLOOR)),
            "heads_at_head_cap":int(np.sum(alloc==CAP)),
            "token_reassigned_between_heads":int(np.abs(alloc-old_n).sum()//2),
            "per_video_delta_pp_min":float(np.min(per_video)),
            "per_video_delta_pp_max":float(np.max(per_video)),
            "videos_improved":int(np.sum(np.asarray(per_video)>1e-10)),
            "videos_worsened":int(np.sum(np.asarray(per_video)<-1e-10))}
    (OUT/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
