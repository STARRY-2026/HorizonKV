#!/usr/bin/env python3
"""Exact 1-token 2D head-grid DP followed by exact global budget DP.

Main experiment: no per-head minimum and no Recent-32 floor; each head may
receive 0..1500 visual tokens, recent <=512, history <=1468. The objective is
equal-video average of full-cache projected answer-utility retention.
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from allocate_token_gradient import load_profiles


OUT = Path("results/myobservations/two_bank_grid_dp_20260923")
HEADS = 112
TOTAL = 112000
MAX_HEAD = 1500
MAX_RECENT = 512
MAX_HISTORY = 1468


def head_grid_dp(recent, history):
    """Run the literal 2D path recurrence on one physical head."""
    d = np.full((MAX_RECENT+1, MAX_HISTORY+1), -np.inf)
    back = np.full(d.shape, 255, dtype=np.uint8)
    d[0, 0] = 0.0
    best = np.empty(MAX_HEAD+1)
    endpoint_r = np.empty(MAX_HEAD+1, dtype=np.int16)
    best[0], endpoint_r[0] = 0.0, 0
    for n in range(1, MAX_HEAD+1):
        rr = np.arange(max(0,n-MAX_HISTORY), min(MAX_RECENT,n)+1)
        qq = n-rr
        via_recent = np.full(rr.shape, -np.inf)
        via_history = np.full(rr.shape, -np.inf)
        r_ok = rr > 0
        h_ok = qq > 0
        via_recent[r_ok] = (d[rr[r_ok]-1,qq[r_ok]]
                            + recent[rr[r_ok]]-recent[rr[r_ok]-1])
        via_history[h_ok] = (d[rr[h_ok],qq[h_ok]-1]
                             + history[qq[h_ok]]-history[qq[h_ok]-1])
        take_recent = via_recent >= via_history
        values = np.maximum(via_recent,via_history)
        d[rr,qq] = values
        back[rr,qq] = take_recent.astype(np.uint8)
        j = int(np.argmax(values))
        best[n] = float(values[j])
        endpoint_r[n] = int(rr[j])
    # This check catches a broken edge or index convention. Current edges
    # come from a fixed additive potential, so every endpoint telescopes.
    for n in (0,1,32,512,750,1000,1500):
        rr = np.arange(max(0,n-MAX_HISTORY), min(MAX_RECENT,n)+1)
        qq = n-rr
        if not np.allclose(d[rr,qq],recent[rr]+history[qq],rtol=0,atol=1e-10):
            raise AssertionError(f"Grid recurrence differs from direct endpoint values at n={n}")
    return best,endpoint_r,back


def global_exact_dp(curves, progress=False):
    """Exact multiple-choice knapsack with impossible states pruned."""
    prev = np.full(TOTAL+1,-np.inf)
    prev[0] = 0.0
    back = np.full((HEADS,TOTAL+1),-1,dtype=np.int16)
    prev_lo = prev_hi = 0
    for head in range(HEADS):
        processed = head+1
        left = HEADS-processed
        lo = max(0,TOTAL-left*MAX_HEAD)
        hi = min(TOTAL,processed*MAX_HEAD)
        nxt = np.full_like(prev,-np.inf)
        chosen = back[head]
        for n in range(MAX_HEAD+1):
            dst_lo = max(lo,prev_lo+n)
            dst_hi = min(hi,prev_hi+n)
            if dst_lo > dst_hi:
                continue
            candidate = prev[dst_lo-n:dst_hi-n+1] + curves[head,n]
            current = nxt[dst_lo:dst_hi+1]
            better = candidate > current
            current[better] = candidate[better]
            chosen[dst_lo:dst_hi+1][better] = n
        prev,prev_lo,prev_hi = nxt,lo,hi
        if progress and (processed%16==0 or processed==HEADS):
            print(f"GLOBAL_DP {processed}/{HEADS} reachable={int(np.isfinite(prev).sum())}",flush=True)
    if not np.isfinite(prev[TOTAL]):
        raise AssertionError("Global budget is infeasible")
    alloc = np.empty(HEADS,dtype=np.int16)
    state = TOTAL
    for head in range(HEADS-1,-1,-1):
        n = int(back[head,state])
        if n < 0:
            raise AssertionError(f"Broken global traceback at head {head}, state {state}")
        alloc[head] = n
        state -= n
    if state != 0 or int(alloc.sum()) != TOTAL:
        raise AssertionError("Global traceback does not spend exact budget")
    return alloc,float(prev[TOTAL])


def reconstruct_path(back, r, q):
    reverse = []
    while r or q:
        choice = int(back[r,q])
        if choice == 1:
            reverse.append("R")
            r -= 1
        elif choice == 0:
            reverse.append("H")
            q -= 1
        else:
            raise AssertionError("Broken head-grid traceback")
    return "".join(reversed(reverse))


def full_head_shares():
    """Only for plotting head-normalized curves, never for optimization."""
    from plot_two_dimensional_utility_surface import collect
    by_video,_ = collect()
    shares = []
    for observations in by_video.values():
        shares.append(np.mean([full/full.sum() for _,_,full,_ in observations],axis=0))
    return np.mean(shares,axis=0)


def representative_heads(allocation,r_alloc,curves,shares):
    positive = np.where(allocation > 0)[0]
    ratio = r_alloc[positive] / allocation[positive]
    chosen = []
    for quantile in (0.05,0.5,0.95):
        target = np.quantile(ratio,quantile)
        order = positive[np.argsort(np.abs(ratio-target))]
        chosen.append(int(next(h for h in order if h not in chosen)))
    # Add the head with the largest total task-facing utility mass.
    weighted = np.argsort(-shares)
    chosen.append(int(next(h for h in weighted if h not in chosen)))
    return chosen


def plot_curves(curves,allocation,r_alloc,shares,representative):
    fig,axes = plt.subplots(2,2,figsize=(11.8,7.8),layout="constrained")
    x = np.arange(MAX_HEAD+1)
    for ax,h in zip(axes.flat,representative):
        total = 100*curves[h]
        ax.plot(x,total,color="#355C7D",linewidth=1.8)
        ax.scatter([allocation[h]],[total[allocation[h]]],color="#D95542",s=38,zorder=5,
                   label="Selected allocation")
        ax.axvline(int(allocation[h]),color="#D95542",linestyle="--",linewidth=.8,alpha=.7)
        ax.set(xlim=(0,MAX_HEAD),xlabel="Head capacity n (tokens)",
               ylabel="Global full-utility contribution (pp)",
               title=f"Head {h//4}:{h%4}  |  full-utility share {shares[h]:.2%}")
        ax.grid(alpha=.18)
        ax.text(.03,.96,
                f"Selected: n={allocation[h]}, recent={r_alloc[h]}, history={allocation[h]-r_alloc[h]}",
                transform=ax.transAxes,va="top",ha="left",fontsize=8,
                bbox={"boxstyle":"round,pad=.35","facecolor":"white","edgecolor":"#D0D0D0","alpha":.85})
    fig.suptitle("Representative exact 1-token n–v curves (full-cohort calibration)",fontsize=13)
    fig.savefig(OUT/"representative_n_v_curves.png",dpi=250)
    fig.savefig(OUT/"representative_n_v_curves.pdf")
    plt.close(fig)


def plot_allocation(alloc, recent):
    heads = np.arange(HEADS)
    history = alloc-recent
    fig,ax = plt.subplots(figsize=(14.6,4.8),layout="constrained")
    ax.bar(heads,history,width=.85,color="#355C7D",label="History capacity")
    ax.bar(heads,recent,bottom=history,width=.85,color="#E5A54B",label="Recent capacity")
    ax.axhline(1000,color="#555555",linestyle="--",linewidth=.8,label="Mean budget: 1000")
    ax.set(xlim=(-1,HEADS),ylim=(0,1550),xlabel="Physical KV head (layer × 4 + head)",
           ylabel="Assigned visual KV slots",title="Exact global-DP allocation: 112,000 visual slots")
    ax.set_xticks(np.arange(0,HEADS,8))
    ax.legend(ncol=3,loc="upper center",frameon=False)
    fig.savefig(OUT/"per_head_allocation.png",dpi=250)
    fig.savefig(OUT/"per_head_allocation.pdf")
    plt.close(fig)


def plot_normalized_curves(curves,alloc,recent,shares,representative):
    fig,ax = plt.subplots(figsize=(8.8,5.5),layout="constrained")
    x = np.arange(MAX_HEAD+1)
    colors = ("#355C7D","#C96440","#3A9C89","#8164A5")
    for h,color in zip(representative,colors):
        y = curves[h]/shares[h]
        ax.plot(x,y,color=color,linewidth=1.9,
                label=f"Head {h//4}:{h%4}  (n={alloc[h]}, R={recent[h]}, H={alloc[h]-recent[h]})")
        ax.scatter([alloc[h]],[y[alloc[h]]],color=color,s=31,zorder=5)
    ax.set(xlim=(0,MAX_HEAD),ylim=(0,1),xlabel="Head capacity n (tokens)",
           ylabel="Retained fraction of this head's full-cache utility",
           title="The same n–v curves, normalized only for shape comparison")
    ax.grid(alpha=.18)
    ax.legend(loc="upper left",frameon=False,fontsize=8)
    fig.savefig(OUT/"representative_n_v_normalized.png",dpi=250)
    fig.savefig(OUT/"representative_n_v_normalized.pdf")
    plt.close(fig)


def small_test():
    # Independent brute force verifies the head recurrence and the global
    # convolution logic before the full expensive run.
    recent = np.array([0.,2.,3.,8.])
    history = np.array([0.,4.,5.,6.,12.])
    d = np.full((len(recent),len(history)),-np.inf)
    d[0,0] = 0
    for n in range(1,7):
        for r in range(len(recent)):
            q = n-r
            if not 0 <= q < len(history):
                continue
            options = []
            if r:
                options.append(d[r-1,q]+recent[r]-recent[r-1])
            if q:
                options.append(d[r,q-1]+history[q]-history[q-1])
            d[r,q] = max(options)
    for n in range(7):
        assert np.isclose(max(d[r,n-r] for r in range(len(recent)) if 0<=n-r<len(history)),
                          max(recent[r]+history[n-r] for r in range(len(recent)) if 0<=n-r<len(history)))
    # Tiny exact multiple-choice knapsack vs exhaustive enumeration.
    toy = np.array([[0.,1.,3.],[0.,2.,2.],[0.,1.,4.]])
    dp = np.array([0.,-np.inf,-np.inf,-np.inf,-np.inf])
    for head in range(3):
        nxt = np.full_like(dp,-np.inf)
        for n in range(3):
            nxt[n:] = np.maximum(nxt[n:],dp[:5-n]+toy[head,n])
        dp = nxt
    brute = max(toy[0,a]+toy[1,b]+toy[2,c]
                for a in range(3) for b in range(3) for c in range(3) if a+b+c==4)
    assert np.isclose(dp[4],brute)


def main():
    small_test()
    profiles,_ = load_profiles()
    videos = sorted(profiles)
    recent = np.mean([profiles[v]["recent"] for v in videos],axis=0)
    history = np.mean([profiles[v]["history"] for v in videos],axis=0)
    curves = np.empty((HEADS,MAX_HEAD+1),dtype=np.float64)
    endpoints = np.empty((HEADS,MAX_HEAD+1),dtype=np.int16)
    backpointers = np.empty((HEADS,MAX_RECENT+1,MAX_HISTORY+1),dtype=np.uint8)
    for h in range(HEADS):
        curves[h],endpoints[h],backpointers[h] = head_grid_dp(recent[h],history[h])
    print("HEAD_GRIDS_COMPLETE",flush=True)
    alloc,objective = global_exact_dp(curves,progress=True)
    rr = endpoints[np.arange(HEADS),alloc]
    qq = alloc-rr
    if np.any(rr<0) or np.any(qq<0) or np.any(rr>MAX_RECENT) or np.any(qq>MAX_HISTORY):
        raise AssertionError("Invalid selected bank split")
    if not np.isclose(objective,(recent[np.arange(HEADS),rr]+history[np.arange(HEADS),qq]).sum(),atol=1e-9):
        raise AssertionError("Global and head objectives disagree")
    shares = full_head_shares()
    reps = representative_heads(alloc,rr,curves,shares)
    OUT.mkdir(parents=True,exist_ok=True)
    plot_curves(curves,alloc,rr,shares,reps)
    plot_normalized_curves(curves,alloc,rr,shares,reps)
    plot_allocation(alloc,rr)
    per_head = []
    paths = {}
    for h in range(HEADS):
        path = reconstruct_path(backpointers[h],int(rr[h]),int(qq[h]))
        if len(path)!=alloc[h] or path.count("R")!=rr[h] or path.count("H")!=qq[h]:
            raise AssertionError("Path does not match endpoint")
        paths[str(h)] = path
        per_head.append({"head_id":h,"layer":h//4,"kv_head":h%4,
                         "capacity_n":int(alloc[h]),"recent_r":int(rr[h]),
                         "history_q":int(qq[h]),
                         "history_share":float(qq[h]/alloc[h]) if alloc[h] else None,
                         "head_full_utility_share":float(shares[h]),
                         "selected_proxy":float(curves[h,alloc[h]])})
    with (OUT/"per_head_allocation.csv").open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=list(per_head[0]))
        writer.writeheader();writer.writerows(per_head)
    (OUT/"selected_paths.json").write_text(json.dumps(paths,ensure_ascii=False)+"\n")
    np.savez_compressed(OUT/"head_n_v_curves.npz",capacity=np.arange(MAX_HEAD+1),
                        values=curves,endpoint_recent=endpoints,full_head_share=shares)
    positive = alloc>0
    report = {"protocol":{"videos":len(videos),"observations":sum(profiles[v]["observations"] for v in videos),
                           "total_visual_slots":TOTAL,"head_capacity_range":[0,MAX_HEAD],
                           "recent_candidate_max":MAX_RECENT,"history_candidate_max":MAX_HISTORY,
                           "grid_step":1,"optimizer":"head grid DP plus exact global multiple-choice DP",
                           "PAVA_or_concave_relaxation":False,
                           "caveat":"In-sample full-cache answer-utility proxy, not compressed-cache QA"},
            "allocation":{"total":int(alloc.sum()),"recent_total":int(rr.sum()),
                          "history_total":int(qq.sum()),
                          "history_share":float(qq.sum()/alloc.sum()),
                          "recent_share":float(rr.sum()/alloc.sum()),
                          "head_capacity_quantiles":np.quantile(alloc,[0,.1,.25,.5,.75,.9,1]).tolist(),
                          "head_recent_quantiles":np.quantile(rr,[0,.1,.25,.5,.75,.9,1]).tolist(),
                          "heads_zero_capacity":int(np.sum(alloc==0)),
                          "heads_at_cap":int(np.sum(alloc==MAX_HEAD)),
                          "objective":objective},
            "representative_heads":[per_head[h] for h in reps]}
    (OUT/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
