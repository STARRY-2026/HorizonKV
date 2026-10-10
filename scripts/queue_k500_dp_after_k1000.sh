#!/usr/bin/env bash
set -euo pipefail

cd /home/yujun/program/hermes
upstream_session=k1000_dp_cachecomp_first250_20260924
upstream_dir=results/qwen2.5_vl_7b/streamingbench/fps0.5-kv1000-hermes-kvheadbudget-dp_k1000_exact_budget_scores-sparsemm-union1-r0.2-w32-raggedprefill-headalpha-recent_ratio_profile-d0.1-t1-a0-1-kd2-dp-cachecomp-alpha-k1000-native-first250-20260924

echo "$(date -u +'%F %T UTC') Waiting for ${upstream_session} to finish."
while tmux has-session -t "$upstream_session" 2>/dev/null; do
  sleep 30
done

if [[ ! -s "$upstream_dir/results.csv" || ! -s "$upstream_dir/eval_results.txt" ]]; then
  echo "$(date -u +'%F %T UTC') K=1000 did not produce completed result files; K=500 not started." >&2
  exit 1
fi
if ! rg -q '#Samples: 1255' "$upstream_dir/eval_results.txt"; then
  echo "$(date -u +'%F %T UTC') K=1000 evaluation does not contain all 1255 questions; K=500 not started." >&2
  exit 1
fi

echo "$(date -u +'%F %T UTC') K=1000 completed; starting K=500 DP+alpha/k on 150 videos."
bash scripts/run_k500_dp_cachecomp_first150.sh
