#!/usr/bin/env bash
set -u

cd /home/yujun/program/hermes
current_pid="${1:?current experiment PID required}"
queue_status=logs/lu_uniform_joint_21_150_k1000/queue.status
mkdir -p logs/lu_uniform_joint_21_150_k1000
printf 'queued-after-pid=%s %s\n' "$current_pid" "$(date -u +%FT%TZ)" > "$queue_status"

while kill -0 "$current_pid" 2>/dev/null; do
  sleep 30
done

printf 'dependency-finished %s\n' "$(date -u +%FT%TZ)" >> "$queue_status"
setsid bash scripts/run_lu_uniform_joint_21_150_k1000.sh \
  > logs/lu_uniform_joint_21_150_k1000/launcher.log 2>&1 < /dev/null &
pilot_pid=$!
printf 'launched pid=%s %s\n' "$pilot_pid" "$(date -u +%FT%TZ)" >> "$queue_status"

while kill -0 "$pilot_pid" 2>/dev/null; do
  sleep 30
done

if grep -q '^complete ' logs/lu_uniform_joint_21_150_k1000/run.status; then
  if /home/yujun/.conda/envs/hermes-qwen/bin/python \
      scripts/analyze_lu_uniform_joint_21_150.py \
      > logs/lu_uniform_joint_21_150_k1000/paired_analysis.log 2>&1; then
    printf 'analysis-complete %s\n' "$(date -u +%FT%TZ)" >> "$queue_status"
  else
    printf 'analysis-failed %s\n' "$(date -u +%FT%TZ)" >> "$queue_status"
  fi
else
  printf 'run-failed-or-incomplete %s\n' "$(date -u +%FT%TZ)" >> "$queue_status"
fi
