#!/usr/bin/env bash
set -u

cd /home/yujun/program/hermes
pilot_pid="${1:?pilot PID required}"
pilot_status=logs/lu_uniform_joint_first20_k1000/run.status

while kill -0 "$pilot_pid" 2>/dev/null; do
  sleep 30
done

if grep -q '^complete ' "$pilot_status"; then
  if /home/yujun/.conda/envs/hermes-qwen/bin/python \
      scripts/analyze_lu_uniform_joint_first20.py \
      > logs/lu_uniform_joint_first20_k1000/paired_analysis.log 2>&1; then
    printf 'analysis-complete %s\n' "$(date -u +%FT%TZ)" >> "$pilot_status"
  else
    printf 'analysis-failed %s\n' "$(date -u +%FT%TZ)" >> "$pilot_status"
  fi
fi

setsid bash scripts/run_lu_k2000_51_150.sh \
  > logs/lu_k2000_51_150/resume_after_joint_launcher.log 2>&1 < /dev/null &
printf 'resume-after-joint-launched pid=%s %s\n' "$!" "$(date -u +%FT%TZ)" \
  >> logs/lu_k2000_51_150/run.status
