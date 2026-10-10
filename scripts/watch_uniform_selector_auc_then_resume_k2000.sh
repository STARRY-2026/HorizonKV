#!/usr/bin/env bash
set -u

cd /home/yujun/program/hermes

uniform_pid="${1:?uniform selector PID required}"
uniform_status=logs/uniform_selector_auc_50v/run.status
uniform_report=results/myobservations/uniform_selector_auc_50v_20260920/uniform_selector_auc_summary.json

while kill -0 "$uniform_pid" 2>/dev/null; do
  sleep 30
done

if [[ -f "$uniform_report" ]] && grep -q '"status": "complete"' "$uniform_report"; then
  printf 'complete %s\n' "$(date -u +%FT%TZ)" >> "$uniform_status"
else
  printf 'failed-or-partial %s\n' "$(date -u +%FT%TZ)" >> "$uniform_status"
fi

# The interrupted runner resumes from its existing per-video CSV and skips
# completed video IDs.
setsid bash scripts/run_lu_k2000_51_150.sh \
  > logs/lu_k2000_51_150/resume_launcher.log 2>&1 < /dev/null &
printf 'resume-launched pid=%s %s\n' "$!" "$(date -u +%FT%TZ)" \
  >> logs/lu_k2000_51_150/run.status
