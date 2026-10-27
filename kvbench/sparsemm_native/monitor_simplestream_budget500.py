"""Refresh the detached K=500 SimpleStream recent-only run's live status."""

import json
import statistics
import time
from pathlib import Path


ROOT = Path('/home/yujun/program/kvbench/results/ovobench_simplestream_budget500_20261025/full')


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    while True:
        rows = read_jsonl(ROOT / 'worker/results_incremental.jsonl')
        audits = read_jsonl(ROOT / 'worker/ragged_query_audit.jsonl')
        if len(audits) > len(rows):
            audits = audits[:len(rows)]
        slots = [row['actual_visual_slots'] / 112 for row in audits]
        state = dict(completed=len(rows), total=1468,
                     actual_visual_tokens_per_head_mean=statistics.mean(slots) if slots else None,
                     actual_visual_tokens_per_head_max=max(slots) if slots else None,
                     recent_only_running=len(rows) < 1468,
                     history_arms_running=False)
        (ROOT / 'progress_live.json').write_text(json.dumps(state, ensure_ascii=False, indent=2))
        average = '待统计' if not slots else f'{state["actual_visual_tokens_per_head_mean"]:.1f}'
        (ROOT / 'progress_live.md').write_text(
            f'近期区 K=500 对照：{len(rows)}/1468 题；实际平均视觉 KV {average} token/头。\n'
            '历史辅助分支尚未启动。\n')
        if len(rows) >= 1468:
            return
        time.sleep(15)


if __name__ == '__main__':
    main()
