"""Refresh the detached pseudo-query 250/250 history run's live status."""

import json
import statistics
import time
from pathlib import Path


ROOT = Path('/home/yujun/program/kvbench/results/ovobench_simplestream_pseudo_history250_20261025/full')


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
        history = [row['actual_history_visual_slots'] / 112 for row in audits]
        recent = [row['actual_recent_visual_slots'] / 112 for row in audits]
        state = dict(completed=len(rows), total=1468,
                     actual_visual_tokens_per_head_mean=statistics.mean(slots) if slots else None,
                     actual_visual_tokens_per_head_max=max(slots) if slots else None,
                     actual_history_tokens_per_head_mean=statistics.mean(history) if history else None,
                     actual_recent_tokens_per_head_mean=statistics.mean(recent) if recent else None,
                     recent_only_running=False,
                     history_arms_running=len(rows) < 1468)
        (ROOT / 'progress_live.json').write_text(json.dumps(state, ensure_ascii=False, indent=2))
        average = '待统计' if not slots else f'{state["actual_visual_tokens_per_head_mean"]:.1f}'
        (ROOT / 'progress_live.md').write_text(
            f'伪查询低近期头 250+250 双缓存：{len(rows)}/1468 题；实际平均视觉 KV {average} token/头。\n'
            '历史候选是视频更早的最多16个等间隔采样帧；低近期头历史250、近期250，高近期头近期500。\n')
        if len(rows) >= 1468:
            return
        time.sleep(15)


if __name__ == '__main__':
    main()
