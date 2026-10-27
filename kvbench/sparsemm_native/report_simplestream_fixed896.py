"""CPU-only official scoring and matched-question reporting for fixed-896 K500.

The fps=.5 native-resolution run is the controlled baseline.  The older fps=1
dynamic-resolution run is optional context and is never called a resize-only
comparison.  Partial runs are scored on their own completed-question intersection.
"""

import argparse
from collections import Counter
import csv
import io
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NATIVE = ROOT / 'results/ovobench_simplestream_k500_input_only_fps05_native_20261026/full'
DEFAULT_DYNAMIC = ROOT / 'results/ovobench_simplestream_pseudo_budget500_20261025/full'
TASKS = ('EPM', 'ASI', 'HLD', 'STU', 'OJR', 'ATR', 'ACR', 'OCR', 'FPD')
BACKWARD = set(TASKS[:3])
COUNTS = dict(zip(TASKS, (297, 148, 186, 178, 184, 116, 109, 149, 101)))
SELECTOR = 'fixed_mixed_pseudo_qk_fp32_gqa_mean_avgpool5_topk_protect32'


def read_rows(path):
    """Ignore only an unterminated final append while the worker is running."""
    rows = []
    for line in Path(path).read_bytes().splitlines(keepends=True):
        if not line.endswith(b'\n'):
            break
        rows.append(json.loads(line))
    return rows


def load_run(root):
    root = Path(root)
    if (root / 'queries.jsonl').exists():
        rows = read_rows(root / 'queries.jsonl')
    else:
        values = read_rows(root / 'worker/results_incremental.jsonl')
        audits = read_rows(root / 'worker/ragged_query_audit.jsonl')
        observations = read_rows(root / 'worker/query_observations.jsonl')
        if not (len(values) == len(audits) == len(observations)):
            raise ValueError('Legacy observations, predictions and audits must match')
        rows = []
        for value, audit, observation in zip(values, audits, observations):
            if (str(value['id']) != observation['query_id'] or
                    audit['prompt_sha256'] != observation['prompt_sha256']):
                raise ValueError('Legacy audit order or prompt mismatch')
            rows.append(dict(query_id=str(value['id']), task=value['task'],
                target=value['ground_truth'], response=value['response'],
                seconds=observation['total_query_seconds'], input_audit=observation,
                budget_audit=audit))
    by_id = {}
    for row in rows:
        qid = str(row['query_id'])
        if qid in by_id or row['task'] not in TASKS:
            raise ValueError('Duplicate question or unsupported task')
        if (not isinstance(row['response'], str) or not isinstance(row['target'], str)
                or len(row['target']) != 1 or not 'A' <= row['target'] <= 'Z'):
            raise ValueError('Expected raw MCQA response and letter target')
        row = dict(row, query_id=qid, correct=row['target'] in row['response'])
        by_id[qid] = row
    return by_id


def score(rows):
    rows = list(rows)
    per_task = {}
    for task in TASKS:
        subset = [row for row in rows if row['task'] == task]
        correct = sum(row['target'] in row['response'] for row in subset)
        per_task[task] = dict(correct=correct, total=len(subset),
            accuracy=correct / len(subset) if subset else None)
    backward = [per_task[t]['accuracy'] for t in TASKS[:3]]
    realtime = [per_task[t]['accuracy'] for t in TASKS[3:]]
    complete_groups = all(value is not None for value in backward + realtime)
    audit_values = [row['budget_audit'] for row in rows]
    frames = [token for row in rows for token in row['budget_audit']['frame_visual_tokens']]
    durations = [row['seconds'] for row in rows]
    return dict(questions=len(rows), correct=sum(row['target'] in row['response'] for row in rows),
        question_accuracy=sum(row['target'] in row['response'] for row in rows) / len(rows) if rows else None,
        per_task=per_task,
        nine_task_macro_average=statistics.mean(backward + realtime) if complete_groups else None,
        official_two_group_macro_average=(statistics.mean(backward) + statistics.mean(realtime)) / 2
            if complete_groups else None,
        actual_visual_tokens_per_head_mean=statistics.mean(a['actual_visual_slots'] / 112 for a in audit_values)
            if rows else None,
        frame_visual_tokens_mean=statistics.mean(frames) if frames else None,
        frame_visual_tokens_distribution=dict(sorted(Counter(frames).items())),
        query_seconds_total=sum(durations),
        query_seconds_mean=statistics.mean(durations) if durations else None)


def validate_controlled_pair(current, baseline):
    """Spatial-dependent values intentionally differ; verify unaffected inputs."""
    for key in ('query_id', 'source_id', 'probe_id', 'task', 'group', 'timestamp', 'clip_end', 'target'):
        if current.get(key) != baseline.get(key):
            raise ValueError(f'Question {current["query_id"]}: mismatched {key}')
    for key in ('source_clip', 'requested_fps', 'raw_fps', 'raw_total_frames', 'full_sampled_frames',
                'selected_raw_frame_indices', 'selected_chunk_ids', 'sampled_timestamps',
                'source_hw', 'selected_frames', 'prompt_sha256', 'visual_encoder', 'position_layout'):
        if current['input_audit'].get(key) != baseline['input_audit'].get(key):
            raise ValueError(f'Question {current["query_id"]}: mismatched input {key}')
    for row in (current, baseline):
        audit = row['budget_audit']
        if (audit['selector'] != SELECTOR or audit['visual_budget_per_head'] != 500
                or audit['low_head_count'] != 56 or not audit['independent_single_frame_prefill']
                or not audit['pruned_before_question']
                or audit.get('actual_history_visual_slots', 0) != 0
                or audit.get('question_text_used_for_selection', False)
                or audit['actual_visual_slots'] > 56000):
            raise ValueError(f'Question {current["query_id"]}: cache/selector protocol changed')


def validate_protocol(current_root, baseline_root):
    current = json.loads((Path(current_root) / 'protocol.json').read_text())
    baseline = json.loads((Path(baseline_root) / 'protocol.json').read_text())
    for key in ('fps', 'visual_budget_per_head', 'configured_visual_slots', 'recent_bucket_count',
                'bucket_seconds', 'max_new_tokens', 'model', 'model_files_sha256',
                'annotation_sha256', 'required_dependencies', 'prompt', 'visual_encoder',
                'positions', 'selector', 'head_groups', 'historical_visual_kv',
                'question_used_for_selection', 'source_sampling'):
        if key not in current or current[key] != baseline[key]:
            raise ValueError(f'Controlled protocol differs in {key}')
    # The added runner and RGB transformation may differ. Every original
    # inference source and the exact head-allocation table must stay pinned.
    for path, digest in baseline['code_sha256'].items():
        if current['code_sha256'].get(path) != digest:
            raise ValueError(f'Original inference/head-group source changed: {path}')
    return dict(original_protocol_fields_equal=True, original_code_and_head_table_hashes_equal=True)


def compare(current, baseline, *, controlled):
    common = [qid for qid in current if qid in baseline]
    counts = {task: dict(both_correct=0, current_only=0, baseline_only=0, both_wrong=0)
              for task in TASKS}
    table = []
    for qid in common:
        left, right = current[qid], baseline[qid]
        if left['task'] != right['task'] or left['target'] != right['target']:
            raise ValueError(f'Question {qid}: task/target mismatch')
        if controlled:
            validate_controlled_pair(left, right)
        a, b = left['target'] in left['response'], right['target'] in right['response']
        bucket = 'both_correct' if a and b else 'current_only' if a else 'baseline_only' if b else 'both_wrong'
        counts[left['task']][bucket] += 1
        table.append(dict(query_id=qid, task=left['task'], target=left['target'],
            current_response=left['response'], baseline_response=right['response'],
            current_correct=a, baseline_correct=b, outcome=bucket,
            current_tokens_per_head=left['budget_audit']['actual_visual_slots'] / 112,
            baseline_tokens_per_head=right['budget_audit']['actual_visual_slots'] / 112,
            current_seconds=left['seconds'], baseline_seconds=right['seconds']))
    return dict(comparison_scope='completed-question intersection', controlled_resize_only=controlled,
        matched_questions=len(common), input_identity_checks_passed=True,
        current=score(current[qid] for qid in common), baseline=score(baseline[qid] for qid in common),
        paired_by_task=counts, current_only=sum(c['current_only'] for c in counts.values()),
        baseline_only=sum(c['baseline_only'] for c in counts.values())), table


def pct(value):
    return '—' if value is None else f'{100 * value:.2f}%'


def generate(root, baseline_root=DEFAULT_NATIVE, dynamic_root=DEFAULT_DYNAMIC, *, allow_partial=False):
    root = Path(root)
    current, baseline = load_run(root), load_run(baseline_root)
    if len(current) < 1468 and not allow_partial:
        raise ValueError('Incomplete run; use --allow-partial for completed-question reporting')
    if len(current) > 1468 or any(qid not in baseline for qid in current):
        raise ValueError('New run contains unexpected questions')
    if len(baseline) != 1468 or Counter(r['task'] for r in baseline.values()) != COUNTS:
        raise ValueError('Controlled baseline must be the completed nine-task 1468-question run')
    protocol_checks = validate_protocol(root, baseline_root)
    native, table = compare(current, baseline, controlled=True)
    output = dict(version='simplestream_fixed896_comparison_v1', done=len(current), total=1468,
        complete=len(current) == 1468,
        scoring='official nine-task raw case-sensitive target-in-response, no extraction',
        partial_macro_note='Macro averages are null until all 9 tasks are represented; partial means are not full results.',
        current_root=str(root.resolve()), controlled_baseline_root=str(Path(baseline_root).resolve()),
        protocol_checks=protocol_checks,
        current=score(current.values()), controlled_native_same_fps=native,
        controlled_baseline_full=score(baseline.values()))
    if dynamic_root is not None and Path(dynamic_root).exists():
        dynamic = load_run(dynamic_root)
        contextual, _ = compare(current, dynamic, controlled=False)
        output['dynamic_fps1_background'] = dict(root=str(Path(dynamic_root).resolve()),
            note='Background only: FPS, sampling and resize differ; cannot attribute difference to resize alone.',
            full=score(dynamic.values()), matched=contextual)
    text = [f'# 固定896 SimpleStream 一帧／四帧 K500：{len(current)}/1468题', '',
        '官方原始回答按目标字母是否出现评分。以下两列只比较已完成的相同题目；未完成任务记为“—”。', '',
        '| 任务 | 已完成题数 | 固定896 | fps=0.5原分辨率 | 当前单独答对 | 基线单独答对 |',
        '|---|---:|---:|---:|---:|---:|']
    for task in TASKS:
        a, b, pair = native['current']['per_task'][task], native['baseline']['per_task'][task], native['paired_by_task'][task]
        text.append(f'| {task} | {a["total"]}/{COUNTS[task]} | {pct(a["accuracy"])} | {pct(b["accuracy"])} | {pair["current_only"]} | {pair["baseline_only"]} |')
    for label, metric in [('逐题正确率', 'question_accuracy'), ('九类等权均分', 'nine_task_macro_average'),
                          ('官方两组等权均分', 'official_two_group_macro_average')]:
        text.append(f'| {label} | {len(current)}/1468 | {pct(native["current"][metric])} | {pct(native["baseline"][metric])} | | |')
    text += ['', f'实际视觉预算：当前 {native["current"]["actual_visual_tokens_per_head_mean"]} token/头；'
                 f'同题基线 {native["baseline"]["actual_visual_tokens_per_head_mean"]} token/头。', '',
             '此处 K500 指回答前保留的视觉 KV；完整视觉 prefill 的临时缓存和文本 token 另计。', '',
             '受控对照逐题核查了原视频、采样索引、时间戳、问题哈希、逐图编码、位置实现、分支与选择器设置。'
             '缩放改变视觉网格，因此具体空间位置编号和候选数随之变化。', '',
             'fps=1动态缩放版仅作背景，不能据此单独归因分辨率。详细数据见 comparison_summary.json。']
    root.mkdir(parents=True, exist_ok=True)
    (root / 'comparison_summary.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    (root / 'comparison_report.md').write_text('\n'.join(text) + '\n')
    buffer = io.StringIO()
    if table:
        writer = csv.DictWriter(buffer, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    (root / 'comparison_per_query.csv').write_text(buffer.getvalue())
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--baseline-root', type=Path, default=DEFAULT_NATIVE)
    parser.add_argument('--dynamic-root', type=Path, default=DEFAULT_DYNAMIC)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    result = generate(args.output_root, args.baseline_root, args.dynamic_root, allow_partial=args.allow_partial)
    print(json.dumps(dict(done=result['done'], total=result['total'], complete=result['complete'])))


if __name__ == '__main__':
    main()
