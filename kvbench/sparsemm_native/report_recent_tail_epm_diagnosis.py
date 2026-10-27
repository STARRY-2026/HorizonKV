"""CPU-only Chinese report for the deliberately selected recent-tail EPM probes."""

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
from statistics import mean

from sparsemm_native.run_ovobench import official_correct


DEFAULT_ROOT = (Path(__file__).resolve().parents[1] / 'results' /
                'ovobench_epm_current_diagnosis_20261026')
BASELINE = 'native_pseudo_qpos'
LABELS = {
    BASELINE: '当前原配置：视频位置＋伪查询 K500',
    'native_pseudo_token_end': '视频位置＋探针从视觉 token 末尾起步',
    'native_uniform': '视频位置＋均匀 K500',
    'native_unpruned': '视频位置＋不裁剪（超 K500，仅归因）',
    'simplestream_contiguous_pseudo_qpos': '旧位置布局＋伪查询同问题起点',
    'simplestream_contiguous_pseudo_token_end': '旧位置布局＋探针从视觉 token 末尾起步',
    'simplestream_contiguous_uniform': '旧位置布局＋均匀 K500',
    'compact_native_pseudo_qpos': '缩至旧尺寸＋视频位置＋伪查询 K500',
    'compact_native_uniform': '缩至旧尺寸＋视频位置＋均匀 K500',
}
GROUPS = {
    'old_correct_current_wrong': '旧对、当前错（预选12题）',
    'both_correct': '两版都对（预选4题）',
    'both_wrong': '两版都错（预选4题）',
}


def load_json(path):
    return json.loads(path.read_text())


def checked_correct(task, response, target, stored=None, context=''):
    result = bool(official_correct(task, response, target))
    if stored is not None and (not isinstance(stored, bool) or stored != result):
        raise ValueError(f'Raw-response score differs from stored bool: {context}')
    return result


def category(old, current):
    return ('old_correct_current_wrong' if old and not current else
            'both_correct' if old and current else
            'both_wrong' if not old and not current else 'current_only_correct')


def tokens(arm):
    return arm['budget_audit']['actual_visual_slots'] / 112


def arm_stats(rows, name):
    selected = [row for row in rows if name in row['arms']]
    if not selected:
        return None
    arms = [row['arms'][name] for row in selected]
    bases = [row['arms'][BASELINE] for row in selected]
    fixed = [row['query_id'] for row, arm, base in zip(selected, arms, bases)
             if not base['correct'] and arm['correct']]
    broke = [row['query_id'] for row, arm, base in zip(selected, arms, bases)
             if base['correct'] and not arm['correct']]
    return dict(questions=len(arms), correct=sum(arm['correct'] for arm in arms),
                accuracy=mean(arm['correct'] for arm in arms),
                fixed_wrong=len(fixed), broke_correct=len(broke),
                changed_answer=sum(arm['response'] != base['response']
                                   for arm, base in zip(arms, bases)),
                mean_tokens_per_head=mean(tokens(arm) for arm in arms),
                min_tokens_per_head=min(tokens(arm) for arm in arms),
                max_tokens_per_head=max(tokens(arm) for arm in arms),
                fixed_ids=fixed, broke_ids=broke)


def validate(rows, audit, saved_summary):
    per_case = audit['per_case']
    metadata = ({str(item['query_id']): item for item in per_case}
                if isinstance(per_case, list) else per_case)
    ids = [str(row['query_id']) for row in rows]
    if len(ids) != len(set(ids)) or not set(ids) <= set(metadata):
        raise ValueError('Duplicate or unselected diagnostic query')
    uniform_checked = pseudo_checked = bool_checked = 0
    for row in rows:
        qid, task, target = str(row['query_id']), row['task'], row['target']
        meta = metadata[qid]
        if task != 'EPM' or target != meta['target']:
            raise ValueError(f'Question task/target mismatch: {qid}')
        old = checked_correct(task, row['old_response'], target,
                              meta.get('old_correct'), f'{qid}:old')
        current = checked_correct(task, row['reference_response'], target,
                                  meta.get('current_correct'), f'{qid}:current')
        if (row['old_response'] != meta['old_response']
                or row['reference_response'] != meta['current_response']
                or meta['category'] != category(old, current)):
            raise ValueError(f'Question reference response/category mismatch: {qid}')
        row['category'] = meta['category']
        for name, arm in row['arms'].items():
            if name not in LABELS:
                raise ValueError(f'Unknown diagnosis arm: {name}')
            if 'correct' not in arm:
                raise ValueError(f'Missing stored correctness: {qid}:{name}')
            arm['correct'] = checked_correct(task, arm['response'], target,
                                             arm['correct'], f'{qid}:{name}')
            bool_checked += 1
            if name != 'native_unpruned' and tokens(arm) > 500 + 1e-9:
                raise ValueError(f'K500 budget exceeded: {qid}:{name}')
            if arm['budget_audit'].get('actual_history_visual_slots', 0) != 0:
                raise ValueError(f'Unexpected historical visual KV: {qid}:{name}')
        baseline = row['arms'][BASELINE]
        if (baseline['response'] != row['reference_response']
                or not all(baseline.get('baseline_checks', {}).values())):
            raise ValueError(f'Current answer could not be reproduced: {qid}')
        native = row['arms'].get('native_uniform')
        legacy = row['arms'].get('simplestream_contiguous_uniform')
        if native and legacy:
            for key in ('recent_selection_sha256', 'actual_visual_slots'):
                if native['budget_audit'][key] != legacy['budget_audit'][key]:
                    raise ValueError(f'Uniform position-only comparison changed {key}: {qid}')
            uniform_checked += 1
        for layout in ('native', 'simplestream_contiguous'):
            first = row['arms'].get(layout + '_pseudo_qpos')
            second = row['arms'].get(layout + '_pseudo_token_end')
            if first and second:
                for key in ('one_positions_sha256', 'four_positions_sha256',
                            'question_position'):
                    if first['budget_audit'][key] != second['budget_audit'][key]:
                        raise ValueError(f'Probe-only comparison changed {key}: {qid}:{layout}')
                pseudo_checked += 1
    stats = {name: arm_stats(rows, name) for name in LABELS
             if any(name in row['arms'] for row in rows)}
    if saved_summary is not None:
        if saved_summary['questions'] != len(rows) or set(saved_summary['arms']) != set(stats):
            raise ValueError('Saved summary question/arm coverage mismatch')
        for name, summary in stats.items():
            for key, stored in saved_summary['arms'][name].items():
                if key not in summary:
                    continue
                actual = summary[key]
                matches = (math.isclose(actual, stored, rel_tol=1e-10, abs_tol=1e-10)
                           if isinstance(actual, float) else actual == stored)
                if not matches:
                    raise ValueError(f'Saved summary mismatch: {name}:{key}')
    return metadata, stats, dict(score_bools=bool_checked, uniform_pairs=uniform_checked,
                                pseudo_pairs=pseudo_checked)


def escaped(value):
    return str(value).replace('|', '\\|').replace('\n', '<br>')


def score(value):
    return f"{value['correct']}/{value['questions']}（{value['accuracy']:.2%}）" if value else '—'


def make_report(root):
    root = Path(root)
    rows = [json.loads(line) for line in (root / 'answers.jsonl').read_text().splitlines()
            if line.strip()]
    if not rows:
        raise ValueError('No completed diagnostic answers yet')
    audit = load_json(root / 'selection_audit.json')
    saved = load_json(root / 'summary.json') if (root / 'summary.json').exists() else None
    metadata, stats, checks = validate(rows, audit, saved)
    names = list(stats)
    selected = audit.get('selected_category_counts', dict(Counter(
        meta['category'] for meta in metadata.values())))
    completed = Counter(row['category'] for row in rows)
    lines = [
        '# 近期控制版 EPM 单因素排查', '',
        f'已完成 {len(rows)}/{len(metadata)} 道诊断题。原全量实验保持暂停。', '',
        '**这是刻意富集退步题的诊断集，不能将这里的正确率当作 EPM 全量准确率。** '
        '预选组成是旧对当前错12题、两版都对4题、两版都错4题；'
        '本报告仅用同题改答和恢复情况定位原因。', '',
        '所有正确性均从完整原始回答重新调用 `official_correct` 计算；'
        f"已核对 {checks['score_bools']} 个实验分支的存储布尔值、旧/当前参考结果"
        + ('及运行器 summary.json。' if saved is not None else '；summary.json 尚未产出。'), '',
        '## 实验边界', '',
        '- 所有分支使用 fps=0.5、相同问题与选项、官方提示词、相同头分组和问答解码器，历史 KV 为0。',
        '- 高分辨率各分支共享同一批 RGB、采样位置及视觉特征。`native_uniform` 与 '
        '`simplestream_contiguous_uniform` 只改位置编号；'
        f"已核对 {checks['uniform_pairs']} 题的选择索引哈希与实际预算完全相同。",
        '- `pseudo_token_end` 在相同视觉位置和问题起点下，只把固定伪查询起点改为 '
        '`prefix + 四帧分支候选 token 数`；'
        f"已核对 {checks['pseudo_pairs']} 对位置/问题起点一致。",
        '- “旧位置布局”只把当前视频时间单元放入旧单 vision_start 连续前缀的位置布局：'
        '首个 unit 为三维位置，后续视觉 token 为一维连续位置。'
        '它仍使用当前两帧融合视频特征，不等于恢复旧逐图编码。',
        '- `compact` 保持同 RGB 与采样，只缩至该题旧运行记录的 `old_resize_hw`，'
        '继续使用当前两帧融合视频编码；它没有恢复旧 fps=1、旧视频文件或旧采样。',
        '- `native_unpruned` 保留全部候选，实际缓存可超过 K500，**仅用于归因，不能作为公平预算成绩**。',
        '- 下述“恢复/损坏/改答”均相对于当前原配置：错→对 / 对→错 / 原始回答字符串发生变化。', '',
        '## 配对结果与实际预算', '',
        '| 分支 | 正确数/题数（正确率） | 恢复 | 损坏 | 改答 | 平均视觉 token/头 | 范围 |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for name, value in stats.items():
        lines.append(f"| {LABELS[name]} | {score(value)} | {value['fixed_wrong']} | "
                     f"{value['broke_correct']} | {value['changed_answer']} | "
                     f"{value['mean_tokens_per_head']:.2f} | "
                     f"{value['min_tokens_per_head']:.2f}～{value['max_tokens_per_head']:.2f} |")
    lines += ['', '预算只统计回答前已保留的视觉 KV，单位为112个KV头的平均 token 数；'
              '文字前缀、真实问题、生成 token 和临时完整 prefill 不计入该值。', '',
              '## 按预选组分层', '',
              '| 分支 | ' + ' | '.join(GROUPS.values()) + ' |',
              '|---|' + '---:|' * len(GROUPS)]
    for name in names:
        values = [score(arm_stats([row for row in rows if row['category'] == group], name))
                  for group in GROUPS]
        lines.append('| ' + LABELS[name] + ' | ' + ' | '.join(values) + ' |')
    lines += ['', '各组已完成/预选：' + '；'.join(
        f'{label} {completed[group]}/{selected.get(group, 0)}' for group, label in GROUPS.items()) + '。', '',
        '## 相对当前原配置的恢复与损坏题', '']
    for name in names:
        if name == BASELINE:
            continue
        value = stats[name]
        lines.append(f"- {LABELS[name]}：恢复 {', '.join(value['fixed_ids']) or '无'}；"
                     f"损坏 {', '.join(value['broke_ids']) or '无'}。")
    lines += ['', '## 逐题原始回答', '',
              '回答保留原始大小写与完整内容；换行在表格中显示为 `<br>`。', '',
              '| 题号 | 分组 | 标准答案 | 旧版 | 当前参考 | ' +
              ' | '.join(LABELS[name] for name in names) + ' |',
              '|---|' + '---|' * (4 + len(names))]
    csv_rows = []
    for row in rows:
        cells = [row['query_id'], GROUPS.get(row['category'], row['category']),
                 row['target'], row['old_response'], row['reference_response']]
        cells += [row['arms'][name]['response'] if name in row['arms'] else '—' for name in names]
        lines.append('| ' + ' | '.join(escaped(cell) for cell in cells) + ' |')
        record = dict(query_id=row['query_id'], category=row['category'], target=row['target'],
                      old_response=row['old_response'], current_response=row['reference_response'])
        for name in names:
            arm = row['arms'].get(name)
            record.update({name + '_response': arm['response'] if arm else '',
                           name + '_correct': arm['correct'] if arm else '',
                           name + '_actual_tokens_per_head': tokens(arm) if arm else ''})
        csv_rows.append(record)
    (root / 'diagnosis_report.md').write_text('\n'.join(lines) + '\n')
    with (root / 'per_query.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    return dict(questions=len(rows), arms=stats, checks=checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    result = make_report(args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
