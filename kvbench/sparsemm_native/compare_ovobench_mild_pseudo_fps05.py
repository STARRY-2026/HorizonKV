"""Compare the nine-task native fps=0.5 run on exactly paired OVO queries."""

import argparse
import json
from pathlib import Path

from sparsemm_native import run_ovobench as official


ROOT = Path(__file__).resolve().parents[1]
ARM = 'mild_pseudo_history_native_fps05_k500'
CONTROLS = {
    'SparseMM': ROOT / 'results/ovobench_hermes9_k500_fps05_p151200_20261019/native',
    '旧DP': ROOT / 'results/ovobench_hermes9_k500_fps05_p151200_20261019/dp',
    '旧近期四帧DP': ROOT / 'results/ovobench_recent4_official_full_k500_fps05_p151200_20261021/dp_recent4',
}


def rows(path):
    data = [json.loads(line) for line in path.open() if line.strip()]
    by_id = {str(row['query_id']): row for row in data}
    if len(data) != len(by_id):
        raise ValueError(f'Duplicate IDs in {path}')
    return data, by_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    arm_dir = args.output_root.resolve() / ARM
    new_rows, new = rows(arm_dir / 'queries.jsonl')
    new_protocol = json.loads((arm_dir / 'protocol.json').read_text())
    if new_protocol['fps'] != 0.5 or new_protocol['video_max_pixels'] != 151200:
        raise ValueError('New run changed the paired video input')
    arms = {ARM: new}
    for name, path in CONTROLS.items():
        protocol = json.loads((path / 'protocol.json').read_text())
        if protocol['fps'] != 0.5 or protocol['video_max_pixels'] != 151200:
            raise ValueError(f'{name} input protocol differs')
        if protocol['model_config_sha256'] != new_protocol['model_config_sha256']:
            raise ValueError(f'{name} model config differs')
        _, controls = rows(path / 'queries.jsonl')
        arms[name] = controls
    ids = [str(row['query_id']) for row in new_rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate IDs in new run')
    excluded = []
    rescoring = {}
    for name, control in arms.items():
        rescoring[name] = []
        if not set(ids).issubset(control):
            raise ValueError(f'{name} lacks new run IDs')
        for query_id in ids:
            actual, expected = control[query_id], new[query_id]
            for key in ('source_id', 'probe_id', 'task', 'group', 'target', 'frames'):
                if actual[key] != expected[key]:
                    if (query_id == '1303' and key == 'target'
                            and name in ('SparseMM', '旧DP')
                            and actual[key] == 'A' and expected[key] == 'D'):
                        excluded.append(query_id)
                        continue
                    raise ValueError(f'{name} {query_id}: {key} differs')
            for key in ('timestamp', 'clip_end'):
                if abs(float(actual[key]) - float(expected[key])) > 1e-7:
                    raise ValueError(f'{name} {query_id}: {key} differs')
            if actual.get('prompt_sha256') and actual['prompt_sha256'] != expected['prompt_sha256']:
                raise ValueError(f'{name} {query_id}: prompt differs')
            correct = official.official_correct(actual['task'], actual['response'], actual['target'])
            if correct != bool(actual['correct']):
                rescoring[name].append(dict(query_id=query_id, task=actual['task'],
                    response=actual['response'], target=actual['target'],
                    stored_correct=actual['correct'], official_correct=correct))
            # Keep the original files intact. Every table below uses a copy
            # rescored from the raw response, never a legacy correctness flag.
            control[query_id] = dict(actual, correct=correct)
    ids = [query_id for query_id in ids if query_id not in set(excluded)]
    tasks = sorted({new[query_id]['task'] for query_id in ids})
    lines = ['# OVO-Bench fps=0.5 最终对比', '',
             '按官方规则从原始回答统一重评分。覆盖九类回溯/即时任务，不含 REC、SSR、CRR。',
             '九类两组均分：先分别对回溯三类、即时六类的准确率取平均，再将两组等权平均；不等于十二类全量官方总分。', '',
             '## 同官方提示词与帧处理的全量对照', '',
             '新方法与旧近期四帧 DP 的提示词哈希逐题相同，采样帧数、模型配置、fps=0.5、151200 像素上限已核对。', '',
             '| 方法 | 正确数/题数 | 逐题正确率 | 九类两组均分 |',
             '|---|---:|---:|---:|']
    full_ids = [str(row['query_id']) for row in new_rows]
    for name in (ARM, '旧近期四帧DP'):
        selected = [arms[name][query_id] for query_id in full_ids]
        score = official.summary(selected, 1468)
        grouped = score['hermes_category_average']
        label = '新方法' if name == ARM else name
        lines.append(f'| {label} | {sum(int(r["correct"]) for r in selected)}/{len(full_ids)} | '
                     f'{score["question_accuracy"]:.2%} | '
                     f'{format(grouped, ".2%") if grouped is not None else "尚未覆盖全部九类"} |')
    lines.extend(['', '| 题型 | 题数 | 新方法 | 旧近期四帧DP | 差值（百分点） |',
                  '|---|---:|---:|---:|---:|'])
    strict_scores = {name: official.summary([arms[name][query_id] for query_id in full_ids], 1468)
                     for name in (ARM, '旧近期四帧DP')}
    for task in tasks:
        count = sum(new[query_id]['task'] == task for query_id in full_ids)
        current = strict_scores[ARM]['task_accuracy'][task]
        previous = strict_scores['旧近期四帧DP']['task_accuracy'][task]
        lines.append(f'| {task} | {count} | {current:.2%} | {previous:.2%} | '
                     f'{100*(current-previous):+.2f} |')
    lines.extend(['', '## 与旧 SparseMM / DP 的参考比较', '',
                  '**旧 SparseMM 和旧 DP 使用 HERMES 提示词；以下虽统一重评分并对齐采样帧，仍存在提示词差异，不能视为严格受控比较。**', '',
                  f'共同题数 {len(ids)}。', ''])
    if excluded:
        lines.append('')
        lines.append('排除题目 1303：旧 SparseMM/DP 使用旧版选项与答案 A；修订版标注与新方法使用答案 D。')
        lines.append('')
    lines.extend(['| 方法 | 同题正确数/题数 | 逐题正确率 | 九类两组均分 |',
                  '|---|---:|---:|---:|'])
    all_scores = {}
    for name, control in arms.items():
        selected = [control[query_id] for query_id in ids]
        score = official.summary(selected, 1468)
        all_scores[name] = score
        grouped = score['hermes_category_average']
        label = '新方法' if name == ARM else name
        lines.append(f'| {label} | {sum(int(r["correct"]) for r in selected)}/{len(ids)} | '
                     f'{score["question_accuracy"]:.2%} | '
                     f'{format(grouped, ".2%") if grouped is not None else "尚未覆盖全部九类"} |')
    lines.extend(['', '| 题型 | 题数 | SparseMM | 旧DP | 旧近期四帧DP | 新方法 |',
                  '|---|---:|---:|---:|---:|---:|'])
    for task in tasks:
        count = sum(new[query_id]['task'] == task for query_id in ids)
        values = [all_scores[name]['task_accuracy'][task] for name in
                  ('SparseMM', '旧DP', '旧近期四帧DP', ARM)]
        lines.append(f'| {task} | {count} | ' + ' | '.join(f'{v:.2%}' for v in values) + ' |')
    if full_ids:
        actual = [new[query_id]['budget_audit']['actual_visual_slots'] / 112 for query_id in full_ids]
        history = [new[query_id]['budget_audit']['actual_history_visual_slots'] / 112 for query_id in full_ids]
        recent = [new[query_id]['budget_audit']['actual_recent_visual_slots'] / 112 for query_id in full_ids]
        lines.extend(['', f'新方法实际视觉 KV：平均 {sum(actual)/len(actual):.1f}/头，'
                      f'其中历史 {sum(history)/len(history):.1f}、近期 {sum(recent)/len(recent):.1f}；'
                      f'范围 {min(actual):.1f}–{max(actual):.1f}/头。',
                      '配置上限与实际占用不同；准确率表只保证输入帧处理和配置上限对齐，'
                      '不将实际 KV 占用表述为完全相等。'])
    lines.extend(['', '重评分变化（原始结果文件保持不变）：' + '；'.join(
        f'{name} {len(changes)}题' for name, changes in rescoring.items()) + '。',
        '官方选择题规则为目标字母是否出现在回答中；这与旧 HERMES 的选项字母提取规则不同。'])
    (arm_dir / 'official_rescoring_changes.json').write_text(
        json.dumps(rescoring, indent=2, ensure_ascii=False) + '\n')
    (arm_dir / 'final_comparison_metrics.json').write_text(json.dumps(dict(
        full_query_count=len(full_ids), strict_official_prompt_comparison=strict_scores,
        reference_common_query_count=len(ids), excluded_reference_query_ids=sorted(set(excluded)),
        reference_prompt_warning='SparseMM and old DP used HERMES prompts; not a controlled comparison',
        reference_scores=all_scores,
        rescoring_changes={name: len(changes) for name, changes in rescoring.items()},
        actual_mean_visual_tokens_per_head=sum(actual)/len(actual) if full_ids else None),
        indent=2, ensure_ascii=False) + '\n')
    result = arm_dir / 'paired_comparison.md'
    result.write_text('\n'.join(lines) + '\n')
    print(result)


if __name__ == '__main__':
    main()
