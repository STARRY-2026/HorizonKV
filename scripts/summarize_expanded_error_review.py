"""Validate and summarize 33 reused + 87 newly inspected DP failure cases.

This aggregates manual observations; it does not automatically infer causes.
"""
import csv
import hashlib
import html
import json
from collections import Counter
from pathlib import Path

AUDIT = Path('/home/yujun/program/hermes/results/myobservations/dp_failure_audit_20261010')
BASE = AUDIT / 'expanded_review_120'
OLD = AUDIT / 'absolute_errors'
GROUPS = {
    'sampling_gap': '局部采样缺口',
    'count_state': '累计计数或事件状态更新候选',
    'motion': '动作阶段、方向或快速运动候选',
    'detail_binding': '细节识别或实体关系绑定候选',
    'causal': '因果关联候选',
    'annotation': '标注、选项或参照系疑点',
    'prospective': '未来预测不确定性',
    'insufficient': '当前审阅证据不足',
}
STATUSES = {
    'absent_entire_input': '完整采样前缀缺少关键视觉证据',
    'local_sampling_gap': '截点附近关键画面未采到',
    'present_in_sampled_input': '已在实际采样帧观察到支持gold的相关证据',
    'uncertain': '未决或未完成同等强度的采样核对',
}
JOBS = {'action_clips': 13, 'text_attribute_spatial': 26,
        'causal_object_event': 23, 'counting_prospective': 25}


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def key(row):
    return int(row['video_id']), int(row['question_index'])


def cid(row):
    return 'v%d/q%d' % key(row)


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def count(rows, field):
    return dict(Counter(r[field] for r in rows))


def subset_stats(rows):
    return dict(n=len(rows), primary_groups=count(rows, 'primary_group_code'),
                input_evidence_status=count(rows, 'input_evidence_status'),
                review_scope=count(rows, 'review_scope'), case_ids=[cid(r) for r in rows])


def prior_reviews():
    old_reviews = json.loads((OLD / 'manual_case_review.json').read_text())['cases']
    old_counts = json.loads((OLD / 'cause_count_status.json').read_text())
    group_by_id = {}
    for group_code, spec in zip(GROUPS, old_counts['diagnostic_groups']):
        assert GROUPS[group_code] == spec['diagnostic_group']
        for case_id in spec['case_ids']:
            group_by_id[case_id] = group_code
    result = []
    # Other reused observations remain valid, but no new exact-PTS audit is
    # inferred from source-time screenshots. Keep this evidence axis conservative.
    explicit_present = {(11, 2), (56, 3), (124, 2)}
    for r in old_reviews:
        k = key(r)
        group = group_by_id[cid(r)]
        status = 'uncertain'
        if group == 'sampling_gap':
            status = 'absent_entire_input' if k == (173, 0) else 'local_sampling_gap'
        elif k in explicit_present:
            status = 'present_in_sampled_input'
        paths = [str(p) for p in (OLD / 'frames').glob('exact_v%d_q%d_*.jpg' % k)]
        if k == (11, 2):
            paths += [str(p) for p in (OLD / 'frames').glob('v11_model_prefix196_sheet*.jpg')]
        result.append(dict(video_id=k[0], question_index=k[1],
                           primary_group_code=group, input_evidence_status=status,
                           review_scope='entire_sampled_prefix' if k == (11, 2) else 'prior_review_reused',
                           observations=r['evidence'], interpretation_and_limit=r['interpretation_and_limit'],
                           evidence_status_note='沿用旧审阅；除明确全输入/精确采样例，其余本轴保守列未决，不否定原视觉观察。',
                           viewed_sheet_paths=paths, kv_retention_verified=False,
                           causal_intervention_run=False, review_batch='prior33'))
    assert len(result) == 33
    return result


def main():
    selected = read_csv(BASE / 'selected120.csv')
    new_selected = read_csv(BASE / 'new87.csv')
    source = {key(r): r for r in read_csv(AUDIT / 'case_ledger.csv')}
    assert len(selected) == len({key(r) for r in selected}) == 120
    assert len(new_selected) == len({key(r) for r in new_selected}) == 87
    assert len({r['video_id'] for r in selected}) == 100
    for r in selected:
        s = source[key(r)]
        for field in ['source_row', 'video_id', 'question_index', 'question', 'choices', 'answer',
                      'task', 'end_time', 'dp1000_answer', 'hermes1000_answer', 'lu1000_answer']:
            assert r[field] == s[field], (cid(r), field)
        assert r['dp1000_correct'] == '0'
    new_reviews = []
    manifests = []
    for job, expected in JOBS.items():
        part = json.loads((BASE / ('reviews_' + job + '.json')).read_text())
        assert isinstance(part, list) and len(part) == expected, job
        assert {key(r) for r in part} == {key(r) for r in read_csv(BASE / (job + '.csv'))}, job
        for r in part:
            assert r['primary_group_code'] in GROUPS, r
            assert r['input_evidence_status'] in STATUSES, r
            assert r['kv_retention_verified'] is False
            assert r['causal_intervention_run'] is False
            assert r['observations'] and r['interpretation_and_limit']
            assert r['viewed_sheet_paths'], cid(r)
            directory = BASE / ('frames_' + job) / ('v%d_q%d' % key(r))
            for path in r['viewed_sheet_paths']:
                p = Path(path)
                assert p.is_absolute() and p.is_file(), (cid(r), path)
                assert directory in p.parents, (cid(r), path)
            manifest = json.loads((directory / 'manifest.json').read_text())
            assert key(manifest) == key(r)
            s = source[key(r)]
            assert manifest['question'] == s['question']
            assert manifest['choices'] == json.loads(s['choices'])
            assert manifest['gold'] == s['answer']
            assert manifest['end_time'] == float(s['end_time'])
            for frame in manifest['records']:
                assert Path(frame['image_path']).is_file()
                # Extra SOURCE frames must be legally at/before the question.
                if not frame['experiment_input_frame']:
                    assert frame['pts_seconds'] <= manifest['end_time'] + 1e-6
                else:
                    assert frame['frame_index'] == frame['sample_ordinal'] * manifest['stride_frames']
            if r['review_scope'] == 'entire_sampled_prefix':
                viewed = set(r['viewed_sheet_paths'])
                covered = {i for sheet in manifest['sheets'] if sheet['path'] in viewed
                           for i in sheet['frame_indices']}
                covered.update(frame['frame_index'] for frame in manifest['records']
                               if frame['image_path'] in viewed)
                expected_frames = {i * manifest['stride_frames']
                                   for i in range(manifest['actual_sample_count'])}
                assert expected_frames <= covered, (cid(r), expected_frames - covered)
            r['review_batch'] = 'new87'
            r['reviewer_job'] = job
            r['last_input_pts'] = manifest['last_input_pts']
            r['manifest_path'] = str(directory / 'manifest.json')
            r['sampled_frames_after_cut'] = manifest['sampled_frames_after_cut']
            manifests.append(manifest)
        new_reviews.extend(part)
    assert {key(r) for r in new_reviews} == {key(r) for r in new_selected}
    all_reviews = prior_reviews() + new_reviews
    assert len(all_reviews) == len({key(r) for r in all_reviews}) == 120
    assert {key(r) for r in all_reviews} == {key(r) for r in selected}
    reviews_by_key = {key(r): r for r in all_reviews}
    rows = []
    for s in selected:
        r = dict(s)
        r.update(reviews_by_key[key(s)])
        r['diagnostic_group'] = GROUPS[r['primary_group_code']]
        r['input_evidence_label'] = STATUSES[r['input_evidence_status']]
        rows.append(r)
    new = [r for r in rows if r['review_batch'] == 'new87']
    prior = [r for r in rows if r['review_batch'] == 'prior33']
    common = [r for r in rows if r['outcome_H_L_D'] == '000']
    differential = [r for r in rows if r['outcome_H_L_D'] != '000']
    h_regressions = [r for r in rows if r['hermes1000_correct'] == '1']
    l_regressions = [r for r in rows if r['lu1000_correct'] == '1']
    assert len(common) == 86 and len(differential) == 34
    assert len(h_regressions) == 25 and len(l_regressions) == 20
    all_differential = {key(r) for r in source.values()
                        if r['dp1000_correct'] == '0' and r['outcome_H_L_D'] != '000'}
    assert all_differential == {key(r) for r in differential}
    summary = dict(
        scope=dict(all_dp_errors=369, reviewed=120, prior_reused=33, newly_reviewed=87,
                   unique_videos=100, tasks=10, error_coverage=120 / 369,
                   common_errors_reviewed=86, differential_errors_reviewed=34,
                   all_hermes_correct_dp_wrong_covered=25, all_lu_correct_dp_wrong_covered=20,
                   differential_overlap_h_and_l=11,
                   selected_purposively=True, causal_attributions=False,
                   question_index_base=0),
        definitions=dict(diagnostic_groups=GROUPS, input_evidence_status=STATUSES,
                         present_does_not_mean='相关gold支持证据进入原始采样；不表示完整充分、不表示视觉预处理后可读、不表示最终KV保留。',
                         absent_entire_input_does_not_mean='缺少关键答题视觉片段；不表示输入中无任何像素或没有可供猜测的上下文。',
                         categories='每题只分一个主要诊断组；多个机制可以共存。',
                         prior_status='旧33题的输入状态只对明确精确采样/全输入案例下判断，其余保守未决；新87题全部使用实际采样与合法SOURCE对照。'),
        total=subset_stats(rows), prior33=subset_stats(prior), new87=subset_stats(new),
        common=subset_stats(common), differential=subset_stats(differential),
        hermes_correct_dp_wrong=subset_stats(h_regressions),
        lu_correct_dp_wrong=subset_stats(l_regressions),
        task_counts=count(rows, 'task'),
        group_cases={g: [cid(r) for r in rows if r['primary_group_code'] == g] for g in GROUPS},
        evidence_cases={s: [cid(r) for r in rows if r['input_evidence_status'] == s] for s in STATUSES},
        extraction=dict(new_case_manifests=len(manifests),
                        decoded_case_frame_records=sum(len(m['records']) for m in manifests),
                        generated_contact_sheets=sum(len(m['sheets']) for m in manifests),
                        viewed_image_references=sum(len(r['viewed_sheet_paths']) for r in new),
                        full_sampled_prefix_coverage_verified=sum(r['review_scope'] == 'entire_sampled_prefix' for r in new),
                        input_frames_after_cut_cases=[cid(m) for m in manifests if m['sampled_frames_after_cut']]),
        causal_status=dict(confirmed_latest_dp_key_evidence_deleted_count=None,
                           confirmed_latest_dp_sufficient_kv_but_readout_fails_count=None,
                           newly_run_model_interventions=0,
                           null_means='尚未完成机制分离，不能填0或用诊断组数量代替。'),
        limitations=[
            '120题目的性选择，其中差异错题全覆盖；共同错题样本仍不能代表全部335题的机制频率。',
            '计数题优先审阅完整实际采样前缀；其他题通常只看尾部和分散锚点，长历史因果/事件题增加约8秒间隔概览。逐题保留范围。',
            '原始帧被采到与视觉token最终在DP KV中保留是不同阶段，本轮没有运行full-KV、token恢复或同预算干预。',
            '局部采样缺口不自动证明独立致错，更早前缀可能有重复证据，H答对也可能来自上下文推断。',
            '标注疑点仅表示需复核，不宣称基准答案已被证明错误。',
        ],
        provenance=dict(source_ledger=str(AUDIT / 'case_ledger.csv'),
                        source_ledger_sha256=hashlib.sha256((AUDIT / 'case_ledger.csv').read_bytes()).hexdigest(),
                        selection_protocol=str(BASE / 'selection_protocol.json'),
                        extractor=str(Path(__file__).parent / 'extract_error_review_frames.py')),
    )
    dump(BASE / 'expanded_summary.json', summary)
    dump(BASE / 'expanded_case_reviews.json', rows)
    fields = list(rows[0])
    for r in rows:
        for field in r:
            if field not in fields:
                fields.append(field)
    with (BASE / 'expanded_case_reviews.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in r.items()})
    render_report(rows, summary)
    print(json.dumps({k: summary[k] for k in ['scope', 'total', 'new87', 'differential',
                                            'hermes_correct_dp_wrong', 'extraction']}, ensure_ascii=False, indent=2))


def render_report(rows, summary):
    esc = html.escape
    def a(path, label):
        rel = str(Path(path).relative_to(BASE)) if BASE in Path(path).parents else str(path)
        return '<a href="%s">%s</a>' % (esc(rel, quote=True), esc(label))
    group_rows = ''.join('<tr><td>%s</td><td>%d</td><td>%d</td><td>%d</td></tr>' %
                         (esc(label), summary['prior33']['primary_groups'].get(code, 0),
                          summary['new87']['primary_groups'].get(code, 0),
                          summary['total']['primary_groups'].get(code, 0))
                         for code, label in GROUPS.items())
    evidence_rows = ''.join('<tr><td>%s</td><td>%d</td><td>%d</td></tr>' %
                            (esc(label), summary['new87']['input_evidence_status'].get(code, 0),
                             summary['differential']['input_evidence_status'].get(code, 0))
                            for code, label in STATUSES.items())
    cards = []
    for r in rows:
        status = r['input_evidence_status']
        group = r['primary_group_code']
        links = []
        for index, path in enumerate(r['viewed_sheet_paths']):
            links.append(a(path, Path(path).name))
        fallback = r['viewed_sheet_paths'][0] if r['review_batch'] == 'new87' else None
        img = next((p for p in r['viewed_sheet_paths'] if Path(p).name == 'tail_1.jpg'), fallback)
        image_html = '<img loading="lazy" src="%s" alt="实际输入与合法截点源帧">' % esc(str(Path(img).relative_to(BASE)), quote=True) if img else ''
        choices = json.loads(r['choices'])
        options = '；'.join('%s: %s' % (chr(65 + i), choice) for i, choice in enumerate(choices))
        meta = '%s | %s | 截点%s秒 | H/L/DP=%s | %s' % (cid(r), r['task'], r['end_time'], r['outcome_H_L_D'], r['review_batch'])
        cards.append('''<article data-group="%s" data-status="%s" data-task="%s" data-diff="%s">
<h3>%s</h3><p class="tags">%s · %s</p><p><b>题目：</b>%s</p><p class="options">%s</p>
<p><b>gold：</b>%s<br><b>DP：</b>%s<br><b>HERMES：</b>%s<br><b>旧LU：</b>%s</p>
<p><b>直接观察：</b>%s</p><p><b>解释与边界：</b>%s</p><p class="scope">审阅范围：%s；KV保留未验证；未做模型干预。</p>
<details><summary>证据图片 / 时间戳清单（%d个已审阅图片引用）</summary><p>%s</p>%s%s</details></article>'''
                     % (group, status, esc(r['task'], quote=True), '1' if r['outcome_H_L_D'] != '000' else '0',
                        esc(meta), esc(GROUPS[group]), esc(STATUSES[status]), esc(r['question']), esc(options),
                        esc(r['answer']), esc(r['dp1000_answer']), esc(r['hermes1000_answer']), esc(r['lu1000_answer']),
                        esc(r['observations']), esc(r['interpretation_and_limit']), esc(r['review_scope']),
                        len(links), ' · '.join(links), image_html,
                        ('<p>' + a(r['manifest_path'], '实际帧索引与PTS manifest') + '</p>') if 'manifest_path' in r else ''))
    def options(mapping):
        return ''.join('<option value="%s">%s</option>' % (esc(k, quote=True), esc(v)) for k, v in mapping.items())
    report = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DP错误审阅：120题 / 100视频</title><style>
body{font:16px/1.65 system-ui,sans-serif;max-width:1180px;margin:auto;padding:30px;color:#233048;background:#f5f7fb}
h1{font-size:29px}h2{font-size:22px}h3{font-size:17px}article,section{background:white;padding:22px;margin:18px 0;border:1px solid #dde3ec;border-radius:10px}
table{border-collapse:collapse;width:100%%;font-variant-numeric:tabular-nums}th,td{padding:9px 12px;border-bottom:1px solid #e5e9f0;text-align:left}td:not(:first-child){text-align:right}
.note{background:#fff5da;padding:16px;border-left:4px solid #d79c23}.tags{color:#315b90;font-weight:600}.scope,.options{font-size:14px;color:#5b6574}img{max-width:100%%;height:auto}a{color:#155bc0}select,input{padding:8px;margin:4px;max-width:95%%}#filters{position:sticky;top:0;background:#edf2f8;padding:10px;z-index:2;box-shadow:0 2px 5px #ddd}details{padding-top:8px}summary{cursor:pointer}
</style><h1>DP错误审阅已扩大至120题，覆盖100个视频</h1>
<p>沿用33题，新增实际看图87题，覆盖全部369道DP错误的32.5%%。包含86道H/L/DP共同错误，以及全部34道H或旧LU正确、DP错误的差异题；其中全部25道H正确DP错、20道LU正确DP错（交集11题）。q编号从0开始。</p>
<p class="note"><b>这是逐题观察与候选诊断数量，不是因果归因比例。</b>目的性样本不能外推369题。原始采样帧有证据不等于最终KV保留，局部缺帧也不自动证明独立致错。本轮模型干预0次，最新DP的“关键证据被删除”与“充分保留但读取失败”数量仍未确定。</p>
<section><h2>主要诊断组：每题一个，合计120</h2><table><thead><tr><th>主要诊断组</th><th>旧33题</th><th>新增87题</th><th>合计120题</th></tr></thead><tbody>%s</tbody></table></section>
<section><h2>实际采样证据状态：单独统计新增87题</h2><p>新增87题均核对INPUT与截点前的合法SOURCE。present仅指看到支持gold的相关输入证据，不能推断证据充分、视觉编码正确或最终KV保留。旧33题的输入状态只沿用明确精确采样/全输入例，其余保守未决，因此不把新旧两批混成同等强度的证据率。</p>
<table><thead><tr><th>证据状态</th><th>新增87题</th><th>全部差异错题34题</th></tr></thead><tbody>%s</tbody></table></section>
<section><h2>新增证据对原假设的修正</h2>
<p><b>计数单位需要先定义。</b>v14/q0、v209/q1连续插入多个电影来源，中途没有回到讲师，gold按一个连续插入块计数。v11/q2反过来有五次独立插入、三个电影来源。v22/q2和v232/q1的多个独立插入在实际输入完整存在；计数单位混淆和累计状态失稳需要分开验证。</p>
<p><b>表格与对象绑定错误有重复实例。</b>v372/q3正确的骁龙8Gen1行58fps/6.3W已采到，DP答其他行48fps/5.1W；v357/q1实际输入A15多核4908，三方法答邻行A14的4317。v274/q3把左侧LOW-TO-HIGH答成右侧HIGH-TO-LOW。</p>
<p><b>截点画面错位跨多种任务出现。</b>v350/q1最后输入58秒仍是切割机，59秒源视频才切到已切好的片材；v282/q1最后238秒仍是拆机，239秒源帧才出现目标显卡表；v378/q0最后24秒是静置材料，25秒才开始旋转球体。</p>
<p><b>输入已有新动作时，也会选旧动作。</b>v384/q4末输入已从除去多余材料转为揭起成品贴片；v422/q4的98秒输入已切到红沙发拿衣物，DP仍选前段调整假发。它们支持动作/事件更新方向，尚不能区分缓存删除与读取干扰。</p>
<p><b>相同K下DP重新分配总缓存预算。</b>“可保留的信息更合适”仍是待检验假设；总槽位相同，修复与回退同时发生。扩大人工审阅不能替代同预算token恢复、合法截点补帧以及full-KV对照。</p></section>
<section><h2>验证优先级</h2><ol><li>对截点缺口做合法cut帧替换/补帧，并保持最终视觉/KV预算，测量净修复而非仅看修复题。</li><li>对已采到的差异错题记录H与DP关键帧/区域token去留；做full-KV、末帧独立QA、等预算证据恢复，分离保留与读取。</li><li>对计数题显式固定“连续插入块”规则，逐事件维护可回溯计数状态，检查同一视频累计预测下降。</li><li>单独复核参照系、选项重叠和动作时间窗口疑点，避免把计分争议当缓存瓶颈。</li></ol></section>
<p><a href="expanded_case_reviews.csv">120题逐题CSV</a> · <a href="expanded_case_reviews.json">完整逐题JSON</a> · <a href="expanded_summary.json">统计与校验摘要</a> · <a href="selection_protocol.json">选题协议</a></p>
<div id="filters"><select id="g"><option value="">所有诊断组</option>%s</select><select id="s"><option value="">所有证据状态</option>%s</select><select id="t"><option value="">所有任务</option>%s</select><select id="d"><option value="">全部题</option><option value="1">差异错题</option><option value="0">共同错题</option></select><input id="q" placeholder="搜索题号、答案或观察"><span id="n"></span></div>
<div id="cases">%s</div><script>
const ids=['g','s','t','d','q'];const els=Object.fromEntries(ids.map(x=>[x,document.getElementById(x)]));
function filter(){let n=0;for(const a of document.querySelectorAll('article')){const yes=(!els.g.value||a.dataset.group===els.g.value)&&(!els.s.value||a.dataset.status===els.s.value)&&(!els.t.value||a.dataset.task===els.t.value)&&(!els.d.value||a.dataset.diff===els.d.value)&&(!els.q.value||a.textContent.toLowerCase().includes(els.q.value.toLowerCase()));a.hidden=!yes;if(yes)n++}document.getElementById('n').textContent=n+' / 120题'};for(const e of Object.values(els))e.addEventListener('input',filter);filter();
</script></html>''' % (group_rows, evidence_rows, options(GROUPS), options(STATUSES),
                       options({task: task for task in sorted(summary['task_counts'])}), ''.join(cards))
    (BASE / 'report.html').write_text(report)


if __name__ == '__main__':
    main()
