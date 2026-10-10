#!/usr/bin/env python3
"""Controlled readout and equal-budget surgery on one matched DP QA cache.

Run with the hermes-qwen Python and --launch. Production inference is unchanged.
"""
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/myobservations/in_information_probe_20261010'
OLD = ROOT / 'results/myobservations/dp_token_frame_study_20261010'
ROI = set([1274, 1275, 1276, 1277, 1278, 1343, 1344, 1345, 1346, 1347,
           1412, 1413, 1414, 1415, 1416])
BLANK = {r * 69 + c for r in range(18, 21) for c in range(5, 10)}
SEED = 20261010
OPEN_PROMPTS = [
    'What is the first word displayed at the start of the video? Return the word only, or UNKNOWN if unreadable.',
    'Transcribe the text shown at the very beginning of the video. Return only the text.',
]


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def launch():
    import glob
    OUT.mkdir(parents=True, exist_ok=False)
    annotation = json.loads((OLD / 'annotation.json').read_text())
    sample = next(x for x in annotation if str(x['video_id']) == '269')
    sample['conversations'] = sample['conversations'][:1]
    write_json(OUT / 'annotation.json', [sample])
    argv = json.loads((OLD / 'dp/command.json').read_text())['argv']
    argv[0:2] = [sys.executable, '-u']
    argv[2] = str(Path(__file__).resolve())
    for flag, value in [('--anno_path', OUT / 'annotation.json'),
                        ('--save_dir', OUT / 'predictions'),
                        ('--ragged_token_trace_path', OUT / 'tokens.jsonl'),
                        ('--ragged_readout_trace_path', OUT / 'readout.jsonl')]:
        argv[argv.index(flag) + 1] = str(value)
    pos = argv.index('--source_manifest_path')
    del argv[pos:pos + 2]
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               CUDA_VISIBLE_DEVICES='0', DECORD_EOF_RETRY_MAX='40960')
    libs = glob.glob(str(Path(sys.prefix) / 'lib/python*/site-packages/nvidia/cu[0-9]*/lib'))
    env['LD_LIBRARY_PATH'] = os.pathsep.join(libs + [env.get('LD_LIBRARY_PATH', '')])
    sources = [Path(__file__).resolve(), ROOT / 'head_analysis/hermes_kv_head_ragged.py',
               ROOT / 'head_analysis/hermes_kv_head_ragged_prefill.py',
               ROOT / 'video_qa/base.py', ROOT / 'video_qa/hermes_vqa.py', OUT / 'annotation.json']
    sources += [Path(argv[argv.index(flag) + 1]) for flag in
                ('--kv_head_budget_scores', '--kv_head_recency_scores')]
    sources += [OLD / 'dp/tokens.jsonl', OLD / 'dp/command.json']
    write_json(OUT / 'command.json', dict(argv=argv, seed=SEED,
        sources_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        reference_command=str(OLD / 'dp/command.json'), prompts=OPEN_PROMPTS,
        design='Same replay, uncompressed before sole compression; same QA positions and history.'))
    print('Running cache experiment; log: ' + str(OUT / 'run.log'), flush=True)
    with (OUT / 'run.log').open('x') as log:
        result = subprocess.run(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise SystemExit('Probe failed; inspect ' + str(OUT / 'run.log'))
    print('Completed: ' + str(OUT / 'results.json'), flush=True)


def child():
    import torch
    from head_analysis.hermes_kv_head_ragged import HeadRaggedCache, HeadRaggedLayerCache
    from head_analysis.hermes_kv_head_ragged_prefill import (
        _sync_position_cache_from_ragged, _build_text_position_ids,
        _ragged_language_chunk_forward)
    from video_qa.base import work
    from video_qa.hermes_vqa import HermesVQA

    fields = ('flat_k', 'flat_v', 'head_lens', 'flat_pos', 'flat_source_chunk',
              'flat_attention_ema', 'flat_token_uid')

    def clone(cache):
        assert isinstance(cache, HeadRaggedCache)
        return HeadRaggedCache([HeadRaggedLayerCache(*[
            None if getattr(layer, name) is None else getattr(layer, name).clone()
            for name in fields]) for layer in cache.layers])

    def uid_lists(cache):
        return [[layer.get_token_uid_segment(h).cpu().tolist()
                 for h in range(layer.head_lens.numel())] for layer in cache.layers]

    def select(cache, chosen):
        # Synchronous metadata/K/V pruning; all initial text is always retained.
        result = clone(cache)
        indices = [[torch.tensor(idx, device=layer.flat_k.device, dtype=torch.long)
                    for idx in per_head] for layer, per_head in zip(result.layers, chosen)]
        result.prune_per_head(indices)
        return result

    def filtered(cache, predicate):
        chosen = [[[i for i, u in enumerate(us) if u < 0 or predicate(u)]
                   for us in layer] for layer in uid_lists(cache)]
        return select(cache, chosen)

    def stats(cache):
        lists = uid_lists(cache)
        visual = [u for layer in lists for head in layer for u in head if u >= 0]
        counts = [sum(u >= 0 for u in head) for layer in lists for head in layer]
        roi_counts = [sum(u in ROI for u in head) for layer in lists for head in layer]
        return dict(visual_copies=len(visual), unique_visual_uids=len(set(visual)),
                    roi_copies=sum(roi_counts), roi_unique=len(ROI & set(visual)),
                    nonempty_roi_heads=sum(c > 0 for c in roi_counts),
                    visual_counts_per_head=counts, roi_counts_per_head=roi_counts,
                    ordered_uids_sha256=hashlib.sha256(json.dumps(lists).encode()).hexdigest())

    class ProbeVQA(HermesVQA):
        def load_video(self, video_path, clip=None):
            # The original loader decodes 12m42s, but QA0 consumes exactly these
            # ten indices. Same decoder, fps rounding, stride and RGB pixels.
            assert clip is None and self.sample_fps == 0.5
            frames = super().load_video(video_path, clip=[0, 18.0])
            assert len(frames) == 10
            self.input_manifest = dict(video_path=video_path, shape=list(frames.shape),
                nominal_sample_seconds=list(range(0, 20, 2)),
                sha256=hashlib.sha256(frames.tobytes()).hexdigest(),
                loader_note='Load only the same first ten sampled frames consumed by original QA0.')
            return frames

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.encode_calls = self.compress_calls = 0
            model = self.qa_model
            encode = model.encode_video_chunk
            compress = model.predict_and_compress

            def capture_encode(*a, **kw):
                self.encode_calls += 1
                return encode(*a, **kw)

            def capture_compress(*a, **kw):
                self.compress_calls += 1
                assert self.compress_calls == 1, 'Full prefix assumes a single compression.'
                self.full = clone(model.kv_cache)
                return compress(*a, **kw)

            model.encode_video_chunk = capture_encode
            model.predict_and_compress = capture_compress

        @torch.inference_mode()
        def video_close_qa(self, question, choices, correct_choice):
            model = self.qa_model
            assert self.encode_calls == self.compress_calls == 1
            assert not model.conv_history
            assert not getattr(model, '_answer_history_max_age_seconds', 0)
            dp = clone(model.kv_cache)
            full = self.full
            dp_uids, full_uids = uid_lists(dp), uid_lists(full)
            assert all([u for u in us if u >= 0] == list(range(13455))
                       for layer in full_uids for us in layer)
            old = [json.loads(s) for s in (OLD / 'dp/tokens.jsonl').read_text().splitlines()]
            old = [r for r in old if str(r['video_id']) == '269' and r['question_index'] == 0]
            assert len(old) == 1
            for h in old[0]['heads']:
                assert [u for u in dp_uids[h['layer']][h['kv_head']] if u >= 0] == h['visual_token_uids']
            assert stats(dp)['roi_copies'] == 168 and stats(dp)['roi_unique'] == 15
            assert stats(dp)['visual_copies'] == 112000

            # Compression must only select the existing representations, without changing them.
            retained_checks = 0
            for li, (d, f) in enumerate(zip(dp.layers, full.layers)):
                for h, us in enumerate(dp_uids[li]):
                    full_map = {u: j for j, u in enumerate(full_uids[li][h]) if u >= 0}
                    vi = [j for j, u in enumerate(us) if u >= 0]
                    fi = [full_map[us[j]] for j in vi]
                    for a, b in zip(d.get_segment(h), f.get_segment(h)):
                        assert torch.equal(a[vi], b[fi])
                    assert torch.equal(d.get_position_segment(h)[:, vi], f.get_position_segment(h)[:, fi])
                    retained_checks += len(vi)

            rng = random.Random(SEED)
            matched, matched_roi, restore_roi, restore_bg, surgery = [], [], [], [], []
            for li, (dus, fus) in enumerate(zip(dp_uids, full_uids)):
                lm, lmr, lr, lb = [], [], [], []
                for h, (us, all_us) in enumerate(zip(dus, fus)):
                    present = set(us)
                    need = sorted(ROI - present)
                    nroi = sum(u in ROI for u in us)
                    bg_present = [i for i, u in enumerate(us) if 0 <= u < 2691 and u not in ROI]
                    matched_n = min(nroi, len(bg_present))
                    text_idx = [i for i, u in enumerate(us) if u < 0]
                    lm.append(sorted(text_idx + rng.sample(bg_present, matched_n)))
                    lmr.append(sorted(text_idx + rng.sample(
                        [i for i, u in enumerate(us) if u in ROI], matched_n)))
                    drop_pool = [u for u in us if 0 <= u < 13455 - 32 and u not in ROI]
                    drops = set(rng.sample(drop_pool, len(need)))
                    bg_pool = [u for u in range(2691) if u not in present and u not in ROI]
                    bg_add = set(rng.sample(bg_pool, len(need)))
                    target_set = (present - drops) | set(need)
                    bg_set = (present - drops) | bg_add
                    # Negative UID tags identify initial text; never deduplicate those tokens.
                    lr.append([i for i, u in enumerate(all_us) if u < 0 or u in target_set])
                    lb.append([i for i, u in enumerate(all_us) if u < 0 or u in bg_set])
                    assert len(lr[-1]) == len(lb[-1]) == len(us)
                    surgery.append(dict(layer=li, head=h, removed=sorted(drops),
                                        added_roi=need, added_background=sorted(bg_add)))
                matched.append(lm); matched_roi.append(lmr)
                restore_roi.append(lr); restore_bg.append(lb)
            assert sum(len(x['added_roi']) for x in surgery) == 1512
            write_json(OUT / 'surgery.json', surgery)

            builders = {
                'dp_full': lambda: clone(dp),
                'dp_text_only': lambda: filtered(dp, lambda u: False),
                'dp_first_pair': lambda: filtered(dp, lambda u: u < 2691),
                'dp_roi_only': lambda: filtered(dp, lambda u: u in ROI),
                'dp_blank_roi_only': lambda: filtered(dp, lambda u: u in BLANK),
                'dp_matched_background': lambda: select(dp, matched),
                'dp_matched_roi': lambda: select(dp, matched_roi),
                'dp_no_roi': lambda: filtered(dp, lambda u: u not in ROI),
                'full_prefix': lambda: clone(full),
                'full_roi_only': lambda: filtered(full, lambda u: u in ROI),
                'full_first_pair': lambda: filtered(full, lambda u: u < 2691),
                'dp_restore_roi_equal_budget': lambda: select(full, restore_roi),
                'dp_restore_background_equal_budget': lambda: select(full, restore_bg),
                'dp_full_repeat': lambda: clone(dp),
            }
            offsets_method = model._get_next_global_offset_per_layer
            offsets = list(offsets_method())
            history = copy.deepcopy(model.conv_history)
            trace_names = ('_ragged_readout_trace_config', '_ragged_token_trace_config',
                           '_prospective_lifecycle_trace_config')
            trace_values = {k: getattr(model, k, None) for k in trace_names}
            for k in trace_names:
                setattr(model, k, None)
            model._get_next_global_offset_per_layer = types.MethodType(lambda self: offsets, model)
            mcq = self.format_mcqa_prompt(question, choices)
            prompts = [('mcq', mcq)] + [('open_' + str(i + 1),
                        dict(question=p, prompt=model.get_prompt(p))) for i, p in enumerate(OPEN_PROMPTS)]
            results = dict(video_id=269, question_index=0, question=question, choices=choices,
                           encode_calls=self.encode_calls, compression_calls=self.compress_calls,
                           retained_kv_equal_checks=retained_checks, qa_offsets=offsets,
                           input_manifest=self.input_manifest,
                           reference_matched_heads=112, roi_uids=sorted(ROI), seed=SEED, branches={})

            def install(cache):
                model.kv_cache = cache
                model.conv_history = copy.deepcopy(history)
                _sync_position_cache_from_ragged(model)

            def score_letters(inp):
                tok = model.processor.tokenizer
                ids = torch.tensor([tok(inp['prompt']).input_ids], device=model.device)
                embeds = model.get_input_embeddings()(ids)
                pos = _build_text_position_ids(model, offsets, embeds.shape[1])
                hidden, _ = _ragged_language_chunk_forward(model, embeds, model.kv_cache, pos,
                                                          append=False, collect_scores=False)
                lp = model.lm_head(hidden[:, -1])[0].float().log_softmax(-1)
                token_ids = {c: tok(c, add_special_tokens=False).input_ids for c in 'ABCD'}
                assert all(len(v) == 1 for v in token_ids.values())
                probs = {c: lp[v[0]].exp().item() for c, v in token_ids.items()}
                return dict(token_ids=token_ids, probabilities=probs,
                            candidate_probability_mass=sum(probs.values()),
                            conditional_probabilities={c: p / sum(probs.values()) for c, p in probs.items()})

            try:
                for name, builder in builders.items():
                    cache = builder()
                    info = dict(cache=stats(cache), responses={})
                    for kind, inp in prompts:
                        if name == 'dp_full_repeat' and kind != 'mcq':
                            continue
                        install(clone(cache))
                        started = time.time()
                        scores = score_letters(inp) if kind == 'mcq' else None
                        output = model.question_answering(inp, max_new_tokens=16,
                                                         temperature=0, repetition_penalty=1.1)
                        info['responses'][kind] = dict(output=output, letter_scores=scores,
                                                      seconds=time.time() - started)
                        assert stats(model.kv_cache)['ordered_uids_sha256'] == info['cache']['ordered_uids_sha256']
                        print('PROBE', name, kind, repr(output), flush=True)
                    results['branches'][name] = info
                    write_json(OUT / 'results.partial.json', results)
                    del cache
                baseline = results['branches']['dp_full']['responses']['mcq']['output']
                assert baseline.strip().startswith('A'), baseline
                assert results['branches']['dp_full_repeat']['responses']['mcq']['output'] == baseline
                results['baseline_repeat_identical'] = True
                write_json(OUT / 'results.json', results)
                with (OUT / 'responses.csv').open('w') as f:
                    writer = csv.writer(f)
                    writer.writerow(['branch', 'prompt', 'output', 'visual_copies', 'roi_copies', 'p_A', 'p_C'])
                    for name, info in results['branches'].items():
                        for kind, response in info['responses'].items():
                            probs = (response['letter_scores'] or {}).get('probabilities', {})
                            writer.writerow([name, kind, response['output'], info['cache']['visual_copies'],
                                             info['cache']['roi_copies'], probs.get('A'), probs.get('C')])
                return dict(pred_answer=baseline, pred_choice='A', acc=float(correct_choice == 'A'))
            finally:
                model._get_next_global_offset_per_layer = offsets_method
                for k, v in trace_values.items():
                    setattr(model, k, v)
                install(dp)

    work(ProbeVQA)


if __name__ == '__main__':
    if sys.argv[1:] == ['--launch']:
        launch()
    else:
        child()
