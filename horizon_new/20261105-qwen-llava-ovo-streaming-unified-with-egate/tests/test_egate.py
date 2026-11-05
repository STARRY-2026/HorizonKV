import argparse
import json
from pathlib import Path
import tempfile
import unittest

from core.dispatch import make_plan, summary_backend
from methods.horizonkv_egate.protocol import load_allocation, format_mc_question, route_answer
from methods.horizonkv_egate.positions import token_ids, position_values, retained_metadata
from methods.horizonkv_egate.data import accuracy, validate_rows


class EGateControlTests(unittest.TestCase):
    def test_four_combinations_have_explicit_dispatch_and_asset_hashes(self):
        root = Path(__file__).resolve().parents[1]
        for model in ('qwen25vl', 'llavaov'):
            for benchmark in ('streamingbench', 'ovobench'):
                args = argparse.Namespace(model=model, benchmark=benchmark, method='horizonkv-egate',
                    gpus='4', output_dir='/tmp/egate-test', annotation=None, max_videos=1, cohort='heldout',
                    allow_calibration_overlap=False, accept_equivalent_choices=False)
                plan = make_plan(args, root)
                self.assertEqual(plan['backend'], 'egate')
                self.assertEqual(plan['config']['memory_batch_frames'], 4)
                self.assertIn('assets/per_head_allocation.csv', plan['config']['egate_sources'])
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/'manifest.json').write_text(json.dumps(dict(backend='egate', model='llavaov')))
            self.assertEqual(summary_backend(directory), 'egate')

    def test_frozen_allocation_and_gate_contract(self):
        allocation, digest = load_allocation()
        self.assertEqual(len(allocation), 112)
        self.assertEqual(len(digest), 64)
        self.assertEqual(sum(v['recent_group']=='one_frame' for v in allocation.values()), 56)
        self.assertTrue(all(v['recent_quota']+v['history_quota']==500 for v in allocation.values()))
        for n, abstain in ((2, 'C'), (3, 'D'), (4, 'E'), (5, 'F'), (8, 'I')):
            text, gate = format_mc_question('Where?', ['option']*n, recent_gate=True)
            self.assertEqual(gate, abstain)
            self.assertIn(f'choose {abstain} instead of guessing', text)
            original, no_gate = format_mc_question('Where?', ['option']*n)
            self.assertIsNone(no_gate)
            self.assertNotIn('Insufficient visual evidence', original)
            self.assertEqual(route_answer(abstain, 'B', n), ('B', 'history'))
            self.assertEqual(route_answer('A', 'B', n), ('A', 'recent'))

    def test_dynamic_grid_identity_and_bounded_metadata(self):
        low, high = token_ids(100000, 14, 14), token_ids(100001, 27, 27)
        self.assertEqual(len(set(low+high)), 196+729)
        meta = {100000:(14,14), 100001:(27,27)}
        qwen = position_values([-2,-1]+low+high, 2, meta, 'qwen25vl')
        self.assertEqual(max(max(axis) for axis in qwen), 2+14+26)
        llava = position_values([-2,-1]+low+high, 2, meta, 'llavaov')
        self.assertEqual(llava[0], list(range(2+196+729)))
        kept = retained_metadata({(0,0):dict(uids=high[:3])}, meta)
        self.assertEqual(kept, {100001:(27,27)})

    def test_partial_resume_validates_gate_and_scoring(self):
        q = dict(question='Where?', choices=['yes','no','yes','other'], answer='yes', correct_choice='A')
        v = dict(video_id=1, conversations=[q,q])
        row = dict(video_id='1',question_index=0, question=q['question'], choices=q['choices'],answer='yes',
                   correct_choice='A', task='', recent_answer='E', history_answer='C', gate_source='history',
                   pred_choice='C',pred_answer='C',qa_acc=100,max_visual_per_head=500)
        validate_rows([row], [v], equivalent=True)
        self.assertEqual(accuracy(q, 'C'), 0)
        self.assertEqual(accuracy(q, 'C', equivalent=True), 100)
        with self.assertRaisesRegex(ValueError, 'score'):
            validate_rows([row], [v])
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            validate_rows([row], [v], equivalent=True, complete=True)
        row['pred_choice'] = 'B'
        with self.assertRaisesRegex(ValueError, 'routing'):
            validate_rows([row], [v], equivalent=True)


if __name__ == '__main__':
    unittest.main()
