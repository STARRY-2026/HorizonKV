import argparse
import json
import tempfile
import unittest
from pathlib import Path

from core.dispatch import check_saved_plan, make_plan, save_completed_plan, summary_backend
from core.registry import BASELINE_METHODS


def args(**changes):
    values = dict(benchmark='ovobench', model='qwen25vl', method='sparsemm', gpus='0,1',
                  output_dir='/tmp/unified-test-output', annotation=None, max_videos=None,
                  cohort='heldout', allow_calibration_overlap=False, accept_equivalent_choices=False)
    values.update(changes)
    return argparse.Namespace(**values)


class ControlTests(unittest.TestCase):
    def test_all_24_baseline_combinations(self):
        for benchmark in ('ovobench', 'streamingbench'):
            for model in ('qwen25vl', 'llavaov'):
                for method in BASELINE_METHODS:
                    with self.subTest(benchmark=benchmark, model=model, method=method):
                        plan = make_plan(args(benchmark=benchmark, model=model, method=method), Path('/code'))
                        self.assertEqual(plan['backend'], 'qwen' if model == 'qwen25vl' else 'llava')
                        self.assertEqual(plan['config']['visual_budget'], None if method == 'fullkv' else 500)
                        profile = plan['config']['input_profile']
                        self.assertIs(profile['latest_frame_replay'], False)
                        if model == 'qwen25vl':
                            self.assertEqual(profile['ordinary_target_pixels'], 448**2)
                            self.assertEqual(profile['chunk_frames'], 2)
                        else:
                            self.assertEqual(profile['ordinary_visual_tokens'], 196)
                            self.assertEqual(profile['frame_size'], [384, 384])

    def test_full_qwen_requires_explicit_scoring_policy(self):
        with self.assertRaisesRegex(ValueError, 'equivalent'):
            make_plan(args(benchmark='streamingbench', cohort='full'), Path('/code'))
        plan = make_plan(args(benchmark='streamingbench', cohort='full', accept_equivalent_choices=True), Path('/code'))
        self.assertEqual(plan['backend'], 'qwen_full2495')
        self.assertIs(plan['config']['input_profile']['latest_frame_replay'], False)

    def test_lukv_calibration_overlap_is_explicit(self):
        with self.assertRaisesRegex(ValueError, 'calibration'):
            make_plan(args(benchmark='streamingbench', method='lukv', cohort='full', accept_equivalent_choices=True), Path('/code'))

    def test_llava_does_not_silently_ignore_scoring_policy(self):
        for method in ('adakv', 'pyramidkv'):
            with self.assertRaisesRegex(ValueError, 'silently'):
                make_plan(args(benchmark='streamingbench', model='llavaov', method=method, accept_equivalent_choices=True), Path('/code'))

    def test_invalid_resource_and_dataset_flags(self):
        for change in (dict(gpus='0,00'), dict(gpus='0,'), dict(gpus='-1'), dict(max_videos=0), dict(cohort='full')):
            with self.subTest(change=change), self.assertRaises(ValueError):
                make_plan(args(**change), Path('/code'))

    def test_protocol_cannot_be_inferred_from_partial_question_count(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output/'manifest.json').write_text(json.dumps(dict(model='qwen25vl', benchmark='streamingbench', cohort='full', questions=8)))
            with self.assertRaisesRegex(ValueError, 'provenance'):
                summary_backend(output)
            self.assertEqual(summary_backend(output, 'qwen_full2495'), 'qwen_full2495')

    def test_completed_run_records_and_checks_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root/'result'
            output.mkdir()
            for name in ('run.py', 'core/dispatch.py', 'core/registry.py', 'qwen_full2495/run.py'):
                target = root/name
                target.parent.mkdir(exist_ok=True, parents=True)
                target.write_text('# test fixture\n')
            plan = make_plan(args(benchmark='streamingbench', cohort='full', accept_equivalent_choices=True, output_dir=output), root)
            with self.assertRaisesRegex(RuntimeError, 'manifest'):
                save_completed_plan(plan, root)
            (output/'manifest.json').write_text('{}')
            save_completed_plan(plan, root)
            check_saved_plan(plan)
            self.assertEqual(summary_backend(output), 'qwen_full2495')
            with self.assertRaisesRegex(ValueError, 'conflicts'):
                summary_backend(output, 'qwen')
            changed = make_plan(args(benchmark='streamingbench', cohort='full', accept_equivalent_choices=True, output_dir=output, method='hermes'), root)
            with self.assertRaisesRegex(ValueError, 'different'):
                check_saved_plan(changed)


if __name__ == '__main__':
    unittest.main()
