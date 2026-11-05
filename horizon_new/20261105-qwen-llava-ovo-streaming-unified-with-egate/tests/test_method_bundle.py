"""Verify the vendored implementation is complete and its active paths stay local."""
import ast
import hashlib
import json
import unittest
from pathlib import Path

from methods.registry import PROJECT_ROOT, METHOD_NAMES, LUKV_PROFILE, SPARSEMM_SCORES, llava_impl, local_python_paths, qwen_impl


class BundleTests(unittest.TestCase):
    def test_six_parallel_method_entries(self):
        for method in METHOD_NAMES:
            with self.subTest(method=method):
                self.assertTrue((PROJECT_ROOT/'methods'/method/'README.md').is_file())
                self.assertTrue((llava_impl(method)/'smoke.py').is_file())
        for method in ('hermes', 'fullkv'):
            self.assertTrue((qwen_impl(method)/'inference/qwenvl_hermes.py').is_file())

    def test_pinned_budget_assets(self):
        self.assertEqual(hashlib.sha256(SPARSEMM_SCORES.read_bytes()).hexdigest(), '9cc0f47f24a0af4306ce7367806516d119dc4f57df77d064a3eac59c05b69dab')
        self.assertEqual(hashlib.sha256(LUKV_PROFILE.read_bytes()).hexdigest(), '3bd31517199a0256b25150e67adc2a6fcfc2e011aafbe6f1e3eb6089db93c266')

    def test_source_inventory_matches_vendored_bytes(self):
        records = json.loads((PROJECT_ROOT/'methods/SOURCE_MANIFEST.json').read_text())
        self.assertGreater(len(records), 100)
        for row in records:
            with self.subTest(file=row['target']):
                file = PROJECT_ROOT/row['target']
                self.assertTrue(file.is_file())
                self.assertEqual(hashlib.sha256(file.read_bytes()).hexdigest(), row['vendored_sha256'])

    def test_local_import_roots_exist(self):
        for directory in local_python_paths():
            self.assertTrue(directory.is_dir(), str(directory))

    def test_active_workers_do_not_reference_old_method_paths(self):
        forbidden = ('20261022-ovobench-baselines-v1', '20261026-lukv-cal50-ovobench-v1',
                     '20261019-sparsemm-fp16-chunk16-v2', 'code/MM-ShiftKV-main/LLaVA-NeXT')
        for backend in ('qwen', 'qwen_full2495', 'llava'):
            for name in ('worker.py', 'run.py', 'ovo_qwen_dscache_run.py', 'ovo_run.py'):
                file = PROJECT_ROOT/backend/name
                if not file.exists():
                    continue
                source = file.read_text()
                for old_path in forbidden:
                    self.assertNotIn(old_path, source, str(file))
                ast.parse(source)


if __name__ == '__main__':
    unittest.main()
