"""Bounded OVO interface checks that never load a model or acquire a GPU."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kvbench.ovo import ARM, OvoRunConfig, build_ovo_command, run_ovo


class OvoInterfaceTest(unittest.TestCase):
    def test_explicit_bounded_selection_and_separate_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            q = {"query_id": "1117", "video_path": "/video.mp4", "clip_end": 1.0}
            with patch("sparsemm_native.run_ovobench_sparsemm_fps1.selected_queries",
                       return_value=([q], [q])):
                config = OvoRunConfig(Path(directory) / "run", ("1117",))
                command = build_ovo_command(config)
                self.assertEqual(command[3], "kvbench.ovo_recent1or4_worker")
                self.assertEqual(command[command.index("--max-groups") + 1], "1")
                self.assertEqual(config.selected(), [q])
            with patch("sparsemm_native.run_ovobench_sparsemm_fps1.selected_queries",
                       return_value=([dict(q, clip_end=121.0)], [dict(q, clip_end=121.0)])):
                with self.assertRaisesRegex(ValueError, "120 seconds"):
                    config.selected()
            for ids in ((), tuple(map(str, range(9))), ("1117", "1117")):
                with self.subTest(ids=ids), self.assertRaises(ValueError):
                    OvoRunConfig(Path(directory) / "run", ids).selected()

    def test_dry_run_writes_only_bounded_plan_and_rejects_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            q = {"query_id": "1117", "video_path": "/video.mp4", "clip_end": 1.0}
            config = OvoRunConfig(Path(directory) / "run", ("1117",), dry_run=True)
            with patch("sparsemm_native.run_ovobench_sparsemm_fps1.selected_queries",
                       return_value=([q], [q])), patch("kvbench.ovo.subprocess.run") as process:
                result = run_ovo(config)
                self.assertEqual(result["method"], ARM)
                self.assertTrue(result["full_benchmarks_paused"])
                self.assertEqual(json.loads((config.output_dir / "kvbench_ovo_plan.json").read_text())["query_ids"], ["1117"])
                process.assert_not_called()
                with self.assertRaises(FileExistsError):
                    run_ovo(config)


if __name__ == "__main__":
    unittest.main()
