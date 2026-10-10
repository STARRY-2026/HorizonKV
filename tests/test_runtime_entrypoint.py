import csv
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from video_qa import run_infer


class RuntimeEntrypointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="hermes paths ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.annotation = self.root / "annotation file.json"
        self.annotation.write_text("[]", encoding="utf-8")
        self.output = self.root / "output files"
        self.output.mkdir()
        self.args = run_infer.build_parser().parse_args([
            "--dataset", "ovobench", "--anno_path", str(self.annotation),
            "--output_dir", str(self.output), "--model_path", str(self.root / "model files"),
        ])
        self.args.devices = ["0"]
        # Entry-point behavior is independent of tensor policy validation.
        self.policy_patch = mock.patch.dict(sys.modules, {
            "head_analysis.horizonkv_policy": SimpleNamespace(
                validate_horizonkv_args=lambda args: None,
            ),
        })
        self.policy_patch.start()
        self.addCleanup(self.policy_patch.stop)

    def write_chunk(self, name="1_0.csv", header=("video_id", "pred_answer"),
                    row=("clip", "answer\nwith newline, and comma")):
        path = self.output / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerow(row)
        return path

    def test_failed_command_raises(self):
        with self.assertRaises(subprocess.CalledProcessError):
            run_infer.exec([sys.executable, "-c", "raise SystemExit(7)"])

    def test_worker_failure_rejects_stale_chunk_and_skips_evaluation(self):
        chunk = self.write_chunk()
        process = mock.Mock(exitcode=7)
        with mock.patch.object(run_infer.multiprocessing, "Process", return_value=process), \
                mock.patch.object(run_infer, "exec") as execute:
            with self.assertRaisesRegex(RuntimeError, "chunks failed"):
                run_infer.run_eval(self.args, run_infer.BENCHMARK_CONFIGS["ovobench"])
        execute.assert_not_called()
        self.assertTrue(chunk.exists())
        self.assertFalse((self.output / "results.csv").exists())

    def test_successful_process_without_chunk_skips_evaluation(self):
        process = mock.Mock(exitcode=0)
        with mock.patch.object(run_infer.multiprocessing, "Process", return_value=process), \
                mock.patch.object(run_infer, "exec") as execute:
            with self.assertRaisesRegex(RuntimeError, "chunks failed"):
                run_infer.run_eval(self.args, run_infer.BENCHMARK_CONFIGS["ovobench"])
        execute.assert_not_called()

    def test_paths_and_dp_alpha_k_are_forwarded_as_individual_arguments(self):
        self.write_chunk()
        recency = self.root / "recency profile.csv"
        recency.write_text("profile", encoding="utf-8")
        self.args.kv_head_budget_scores = str(self.root / "DP scores.csv")
        self.args.kv_head_recency_scores = str(recency)
        self.args.kv_head_recency_delta = 0.2
        self.args.kv_head_recency_k_delta = 2.0
        process = mock.Mock(exitcode=0)
        with mock.patch.object(run_infer.multiprocessing, "Process", return_value=process) as create, \
                mock.patch.object(run_infer, "exec") as execute:
            run_infer.run_eval(self.args, run_infer.BENCHMARK_CONFIGS["ovobench"])
        command, sub, device = create.call_args.kwargs["args"]
        for flag, expected in (
            ("--model_path", self.args.model_path),
            ("--anno_path", str(self.annotation)),
            ("--save_dir", str(self.output)),
            ("--kv_head_budget_scores", self.args.kv_head_budget_scores),
            ("--kv_head_recency_scores", str(recency)),
            ("--kv_head_recency_delta", "0.2"),
            ("--kv_head_recency_k_delta", "2.0"),
            ("--kv_head_ragged_prefill", "True"),
        ):
            self.assertEqual(command[command.index(flag) + 1], expected)
        self.assertNotIn("--kv_size", command)
        self.assertTrue(sub)
        self.assertEqual(device, "0")
        evaluation = shlex.split(execute.call_args.args[0])
        self.assertEqual(evaluation[-1], str(self.output / "results.csv"))

    def test_eval_failure_propagates(self):
        self.args.only_eval = True
        self.write_chunk("results.csv")
        with mock.patch.object(run_infer, "exec", side_effect=subprocess.CalledProcessError(2, "eval")):
            with self.assertRaises(subprocess.CalledProcessError):
                run_infer.run_eval(self.args, run_infer.BENCHMARK_CONFIGS["ovobench"])

    def test_only_eval_requires_results(self):
        self.args.only_eval = True
        with mock.patch.object(run_infer, "exec") as execute:
            with self.assertRaisesRegex(FileNotFoundError, "Results file"):
                run_infer.run_eval(self.args, run_infer.BENCHMARK_CONFIGS["ovobench"])
        execute.assert_not_called()

    def test_merge_preserves_multiline_fields_and_removes_complete_chunks(self):
        first = self.write_chunk("2_0.csv")
        second = self.write_chunk("2_1.csv", row=("第二段", "Unicode answer"))
        output = self.output / "results.csv"
        run_infer.merge_chunk_results([first, second], output)
        with output.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        self.assertEqual(rows, [
            ["video_id", "pred_answer"],
            ["clip", "answer\nwith newline, and comma"],
            ["第二段", "Unicode answer"],
        ])
        self.assertFalse(first.exists())
        self.assertFalse(second.exists())

    def test_invalid_merge_preserves_previous_results_and_all_chunks(self):
        first = self.write_chunk("2_0.csv")
        second = self.write_chunk("2_1.csv", header=("wrong", "columns"))
        output = self.output / "results.csv"
        output.write_text("previous results", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Inconsistent result columns"):
            run_infer.merge_chunk_results([first, second], output)
        self.assertEqual(output.read_text(encoding="utf-8"), "previous results")
        self.assertTrue(first.exists())
        self.assertTrue(second.exists())
        self.assertEqual(list(self.output.glob(".results-*")), [])

    def test_empty_results_are_rejected(self):
        chunk = self.output / "1_0.csv"
        chunk.write_text("video_id,pred_answer\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "no result rows"):
            run_infer.merge_chunk_results([chunk], self.output / "results.csv")
        self.assertTrue(chunk.exists())


if __name__ == "__main__":
    unittest.main()
