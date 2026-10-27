"""CPU-only launch, orphan detection and interrupted-append safety checks."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sparsemm_native import supervise_ovobench_dual as controller


class SupervisorHelperTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "queries.jsonl"

    def test_worker_command_runs_only_full_dp_dual_with_pinned_settings(self):
        command = controller.worker_command(self.root, "hermes9", python="/test/python")
        self.assertEqual(command[:4], ["/test/python", "-u", "-m", "sparsemm_native.run_ovobench"])
        for option, value in {"--arm": "dp_dual", "--dataset": "hermes9", "--max-new": "100000",
                              "--budget": "500", "--fps": "0.5", "--chunk-size": "16",
                              "--max-new-tokens": "16", "--video-max-pixels": "151200"}.items():
            self.assertEqual(command[command.index(option) + 1], value)
        self.assertIn("--grouped", command)
        self.assertNotIn("--debug-explain", command)
        self.assertEqual(command[command.index("--output-root") + 1], str(self.root / "hermes9"))

    def test_orphan_worker_is_adopted_from_process_identity_without_registry(self):
        command = controller.worker_command(self.root, "hermes9")
        record = {"pid": 123, "start_ticks": "12345", "command": command}
        with patch.object(Path, "iterdir", return_value=[SimpleNamespace(name="123")]), \
                patch.object(controller, "process_record", return_value=record):
            self.assertEqual(controller.find_worker(self.root, "hermes9"), record)

    def test_conflicting_or_duplicate_workers_are_rejected(self):
        command = controller.worker_command(self.root, "hermes9")
        wrong = command.copy()
        wrong[wrong.index("--fps") + 1] = "1.0"
        records = [
            {"pid": 123, "start_ticks": "12345", "command": wrong},
            {"pid": 124, "start_ticks": "12346", "command": command},
        ]
        with patch.object(Path, "iterdir", return_value=[SimpleNamespace(name="123")]), \
                patch.object(controller, "process_record", return_value=records[0]), \
                self.assertRaisesRegex(ValueError, "Conflicting"):
            controller.find_worker(self.root, "hermes9")
        with patch.object(Path, "iterdir", return_value=[SimpleNamespace(name="123"), SimpleNamespace(name="124")]), \
                patch.object(controller, "process_record", side_effect=[{**records[0], "command": command}, records[1]]), \
                self.assertRaisesRegex(ValueError, "Multiple"):
            controller.find_worker(self.root, "hermes9")

    def test_incomplete_last_append_is_backed_up_without_losing_valid_records(self):
        prefix = b'{"query_id":"a"}\n{"query_id":"b"}\n'
        tail = b'{"query_id":"interrupted'
        self.path.write_bytes(prefix + tail)
        self.assertEqual(len(controller.progress_rows(self.path)), 2)
        repair = controller.repair_jsonl_tail(self.path)
        self.assertEqual(self.path.read_bytes(), prefix)
        self.assertEqual(Path(repair["backup"]).read_bytes(), tail)
        self.assertEqual(repair["removed_bytes"], len(tail))

    def test_valid_last_record_without_newline_is_preserved(self):
        data = b'{"query_id":"a"}\n{"query_id":"b"}'
        self.path.write_bytes(data)
        self.assertEqual(len(controller.progress_rows(self.path)), 2)
        repair = controller.repair_jsonl_tail(self.path)
        self.assertTrue(repair["added_final_newline"])
        self.assertEqual(self.path.read_bytes(), data + b"\n")
        self.assertIsNone(controller.repair_jsonl_tail(self.path))

    def test_large_trace_fast_path_inspects_only_the_final_append(self):
        # Middle trace validation is deliberately left to the trace auditor.
        prefix = b"previous trace records are not parsed here\n" * 2000
        self.path.write_bytes(prefix)
        self.assertIsNone(controller.repair_jsonl_tail(self.path, validate_complete=False))
        tail = b'{"event":"video_prefill","interrupted":'
        self.path.write_bytes(prefix + tail)
        repair = controller.repair_jsonl_tail(self.path, validate_complete=False)
        self.assertEqual(self.path.read_bytes(), prefix)
        self.assertEqual(Path(repair["backup"]).read_bytes(), tail)

    def test_pause_sentinel_exits_without_launch_or_restart(self):
        self.root.joinpath("PAUSE").touch()
        record = {"pid": 123, "start_ticks": "12345", "command": ["worker"]}
        with patch.object(controller, "preflight", return_value={}), \
                patch.object(controller, "find_worker", side_effect=[record, None]), \
                patch.object(controller, "stop_worker") as stop, \
                patch.object(controller.subprocess, "Popen") as launch, \
                patch.object(controller, "event"):
            self.assertEqual(controller.supervise(self.root), 0)
        stop.assert_called_once_with(record)
        launch.assert_not_called()
        status = json.loads(self.root.joinpath("status.json").read_text())
        self.assertEqual(status["state"], "paused")
        self.assertEqual(status["total_queries"], 3035)

    def test_malformed_complete_middle_and_corrupt_tail_records_are_never_discarded(self):
        variants = [b'{"query_id":"a"}\nnot json\n',
                    b'{"query_id":"a"}\n{bad}\n{"query_id":"b"}\n',
                    b'{"query_id":"a"}\n{"query_id": garbage}']
        for data in variants:
            with self.subTest(data=data):
                self.path.write_bytes(data)
                with self.assertRaises(ValueError):
                    controller.repair_jsonl_tail(self.path)
                self.assertEqual(self.path.read_bytes(), data)

    def test_atomic_status_write_replaces_the_previous_complete_document(self):
        controller.write_json(self.path, {"state": "running", "completed": 1})
        controller.write_json(self.path, {"state": "complete", "completed": 3035})
        self.assertEqual(json.loads(self.path.read_text()), {"state": "complete", "completed": 3035})
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_pause_does_not_kill_reused_or_non_detached_process(self):
        record = {"pid": 123, "start_ticks": "12345", "command": ["worker"]}
        with patch.object(controller, "process_record", return_value={**record, "start_ticks": "99999"}), \
                patch.object(controller.os, "killpg") as kill, self.assertRaises(ValueError):
            controller.stop_worker(record)
        kill.assert_not_called()
        with patch.object(controller, "process_record", return_value=record), \
                patch.object(controller.os, "getpgid", return_value=100), \
                patch.object(controller.os, "killpg") as kill, self.assertRaises(ValueError):
            controller.stop_worker(record)
        kill.assert_not_called()


if __name__ == "__main__":
    unittest.main()
