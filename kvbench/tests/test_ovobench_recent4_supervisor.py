"""Two-worker scheduling, safe restart and matched-official reporting checks."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sparsemm_native import supervise_ovobench_recent4 as controller


def query(index=0, task="EPM"):
    return dict(query_id=str(index), source_id=index, probe_id=None, task=task, group="backward",
                timestamp=2.0, clip_end=2.0, target="A", prompt="official prompt", video_path="a.mp4")


def result(q, answer="A"):
    row = {key: q[key] for key in controller.runner.IDENTITY}
    row.update(response=answer, correct=answer == q["target"], frames=1, seconds=1.0,
               prompt_sha256=controller.runner.digest(q["prompt"]))
    return row


class Recent4SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_worker_commands_pin_full_official_grouped_two_process_inference(self):
        for arm in controller.runner.ARMS:
            command = controller.worker_command(self.root, arm, python="/test/python")
            self.assertEqual(command[:4], ["/test/python", "-u", "-m", "sparsemm_native.run_ovobench_recent4"])
            for flag, value in (("--arm", arm), ("--budget", "500"), ("--fps", "0.5"),
                                ("--concurrent-processes", "2"), ("--max-new", "100000")):
                self.assertEqual(command[command.index(flag) + 1], value)
            self.assertIn("--grouped", command)
            self.assertNotIn("--ids", command)
            self.assertEqual(command.count("--reuse-root"), 0 if arm == "dp_recent4" else 2)

    def test_surviving_two_workers_are_adopted_without_pid_registry(self):
        records = {str(i): dict(pid=i, start_ticks=str(i * 10), command=controller.worker_command(self.root, arm))
                   for i, arm in ((101, "native"), (102, "dp_recent4"))}
        with patch.object(Path, "iterdir", return_value=[SimpleNamespace(name=k) for k in records]), \
                patch.object(controller, "process_record", side_effect=lambda pid: records[pid]), \
                patch.object(controller.os, "getpgid", side_effect=lambda pid: pid), \
                patch.object(controller.os, "getsid", side_effect=lambda pid: pid):
            found = controller.find_workers(self.root)
        self.assertEqual(set(found), {"native", "dp_recent4"})

    def test_wrong_worker_settings_and_third_process_are_rejected(self):
        command = controller.worker_command(self.root, "native")
        command[command.index("--fps") + 1] = "1"
        with patch.object(Path, "iterdir", return_value=[SimpleNamespace(name="101")]), \
                patch.object(controller, "process_record", return_value=dict(pid=101, command=command)), \
                self.assertRaisesRegex(ValueError, "Conflicting"):
            controller.find_workers(self.root)
        records = {str(i): dict(pid=i, command=controller.worker_command(self.root, arm))
                   for i, arm in enumerate(controller.runner.ARMS, start=101)}
        with patch.object(Path, "iterdir", return_value=[SimpleNamespace(name=k) for k in records]), \
                patch.object(controller, "process_record", side_effect=lambda pid: records[pid]), \
                patch.object(controller.os, "getpgid", side_effect=lambda pid: pid), \
                patch.object(controller.os, "getsid", side_effect=lambda pid: pid), \
                self.assertRaisesRegex(ValueError, "More than two"):
            controller.find_workers(self.root)

    def test_available_controls_include_verified_nonprefix_reuse(self):
        q = [query(0), query(1)]
        first, later = result(q[0]), result(q[1])
        (self.root / "native").mkdir()
        (self.root / "native/queries.jsonl").write_text(json.dumps(first) + "\n")
        reused = {arm: {} for arm in controller.runner.ARMS}
        reused["native"] = {"1": dict(later, reused_from="verified official")}
        written, available, inferred = controller.collect(self.root, q, reused)
        self.assertEqual(written["native"], 1)
        self.assertEqual(inferred["native"], 1)
        self.assertEqual(set(available["native"]), {"0", "1"})

    def test_reporting_requires_identical_frames_and_never_labels_partial_as_official_total(self):
        q = query()
        available = {arm: {"0": result(q)} for arm in controller.runner.ARMS}
        status = dict(available_queries={arm: 1 for arm in controller.runner.ARMS}, new_inference_queries={})
        report = controller.publish(self.root, [q], available, status)
        self.assertEqual(report["matched"]["matched_queries"], 1)
        self.assertFalse(report["complete"])
        self.assertTrue(all(v is None for v in report["official_total_average"].values()))
        available["dp"]["0"]["frames"] = 2
        with self.assertRaisesRegex(ValueError, "Different frame"):
            controller.publish(self.root, [q], available, status)

    def test_real_preflight_pins_all_3035_questions_and_reuses_only_official_controls(self):
        fixture = controller.ROOT / "results/ovobench_official_full_k500_fps05_p151200_20261019/native/protocol.json"
        if not fixture.is_file():
            self.skipTest("Historical baseline predictions are not part of the source repository")
        manifest, selected, _, reused = controller.preflight(self.root)
        self.assertEqual(len(selected), 3035)
        self.assertEqual(manifest["reused_counts"], dict(native=1594, dp=1595, dp_recent4=0))
        self.assertEqual(manifest["predictions_to_compute"], dict(native=1441, dp=1440, dp_recent4=3035))
        self.assertEqual(len(reused["native"]), 1594)
        self.assertEqual(controller.preflight(self.root)[0], manifest)
        path = self.root / "supervision_manifest.json"
        path.write_text(json.dumps(dict(manifest, concurrent_inference_processes=3)))
        with self.assertRaisesRegex(ValueError, "Pinned"):
            controller.preflight(self.root)

    def test_two_slots_schedule_third_arm_after_completion_and_pause_both_survivors(self):
        one, two, three = (dict(pid=i, command=["worker"], start_ticks=str(i)) for i in (101, 102, 103))
        manifest = dict(reused_counts={arm: 0 for arm in controller.runner.ARMS}, predictions_to_compute={})
        empty = {arm: {} for arm in controller.runner.ARMS}
        written_first = dict(native=0, dp=0, dp_recent4=0)
        written_later = dict(native=3035, dp=0, dp_recent4=5)
        processes = [SimpleNamespace(pid=i, returncode=0, poll=lambda: 0) for i in (101, 102, 103)]
        sleeps = []

        def pause_after_two_loops(_):
            sleeps.append(1)
            if len(sleeps) == 2:
                (self.root / "PAUSE").touch()

        with patch.object(controller, "preflight", return_value=(manifest, [], [], empty)), \
                patch.object(controller, "find_workers", side_effect=[{}, {"dp_recent4": one}, {"dp_recent4": one, "dp": three}]), \
                patch.object(controller, "collect", side_effect=[(written_first, empty, written_first),
                                                                   (written_later, empty, written_later),
                                                                   (written_later, empty, written_later)]), \
                patch.object(controller, "check_pins"), \
                patch.object(controller, "publish", return_value={"matched": {"matched_queries": 0}}), \
                patch.object(controller, "process_record", side_effect=[one, two, three]), \
                patch.object(controller.subprocess, "Popen", side_effect=processes) as popen, \
                patch.object(controller.time, "sleep", side_effect=pause_after_two_loops), \
                patch.object(controller, "stop_worker") as stop:
            self.assertEqual(controller.supervise(self.root, poll_seconds=.01), 0)
        arms = [call.args[0][call.args[0].index("--arm") + 1] for call in popen.call_args_list]
        self.assertEqual(arms, ["dp_recent4", "native", "dp"])
        self.assertTrue(all(call.kwargs["start_new_session"] and call.kwargs["close_fds"]
                            and call.kwargs["stdin"] == controller.subprocess.DEVNULL for call in popen.call_args_list))
        self.assertEqual({call.args[0]["pid"] for call in stop.call_args_list}, {101, 103})
        self.assertEqual(json.loads((self.root / "status.json").read_text())["state"], "paused")


if __name__ == "__main__":
    unittest.main()
