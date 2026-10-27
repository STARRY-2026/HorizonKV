"""Whole-video sharding, no baseline inference, and new-only ETA/report checks."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from sparsemm_native import supervise_ovobench_recent4_only as controller


def query(index):
    return dict(query_id=str(index), source_id=index, probe_id=None, task="EPM", group="backward",
                timestamp=400.0, clip_end=400.0, target="A", prompt="official prompt", video_path=f"v{index}")


def row(q):
    result = {key: q[key] for key in controller.runner.IDENTITY}
    result.update(response="A", correct=True, frames=200, encoded_frames=100, seconds=50,
                  prompt_sha256=controller.runner.digest(q["prompt"]))
    return result


class Recent4OnlyTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_real_full_query_set_is_split_once_and_each_video_stays_on_one_worker(self):
        selected = controller.runner.schedule_queries(controller.runner.legacy.queries())
        jobs = controller.split_videos(selected)
        ids = [qid for job in jobs.values() for qid in job["query_ids"]]
        self.assertEqual(len(ids), 3035)
        self.assertEqual(len(set(ids)), 3035)
        videos = [{q["video_path"] for q in job["queries"]} for job in jobs.values()]
        self.assertFalse(videos[0] & videos[1])
        self.assertEqual(len(videos[0] | videos[1]), 644)
        self.assertEqual(sum(job["expected_encoded_frames"] for job in jobs.values()), 85041)
        for name, job in jobs.items():
            self.assertEqual(controller.runner.schedule_queries(controller.runner.select_queries(
                controller.runner.legacy.queries(), ",".join(job["query_ids"]))), job["queries"])
            command = controller.worker_command(self.root, name, job)
            self.assertEqual(command[command.index("--arm") + 1], "dp_recent4")
            self.assertEqual(command[command.index("--concurrent-processes") + 1], "2")
            self.assertIn("--grouped", command)
            self.assertNotIn("--reuse-root", command)

    def test_real_preparation_pins_new_only_phase_and_is_restart_stable(self):
        manifest, selected, _, reused, jobs, plan = controller.prepare(self.root)
        self.assertEqual(len(selected), 3035)
        self.assertFalse(plan["run_baseline_inference"])
        self.assertEqual(plan["phase"], "new_method_only")
        self.assertEqual(len(reused["native"]), 1594)
        self.assertEqual(len(reused["dp"]), 1595)
        self.assertEqual(sum(len(job["queries"]) for job in jobs.values()), 3035)
        self.assertEqual(controller.prepare(self.root)[5], plan)
        self.assertIn(str(Path(controller.__file__).resolve()), manifest["files_sha256"])
        saved = self.root / "recent4_only_manifest.json"
        saved.write_text(json.dumps(dict(plan, run_baseline_inference=True)))
        with self.assertRaisesRegex(ValueError, "plan changed"):
            controller.prepare(self.root)

    def test_collect_merges_disjoint_shards_and_estimates_from_real_encoded_frames(self):
        jobs = {name: dict(queries=[query(i)], expected_encoded_frames=500)
                for i, name in enumerate(controller.JOBS)}
        for name, job in jobs.items():
            path = self.root / "shards" / name / "dp_recent4/queries.jsonl"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(row(job["queries"][0])) + "\n")
        reused = {arm: {} for arm in controller.runner.ARMS}
        reused["native"]["0"] = row(query(0))
        progress, available, measurements = controller.collect(self.root, jobs, reused)
        self.assertEqual(progress, dict(shard0=1, shard1=1))
        self.assertEqual(len(available["dp_recent4"]), 2)
        self.assertEqual(len(available["native"]), 1)
        for measurement in measurements.values():
            self.assertEqual(measurement["frames_per_second"], 2)
            self.assertEqual(measurement["estimated_remaining_seconds"], 200)

    def test_partial_new_results_are_reported_without_fake_complete_manifest(self):
        q = query(0)
        available = {arm: {"0": row(q)} for arm in controller.runner.ARMS}
        status = dict(available_queries={arm: 1 for arm in controller.runner.ARMS},
                      new_inference_queries=dict(native=0, dp=0, dp_recent4=1))
        (self.root / "dp_recent4").mkdir()
        report = controller.publish(self.root, [q], [q], available, status)
        self.assertEqual(report["matched"]["matched_queries"], 1)
        self.assertFalse(report["complete"])
        self.assertIsNone(report["official_total_average"]["dp_recent4"])
        self.assertTrue((self.root / "dp_recent4/predictions_live.jsonl").is_file())
        self.assertFalse((self.root / "dp_recent4/queries.jsonl").exists())
        self.assertIn("两个基线不启动推理", (self.root / "live_report.md").read_text())


if __name__ == "__main__":
    unittest.main()
