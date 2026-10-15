"""CPU-only interface tests. No subprocess loads a model or acquires a GPU."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from kvbench import DEFAULT_METHODS, METHODS, RunConfig, audit, build_command, run, summarize
from kvbench import api, paths
from kvbench.cli import main


def write_json(path, value):
    Path(path).write_text(json.dumps(value), encoding="utf-8")


class InterfaceTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.annotation = self.root / "annotations.json"
        write_json(self.annotation, [{"video_id": i, "conversations": [{"end_time": 26.0}]} for i in range(3)])

    def tearDown(self):
        self.temporary.cleanup()

    def config(self, **kwargs):
        values = {"anno_path": self.annotation, "output_dir": self.root / "outputs"}
        values.update(kwargs)
        return RunConfig(**values)

    def native_run(self, out, method="adakv", budget=1000, complete=True, audited=True):
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "protocol.json", {"method": method, "budget": budget, "budget_scope": "visual"})
        if complete:
            write_json(out / "summary.json", {"method": method, "budget": budget, "budget_scope": "visual",
                       "videos": 3, "questions": 3, "correct": 2, "accuracy_percent": 200 / 3,
                       "peak_gpu_allocated_gib": 15.85})
        (out / "run.status").write_text("complete" if complete else "dry-run")
        if audited:
            write_json(out / "audit_report.json", {"passed": True, "trace": {"compression_events": 9, "qa_restore_checks": 3}})

    def fake_subprocess(self, argv, **kwargs):
        self.calls.append(argv)
        if argv[3] == "sparsemm_protokv.run_streamingbench":
            out = Path(argv[argv.index("--output-dir") + 1])
            method = argv[argv.index("--method") + 1]
            budget = int(argv[argv.index("--budget") + 1])
            self.native_run(out, method, budget, complete="--dry-run" not in argv, audited=False)
        else:
            out = Path(argv[4])
            write_json(out / "audit_report.json", {"passed": True, "trace": {"compression_events": 9, "qa_restore_checks": 3}})
        return subprocess.CompletedProcess(argv, 0)

    def test_defaults_and_explicit_protokv(self):
        self.assertEqual(DEFAULT_METHODS, ("adakv", "pyramidkv", "lukv", "sparsemm", "uniform"))
        self.assertEqual(self.config().selected_methods, DEFAULT_METHODS)
        self.assertEqual(self.config(method="protokv").selected_methods, ("protokv",))
        self.assertFalse(METHODS["protokv"].default)

    def test_nested_horizonkv_checkout_finds_hermes_resources(self):
        horizon = self.root / "HorizonKV"
        project = horizon / "kvbench"
        evaluator = horizon / "eval/eval_multiple_choice.py"
        project.mkdir(parents=True)
        evaluator.parent.mkdir(parents=True)
        evaluator.write_text("", encoding="utf-8")
        with patch.object(paths, "PROJECT_ROOT", project), patch.dict("os.environ", {"KV_BENCH_RESOURCE_ROOT": str(horizon)}):
            self.assertEqual(paths.hermes_source_dir(), horizon)
            self.assertEqual(paths.default_model_path(), horizon / "models/Qwen2.5-VL-7B-Instruct")
            self.assertEqual(paths.ensure_resources(), horizon)
            self.assertEqual((project / "hermes").resolve(), horizon)

    def test_unknown_duplicate_and_conflicting_methods(self):
        for options in ({"method": "unknown"}, {"methods": ("adakv", "adakv")}, {"methods": ()},
                        {"method": "adakv", "methods": ("uniform",)}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.config(**options).validate()

    def test_commands_preserve_safety_and_asset_overrides(self):
        config = self.config(method="lukv", allow_calibration_transfer=True, lukv_profile_path=self.root / "my profile.json")
        command = build_command(config, "lukv", self.root / "run")
        self.assertIn("--small-validation", command)
        self.assertNotIn("--five-video-timing", command)
        self.assertNotIn("--allow-calibration-transfer", command)
        self.assertEqual(command[command.index("--lukv-profile-path") + 1], str(self.root / "my profile.json"))
        full = build_command(self.config(method="full", budget=1000), "full", self.root / "full")
        self.assertEqual(full[full.index("--method") + 1], "adakv")
        self.assertEqual(full[full.index("--budget") + 1], "1000000")
        self.assertEqual(api._environment()["PYTHONPATH"], str(paths.PROJECT_ROOT))

    def test_unsafe_limits_fail_before_subprocess_or_output(self):
        with patch.object(api.subprocess, "run") as process:
            for options in ({"max_videos": 4}, {"max_videos": 0}, {"budget": -1}, {"sample_fps": float("nan")},
                            {"sample_fps": 0}, {"encode_chunk_size": 0}, {"max_questions": 9}):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    run(self.config(**options))
            self.assertFalse((self.root / "outputs").exists())
            process.assert_not_called()

    def test_total_question_limit_and_endpoint_are_checked_before_execution(self):
        with patch.object(api.subprocess, "run") as process:
            write_json(self.annotation, [{"video_id": i, "conversations": [{"end_time": 1}] * 3} for i in range(3)])
            with self.assertRaisesRegex(ValueError, "8 total"):
                run(self.config(max_questions=3))
            write_json(self.annotation, [{"video_id": 1, "conversations": [{"end_time": 120.01}]}])
            with self.assertRaisesRegex(ValueError, "120"):
                run(self.config())
            process.assert_not_called()

    def test_existing_output_is_never_overwritten(self):
        out = self.root / "outputs"
        out.mkdir()
        sentinel = out / "sentinel"
        sentinel.write_text("keep")
        with patch.object(api.subprocess, "run") as process, self.assertRaises(FileExistsError):
            run(self.config())
        self.assertEqual(sentinel.read_text(), "keep")
        process.assert_not_called()

    def test_config_json_conflict_fails_before_execution(self):
        extra = self.root / "config.json"
        write_json(extra, {"method": "protokv"})
        with self.assertRaisesRegex(ValueError, "conflicts"):
            self.config(config_json=extra).validate()

    def test_batch_run_audits_each_completed_method_and_writes_reports(self):
        self.calls = []
        with patch.object(paths, "ensure_resources"), patch.object(api.subprocess, "run", side_effect=self.fake_subprocess):
            result = run(self.config(methods=("adakv", "lukv"), allow_calibration_transfer=True))
        self.assertEqual(len(self.calls), 4)
        self.assertEqual([command[3] for command in self.calls], ["sparsemm_protokv.run_streamingbench", "sparsemm_protokv.audit",
                         "sparsemm_protokv.run_streamingbench", "sparsemm_lukv.audit"])
        self.assertIn("--allow-calibration-transfer", self.calls[-1])
        self.assertTrue(result["all_audits_passed"])
        self.assertEqual(result["status"], "complete")
        self.assertTrue((self.root / "outputs/report.md").is_file())
        manifest = json.loads((self.root / "outputs/suite_manifest.json").read_text())
        self.assertIn("kvbench/api.py", manifest["interface_sha256"])
        self.assertEqual(manifest["effective_config"]["methods"], ["adakv", "lukv"])
        self.assertTrue(manifest["effective_config"]["allow_calibration_transfer"])

    def test_dry_run_does_not_audit_or_claim_success(self):
        self.calls = []
        with patch.object(paths, "ensure_resources"), patch.object(api.subprocess, "run", side_effect=self.fake_subprocess):
            result = run(self.config(method="adakv", dry_run=True))
        self.assertEqual(len(self.calls), 1)
        self.assertIn("--dry-run", self.calls[0])
        self.assertEqual(result["status"], "dry-run")
        self.assertFalse(result["all_audits_passed"])
        self.assertIsNone(result["runs"][0]["correct"])

    def test_subprocess_failure_preserves_failure_evidence(self):
        with patch.object(paths, "ensure_resources"), patch.object(api.subprocess, "run", side_effect=subprocess.CalledProcessError(3, ["native"])):
            with self.assertRaises(subprocess.CalledProcessError):
                run(self.config(method="adakv"))
        manifest = json.loads((self.root / "outputs/suite_manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(json.loads((self.root / "outputs/summary.json").read_text())["status"], "failed")

    def test_single_run_summary_preserves_native_summary(self):
        out = self.root / "single"
        self.native_run(out)
        before = (out / "summary.json").read_bytes()
        result = summarize(out)
        self.assertEqual((out / "summary.json").read_bytes(), before)
        self.assertEqual(result["runs"][0]["correct"], 2)
        self.assertTrue((out / "kvbench_summary.json").is_file())

    def test_missing_audit_is_not_success(self):
        out = self.root / "single"
        self.native_run(out, audited=False)
        result = summarize(out, write=False)
        self.assertFalse(result["all_audits_passed"])
        self.assertEqual(result["status"], "incomplete")

    def test_manifest_complete_does_not_hide_missing_methods_or_failed_audits(self):
        out = self.root / "collection"
        self.native_run(out / "adakv", audited=True)
        write_json(out / "suite_manifest.json", {"status": "complete", "methods": ["adakv", "uniform"]})
        missing = summarize(out, write=False)
        self.assertFalse(missing["selection_complete"])
        self.assertEqual(missing["status"], "incomplete")
        self.native_run(out / "uniform", method="uniform")
        write_json(out / "uniform/audit_report.json", {"passed": False, "error": "trace invalid"})
        failed = summarize(out, write=False)
        self.assertTrue(failed["selection_complete"])
        self.assertFalse(failed["all_audits_passed"])
        self.assertEqual(failed["status"], "incomplete")

    def test_external_lu_dataset_fails_before_any_inference_without_opt_in(self):
        exclusion = self.root / "excluded.json"
        write_json(exclusion, [{"video_id": 1, "video_path": "/source/original.mp4"}])
        profile = self.root / "profile.json"
        write_json(profile, {"calibration": {"exclude_annotation": str(exclusion)}})
        write_json(self.annotation, [{"video_id": 1117, "video_path": "/source/ovo.mp4", "conversations": [{"end_time": 26}]}])
        with patch.object(api.subprocess, "run") as process, self.assertRaisesRegex(ValueError, "allow-calibration-transfer"):
            run(self.config(lukv_profile_path=profile))
        process.assert_not_called()
        self.assertFalse((self.root / "outputs").exists())

    def test_lu_audit_requires_explicit_transfer_option(self):
        out = self.root / "lu"
        self.native_run(out, method="lukv")
        default = api.build_audit_command(out)
        transferred = api.build_audit_command(out, allow_calibration_transfer=True)
        self.assertNotIn("--allow-calibration-transfer", default)
        self.assertIn("--allow-calibration-transfer", transferred)
        with self.assertRaises(ValueError):
            api.build_audit_command(out, method="adakv")

    def test_full_control_rejects_real_evictions(self):
        out = self.root / "full"
        self.native_run(out, budget=1_000_000)
        write_json(out / "kvbench_method.json", {"requested_method": "full"})
        with patch.object(api.subprocess, "run"), self.assertRaisesRegex(RuntimeError, "evicted"):
            audit(out)

    def test_full_summary_requires_zero_audited_compression_events(self):
        out = self.root / "full_summary"
        self.native_run(out, budget=1_000_000)
        write_json(out / "kvbench_method.json", {"requested_method": "full"})
        for count in (9, None, 0):
            with self.subTest(compression_events=count):
                trace = {} if count is None else {"compression_events": count}
                write_json(out / "audit_report.json", {"passed": True, "trace": trace})
                result = summarize(out, write=False)
                self.assertEqual(result["all_audits_passed"], count == 0)
                self.assertEqual(result["runs"][0]["audit_passed"], count == 0)
                self.assertEqual(result["status"], "complete" if count == 0 else "incomplete")
                if count != 0:
                    self.assertIn("compression_events", result["runs"][0]["audit_error"])

    def test_resource_link_checks_do_not_silently_change_targets(self):
        project = self.root / "project"
        resources = self.root / "resources"
        project.mkdir()
        (resources / "hermes").mkdir(parents=True)
        with patch.object(paths, "PROJECT_ROOT", project), patch.dict("os.environ", {"KV_BENCH_RESOURCE_ROOT": str(resources)}):
            paths.ensure_resources()
            self.assertEqual((project / "hermes").resolve(), resources / "hermes")
            other = self.root / "other"
            (other / "hermes").mkdir(parents=True)
            with patch.dict("os.environ", {"KV_BENCH_RESOURCE_ROOT": str(other)}), self.assertRaisesRegex(ValueError, "explicitly update"):
                paths.ensure_resources()
            (project / "hermes").unlink()
            (project / "hermes").symlink_to(self.root / "missing")
            with self.assertRaisesRegex(ValueError, "Broken"):
                paths.ensure_resources()

    def test_cli_dispatch_and_error_codes(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["methods", "--json"]), 0)
            with patch("kvbench.cli.run", return_value={}) as execute:
                self.assertEqual(main(["run", "--methods", "adakv,uniform", "--dry-run"]), 0)
                self.assertEqual(execute.call_args.args[0].selected_methods, ("adakv", "uniform"))
            self.assertEqual(main(["run", "--methods", "unknown"]), 2)
            with patch("kvbench.cli.run", side_effect=subprocess.CalledProcessError(1, ["native"])):
                self.assertEqual(main(["run", "--method", "adakv"]), 1)


if __name__ == "__main__":
    unittest.main()
