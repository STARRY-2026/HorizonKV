"""Official-manifest, safe reuse, dynamic allocation, and prefix-branch checks."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from sparsemm_native import run_ovobench_recent4 as runner
from sparsemm_native import report_ovobench_recent4 as reporting


def query(index, video=None, task="EPM", clip_end=None):
    return dict(query_id=str(index), source_id=index, probe_id=None, task=task,
                group="forward" if task in runner.legacy.FORWARD else "backward",
                video_path=video or f"video{index}.mp4", prompt=f"official prompt {index}",
                timestamp=float(index + 1), clip_end=float(clip_end or index + 1), target="A")


def row(item, response="A"):
    result = {key: item[key] for key in runner.IDENTITY}
    result.update(response=response, correct=runner.legacy.official_correct(item["task"], response, item["target"]),
                  frames=4, seconds=1.0, prompt_sha256=runner.digest(item["prompt"]))
    return result


class OfficialRecent4Tests(unittest.TestCase):
    def test_official_dataset_is_3035_queries_and_644_global_video_groups(self):
        expected = runner.legacy.queries()
        self.assertEqual(len(expected), 3035)
        scheduled = runner.schedule_queries(expected)
        groups = runner.legacy.grouped_batches(scheduled, 10000)
        self.assertEqual(len(groups), 644)
        self.assertEqual({q["query_id"] for q in scheduled}, {q["query_id"] for q in expected})
        self.assertTrue(all(len({q["video_path"] for q in group}) == 1 for group in groups))
        self.assertEqual(len({group[0]["video_path"] for group in groups}), 644)
        official = {q["query_id"]: q for q in expected}
        self.assertTrue(all(q == official[q["query_id"]] for q in scheduled))

    def test_nonadjacent_same_video_is_grouped_without_changing_query_content(self):
        selected = [query(0, "a"), query(1, "b"), query(2, "a"), query(3, "b")]
        self.assertEqual([q["query_id"] for q in runner.schedule_queries(selected)], ["0", "2", "1", "3"])
        self.assertEqual(runner.schedule_queries(selected, grouped=False), selected)

    def test_cli_is_grouped_by_default_and_pins_comparison_inputs(self):
        argv = ["--arm", "dp_recent4", "--output-root", "/tmp/unused"]
        self.assertTrue(runner.parse_args(argv).grouped)
        self.assertFalse(runner.parse_args(argv + ["--fresh"]).grouped)
        self.assertEqual(runner.parse_args(argv + ["--concurrent-processes", "2"]).concurrent_processes, 2)
        for flag, value in [("--budget", "1000"), ("--fps", "1"), ("--chunk-size", "8"),
                            ("--max-new-tokens", "32"), ("--video-max-pixels", "3136")]:
            with self.subTest(flag=flag), self.assertRaises(SystemExit):
                runner.parse_args(argv + [flag, value])

    def test_protocol_fingerprints_new_dynamic_source_and_scheduling(self):
        args = runner.parse_args(["--arm", "dp_recent4", "--output-root", "/tmp/unused"])
        expected = [query(0)]
        with patch.object(runner.legacy, "sha256", side_effect=lambda path: str(Path(path).resolve())), \
                patch("importlib.metadata.version", return_value="test-version"):
            protocol = runner.build_protocol(args, expected, expected)
        self.assertIn("sparsemm_native/recent4.py", protocol["code_sha256"])
        self.assertIn("sparsemm_native/gpu_slots.py", protocol["code_sha256"])
        self.assertEqual(protocol["concurrent_processes"], 1)
        self.assertEqual(protocol["concurrency_mode"], "exclusive_gpu0")
        self.assertEqual(protocol["query_list_sha256"], runner.digest(expected))
        self.assertEqual(protocol["score_mode"], "official_OVOBenchOfflineScore")

    def test_main_rejects_protocol_drift_before_loading_model(self):
        from tempfile import TemporaryDirectory

        expected = [query(0)]
        with TemporaryDirectory() as temp:
            directory = Path(temp) / "dp_recent4"
            directory.mkdir()
            (directory / "queries.jsonl").write_text(json.dumps(row(expected[0])) + "\n")
            (directory / "protocol.json").write_text(json.dumps({"code_sha256": "original"}))
            with patch.object(runner.legacy, "queries", return_value=expected), \
                    patch.object(runner, "build_protocol", return_value={"code_sha256": "changed"}), \
                    self.assertRaisesRegex(ValueError, "refusing to resume"):
                runner.main(["--arm", "dp_recent4", "--output-root", temp])

    def test_ids_preserve_official_order_and_reject_duplicates_or_unknowns(self):
        expected = [query(0), query(1), query(2)]
        self.assertEqual(runner.select_queries(expected, "2,0"), [expected[0], expected[2]])
        for ids in ("1,1", "0,", "3"):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                runner.select_queries(expected, ids)

    def test_resume_requires_exact_identity_prompt_and_official_scoring(self):
        expected = query(0)
        runner.validate_row(row(expected), expected)
        for field in runner.IDENTITY + ("prompt_sha256", "correct", "frames"):
            with self.subTest(field=field):
                changed = row(expected)
                changed[field] = False if field == "correct" else "changed"
                with self.assertRaises(ValueError):
                    runner.validate_row(changed, expected)

    def test_reuse_accepts_official_results_and_rejects_hermes_or_prompt_drift(self):
        from tempfile import TemporaryDirectory

        expected = [query(0), query(1)]
        reference = dict(model_path="/model", dp_budget_sha256="dp",
                         **{field: field for field in runner.COMMON_FIELDS})
        source = dict(reference, dataset="official_full", arm="native", query_count=2,
                      query_list_sha256=runner.digest(expected))
        with TemporaryDirectory() as temp:
            directory = Path(temp) / "native"
            directory.mkdir()
            progress = row(expected[0])
            progress.pop("prompt_sha256")  # Prior official runs did not save this field.
            (directory / "queries.jsonl").write_text(json.dumps(progress) + "\n")
            protocol = directory / "protocol.json"
            protocol.write_text(json.dumps(source))
            adopted = runner.reusable_rows([temp], "native", expected, reference)
            self.assertEqual(adopted["0"]["response"], "A")
            self.assertEqual(adopted["0"]["seconds"], 0.0)
            self.assertEqual(adopted["0"]["prompt_sha256"], runner.digest(expected[0]["prompt"]))
            for key, value in [("dataset", "hermes9"), ("query_list_sha256", "drift"), ("fps", 1.0)]:
                with self.subTest(key=key):
                    protocol.write_text(json.dumps(dict(source, **{key: value})))
                    with self.assertRaises(ValueError):
                        runner.reusable_rows([temp], "native", expected, reference)

    def test_dynamic_allocation_validation_uses_each_snapshot_instead_of_old_static_table(self):
        allocation = dict(method="dp_recent4_history_proportional", head_budgets=[[500] * 4 for _ in range(28)],
                          recent_budgets=[[360] * 4 for _ in range(28)], history_budgets=[[140] * 4 for _ in range(28)],
                          recent_unit_token_counts=[180, 180], recent_tokens_per_head=360,
                          target_visual_slots=56000, allocated_visual_slots=56000,
                          protect_window_included_in_recent=True, recent_temporal_units=2, short_cache_gate="legacy_global")
        check = dict(validation_passed=True, layer_idx=0, head_budgets=[500] * 4, allocation=allocation)
        bridge = SimpleNamespace(is_recent4=True, num_layers=28, num_kv_heads=4,
                                 budgets=torch.full((28, 4), 375), validation_checks={})

        def reset(item):
            bridge.validation_checks = dict(selector_checks=[item], qa_checks=[dict(returned=True, restored=True)], failures=[])

        reset(copy.deepcopy(check))
        runner.validate_runtime(bridge, encoded_frames=True)
        for field, value in [("recent_unit_token_counts", [180]), ("allocated_visual_slots", 56001),
                             ("protect_window_included_in_recent", False), ("recent_tokens_per_head", 359)]:
            with self.subTest(field=field):
                changed = copy.deepcopy(check)
                changed["allocation"][field] = value
                reset(changed)
                with self.assertRaises(RuntimeError):
                    runner.validate_runtime(bridge, encoded_frames=True)

    def test_official_answer_has_no_mc_prefill_and_clears_conversation_history(self):
        class Bridge:
            conv_history = []

            def get_prompt(self, text, mc=False):
                self.mc = mc
                return text

            def question_answering(self, prompt, max_new_tokens):
                self.conv_history.append(("question", "A"))
                return "A"

        bridge = Bridge()
        with patch.object(runner, "validate_runtime"):
            self.assertEqual(runner.answer(bridge, query(0), 16, encoded_frames=False), "A")
        self.assertFalse(bridge.mc)
        self.assertEqual(bridge.conv_history, [])


class PrefixReuseTests(unittest.TestCase):
    def test_odd_endpoint_branch_restores_cache_and_each_question_reads_its_exact_prefix(self):
        class Batch:
            def __init__(self, data):
                self.data = data

            def asnumpy(self):
                return np.asarray(self.data)

        class Reader:
            def __init__(self, path, num_threads):
                pass

            def __len__(self):
                return 20

            def get_avg_fps(self):
                return 1

            def get_batch(self, indices):
                return Batch(indices)

        class Bridge:
            def __init__(self):
                self.encoded = []
                self._next_position = self._seen_visual_tokens = self.total_processed_frames = 0
                self.last_encoded_frames = self._prefills = self._compressions = 0
                self.last_metadata, self.conv_history = [], []

            def encode_init_prompt(self):
                self.kv_cache = {"ledger": []}

            def encode_video_chunk(self, frames, is_last_chunk):
                values = frames.tolist()
                self.encoded.append(values)
                self.kv_cache["ledger"].extend(values)
                self.total_processed_frames += len(values)

            def predict_and_compress(self):
                pass

            def clear_cache(self):
                self.kv_cache = None

        observed = []

        def answer(bridge, query, max_new_tokens, *, encoded_frames):
            observed.append((query["query_id"], list(bridge.kv_cache["ledger"])))
            return "A"

        bridge = Bridge()
        group = [query(0, "a", clip_end=6), query(1, "a", clip_end=4), query(2, "a", clip_end=8)]
        with patch("sparsemm_baselines.run_streamingbench.sampled_indices", return_value=list(range(10))), \
                patch.dict("sys.modules", {"sparsemm_protokv.models.qwen2_5_vl_stream": SimpleNamespace(BatchScopedVideoReader=Reader)}), \
                patch.object(runner, "answer", side_effect=answer):
            result = runner.run_grouped_video(bridge, group)
        self.assertEqual(observed, [("1", [0, 1]), ("0", [0, 1, 2]), ("2", [0, 1, 2, 3])])
        self.assertEqual([result[str(index)][1] for index in range(3)], [3, 2, 4])
        # Frame 2 is encoded in an odd branch and again in the retained even
        # stream. Other frames are reused; the odd branch never leaks frame 2.
        self.assertEqual(bridge.encoded, [[0, 1], [2], [2, 3]])


class OfficialReportTests(unittest.TestCase):
    def test_official_total_is_hierarchical_and_requires_all_matched_queries(self):
        from tempfile import TemporaryDirectory

        official = runner.legacy.queries()
        selected = runner.schedule_queries(official)
        common = dict(dataset="official_full", score_mode="official_OVOBenchOfflineScore", grouped=True,
                      full_query_list_sha256=runner.digest(official), query_list_sha256=runner.digest(selected),
                      query_ids=[q["query_id"] for q in selected], query_count=3035)
        with TemporaryDirectory() as temp:
            for arm in runner.ARMS:
                directory = Path(temp) / arm
                directory.mkdir()
                (directory / "protocol.json").write_text(json.dumps(dict(common, arm=arm)))
                answers = [row(q, response=q["target"] if q["group"] != "realtime" else "?") for q in selected]
                (directory / "queries.jsonl").write_text("".join(json.dumps(r) + "\n" for r in answers))
            result = reporting.report(temp)
            self.assertTrue(result["complete"])
            self.assertAlmostEqual(result["official_total_average"]["dp_recent4"], 2 / 3)
            self.assertNotAlmostEqual(result["matched"]["dp_recent4"]["accuracy"], 2 / 3)
            directory = Path(temp) / "dp_recent4"
            lines = (directory / "queries.jsonl").read_text().splitlines()
            (directory / "queries.jsonl").write_text("\n".join(lines[:-1]) + "\n")
            incomplete = reporting.report(temp)
            self.assertFalse(incomplete["complete"])
            self.assertTrue(all(score is None for score in incomplete["official_total_average"].values()))


if __name__ == "__main__":
    unittest.main()
