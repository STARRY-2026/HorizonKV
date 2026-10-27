"""Independent failures and matched-subset checks for the three-way reporter."""

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from sparsemm_native import compare_ovobench_three_way as reporting


class ThreeWayComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name) / "baseline"
        self.dual = Path(self.temp.name) / "dual"
        self.expected = [
            dict(query_id=str(index), source_id=index, probe_id=None, task=task,
                 group=group, timestamp=float(index + 1), clip_end=float(index + 1),
                 target="A", prompt=f"Pinned prompt {index}", video_path=f"/video{index}.mp4")
            for index, task, group in [(0, "EPM", "backward"), (1, "STU", "realtime"),
                                      (2, "HLD", "backward")]
        ]
        self.addCleanup(patch.stopall)
        patch.object(reporting, "hermes_queries", return_value=self.expected).start()
        matrix = [[500] * 4 for _ in range(28)]
        self.allocation = dict(head_budgets=matrix, allocated_slots=56000, target_slots=56000,
                               window_size=32, dp_bank_split_used_online=False)
        self.protocols = {}
        self.rows = {}
        for arm in reporting.ARMS:
            protocol = dict(source_sha256="source", model_config_sha256="config", budget=500,
                            fps=0.5, dtype="bfloat16", chunk_size=16, max_new_tokens=16,
                            video_max_pixels=151200, score_mode="HERMES_exact_MC_letter",
                            video_mode="source_prefix_ceil_timestamp", model_path="/model",
                            dataset="hermes9", arm=arm, query_count=3, dp_budget_sha256="original_budget",
                            query_list_sha256=reporting._digest(self.expected))
            protocol["dp_allocation"] = copy.deepcopy(self.allocation) if arm != "native" else None
            if arm == "dp_dual":
                protocol.update(query_ids=[q["query_id"] for q in self.expected],
                                full_query_list_sha256=reporting._digest(self.expected),
                                cache_policy="two_bank", protect_window=32,
                                attention_observation_window=32,
                                protect_window_included_in_recent=True, debug_explain=False,
                                decoding="greedy", repetition_penalty=1.1,
                                reference_dp_budget_sha256="original_budget")
                protocol["dp_allocation"].update(
                    dp_bank_split_used_online=True, protect_window_included_in_recent=True,
                    short_cache_gate="legacy_global", recent_budgets=[[200] * 4 for _ in range(28)],
                    history_budgets=[[300] * 4 for _ in range(28)])
            self.protocols[arm] = protocol
            self.rows[arm] = []
            for index, query in enumerate(self.expected):
                row = {key: query[key] for key in reporting.IDENTITY}
                correct = index != {"native": 0, "dp": 1, "dp_dual": 2}[arm]
                row.update(response="A" if correct else "B", correct=correct, frames=index + 2)
                self.rows[arm].append(row)
        self.write()

    def directory(self, arm):
        return (self.dual if arm == "dp_dual" else self.base) / arm

    def write(self):
        for arm in reporting.ARMS:
            directory = self.directory(arm)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "protocol.json").write_text(json.dumps(self.protocols[arm]))
            (directory / "queries.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in self.rows[arm]))
        trace = {"event": "video_prefill", "layers": [{"allocation": self.allocation}]}
        (self.directory("native") / "cache_trace.jsonl").write_text(json.dumps(trace) + "\n")

    def compare(self):
        self.write()
        return reporting.compare(self.base, self.dual)

    def test_full_matched_scores_and_disagreement_counts(self):
        result = self.compare()
        self.assertTrue(result["full_dataset_complete"])
        self.assertEqual(result["subtotal"]["matched_queries"], 3)
        self.assertEqual(result["subtotal"]["dp_dual_correct"], 2)
        self.assertEqual(result["subtotal"]["dual_only_vs_native"], 1)
        self.assertEqual(result["subtotal"]["native_only_vs_dual"], 1)
        self.assertIn("not pool", result["score_note"])
        self.assertIn("Matched weighted subtotal", reporting.markdown(result))

    def test_selected_subset_and_incomplete_prefix_use_only_matched_ids(self):
        selected = [self.expected[0], self.expected[2]]
        protocol = self.protocols["dp_dual"]
        protocol.update(query_count=2, query_ids=["0", "2"],
                        query_list_sha256=reporting._digest(selected))
        self.rows["dp_dual"] = self.rows["dp_dual"][:1]
        result = self.compare()
        self.assertFalse(result["dual_complete"])
        self.assertEqual(result["query_ids"], ["0"])
        self.assertEqual(result["subtotal"]["native_correct"], 0)
        self.assertEqual(result["subtotal"]["dp_correct"], 1)
        self.assertEqual(result["tasks"]["HLD"]["matched_queries"], 0)
        self.assertIsNone(result["tasks"]["HLD"]["dp_dual_accuracy"])

    def test_empty_progress_is_valid(self):
        self.rows["dp_dual"] = []
        result = self.compare()
        self.assertEqual(result["subtotal"]["matched_queries"], 0)
        self.assertIsNone(result["subtotal"]["dp_dual_accuracy"])

    def test_every_pinned_protocol_field_rejects_change(self):
        for field in reporting.PROTOCOL_FIELDS + ("dataset", "model_path"):
            with self.subTest(field=field):
                original = self.protocols["dp_dual"][field]
                self.protocols["dp_dual"][field] = "changed"
                with self.assertRaisesRegex(ValueError, "mismatch|dataset"):
                    self.compare()
                self.protocols["dp_dual"][field] = original

    def test_prompt_hash_and_subset_order_are_pinned(self):
        self.protocols["dp_dual"]["full_query_list_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "full expected"):
            self.compare()
        self.protocols["dp_dual"]["full_query_list_sha256"] = reporting._digest(self.expected)
        self.protocols["dp_dual"]["query_ids"] = ["1", "0", "2"]
        with self.assertRaisesRegex(ValueError, "ordered subset"):
            self.compare()

    def test_every_identity_field_rejects_change(self):
        for field in reporting.IDENTITY:
            with self.subTest(field=field):
                original = self.rows["dp_dual"][0][field]
                self.rows["dp_dual"][0][field] = "changed"
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    self.compare()
                self.rows["dp_dual"][0][field] = original

    def test_duplicates_frames_and_scoring_are_rejected(self):
        original = copy.deepcopy(self.rows["dp_dual"])
        self.rows["dp_dual"][1] = copy.deepcopy(self.rows["dp_dual"][0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.compare()
        self.rows["dp_dual"] = copy.deepcopy(original)
        self.rows["dp_dual"][0]["frames"] += 1
        with self.assertRaisesRegex(ValueError, "frames differ"):
            self.compare()
        self.rows["dp_dual"] = copy.deepcopy(original)
        self.rows["dp_dual"][0]["correct"] = False
        with self.assertRaisesRegex(ValueError, "correctness disagrees"):
            self.compare()

    def test_capacity_split_and_protection_contract_are_enforced(self):
        allocation = self.protocols["dp_dual"]["dp_allocation"]
        for field, changed in [("dp_bank_split_used_online", False),
                               ("protect_window_included_in_recent", False),
                               ("short_cache_gate", "global"), ("window_size", 31)]:
            with self.subTest(field=field):
                original = allocation[field]
                allocation[field] = changed
                with self.assertRaises(ValueError):
                    self.compare()
                allocation[field] = original
        allocation["head_budgets"][0][0] += 1
        allocation["head_budgets"][0][1] -= 1
        with self.assertRaisesRegex(ValueError, "changes baseline"):
            self.compare()
        allocation["head_budgets"][0][0] -= 1
        allocation["head_budgets"][0][1] += 1
        allocation["recent_budgets"][0][0] = 31
        with self.assertRaisesRegex(ValueError, "recent_budgets"):
            self.compare()

    def test_full_baselines_required_and_native_slots_checked(self):
        self.rows["native"] = self.rows["native"][:1]
        with self.assertRaisesRegex(ValueError, "baseline must contain"):
            self.compare()
        self.rows["native"] = copy.deepcopy(self.rows["dp"])
        self.allocation["allocated_slots"] = 56001
        with self.assertRaisesRegex(ValueError, "Native baseline.*56000"):
            self.compare()

    def test_cli_writes_both_formats(self):
        output = Path(self.temp.name) / "report"
        with patch("builtins.print"):
            reporting.main(["--baseline-root", str(self.base), "--dual-root", str(self.dual),
                            "--output", str(output)])
        self.assertTrue((output / "summary.json").is_file())
        self.assertTrue((output / "summary.md").is_file())


if __name__ == "__main__":
    unittest.main()
