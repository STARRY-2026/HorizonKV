"""Window causality, actual budget accounting, reuse and resumable shard safety."""

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sparsemm_native import run_ovobench_simplestream as runner
from sparsemm_native import supervise_ovobench_simplestream as supervisor


class WindowTests(unittest.TestCase):
    def test_penalty_ignores_prompt_tokens_and_matches_generated_token_sign_rule(self):
        import torch
        processor = runner.GeneratedOnlyRepetitionPenalty(3, torch.bfloat16)
        scores = torch.tensor([[2., -2., 3., -3.]])
        # Tokens 0 and 1 appear in the prompt, but have not been generated.
        untouched = processor(torch.tensor([[0, 1, 0]]), scores.clone())
        self.assertTrue(torch.equal(untouched, scores.to(torch.bfloat16)))
        changed = processor(torch.tensor([[0, 1, 0, 2, 3, 2]]), scores.clone())
        self.assertEqual(float(changed[0, 0]), 2.)
        self.assertEqual(float(changed[0, 1]), -2.)
        self.assertEqual(changed[0, 2], torch.tensor(3., dtype=torch.bfloat16) / 1.1)
        self.assertEqual(changed[0, 3], torch.tensor(-3., dtype=torch.bfloat16) * 1.1)

    def test_windows_use_the_existing_prefix_and_never_future_frames(self):
        indices = list(range(0, 1000, 60))
        count, ordinals, raw = runner.window_plan({"clip_end": 19}, indices)
        self.assertEqual((count, ordinals, raw), (10, [6, 7, 8, 9], [360, 420, 480, 540]))
        self.assertEqual(runner.window_plan({"clip_end": 3}, indices), (2, [0, 1], [0, 60]))
        self.assertEqual(runner.window_plan({"clip_end": 999}, indices)[0], len(indices))
        # Odd endpoints retain four actual images, without repeating the last.
        self.assertEqual(runner.window_plan({"clip_end": 17}, indices)[1], [5, 6, 7, 8])

    def test_actual_kv_measurements_count_k_and_v_and_all_heads(self):
        import torch
        cache = [(torch.zeros(1, 4, 800, 128, dtype=torch.bfloat16),
                  torch.zeros(1, 4, 800, 128, dtype=torch.bfloat16)) for _ in range(2)]
        value = runner.cache_measurement(cache)
        self.assertEqual(value["physical_slots"], 2 * 4 * 800)
        self.assertEqual(value["bytes"], 2 * 2 * 4 * 800 * 128 * 2)
        self.assertEqual(value["tokens_per_head"], 800)

    def test_visual_reuse_preserves_every_fresh_question_window_and_response(self):
        import numpy as np
        from sparsemm_protokv.models import qwen2_5_vl_stream as video
        queries = []
        for i, end in enumerate((12, 14, 14, 9)):
            queries.append(dict(query_id=str(i), source_id=i, probe_id=None, task="EPM", group="backward",
                                video_path="onevideo", timestamp=float(end), clip_end=float(end),
                                prompt=f"Question {i}", target="A"))

        class Reader:
            def __init__(self, *args, **kwargs):
                pass
            def __len__(self):
                return 1000
            def get_avg_fps(self):
                return 30.0
            def get_batch(self, indices):
                return SimpleNamespace(asnumpy=lambda: np.array(indices))

        class Adapter:
            torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))
            def __init__(self):
                self.encoded = 0
                self.calls = []
            def encode_images(self, arrays):
                self.encoded += len(arrays)
                return [int(i) for i in arrays]
            def answer(self, features, prompt):
                self.calls.append((features, prompt))
                visual = len(features) * 180
                return "A", dict(visual_tokens=visual, visual_tokens_per_frame=[180] * len(features),
                    input_tokens=visual + 50, prefill_kv={"tokens_per_head": visual + 50},
                    initial_cache_tokens=0, fresh_lm_prefill=True)

        reused, fresh = Adapter(), Adapter()
        with patch.object(video, "BatchScopedVideoReader", Reader):
            a = list(runner.run_group(reused, queries, reuse_vision=True))
            b = list(runner.run_group(fresh, queries, reuse_vision=False))
        self.assertEqual(reused.calls, fresh.calls)
        self.assertEqual([r["response"] for r in a], [r["response"] for r in b])
        self.assertLess(reused.encoded, fresh.encoded)
        self.assertEqual(a[2]["encoded_frames"], 0)
        self.assertEqual(a[2]["reused_image_frames"], 4)
        self.assertEqual(reused.calls[-1][0], [60, 120, 180, 240])
        corrupted = dict(a[0], initial_cache_tokens=800)
        with self.assertRaisesRegex(ValueError, "fresh empty LM cache"):
            runner.validate_result(corrupted, queries[0])
        corrupted = dict(a[0], window_sample_ordinals=[0, 1, 2, 3])
        with self.assertRaisesRegex(ValueError, "Invalid window"):
            runner.validate_result(corrupted, queries[0])


class SupervisionTests(unittest.TestCase):
    def test_full_dataset_is_disjoint_and_exact_and_workers_only_run_new_control(self):
        official = runner.official.queries()
        selected = runner.schedule_queries(official)
        jobs = supervisor.split_videos(selected)
        ids = [q["query_id"] for job in jobs.values() for q in job["queries"]]
        self.assertEqual(len(ids), 3035)
        self.assertEqual(len(set(ids)), 3035)
        videos = [{q["video_path"] for q in job["queries"]} for job in jobs.values()]
        self.assertFalse(videos[0] & videos[1])
        self.assertEqual(len(videos[0] | videos[1]), 644)
        self.assertEqual(sum(j["expected_encoded_frames"] for j in jobs.values()), 9579)
        for name, job in jobs.items():
            self.assertEqual(runner.schedule_queries(runner.select_queries(official, ",".join(job["query_ids"]))),
                             job["queries"])
            cmd = supervisor.worker_command(Path("/tmp/ovo-test"), name, job)
            self.assertEqual(cmd[3], "sparsemm_native.run_ovobench_simplestream")
            self.assertEqual(cmd[-2:], ["--concurrent-processes", "2"])

    def test_real_preparation_validates_reference_and_refuses_changed_manifest(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, selected, official, reference, jobs = supervisor.prepare(root)
            self.assertEqual(len(reference), 3035)
            self.assertFalse(manifest["reference_dp_inference_running"])
            self.assertFalse(manifest["run_sparsemm_baseline"])
            self.assertEqual(supervisor.prepare(root)[0], manifest)
            report = supervisor.publish(root, official, selected, {}, reference, {"state": "preflight"})
            self.assertEqual(report["matched_queries"], 0)
            self.assertIsNone(report["summaries"]["simplestream"]["official_total_average"])
            self.assertFalse((root / runner.ARM / "queries.jsonl").exists())
            changed = copy.deepcopy(manifest)
            changed["concurrent_processes"] = 3
            (root / "supervision_manifest.json").write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "Pinned code/assets/shards changed"):
                supervisor.prepare(root)

    def test_matched_statistics_do_not_mix_unanswered_dp_questions(self):
        reference = {"0": {"correct": True}, "1": {"correct": False}, "2": {"correct": True}}
        rows = [{"query_id": "0", "correct": False}, {"query_id": "1", "correct": True}]
        value = supervisor.comparison_metrics(rows, reference)
        self.assertEqual(value["n"], 2)
        self.assertEqual(value["dp_accuracy"], .5)
        self.assertEqual(value["simplestream_accuracy"], .5)
        self.assertEqual(value["dp_only_correct"], 1)
        self.assertEqual(value["simplestream_only_correct"], 1)


if __name__ == "__main__":
    unittest.main()
