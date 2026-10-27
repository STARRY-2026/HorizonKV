"""Check fixed pixels, physical occupancy and native grouped-prefix parity."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from sparsemm_native import run_ovobench as official
from sparsemm_native import run_ovobench_sparsemm_fixed896 as fixed


def test_budget_audit_counts_physical_visual_slots_without_capping_heads():
    values = [100, 200, 500, 1200]
    cache = SimpleNamespace(prefix_tokens=7, layers=[SimpleNamespace(
        head_lens=torch.tensor([v + 7 for v in values])) for _ in range(28)], nbytes=lambda: 900)
    bridge = SimpleNamespace(kv_cache=cache, budgets=torch.tensor([values] * 28))
    audit = fixed.cache_budget_audit(bridge)
    assert audit["actual_tokens_per_head"] == 500
    assert audit["maximum_per_head"] == 1200
    assert audit["actual_visual_slots"] == 56000
    assert audit["individual_quota_cap"] is None
    assert audit["actual_includes_question_tokens"] is False


@pytest.mark.parametrize("deliver_each", [False, True])
def test_grouped_even_prefix_and_odd_branch_match_original_native_flow(deliver_each):
    class Reader:
        def __init__(self, path, num_threads):
            pass

        def __len__(self):
            return 40

        def get_avg_fps(self):
            return 1

        def get_batch(self, indices):
            return SimpleNamespace(asnumpy=lambda: np.asarray(indices))

    class Bridge:
        def __init__(self):
            self.encoded, self.fixed_input_checks = [], []
            self._next_position = self._seen_visual_tokens = self.total_processed_frames = 0
            self.last_encoded_frames = self._prefills = self._compressions = 0
            self.last_metadata, self.conv_history = [], []

        def encode_init_prompt(self):
            self.kv_cache = {"ledger": []}

        def encode_video_chunk(self, frames, is_last_chunk):
            values = frames.tolist()
            self.encoded.append(values)
            self.kv_cache["ledger"].extend(values)
            for offset in range(0, len(values), 2):
                count = len(values[offset:offset + 2])
                self.fixed_input_checks.append(dict(raw_frames=count, visual_tokens=1024))
            self.total_processed_frames += len(values)

        def predict_and_compress(self):
            pass

        def clear_cache(self):
            self.kv_cache = None

    def query(qid, end):
        return dict(query_id=qid, source_id=int(qid), probe_id=None, task="EPM", group="backward",
                    timestamp=float(end), clip_end=float(end), target="A", video_path="source.mp4", prompt="Question")

    # Deliberately preserve nonchronological manifest order: actual execution
    # must be target2 -> target3 -> target4 and return rows in manifest order.
    group = [query("0", 6), query("1", 4), query("2", 8)]
    observations = []

    def answer(bridge, query, max_new_tokens, encoded_frames, *unused):
        observations.append((query["query_id"], list(bridge.kv_cache["ledger"])))
        bridge.conv_history.append((query["query_id"], "A"))
        return "A"

    with patch("sparsemm_native.video_reader.BatchScopedVideoReader", Reader), \
         patch.object(official, "_answer_grouped", side_effect=answer), \
         patch.object(fixed, "cache_budget_audit", return_value={"actual_tokens_per_head": 500}), \
         patch.object(torch.cuda, "synchronize"), patch.object(torch.cuda, "reset_peak_memory_stats"), \
         patch.object(torch.cuda, "max_memory_allocated", return_value=100):
        old = Bridge()
        original = official.run_grouped_video(old, group, .5, 16, 16)
        old_observed = observations[:]
        observations.clear()
        new = Bridge()
        delivered = []

        def on_result(row):
            # Incremental delivery precedes encoding of the next answer and
            # observes the restored even base, including after an odd answer.
            delivered.append((row["query_id"], list(new.kv_cache["ledger"]),
                              new.total_processed_frames, len(new.encoded)))

        rows = fixed.run_group(new, group, on_result=on_result if deliver_each else None)

    assert observations == old_observed == [("1", [0, 2]), ("0", [0, 2, 4]), ("2", [0, 2, 4, 6])]
    assert new.encoded == old.encoded == [[0, 2], [4], [4, 6]]
    assert [r["query_id"] for r in rows] == ["0", "1", "2"]
    assert [r["frames"] for r in rows] == [original[str(i)][1] for i in range(3)] == [3, 2, 4]
    assert [r["input_audit"]["prefix_encoded_visual_tokens"] for r in rows] == [2048, 1024, 2048]
    assert [r["encoded_frames"] for r in rows] == [1, 2, 2]
    assert new.fixed_input_checks == []
    if deliver_each:
        assert delivered == [("1", [0, 2], 2, 1), ("0", [0, 2], 2, 2),
                             ("2", [0, 2, 4, 6], 4, 3)]
