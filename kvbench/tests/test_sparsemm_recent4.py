"""CPU checks for uniform two-unit recent windows and global history quotas.

The cache tests track actual token identities across a moving window, changing
temporal-unit sizes, a branch, and an increase in historical capacity.
"""

import copy
import csv
import hashlib
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from sparsemm_native.recent4 import Recent4Cache, allocate_recent4_budgets
from sparsemm_protokv.config import StreamingConfig


ROOT = Path(__file__).resolve().parents[2]
KVROOT = ROOT / "kvbench"
WEIGHT_PATH = ROOT / (
    "hermes/results/myobservations/"
    "two_bank_grid_dp_k500_floor375_recent32_20261020/"
    "per_head_allocation_fixed_capacity.csv"
)
PAUSED_SNAPSHOT = KVROOT / (
    "results/ovobench_dp_dual_full_k500_fps05_p151200_20261020/"
    "hermes9/dp_dual/source_snapshot"
)


def token_keys(start, count):
    ids = torch.arange(start, start + count, dtype=torch.float32)
    return torch.stack((ids, torch.zeros_like(ids)), dim=-1)


def actual_history_weights():
    weights = torch.full((28, 4), -1, dtype=torch.long)
    with WEIGHT_PATH.open(newline="") as handle:
        for row in csv.DictReader(handle):
            weights[int(row["layer"]), int(row["kv_head"])] = int(row["history_q"])
    if bool((weights < 0).any()):
        raise AssertionError("The tested history-weight table is incomplete")
    return weights


class Recent4AllocationTests(unittest.TestCase):
    def test_exact_global_total_and_real_weight_examples(self):
        weights = actual_history_weights()
        self.assertEqual(int(weights.sum()), 29896)
        for tokens, recent, history_total, maximum in (
                (192, 384, 12992, 312),
                (180, 360, 15680, 377),
                (66, 132, 41216, 990)):
            with self.subTest(unit_tokens=tokens):
                total, window, metadata = allocate_recent4_budgets(weights, [tokens, tokens])
                history = total - window
                self.assertEqual(total.dtype, torch.long)
                self.assertEqual(window.dtype, torch.long)
                self.assertEqual(total.shape, torch.Size((28, 4)))
                self.assertTrue(torch.equal(window, torch.full_like(weights, recent)))
                self.assertEqual(int(total.sum()), 56000)
                self.assertEqual(int(history.sum()), history_total)
                self.assertEqual(int(history.max()), maximum)
                self.assertEqual(metadata["recent_tokens_per_head"], recent)
                self.assertEqual(metadata["target_visual_slots"], 56000)
                self.assertEqual(metadata["allocated_visual_slots"], 56000)
                self.assertEqual(metadata["history_weight_sum"], 29896)

    def test_metadata_exposes_the_actual_loaded_capacities(self):
        weights = torch.tensor([[0, 3], [7, 11]], dtype=torch.long)
        total, recent, metadata = allocate_recent4_budgets(weights, [180, 192])
        self.assertEqual(metadata["head_budgets"], total.tolist())
        self.assertEqual(metadata["recent_budgets"], recent.tolist())
        self.assertEqual(metadata["history_budgets"], (total - recent).tolist())
        self.assertEqual(metadata["recent_unit_token_counts"], [180, 192])
        self.assertEqual(metadata["rounding"], "largest_remainder_layer_head_ties")

    def test_zero_history_weights_remain_zero(self):
        weights = actual_history_weights()
        total, recent, _ = allocate_recent4_budgets(weights, [192, 192])
        history = total - recent
        self.assertEqual(history[0, 1].item(), 0)
        self.assertEqual(history[1, 2].item(), 0)
        self.assertTrue(torch.equal(history[weights == 0], torch.zeros_like(history[weights == 0])))

    def test_largest_remainder_uses_deterministic_layer_head_ties(self):
        # 1401 history slots / weight sum 6: both weight-1 heads have
        # remainder 3, and only one of them can receive the final position.
        weights = torch.tensor([[1, 1, 4]], dtype=torch.long)
        for _ in range(4):
            total, recent, _ = allocate_recent4_budgets(weights, [33])
            self.assertEqual((total - recent).tolist(), [[234, 233, 934]])

    def test_integer_quotas_stay_within_one_position_of_exact_proportions(self):
        weights = actual_history_weights()
        total, recent, _ = allocate_recent4_budgets(weights, [180, 192])
        history = total - recent
        pool, denominator = int(history.sum()), int(weights.sum())
        self.assertTrue(bool((torch.abs(history * denominator - weights * pool)
                              < denominator).all()))

    def test_old_per_head_capacity_bounds_are_not_applied(self):
        total, recent, _ = allocate_recent4_budgets(
            torch.tensor([[0, 1]], dtype=torch.long), [66, 66])
        self.assertEqual(recent.tolist(), [[132, 132]])
        self.assertEqual(total.tolist(), [[132, 868]])
        self.assertEqual(int(total.sum()), 1000)

    def test_first_unit_reserves_only_available_recent_unit(self):
        total, recent, metadata = allocate_recent4_budgets(
            torch.tensor([[1, 3]], dtype=torch.long), [192])
        self.assertEqual(recent.tolist(), [[192, 192]])
        self.assertEqual(metadata["recent_unit_token_counts"], [192])
        self.assertEqual(int(total.sum()), 1000)

    def test_variable_size_units_are_added_instead_of_doubling_newest(self):
        _, recent, metadata = allocate_recent4_budgets(
            torch.tensor([[1, 3]], dtype=torch.long), [180, 66])
        self.assertEqual(recent.tolist(), [[246, 246]])
        self.assertEqual(metadata["recent_unit_token_counts"], [180, 66])

    def test_protect32_is_included_in_recent_instead_of_added(self):
        total, recent, _ = allocate_recent4_budgets(
            torch.tensor([[1, 3]], dtype=torch.long), [32])
        self.assertEqual(recent.tolist(), [[32, 32]])
        self.assertEqual(int((total - recent).sum()), 2 * (500 - 32))
        self.assertEqual(int(total.sum()), 1000)

    def test_recent_equal_to_mean_budget_leaves_no_history(self):
        total, recent, _ = allocate_recent4_budgets(
            torch.tensor([[0, 3]], dtype=torch.long), [250, 250])
        self.assertEqual(total.tolist(), [[500, 500]])
        self.assertTrue(torch.equal(total, recent))

    def test_infeasible_recent_sizes_fail_explicitly(self):
        weights = torch.ones((2, 2), dtype=torch.long)
        for units in ([31], [250, 251], [501]):
            with self.subTest(units=units), self.assertRaises(ValueError):
                allocate_recent4_budgets(weights, units)

    def test_bad_unit_ledger_is_rejected(self):
        weights = torch.ones((2, 2), dtype=torch.long)
        for units in ([], [40, 40, 40], [0, 40], [-1, 40], [32.5], [True, 40]):
            with self.subTest(units=units), self.assertRaises(ValueError):
                allocate_recent4_budgets(weights, units)

    def test_invalid_history_weights_are_rejected(self):
        invalid = (
            torch.zeros((2, 2), dtype=torch.long),
            torch.tensor([[1, -1]], dtype=torch.long),
            torch.ones((2, 2), dtype=torch.float32),
            torch.ones(4, dtype=torch.long),
            torch.empty((0, 2), dtype=torch.long),
            torch.ones((2, 2), dtype=torch.bool),
        )
        for weights in invalid:
            with self.subTest(weights=weights), self.assertRaises((ValueError, TypeError)):
                allocate_recent4_budgets(weights, [64])

    def test_allocator_does_not_mutate_inputs(self):
        weights = torch.tensor([[0, 3], [7, 11]], dtype=torch.long)
        before, units = weights.clone(), [66, 180]
        allocate_recent4_budgets(weights, units)
        self.assertTrue(torch.equal(weights, before))
        self.assertEqual(units, [66, 180])


class Recent4CacheTests(unittest.TestCase):
    def make_cache(self, budget=96, gate="per_head_capacity"):
        # Head 0 has no historical weight and must remain a pure window.
        total = torch.tensor([[budget * 3 // 4, budget * 5 // 4]], dtype=torch.long)
        source_recent = torch.tensor([[budget * 3 // 4, 32]], dtype=torch.long)
        config = StreamingConfig(
            method="sparsemm", budget=budget, window_size=32, kernel_size=1,
            sparsemm_budget_path="cpu-recent4-weights.csv",
            sparsemm_cache_policy="two_bank", sparsemm_two_bank_gate=gate,
        )
        return Recent4Cache(
            config, num_layers=1, num_kv_heads=2, head_dim=2,
            device="cpu", dtype=torch.float32, budgets=total,
            recent_budgets=source_recent,
        )

    def append(self, cache, count, *, initialize=False):
        start = cache.logical_tokens
        keys = token_keys(start, count)[None, None].expand(1, 2, count, 2).clone()
        positions = torch.arange(start, start + count, dtype=torch.float32).repeat(3, 1)
        if initialize:
            cache.set_context("init", positions)
        else:
            cache.begin_visual_unit(count)
            cache.set_context("video", positions, xy=torch.zeros((count, 2)))
        cache.append_layer(0, keys, keys + 1000, q=torch.ones((1, 4, count, 2)))
        if initialize:
            cache.finish_init(count)
            return None
        return cache.finish_video()

    def ids(self, cache, head):
        return cache.layers[0].get_segment(head)[0][:, 0].tolist()

    def test_cache_starts_with_empty_ledger_and_nominal_capacities(self):
        cache = self.make_cache(budget=500)
        self.assertEqual(cache.recent4_unit_token_counts, [])
        self.assertEqual(cache.budgets.tolist(), [[500, 500]])
        self.assertEqual(cache.recent_budgets.tolist(), [[32, 32]])

    def test_ledger_uses_last_two_sizes_and_exact_current_quota(self):
        cache = self.make_cache(budget=500)
        for unit, expected in ((192, [192]), (180, [192, 180]), (66, [180, 66])):
            cache.begin_visual_unit(unit)
            self.assertEqual(cache.recent4_unit_token_counts, expected)
            self.assertEqual(cache.recent_budgets.tolist(), [[sum(expected)] * 2])
            self.assertEqual(int(cache.budgets.sum()), 1000)
            self.assertEqual((cache.budgets - cache.recent_budgets)[0, 0].item(), 0)

    def test_invalid_budget_update_does_not_corrupt_previous_ledger_or_quota(self):
        cache = self.make_cache(budget=500)
        cache.begin_visual_unit(192)
        ledger = list(cache.recent4_unit_token_counts)
        budgets, recent = cache.budgets.clone(), cache.recent_budgets.clone()
        with self.assertRaises(ValueError):
            cache.begin_visual_unit(400)
        self.assertEqual(cache.recent4_unit_token_counts, ledger)
        self.assertTrue(torch.equal(cache.budgets, budgets))
        self.assertTrue(torch.equal(cache.recent_budgets, recent))

    def test_deepcopy_branch_does_not_change_even_prefix_state(self):
        cache = self.make_cache(budget=500)
        self.append(cache, 3, initialize=True)
        self.append(cache, 180)
        self.append(cache, 192)
        baseline_ids = [self.ids(cache, h) for h in range(2)]
        branch = copy.deepcopy(cache)
        self.append(branch, 66)
        self.assertEqual(branch.recent4_unit_token_counts, [192, 66])
        self.assertEqual(cache.recent4_unit_token_counts, [180, 192])
        self.assertEqual(cache.recent_budgets.tolist(), [[372, 372]])
        self.assertEqual([self.ids(cache, h) for h in range(2)], baseline_ids)
        fresh = self.make_cache(budget=500)
        self.assertEqual(fresh.recent4_unit_token_counts, [])

    def test_first_unit_and_zero_history_head_fill_without_duplicate_tokens(self):
        cache = self.make_cache()
        self.append(cache, 3, initialize=True)
        self.append(cache, 32)
        self.assertEqual(cache.recent_budgets.tolist(), [[32, 32]])
        self.assertEqual(self.ids(cache, 0), list(range(35)))
        self.append(cache, 32)
        self.assertEqual(self.ids(cache, 0), list(range(67)))
        self.append(cache, 32)
        self.assertEqual(self.ids(cache, 0), [0, 1, 2] + list(range(35, 99)))
        self.assertEqual(len(set(self.ids(cache, 0))), len(self.ids(cache, 0)))

    def test_recent_to_history_competes_and_evicted_tokens_never_reappear(self):
        cache = self.make_cache()
        self.append(cache, 3, initialize=True)

        def attention_scores(key, queries, prefix, recent, config, layer_idx, head):
            del queries, config, layer_idx, head
            ids = key[prefix:len(key) - recent, 0]
            # The oldest token outranks everything; recent window tokens
            # retain their places independently of these historical scores.
            return torch.where(ids == 3, ids.new_full(ids.shape, 1000), ids)

        with patch("sparsemm_native.two_bank._head_scores", side_effect=attention_scores):
            for _ in range(5):
                report = self.append(cache, 32)
            ids = self.ids(cache, 1)
            self.assertIn(3, ids[3:-64])
            self.assertIn(67, ids[3:-64])  # Just left the recent window.
            self.assertNotIn(4, ids)
            self.assertEqual(ids[-64:], list(range(99, 163)))
            self.assertEqual(len(ids) - 3, 128)
            self.assertTrue(all(report["layers"][0]["position_checks"].values()))
            old_history = int((cache.budgets - cache.recent_budgets)[0, 1])
            self.append(cache, 16)  # Smaller window gives history more slots.
        self.assertEqual(cache.recent4_unit_token_counts, [32, 16])
        self.assertGreater(int((cache.budgets - cache.recent_budgets)[0, 1]), old_history)
        self.assertNotIn(4, self.ids(cache, 1))
        self.assertEqual(self.ids(cache, 1)[-48:], list(range(131, 179)))
        self.assertEqual(self.ids(cache, 0), [0, 1, 2] + list(range(131, 179)))
        for head in range(2):
            keys, values = cache.layers[0].get_segment(head)
            self.assertTrue(torch.equal(values, keys + 1000))
            self.assertTrue(torch.equal(keys[:, 0], cache.layers[0].get_position_segment(head)[0]))
            self.assertLessEqual(len(keys) - 3, int(cache.budgets[0, head]))

    def test_legacy_warmup_remains_explicit_and_does_not_change_quota(self):
        cache = self.make_cache(budget=500, gate="legacy_global")
        self.append(cache, 3, initialize=True)
        self.append(cache, 192)
        self.append(cache, 192)
        report = self.append(cache, 66)
        self.assertEqual(cache.recent_budgets.tolist(), [[258, 258]])
        self.assertEqual(int(cache.budgets.sum()), 1000)
        self.assertTrue(report["layers"][0]["reference_warmup_bypass"])
        self.assertEqual(self.ids(cache, 0), list(range(453)))


class FrozenRuntimeTests(unittest.TestCase):
    @unittest.skipUnless(PAUSED_SNAPSHOT.is_dir(), "Paused experiment source snapshot unavailable")
    def test_paused_experiment_runtime_sources_are_byte_unchanged(self):
        sources = (
            "sparsemm_native/cache.py", "sparsemm_native/policy.py",
            "sparsemm_native/two_bank.py", "sparsemm_native/streaming.py",
            "sparsemm_native/run_ovobench.py", "sparsemm_baselines/core.py",
            "sparsemm_baselines/cache.py", "sparsemm_baselines/adapter.py",
            "sparsemm_protokv/cache.py", "sparsemm_protokv/config.py",
            "sparsemm_protokv/streaming.py", "sparsemm_protokv/monkeypatch.py",
        )
        for name in sources:
            with self.subTest(source=name):
                expected = hashlib.sha256((PAUSED_SNAPSHOT / name).read_bytes()).hexdigest()
                actual = hashlib.sha256((KVROOT / name).read_bytes()).hexdigest()
                self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
