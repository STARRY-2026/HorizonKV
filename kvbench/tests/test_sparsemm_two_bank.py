"""CPU invariants for DP recent/history quotas in the native SparseMM selector.

These checks exercise token identities across updates, not just output lengths:
newest tokens must survive even when their attention score is low, and a token
leaving the sliding window must compete against retained historical tokens.
"""

import csv
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from sparsemm_baselines.core import BaselineConfig, _head_scores
from sparsemm_native.policy import select_layer as select_capacity_only_layer
from sparsemm_native.cache import SparseMMCache
from sparsemm_native.two_bank import load_two_bank_budgets, select_two_bank_layer
from sparsemm_protokv.config import StreamingConfig


def token_keys(length, offset=0):
    """The first key coordinate provides a persistent token ID across updates."""
    ids = torch.arange(offset, offset + length, dtype=torch.float32)
    return torch.stack((ids, torch.zeros_like(ids)), dim=-1)


def increasing_id_scores(key, queries, prefix, recent, config, layer_idx, head):
    del queries, config, layer_idx, head
    return key[prefix:key.shape[0] - recent, 0].clone()


class TwoBankBudgetLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "two_bank.csv"
        self.rows = [
            {"layer": 0, "kv_head": 0, "capacity_n": 40, "recent_r": 32, "history_q": 8},
            {"layer": 0, "kv_head": 1, "capacity_n": 56, "recent_r": 40, "history_q": 16},
            {"layer": 1, "kv_head": 0, "capacity_n": 48, "recent_r": 48, "history_q": 0},
            {"layer": 1, "kv_head": 1, "capacity_n": 48, "recent_r": 35, "history_q": 13},
        ]

    def write(self, fields=None):
        with self.path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields or list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def load(self):
        return load_two_bank_budgets(self.path, 2, 2, 48, window_size=32)

    def test_exact_total_and_recent_history_split_are_loaded(self):
        self.write()
        capacity, recent, metadata = self.load()
        self.assertEqual(capacity.tolist(), [[40, 56], [48, 48]])
        self.assertEqual(recent.tolist(), [[32, 40], [48, 35]])
        self.assertEqual(int(capacity.sum()), 4 * 48)
        self.assertEqual((capacity - recent).tolist(), [[8, 16], [0, 13]])
        self.assertIsInstance(metadata, dict)

    def test_shuffled_rows_use_explicit_layer_and_head_indices(self):
        self.rows.reverse()
        self.write()
        capacity, recent, _ = self.load()
        self.assertEqual(capacity.tolist(), [[40, 56], [48, 48]])
        self.assertEqual(recent.tolist(), [[32, 40], [48, 35]])

    def test_protect_window_is_within_recent_budget_not_added(self):
        self.write()
        capacity, recent, _ = self.load()
        self.assertEqual(int((recent + (capacity - recent)).sum()), 192)
        self.assertGreaterEqual(int(recent.min()), 32)

    def test_wrong_global_total_is_rejected(self):
        self.rows[0]["capacity_n"] += 1
        self.rows[0]["history_q"] += 1
        self.write()
        with self.assertRaises(ValueError):
            self.load()

    def test_invalid_split_and_unprotected_recent_are_rejected(self):
        for recent, history in ((32, 7), (31, 9), (41, -1), ("32.0", 8)):
            with self.subTest(recent=recent, history=history):
                self.rows[0]["recent_r"] = recent
                self.rows[0]["history_q"] = history
                self.write()
                with self.assertRaises(ValueError):
                    self.load()

    def test_bank_columns_are_required(self):
        for missing in (("recent_r",), ("history_q",), ("recent_r", "history_q")):
            with self.subTest(missing=missing):
                rows = [{k: v for k, v in row.items() if k not in missing} for row in self.rows]
                with self.path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                with self.assertRaises(ValueError):
                    self.load()

    def test_missing_duplicate_and_out_of_range_heads_are_rejected(self):
        original = [row.copy() for row in self.rows]
        variants = [original[:-1], original + [original[0].copy()],
                    original[:-1] + [{**original[-1], "layer": 2}]]
        for rows in variants:
            with self.subTest(rows=rows):
                self.rows = rows
                self.write()
                with self.assertRaises(ValueError):
                    self.load()

    def test_malformed_row_and_duplicate_columns_are_rejected(self):
        self.write()
        self.path.write_text(self.path.read_text() + "0,0,40,32\n")
        with self.assertRaises(ValueError):
            self.load()
        self.path.write_text("layer,kv_head,capacity_n,recent_r,history_q,recent_r\n")
        with self.assertRaises(ValueError):
            self.load()

    def test_short_cache_gate_is_explicit_and_invalid_values_are_rejected(self):
        self.write()
        for gate in ("per_head_capacity", "legacy_global"):
            with self.subTest(gate=gate):
                _, _, metadata = load_two_bank_budgets(self.path, 2, 2, 48,
                                                       short_cache_gate=gate)
                self.assertEqual(metadata["short_cache_gate"], gate)
        with self.assertRaises(ValueError):
            load_two_bank_budgets(self.path, 2, 2, 48, short_cache_gate="unknown")


class TwoBankSelectorTests(unittest.TestCase):
    def setUp(self):
        self.queries = torch.ones((1, 2, 2), dtype=torch.float32)

    def select(self, keys, capacities, recents, **kwargs):
        return select_two_bank_layer(keys, self.queries, capacities, recents,
                                     budget=48, window_size=32, kernel_size=1, **kwargs)

    def test_recent_window_survives_even_with_lower_attention_scores(self):
        keys = [token_keys(70)]
        with patch("sparsemm_native.two_bank._head_scores", side_effect=increasing_id_scores):
            selected, _ = self.select(keys, [40], [36])
        # Four historical winners, followed by the latest 36 tokens.
        self.assertEqual(selected[0][:4].tolist(), [33, 32, 31, 30])
        self.assertEqual(selected[0][4:].tolist(), list(range(34, 70)))
        self.assertEqual(selected[0].unique().numel(), 40)

    def test_token_leaving_recent_competes_with_existing_history(self):
        importance = {0: 0, 1: 100, 2: 10, 3: 90}

        def scores(key, queries, prefix, recent, config, layer_idx, head):
            del queries, config, layer_idx, head
            return torch.tensor([importance.get(int(token), -100)
                                 for token in key[prefix:len(key) - recent, 0]])

        with patch("sparsemm_native.two_bank._head_scores", side_effect=scores):
            first, _ = self.select([token_keys(35)], [34], [32])
            retained = token_keys(35)[first[0]]
            self.assertEqual(retained[:2, 0].tolist(), [1, 2])
            updated = torch.cat((retained, token_keys(1, offset=35)))
            second, _ = self.select([updated], [34], [32])
        ids = updated[second[0], 0].tolist()
        self.assertEqual(ids[:2], [1, 3])
        self.assertEqual(ids[2:], list(range(4, 36)))
        self.assertNotIn(2, ids)

    def test_zero_history_quota_is_exact_recent_sliding_window(self):
        with patch("sparsemm_native.two_bank._head_scores", side_effect=increasing_id_scores):
            selected, _ = self.select([token_keys(80)], [40], [40])
        self.assertEqual(selected[0].tolist(), list(range(40, 80)))

    def test_short_and_exactly_full_heads_retain_chronological_order(self):
        self.queries = torch.ones((2, 2, 2), dtype=torch.float32)
        keys = [token_keys(38), token_keys(56)]
        with patch("sparsemm_native.two_bank._head_scores", side_effect=increasing_id_scores):
            selected, _ = self.select(keys, [40, 56], [32, 40])
        self.assertEqual(selected[0].tolist(), list(range(38)))
        self.assertEqual(selected[1].tolist(), list(range(56)))

    def test_ragged_heads_have_independent_quotas_and_protected_prefix(self):
        prefix = 3
        self.queries = torch.ones((4, 2, 2), dtype=torch.float32)
        keys = [token_keys(100 + prefix), token_keys(20 + prefix)]
        with patch("sparsemm_native.two_bank._head_scores", side_effect=increasing_id_scores):
            selected, _ = self.select(keys, [40, 56], [32, 40], prefix_tokens=prefix)
        self.assertEqual(selected[0].numel(), prefix + 40)
        self.assertEqual(selected[1].numel(), prefix + 20)
        self.assertEqual(selected[0][:prefix].tolist(), list(range(prefix)))
        self.assertEqual(selected[0][-32:].tolist(), list(range(71, 103)))
        self.assertEqual(selected[1].tolist(), list(range(23)))
        for indices in selected:
            self.assertEqual(indices.unique().numel(), indices.numel())

    def test_each_head_compresses_before_legacy_global_short_cache_gate(self):
        keys = [token_keys(45)]
        # Legacy SparseMM bypasses at 45 visual tokens with global K=64;
        # the new method must already enforce this head's N=40 quota.
        old, _ = select_capacity_only_layer(keys, self.queries, [40], budget=64,
                                            window_size=32, kernel_size=1)
        new, _ = select_two_bank_layer(keys, self.queries, [40], [32], budget=64,
                                       window_size=32, kernel_size=1)
        self.assertEqual(old[0].numel(), 45)
        self.assertEqual(new[0].numel(), 40)
        self.assertEqual(new[0][-32:].tolist(), list(range(13, 45)))

    def test_history_uses_existing_sparsemm_fp32_gqa_and_torch_sort(self):
        generator = torch.Generator().manual_seed(714)
        keys = [torch.randn((74, 8), generator=generator),
                torch.randn((81, 8), generator=generator)]
        queries = torch.randn((6, 7, 8), generator=generator)
        prefix, capacities, recents = 2, [40, 56], [32, 43]
        config = BaselineConfig(method="adakv", budget=48, window_size=32,
                                kernel_size=5, pooling="avgpool", gqa_reduce="mean",
                                budget_scope="visual", score_precision="float32")
        selected, _ = select_two_bank_layer(keys, queries, capacities, recents,
                                            prefix_tokens=prefix, budget=48)
        for head, (key, capacity, recent) in enumerate(zip(keys, capacities, recents)):
            score = _head_scores(key, queries[head * 3:(head + 1) * 3], prefix,
                                 recent, config, 0, head)
            history = torch.sort(score, descending=True).indices[:capacity - recent] + prefix
            expected = torch.cat((torch.arange(prefix), history,
                                  torch.arange(len(key) - recent, len(key))))
            self.assertTrue(torch.equal(selected[head], expected))

    def test_recent32_matches_previous_capacity_only_selector_when_both_compress(self):
        generator = torch.Generator().manual_seed(196)
        keys = [torch.randn((83, 4), generator=generator),
                torch.randn((77, 4), generator=generator)]
        queries = torch.randn((4, 6, 4), generator=generator)
        options = dict(prefix_tokens=3, budget=48, window_size=32, kernel_size=5)
        old, before_metadata = select_capacity_only_layer(keys, queries, [40, 56], **options)
        new, _ = select_two_bank_layer(keys, queries, [40, 56], [32, 32], **options)
        again, after_metadata = select_capacity_only_layer(keys, queries, [40, 56], **options)
        for previous, current, repeated in zip(old, new, again):
            self.assertTrue(torch.equal(previous, current))
            self.assertTrue(torch.equal(previous, repeated))
        self.assertEqual(before_metadata, after_metadata)

    def test_legacy_gate_recent32_exactly_matches_original_before_and_after_warmup(self):
        generator = torch.Generator().manual_seed(357)
        prefix, capacities = 3, [375, 750]
        for visual in (400, 520):
            with self.subTest(visual=visual):
                keys = [torch.randn((visual + prefix, 8), generator=generator)
                        for _ in range(2)]
                queries = torch.randn((4, 6, 8), generator=generator)
                options = dict(prefix_tokens=prefix, budget=500, window_size=32,
                               kernel_size=5)
                original, old_metadata = select_capacity_only_layer(keys, queries,
                                                                    capacities, **options)
                dual, metadata = select_two_bank_layer(keys, queries, capacities, [32, 32],
                                                       short_cache_gate="legacy_global", **options)
                self.assertEqual(metadata["reference_warmup_bypass"], visual == 400)
                self.assertEqual(metadata["bypassed_short_cache"],
                                 old_metadata["bypassed_short_cache"])
                for before, after in zip(original, dual):
                    self.assertTrue(torch.equal(before, after))
                if visual == 400:
                    # Match the reference warmup even though 400 exceeds N=375.
                    self.assertEqual(dual[0].numel() - prefix, 400)
                    self.assertEqual(dual[0].tolist(), list(range(403)))
                else:
                    self.assertEqual(dual[0].numel() - prefix, 375)
                    self.assertEqual(dual[1].numel() - prefix, 520)
                    # Even an under-capacity head uses legacy score order once
                    # the global gate opens. A per-head bypass would fail here.
                    self.assertNotEqual(dual[1].tolist(), list(range(523)))

    def test_legacy_gate_uses_protect32_threshold_independent_of_larger_recent_quotas(self):
        self.queries = torch.ones((2, 2, 2), dtype=torch.float32)
        for visual in (400, 499, 500, 520):
            with self.subTest(visual=visual), patch(
                    "sparsemm_native.two_bank._head_scores", side_effect=increasing_id_scores):
                keys = [token_keys(visual), token_keys(visual)]
                selected, metadata = select_two_bank_layer(
                    keys, self.queries, [375, 750], [200, 400], budget=500,
                    kernel_size=1, short_cache_gate="legacy_global")
                warmup = visual < 500
                self.assertEqual(metadata["reference_warmup_bypass"], warmup)
                self.assertEqual(metadata["recent_counts"], [200, 400])
                if warmup:
                    for indices in selected:
                        self.assertEqual(indices.tolist(), list(range(visual)))
                else:
                    self.assertEqual(selected[0].numel(), 375)
                    self.assertEqual(selected[1].numel(), visual)
                    self.assertEqual(selected[0][-200:].tolist(), list(range(visual - 200, visual)))
                    self.assertEqual(selected[1][-400:].tolist(), list(range(visual - 400, visual)))
                    self.assertEqual(metadata["retained_history_counts"], [175, visual - 400])

    def test_unknown_selector_gate_is_rejected(self):
        with self.assertRaises(ValueError):
            self.select([token_keys(70)], [40], [32], short_cache_gate="unknown")

    def test_equal_history_scores_follow_existing_torch_sort_ties(self):
        keys = [token_keys(70)]

        def tied_scores(key, queries, prefix, recent, config, layer_idx, head):
            del queries, config, layer_idx, head
            return torch.zeros(len(key) - prefix - recent)

        with patch("sparsemm_native.two_bank._head_scores", side_effect=tied_scores):
            selected, _ = self.select(keys, [40], [32])
        self.assertTrue(torch.equal(selected[0][:8], torch.sort(torch.zeros(38), descending=True).indices[:8]))

    def test_invalid_recent_quotas_fail_without_silent_clamping(self):
        for recent in ([31], [41], [-1], [32.5], [True], [], [32, 32]):
            with self.subTest(recent=recent), self.assertRaises(ValueError):
                self.select([token_keys(70)], [40], recent)


class TwoBankCacheIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.capacity = torch.tensor([[34, 62], [48, 48]])
        self.recent = torch.tensor([[32, 40], [48, 32]])

    def make_cache(self, cache_class=SparseMMCache, two_bank=True,
                   gate="per_head_capacity", budget=48):
        config = StreamingConfig(method="sparsemm", budget=budget,
                                 sparsemm_budget_path="cpu-test-budget.csv",
                                 sparsemm_cache_policy="two_bank" if two_bank else "capacity",
                                 sparsemm_two_bank_gate=gate,
                                 kernel_size=1)
        options = {"recent_budgets": self.recent} if two_bank else {}
        return cache_class(config, num_layers=2, num_kv_heads=2, head_dim=2,
                           device="cpu", dtype=torch.float32, budgets=self.capacity,
                           **options)

    def append(self, cache, phase, start, count, keys=None, queries=None):
        positions = torch.arange(start, start + count, dtype=torch.float32).repeat(3, 1)
        options = {"xy": torch.zeros((count, 2))} if phase == "video" else {}
        cache.set_context(phase, positions, **options)
        if keys is None:
            keys = token_keys(count, offset=start)[None, None].expand(1, 2, count, 2).clone()
        if queries is None:
            queries = torch.ones((1, 4, count, 2))
        for layer in range(2):
            cache.append_layer(layer, keys, keys + 1000, q=queries)
        if phase == "init":
            cache.finish_init(count)
        elif phase == "video":
            return cache.finish_video()

    def assert_same_snapshot(self, left, right):
        self.assertEqual(len(left), len(right))
        for left_layer, right_layer in zip(left, right):
            for before, after in zip(left_layer, right_layer):
                if isinstance(before, torch.Tensor):
                    self.assertTrue(torch.equal(before, after))
                else:
                    self.assertEqual(before, after)

    def test_multi_patch_cache_migrates_recent_to_history_and_restores_queries(self):
        cache = self.make_cache()
        self.append(cache, "init", 0, 3)
        importance = {3: 0, 4: 100, 5: 10, 6: 90}

        def scores(key, queries, prefix, recent, config, layer_idx, head):
            del queries, config, layer_idx, head
            return torch.tensor([importance.get(int(token), -100)
                                 for token in key[prefix:len(key) - recent, 0]])

        with patch("sparsemm_native.two_bank._head_scores", side_effect=scores):
            first = self.append(cache, "video", 3, 35)
            first_keys, _ = cache.layers[0].get_segment(0)
            self.assertEqual(first_keys[3:5, 0].tolist(), [4, 5])
            second = self.append(cache, "video", 38, 1)
        first_keys, _ = cache.layers[0].get_segment(0)
        self.assertEqual(first_keys[3:5, 0].tolist(), [4, 6])
        self.assertNotIn(5, first_keys[:, 0].tolist())
        self.assertEqual(first_keys[5:, 0].tolist(), list(range(7, 39)))
        self.assertEqual(cache.head_lengths(), [[37, 39], [39, 39]])
        self.assertEqual(cache.seen_visual_tokens, 36)
        self.assertEqual(cache.logical_tokens, 39)
        self.assertEqual(first["input_tokens"], 35)
        self.assertEqual(second["input_tokens"], 1)
        for layer_index, (layer, metadata) in enumerate(zip(cache.layers, second["layers"])):
            self.assertEqual(metadata["recent_budgets"], self.recent[layer_index].tolist())
            self.assertEqual(metadata["history_budgets"],
                             (self.capacity[layer_index] - self.recent[layer_index]).tolist())
            self.assertTrue(all(metadata["position_checks"].values()))
            for head in range(2):
                key, value = layer.get_segment(head)
                positions = layer.get_position_segment(head)
                self.assertTrue(torch.equal(value, key + 1000))
                self.assertTrue(torch.equal(key[:, 0], positions[0]))
                self.assertEqual(positions[:, :3].tolist(), [[0., 1., 2.]] * 3)
                self.assertLessEqual(len(key) - 3, int(self.capacity[layer_index, head]))

        saved = [tuple(value.clone() if isinstance(value, torch.Tensor) else value
                       for value in layer) for layer in cache.snapshot()]
        with cache.query():
            self.append(cache, "query", 39, 2)
            cache.advance_query(2)
            self.assertEqual(cache.logical_tokens, 41)
            self.assertEqual(cache.head_lengths(), [[39, 41], [41, 41]])
        self.assertEqual(cache.logical_tokens, 39)
        self.assert_same_snapshot(saved, cache.snapshot())

    def test_invalid_two_bank_configurations_are_rejected(self):
        invalid = [dict(method="sparsemm", sparsemm_cache_policy="unknown"),
                   dict(method="uniform", sparsemm_cache_policy="two_bank",
                        sparsemm_budget_path="cpu-test-budget.csv"),
                   dict(method="sparsemm", sparsemm_cache_policy="two_bank"),
                   dict(method="sparsemm", sparsemm_cache_policy="two_bank",
                        sparsemm_budget_path="cpu-test-budget.csv",
                        sparsemm_two_bank_gate="unknown")]
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ValueError):
                StreamingConfig(**options)

    def test_cache_reference_warmup_overflow_is_flagged_then_bank_quotas_apply(self):
        self.capacity = torch.tensor([[375, 625], [375, 625]])
        self.recent = torch.tensor([[200, 400], [200, 400]])
        cache = self.make_cache(gate="legacy_global", budget=500)
        self.append(cache, "init", 0, 3)
        with patch("sparsemm_native.two_bank._head_scores", side_effect=increasing_id_scores):
            warmup = self.append(cache, "video", 3, 400)
            self.assertEqual(cache.head_lengths(), [[403, 403], [403, 403]])
            active = self.append(cache, "video", 403, 120)
        self.assertEqual(cache.head_lengths(), [[378, 523], [378, 523]])
        for metadata in warmup["layers"]:
            self.assertTrue(metadata["reference_warmup_bypass"])
            self.assertTrue(all(metadata["position_checks"].values()))
            self.assertEqual(metadata["retained_history_counts"], [200, 0])
        for metadata in active["layers"]:
            self.assertFalse(metadata["reference_warmup_bypass"])
            self.assertTrue(all(metadata["position_checks"].values()))
            self.assertEqual(metadata["recent_counts"], [200, 400])
            self.assertEqual(metadata["retained_history_counts"], [175, 120])
            self.assertEqual(metadata["retained_visual_counts"], [375, 520])

    @unittest.skipUnless(Path("/tmp/dp_two_bank_original_20261020/cache.py").is_file(),
                         "Local pre-change cache snapshot is unavailable")
    def test_capacity_policy_matches_pre_change_cache_across_updates(self):
        """Local regression against the preserved source used before this change."""
        source = "/tmp/dp_two_bank_original_20261020/cache.py"
        spec = importlib.util.spec_from_file_location("_pre_change_sparsemm_cache", source)
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        previous = self.make_cache(original.SparseMMCache, two_bank=False)
        current = self.make_cache(two_bank=False)
        self.append(previous, "init", 0, 3)
        self.append(current, "init", 0, 3)
        generator = torch.Generator().manual_seed(205)
        start = 3
        for count in (35, 1, 9, 40):
            keys = torch.randn((1, 2, count, 2), generator=generator)
            queries = torch.randn((1, 4, count, 2), generator=generator)
            before = self.append(previous, "video", start, count, keys, queries)
            after = self.append(current, "video", start, count, keys, queries)
            self.assertEqual(before, after)
            self.assert_same_snapshot(previous.snapshot(), current.snapshot())
            start += count


if __name__ == "__main__":
    unittest.main()
