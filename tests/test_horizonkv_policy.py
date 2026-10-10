import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from head_analysis.hermes_kv_head_ragged import HeadRaggedCache, HeadRaggedLayerCache
from head_analysis.horizonkv_policy import (
    HERMES_GLOBAL_PROBE,
    HERMES_LOCAL_PROBE,
    aggregate_probe_attention,
    get_horizonkv_pseudo_probes,
    hermes_layer_recency_alpha,
    horizonkv_needs_compression,
    load_horizonkv_policy,
    select_horizonkv_keep_indices,
    select_memory_tokens,
    update_attention_ema,
    validate_horizonkv_args,
)


ROOT = Path(__file__).resolve().parents[1]
PILOT_POLICY = (
    ROOT
    / "results/myobservations/horizonkv_pilot_10v_fixed896"
    / "pseudo_trend_memory_budget_6000_qwen_adaptive_recent2691.csv"
)


class HorizonPolicyTests(unittest.TestCase):
    def test_horizon_is_opt_in_and_enables_ragged_prefill(self):
        disabled = SimpleNamespace(horizonkv_policy=None, kv_head_ragged_prefill=False)
        validate_horizonkv_args(disabled)
        self.assertFalse(disabled.kv_head_ragged_prefill)

        enabled = SimpleNamespace(
            horizonkv_policy=str(PILOT_POLICY),
            model="qwen2.5_vl_7b",
            compress_mode="hermes",
            horizonkv_recent_capacity=2691,
            horizonkv_ema_gamma=0.9,
            kv_head_ragged_prefill=False,
        )
        validate_horizonkv_args(enabled)
        self.assertTrue(enabled.kv_head_ragged_prefill)

    def test_horizon_rejects_existing_budget_method(self):
        args = SimpleNamespace(
            horizonkv_policy=str(PILOT_POLICY),
            model="qwen2.5_vl_7b",
            compress_mode="hermes",
            horizonkv_recent_capacity=2691,
            horizonkv_ema_gamma=0.9,
            kv_head_ragged_prefill=False,
            kv_head_budget_scores="old.csv",
        )
        with self.assertRaisesRegex(ValueError, "kv_head_budget_scores"):
            validate_horizonkv_args(args)

    def test_pilot_policy_is_complete_and_conserves_budget(self):
        policy = load_horizonkv_policy(
            PILOT_POLICY,
            expected_average_budget=6000,
        )
        self.assertEqual(policy["head_types"].shape, (28, 4))
        self.assertEqual(int(policy["total_budgets"].sum()), 672000)
        self.assertTrue(np.all(policy["recent_budgets"] == 2691))
        self.assertTrue(
            np.array_equal(
                policy["recent_budgets"] + policy["memory_budgets"],
                policy["total_budgets"],
            )
        )

    def test_hermes_probe_text_uses_static_local_mixed_global(self):
        probes = get_horizonkv_pseudo_probes("hermes_routed")
        self.assertEqual(probes[0], HERMES_LOCAL_PROBE)
        self.assertEqual(probes[1], HERMES_LOCAL_PROBE + "; " + HERMES_GLOBAL_PROBE)
        self.assertEqual(probes[2], HERMES_GLOBAL_PROBE)

    def test_hermes_probe_routing_uses_layer_boundaries(self):
        local = torch.tensor([0.0, 1.0])
        mixed = torch.tensor([1.0, 0.0])
        global_score = torch.tensor([0.25, 0.75])
        probe_scores = [
            [[values] for _ in range(20)]
            for values in (local, mixed, global_score)
        ]
        visual_indices = torch.tensor([0, 1])

        for layer_idx, expected in (
            (7, torch.tensor([0.0, 1.0])),
            (8, torch.tensor([1.0, 0.0])),
            (18, torch.tensor([1.0, 0.0])),
            (19, torch.tensor([0.0, 1.0])),
        ):
            actual = aggregate_probe_attention(
                probe_scores,
                layer_idx,
                0,
                visual_indices,
                probe_style="hermes_routed",
                short_term_threshold=8,
                long_term_threshold=19,
            )
            self.assertTrue(torch.allclose(actual, expected))

    def test_hermes_layer_recency_alpha_matches_runtime_policy(self):
        self.assertEqual(hermes_layer_recency_alpha(7), 1.0)
        self.assertEqual(hermes_layer_recency_alpha(8), 0.75)
        self.assertAlmostEqual(
            hermes_layer_recency_alpha(18),
            0.75 - 0.6 * (10 / 11),
        )
        self.assertEqual(hermes_layer_recency_alpha(19), 0.0)

    def test_new_chunk_initializes_ema_from_current_attention(self):
        old = torch.tensor([0.8, 0.2, 0.0])
        current = torch.tensor([0.4, 0.6, 1.0])
        chunks = torch.tensor([0, 1, 2])
        updated = update_attention_ema(old, current, chunks, 2, 0.9)
        expected = torch.tensor([0.76, 0.24, 1.0])
        self.assertTrue(torch.allclose(updated, expected))

    def test_rising_far_cold_anchor_is_temporally_distributed(self):
        candidates = torch.arange(8)
        chunks = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        ema = torch.tensor([0.9, 0.1, 0.8, 0.1, 0.7, 0.1, 0.6, 0.1])
        selected = select_memory_tokens(
            candidates,
            chunks,
            ema,
            current_chunk_id=7,
            memory_budget=4,
            head_type="rising_far",
        )
        self.assertEqual(selected.numel(), 4)
        # Half of the four-token budget is a cold anchor. Those two anchors
        # are distributed over distinct source chunks; global fill may add more.
        self.assertGreaterEqual(len(set(chunks[selected].tolist())), 2)

    def test_age_penalty_can_be_disabled_independently(self):
        candidates = torch.arange(3)
        chunks = torch.tensor([0, 1, 2])
        ema = torch.tensor([1.0, 0.9, 0.0])

        with_penalty = select_memory_tokens(
            candidates,
            chunks,
            ema,
            current_chunk_id=2,
            memory_budget=1,
            head_type="sustained_gentle",
            apply_age_penalty=True,
        )
        without_penalty = select_memory_tokens(
            candidates,
            chunks,
            ema,
            current_chunk_id=2,
            memory_budget=1,
            head_type="sustained_gentle",
            apply_age_penalty=False,
        )

        self.assertEqual(with_penalty.tolist(), [2])
        self.assertEqual(without_penalty.tolist(), [1])

    def test_all_memory_coverage_reaches_warm_history(self):
        candidates = torch.arange(6)
        chunks = torch.tensor([0, 0, 1, 1, 2, 2])
        ema = torch.tensor([0.0, 0.0, 0.4, 0.3, 1.0, 0.9])

        cold_only = select_memory_tokens(
            candidates,
            chunks,
            ema,
            current_chunk_id=2,
            memory_budget=3,
            head_type="rising_far",
            apply_age_penalty=False,
            temporal_coverage="cold",
        )
        all_memory = select_memory_tokens(
            candidates,
            chunks,
            ema,
            current_chunk_id=2,
            memory_budget=3,
            head_type="rising_far",
            apply_age_penalty=False,
            temporal_coverage="all",
        )

        self.assertNotIn(0, chunks[cold_only].tolist())
        self.assertIn(0, chunks[all_memory].tolist())

    def test_ragged_sidecars_follow_append_prune_and_truncate(self):
        layer = HeadRaggedLayerCache(
            flat_k=torch.arange(8, dtype=torch.float32).view(4, 2),
            flat_v=torch.arange(8, dtype=torch.float32).view(4, 2) + 100,
            head_lens=torch.tensor([2, 2]),
            flat_source_chunk=torch.full((4,), -1),
            flat_attention_ema=torch.zeros(4),
        )
        key = torch.arange(12, dtype=torch.float32).view(1, 2, 3, 2) + 20
        value = key + 100
        layer.append_chunk(key, value, source_chunk_id=0)
        chunks0, _ = layer.get_horizon_aux_segment(0)
        chunks1, _ = layer.get_horizon_aux_segment(1)
        self.assertEqual(chunks0.tolist(), [-1, -1, 0, 0, 0])
        self.assertEqual(chunks1.tolist(), [-1, -1, 0, 0, 0])

        layer.prune_per_head([
            torch.tensor([0, 2, 4]),
            torch.tensor([1, 3]),
        ])
        self.assertEqual(layer.head_lens.tolist(), [3, 2])
        self.assertEqual(layer.get_horizon_aux_segment(0)[0].tolist(), [-1, 0, 0])
        self.assertEqual(layer.get_horizon_aux_segment(1)[0].tolist(), [-1, 0])

        layer.truncate(torch.tensor([2, 1]))
        self.assertEqual(layer.head_lens.tolist(), [2, 1])
        self.assertEqual(layer.flat_source_chunk.tolist(), [-1, 0, -1])
        self.assertEqual(layer.flat_attention_ema.numel(), 3)

    def test_selector_keeps_fixed_fifo_tail_and_fixed_memory(self):
        chunks = torch.tensor([-1, 0, 0, 1, 1, 2, 2, 2], dtype=torch.int32)
        layer = HeadRaggedLayerCache(
            flat_k=torch.arange(16, dtype=torch.float32).view(8, 2),
            flat_v=torch.arange(16, dtype=torch.float32).view(8, 2),
            head_lens=torch.tensor([8]),
            flat_source_chunk=chunks,
            flat_attention_ema=torch.zeros(8),
        )
        config = {
            "current_chunk_id": 2,
            "ema_gamma": 0.9,
            "head_types": np.asarray([["normal_sustained"]], dtype=object),
            "recent_budgets": np.asarray([[3]], dtype=np.int32),
            "memory_budgets": np.asarray([[2]], dtype=np.int32),
            "total_budgets": np.asarray([[5]], dtype=np.int32),
        }
        model = SimpleNamespace(
            kv_cache=HeadRaggedCache([layer]),
            _horizonkv_config=config,
        )
        score = torch.linspace(0.0, 1.0, 8)
        probe_scores = [[[score]], [[score.flip(0)]], [[torch.ones(8)]]]

        self.assertTrue(horizonkv_needs_compression(model))
        keep, budgets, visible = select_horizonkv_keep_indices(model, probe_scores)
        selected = keep[0][0]
        self.assertEqual(int(budgets[0, 0]), 5)
        self.assertEqual(visible, [6])  # text prefix + five visual tokens
        self.assertTrue(torch.equal(selected[-3:], torch.tensor([5, 6, 7])))


if __name__ == "__main__":
    unittest.main()
