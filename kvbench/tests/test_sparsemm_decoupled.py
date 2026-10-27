"""Independent representation computation, bounded fusion and temporal forks."""

import copy
import unittest

import torch
from transformers import Qwen2_5_VLTextConfig
from transformers.cache_utils import DynamicCache
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel

from sparsemm_native.decoupled import (
    IndependentHistoryCache, VisualUnit, concatenate_query_cache, prefill_recent,
)
from sparsemm_protokv.config import StreamingConfig
from sparsemm_native.run_ovobench_decoupled import pilot_queries
from sparsemm_native.run_ovobench import queries as official_queries


def unit(start, tokens=32, value=1.0, dim=32):
    return VisualUnit(torch.full((1, tokens, dim), value),
                      torch.full((3, 1, tokens), float(start * 100 + 10)),
                      (1, 8, tokens // 2), start, 2)


def cache(layers=2, heads=2, dim=8, weights=None):
    value = IndependentHistoryCache(
        StreamingConfig(method="sparsemm", budget=500, sparsemm_two_bank_gate="legacy_global"),
        layers, heads, dim, "cpu", torch.float32,
        history_weights=weights if weights is not None else torch.ones((layers, heads), dtype=torch.long))
    p = 3
    value.prefix_tokens = p
    value._initialized = True
    value.logical_tokens = p
    for layer in value.layers:
        k = torch.zeros(1, heads, p, dim)
        positions = torch.arange(p).float()[None].expand(3, -1)
        layer.append_chunk(k, k.clone(), positions)
    return value


class RecentIsolationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        config = Qwen2_5_VLTextConfig(
            vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
            rope_scaling={"type": "mrope", "mrope_section": [1, 1, 2]},
            attn_implementation="sdpa")
        self.model = Qwen2_5_VLTextModel(config).eval()
        with torch.inference_mode():
            result = self.model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=True,
                                past_key_values=DynamicCache(config=config), return_dict=True)
        self.prefix = tuple((k.clone(), v.clone()) for k, v in result.past_key_values)

    def test_prefill_has_only_prefix_and_exact_window_after_arbitrary_history_changes(self):
        units = [unit(4), unit(6, value=2.)]
        seen = []
        def observe(module, args, kwargs):
            seen.append(kwargs["past_key_values"].get_seq_length())
        hook = self.model.register_forward_pre_hook(observe, with_kwargs=True)
        a = prefill_recent(self.model, self.prefix, units)
        history = cache()
        for layer in history.layers:
            layer.flat_k.add_(100000)
            layer.flat_v.sub_(100000)
        b = prefill_recent(self.model, self.prefix, units)
        hook.remove()
        self.assertEqual(seen, [3, 3])
        self.assertEqual(a.get_seq_length(), 67)
        self.assertIsNot(a, b)
        for (ak, av), (bk, bv), (pk, pv) in zip(a, b, self.prefix):
            self.assertTrue(torch.equal(ak, bk) and torch.equal(av, bv))
            self.assertTrue(torch.equal(ak[:, :, :3], pk) and torch.equal(av[:, :, :3], pv))

    def test_sliding_window_is_rebuilt_without_influence_from_dropped_unit(self):
        initial = prefill_recent(self.model, self.prefix, [unit(0, value=100.), unit(2)])
        first = unit(2)
        second = unit(4, value=2.)
        rebuilt = prefill_recent(self.model, self.prefix, [first, second])
        independently = prefill_recent(self.model, self.prefix, [first, second])
        for (ak, av), (bk, bv) in zip(rebuilt, independently):
            self.assertTrue(torch.equal(ak, bk) and torch.equal(av, bv))
        self.assertFalse(torch.equal(initial[1][1], rebuilt[1][1]))

    def test_native_history_forward_is_unchanged_when_current_recent_features_change(self):
        from sparsemm_protokv.qwen2_5 import install, uninstall
        install()
        self.addCleanup(uninstall)
        a = cache()
        a.push_unit(unit(0))
        a.push_unit(unit(2))
        outgoing = a.push_unit(unit(4))
        b = copy.deepcopy(a)
        for current in b.units:
            current.features.add_(1000.)
        with torch.inference_mode():
            for value in (a, b):
                value.set_context("video", outgoing.positions[:, 0], xy=torch.zeros(32, 2), time=2)
                self.model(inputs_embeds=outgoing.features, position_ids=outgoing.positions,
                           past_key_values=value, cache_position=torch.arange(3, 35),
                           attention_mask={"full_attention": None}, use_cache=True, return_dict=True)
                value.finish_video()
        for left, right in zip(a.layers, b.layers):
            self.assertTrue(torch.equal(left.flat_k, right.flat_k))
            self.assertTrue(torch.equal(left.flat_v, right.flat_v))

    def test_fresh_dense_window_matches_native_streaming_when_no_history_exists(self):
        from sparsemm_native.recent4 import Recent4Cache
        from sparsemm_protokv.qwen2_5 import install, uninstall
        install()
        self.addCleanup(uninstall)
        original = Recent4Cache(
            StreamingConfig(method="sparsemm", budget=500, sparsemm_cache_policy="two_bank",
                            sparsemm_budget_path="synthetic_test", sparsemm_two_bank_gate="legacy_global"),
            2, 2, 8, "cpu", torch.float32, budgets=torch.full((2, 2), 500),
            recent_budgets=torch.full((2, 2), 32))
        original.prefix_tokens, original.logical_tokens, original._initialized = 3, 3, True
        for layer, (k, v) in zip(original.layers, self.prefix):
            layer.append_chunk(k, v, torch.arange(3).float()[None].expand(3, -1))
        units = [unit(0), unit(2, value=2.)]
        with torch.inference_mode():
            for current in units:
                original.begin_visual_unit(current.tokens)
                original.set_context("video", current.positions[:, 0], xy=torch.zeros(32, 2), time=2)
                offset = original.get_seq_length()
                self.model(inputs_embeds=current.features, position_ids=current.positions,
                           past_key_values=original, cache_position=torch.arange(offset, offset + current.tokens),
                           attention_mask={"full_attention": None}, use_cache=True, return_dict=True)
                original.finish_video()
            rebuilt = prefill_recent(self.model, self.prefix, units)
        for layer, (k, v) in zip(original.layers, rebuilt):
            for h in range(2):
                old_k, old_v = layer.get_segment(h)
                self.assertTrue(torch.allclose(old_k, k[0, h], atol=1e-5, rtol=1e-4))
                self.assertTrue(torch.allclose(old_v, v[0, h], atol=1e-5, rtol=1e-4))


class HistoryAndFusionTests(unittest.TestCase):
    def test_global_guard_prevents_mixed_layer_warmup_from_exceeding_mean_500(self):
        value = cache(weights=torch.tensor([[0, 0], [1, 1]]))
        guards = []
        for index in range(24):
            outgoing = value.push_unit(unit(index * 2, tokens=66))
            if outgoing is None:
                continue
            guards.append(value.global_capacity_guard)
            value.set_context("video", outgoing.positions[:, 0], xy=torch.zeros(66, 2), time=index*2)
            keys = torch.ones(1, 2, 66, 8)
            queries = torch.ones(1, 4, 66, 8)
            for layer in range(2):
                value.append_layer(layer, keys, keys, q=queries)
            value.finish_video()
            visual = sum(int(n) - value.prefix_tokens + 132 for layer in value.layers for n in layer.head_lens)
            self.assertLessEqual(visual, 2 * 2 * 500)
        self.assertTrue(any(guards))

    def test_window_uses_features_and_fork_restores_allocation_and_odd_state(self):
        value = cache()
        self.assertIsNone(value.push_unit(unit(0)))
        self.assertIsNone(value.push_unit(unit(2)))
        parent = value
        odd = copy.deepcopy(parent)
        outgoing = odd.push_unit(VisualUnit(torch.ones(1, 32, 32),
            torch.full((3, 1, 32), 410.), (1, 8, 16), 4, 1))
        self.assertEqual(outgoing.frame_start, 0)
        self.assertEqual([u.frame_start for u in parent.units], [0, 2])
        self.assertEqual([u.frame_start for u in odd.units], [2, 4])
        self.assertEqual(sum(u.real_frames for u in odd.units), 3)
        self.assertEqual(parent.revision, 2)
        self.assertEqual(odd.revision, 3)
        self.assertEqual(int(odd.budgets.sum()), 2 * 2 * 500)

    def test_history_never_contains_recent_and_respects_global_quotas_and_zero_weights(self):
        value = cache(weights=torch.tensor([[0, 1], [1, 1]]))
        value.push_unit(unit(0, tokens=180))
        value.push_unit(unit(2, tokens=180))
        outgoing = value.push_unit(unit(4, tokens=180))
        q = torch.ones(1, 4, 180, 8)
        k = torch.ones(1, 2, 180, 8)
        value.set_context("video", outgoing.positions[:, 0], xy=torch.zeros(180, 2), time=2)
        for layer in range(2):
            value.append_layer(layer, k, k, q=q)
        value.finish_video()
        self.assertEqual(value.layers[0].head_lens[0], value.prefix_tokens)
        self.assertTrue(all(m["history_reads_recent"] is False for m in value.last_metadata["layers"]))
        # Passing a current recent unit as history must be rejected.
        value.set_context("video", value.units[0].positions[:, 0], xy=torch.zeros(180, 2), time=4)
        with self.assertRaisesRegex(AssertionError, "recent token"):
            value.append_layer(0, k, k, q=q)

    def test_fusion_has_one_prefix_actual_budget_and_read_only_query_transaction(self):
        history = cache()
        history.push_unit(unit(4))
        history.push_unit(unit(6))
        for layer in history.layers:
            k = torch.ones(1, 2, 10, 8)
            pos = torch.full((3, 10), 10.)
            layer.append_chunk(k, k * 2, pos)
        recent = [(torch.full((1, 2, 67, 8), 3.), torch.full((1, 2, 67, 8), 4.)) for _ in range(2)]
        before = [(layer.flat_k.clone(), layer.flat_v.clone()) for layer in history.layers]
        fused, audit = concatenate_query_cache(history, recent, 2000)
        self.assertEqual(audit["actual_visual_slots"], 2 * 2 * (10 + 64))
        self.assertEqual(audit["prefix_copies_in_query"], 1)
        for layer in fused.layers:
            self.assertEqual(layer.head_lens.tolist(), [77, 77])
            k, v = layer.get_segment(0)
            self.assertTrue(torch.equal(k[:3], torch.zeros(3, 8)))
            self.assertTrue(torch.equal(k[3:13], torch.ones(10, 8)))
            self.assertTrue(torch.equal(k[13:], torch.full((64, 8), 3.)))
        saved = fused.layers[0].flat_k
        with fused.query():
            positions = torch.full((3, 1), 3000.)
            fused.set_context("query", positions)
            k = torch.ones(1, 2, 1, 8)
            for index in range(2):
                fused.append_layer(index, k, k)
            fused.advance_query(1)
        self.assertIs(fused.layers[0].flat_k, saved)
        for layer, (k, v) in zip(history.layers, before):
            self.assertTrue(torch.equal(layer.flat_k, k) and torch.equal(layer.flat_v, v))

    def test_fusion_rejects_time_overlap_and_actual_over_budget(self):
        history = cache()
        history.push_unit(unit(4))
        history.push_unit(unit(6))
        recent = [(torch.zeros(1, 2, 67, 8), torch.zeros(1, 2, 67, 8)) for _ in range(2)]
        history.layers[0].append_chunk(torch.zeros(1, 2, 1, 8), torch.zeros(1, 2, 1, 8),
                                       torch.full((3, 1), 500.))
        with self.assertRaisesRegex(AssertionError, "time ranges overlap"):
            concatenate_query_cache(history, recent, 1000)
        history = cache()
        history.push_unit(unit(4))
        history.push_unit(unit(6))
        for layer in history.layers:
            k = torch.ones(1, 2, 500, 8)
            layer.append_chunk(k, k, torch.full((3, 500), 10.))
        with self.assertRaisesRegex(AssertionError, "exceeded mean K500"):
            concatenate_query_cache(history, recent, 1000)


class PilotSelectionTests(unittest.TestCase):
    def test_balanced_predeclared_selection_has_history_and_does_not_depend_on_answers(self):
        import collections
        official = official_queries()
        chosen = pilot_queries(official)
        self.assertEqual(len(chosen), 24)
        self.assertEqual(set(collections.Counter(q["task"] for q in chosen).values()), {2})
        self.assertTrue(all(15 <= q["clip_end"] <= 64 for q in chosen))
        changed = [dict(q, target="deliberately_changed") for q in official]
        self.assertEqual([q["query_id"] for q in pilot_queries(changed)], [q["query_id"] for q in chosen])
        groups = collections.Counter(q["video_path"] for q in chosen)
        self.assertTrue(any(n > 1 for n in groups.values()))


if __name__ == "__main__":
    unittest.main()
