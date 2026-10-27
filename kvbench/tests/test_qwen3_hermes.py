"""CPU checks for Qwen3 HERMES depth allocation and eviction scores."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sparsemm_baselines.cache import RaggedCache
from sparsemm_baselines.qwen3_hermes import Qwen3Hermes, hermes_layer_scores, layer_groups, predicted_queries


class Qwen3HermesTest(unittest.TestCase):
    def test_requested_depth_split_for_36_layers(self):
        shallow_end, deep_start = layer_groups(36)
        self.assertEqual((shallow_end, deep_start), (3, 25))
        self.assertEqual((shallow_end, deep_start - shallow_end, 36 - deep_start), (3, 22, 11))

    def test_depth_scores_use_local_middle_and_global_queries(self):
        ascending = torch.tensor([0.1, 0.2, 0.3, 0.4])
        descending = ascending.flip(0)
        local = [ascending] * 3
        global_ = [descending] * 3
        mixed = [descending] * 3
        scores, kinds = hermes_layer_scores(local, global_, mixed, [4] * 3, 1, 2)
        self.assertEqual(kinds, ["short-term", "mid-term", "long-term"])
        self.assertEqual(int(scores[0].argmax()), 3)
        self.assertEqual(int(scores[-1].argmax()), 0)
        self.assertTrue(all(torch.isfinite(score).all() for score in scores))

    def test_prediction_uses_recent_answer(self):
        local, global_ = predicted_queries([("What color?", "B", ["red", "blue"])])
        self.assertIn("blue", local)
        self.assertIn("blue", global_)

    def test_pseudo_scoring_prunes_each_layer_without_changing_prefix(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(8, 4)

            @property
            def device(self):
                return self.embedding.weight.device

            def get_input_embeddings(self):
                return self.embedding

        def attention():
            return SimpleNamespace(q_norm=torch.nn.Identity(), k_norm=torch.nn.Identity(),
                                   q_proj=torch.nn.Linear(4, 4, bias=False),
                                   k_proj=torch.nn.Linear(4, 4, bias=False), scaling=0.5)

        hermes = object.__new__(Qwen3Hermes)
        hermes.model = TinyModel()
        hermes.processor = SimpleNamespace(tokenizer=lambda *a, **kw: SimpleNamespace(
            input_ids=torch.tensor([[1, 2]])))
        hermes.num_layers = 36
        hermes.num_query_heads = 1
        hermes.num_kv_heads = 1
        hermes.head_dim = 4
        hermes.visual_start_idx = 1
        hermes._next_position = 5
        hermes.kv_size = 2
        hermes.short_term_threshold, hermes.long_term_threshold = layer_groups(36)
        hermes.conv_history = []
        hermes.trace_path = None
        hermes._prefills = hermes._compressions = 0
        hermes.kv_cache = RaggedCache(36, 1, 4, torch.device("cpu"), torch.float32)
        hermes.language_model = SimpleNamespace(
            layers=[SimpleNamespace(input_layernorm=torch.nn.Identity(), self_attn=attention())
                    for _ in range(36)],
            rotary_emb=lambda embeddings, positions: (None, None))
        keys = torch.randn(1, 1, 5, 4)
        for layer in hermes.kv_cache.layers:
            layer.append_chunk(keys, keys, torch.arange(5).expand(3, -1))
        with patch("sparsemm_baselines.qwen3_hermes.apply_rotary_pos_emb",
                   side_effect=lambda q, k, cos, sin: (q, k)):
            hermes.predict_and_compress()
        self.assertEqual(hermes._get_cache_seq_len_per_layer()[:3], [3, 3, 3])
        self.assertEqual(hermes._get_cache_seq_len_per_layer()[25:], [2] * 11)
        self.assertTrue(all(layer.get_position_segment(0)[:, 0].eq(0).all()
                            for layer in hermes.kv_cache.layers))


if __name__ == "__main__":
    unittest.main()
