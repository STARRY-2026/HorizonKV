"""Checks the per-head HERMES adaptation used by LU calibration and eviction."""

import unittest

import torch

from sparsemm_baselines.qwen3_hermes import hermes_layer_scores, layer_groups
from sparsemm_lukv.qwen3_hermes_ranking import hermes_head_scores


class LUKVHermesPseudoTests(unittest.TestCase):
    def test_matches_layer_shared_hermes_when_heads_agree(self):
        torch.manual_seed(11)
        layers = 10
        local = [torch.rand(7) for _ in range(layers)]
        global_ = [torch.rand(7) for _ in range(layers)]
        mixed = [torch.rand(7) for _ in range(layers)]
        shallow, deep = layer_groups(layers)
        expected, kinds = hermes_layer_scores(local, global_, mixed, [7] * layers,
                                               shallow, deep)
        actual, actual_kinds = hermes_head_scores(
            [[row, row] for row in local], [[row, row] for row in global_],
            [[row, row] for row in mixed])
        self.assertEqual(kinds, actual_kinds)
        for layer in range(layers):
            for head in range(2):
                torch.testing.assert_close(actual[layer][head], expected[layer], atol=1e-7, rtol=0)

    def test_distinct_heads_keep_distinct_scores(self):
        layers = 10
        forward = torch.tensor([.1, .2, .3, .4])
        backward = forward.flip(0)
        local = [[forward, backward] for _ in range(layers)]
        global_ = [[forward, backward] for _ in range(layers)]
        mixed = [[forward, backward] for _ in range(layers)]
        scores, _ = hermes_head_scores(local, global_, mixed)
        self.assertEqual(int(scores[-1][0].argmax()), 3)
        self.assertEqual(int(scores[-1][1].argmax()), 0)


if __name__ == "__main__":
    unittest.main()
