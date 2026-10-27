"""Window partition invariants for the experimental Qwen3 vision topology."""

import unittest

import torch

from sparsemm_baselines.qwen3_window4 import window_index_and_lengths


class WindowFourTests(unittest.TestCase):
    def test_windows_cover_multiframe_merged_grid(self):
        grid = torch.tensor([[2, 16, 24], [1, 8, 8]])
        index, lengths = window_index_and_lengths(grid)
        merged_tokens = 2 * 8 * 12 + 4 * 4
        self.assertEqual(len(index), merged_tokens)
        self.assertTrue(torch.equal(torch.sort(index).values, torch.arange(merged_tokens)))
        self.assertEqual(sum(lengths), merged_tokens * 4)
        self.assertLessEqual(max(lengths), 8 * 8)
        self.assertEqual(len(lengths), 2 * 2 * 3 + 1)

    def test_rejects_partial_merge_units(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            window_index_and_lengths(torch.tensor([[1, 15, 16]]))


if __name__ == "__main__":
    unittest.main()
