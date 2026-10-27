"""Small exactness checks for the Qwen3 attention-ranked DP pilot."""

import unittest

import numpy as np

from tools.qwen3vl_dp_attention_pilot import (
    curves_for_observation, exact_allocation, shared_split_curves,
)


class Qwen3DPPilotTests(unittest.TestCase):
    def test_exact_capacity_allocation_matches_exhaustive_search(self):
        curves = np.array([[0., 1., 4., 4.5], [0., 3., 3.5, 7.]], dtype=np.float64)
        chosen, value = exact_allocation(curves, average=2, floor=1, cap=3)
        brute = max((curves[0, left] + curves[1, 4 - left], left)
                    for left in range(1, 4))
        self.assertAlmostEqual(value, brute[0])
        self.assertEqual(int(chosen.sum()), 4)
        self.assertAlmostEqual(float(curves[np.arange(2), chosen].sum()), value)

    def test_shared_recent_split_uses_one_endpoint_for_both_observations(self):
        first = np.array([[1., 1., 0., 0., 3.]], dtype=np.float64)
        second = np.array([[0., 0., 3., 1., 0.]], dtype=np.float64)
        ranking = np.array([[0, 1, 2, 3]], dtype=np.int64)
        observations = [({}, ranking, first), ({}, ranking, second)]
        values, endpoints = shared_split_curves(observations, floor=2, cap=3, recent_min=1)
        self.assertEqual(values.shape, (1, 4))
        for capacity in (2, 3):
            recent = int(endpoints[0, capacity])
            self.assertGreaterEqual(recent, 1)
            self.assertLessEqual(recent, capacity)
            direct = []
            for _, order, utility in observations:
                row = utility[0]
                eligible = order[0][order[0] < len(row) - recent]
                kept = np.r_[np.arange(len(row) - recent, len(row)), eligible[:capacity - recent]]
                direct.append(float(row[kept].sum() / row.sum()))
            self.assertAlmostEqual(values[0, capacity], float(np.mean(direct)))


if __name__ == "__main__":
    unittest.main()
