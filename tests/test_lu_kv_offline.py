import itertools
import unittest

import numpy as np

from head_analysis.lu_kv_offline import concave_blocks, integer_average, make_curves, solve_curves


class LUOfflineTests(unittest.TestCase):
    def test_envelope_and_bound(self):
        gains = np.array([1., 8., 2., 4., 1.])
        blocks = concave_blocks(gains)
        slopes = np.concatenate([np.full(n, s / n) for s, n in blocks])
        self.assertTrue(np.all(np.diff(slopes) <= 0))
        self.assertTrue(np.all(np.cumsum(slopes) >= np.cumsum(gains) - 1e-9))
        self.assertAlmostEqual(slopes.sum(), gains.sum())

    def test_concave_exact_and_nonconcave_bound(self):
        for gains in ([[5, 3, 1], [4, 2, 0]], [[1, 8, 2], [4, 0, 7]]):
            curves = np.column_stack([np.zeros(2), np.cumsum(gains, axis=1)])
            for total in range(7):
                b, diag = solve_curves(curves, total, 0)
                brute = max(sum(curves[h, x] for h, x in enumerate(bs))
                            for bs in itertools.product(range(4), repeat=2) if sum(bs) == total)
                self.assertEqual(b.sum(), total)
                self.assertGreaterEqual(diag["relaxed_upper_bound"] + 1e-9, brute)
                self.assertLessEqual(diag["actual_utility"], brute + 1e-9)
                if gains[0][0] == 5:
                    self.assertAlmostEqual(diag["actual_utility"], brute)

    def test_rounding_and_protection(self):
        b = integer_average([np.array([2, 3, 4]), np.array([4, 2, 3])], 9, 2)
        self.assertEqual(b.sum(), 9)
        self.assertTrue(np.all(b >= 2))
        u = np.array([[1, 5, 2, 3]], dtype=float)
        c = make_curves(np.array([[1, 0, 2]]), u, 1)
        np.testing.assert_equal(c, [[0, 3, 8, 9, 11]])


if __name__ == "__main__":
    unittest.main()
