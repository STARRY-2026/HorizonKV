"""CPU checks for paper frame budgets and atomic Qwen temporal-merge units."""
import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from methods.h2svc.policy import FrameUnit, HeadBank, classify_heads, curvature_importance


@unittest.skipIf(torch is None, 'Requires the inference host torch runtime')
class H2SVCPolicyTests(unittest.TestCase):
    def unit(self, first, last, value):
        vec = torch.tensor([[float(value), 0.0]])
        return FrameUnit(first, last, vec, vec, (last,))

    def test_qwen_pairs_are_evicted_as_indivisible_units(self):
        for kind, budget in [('instantaneous', 4), ('short_term', 16), ('episodic', 64)]:
            bank = HeadBank(kind)
            for first in range(0, 80, 2):
                bank.append(self.unit(first, first+1, first))
                self.assertLessEqual(bank.retained_frames, budget)
                self.assertTrue(all(unit.span == 2 for unit in bank.units))
            self.assertEqual(bank.retained_frames, budget)
            self.assertEqual(bank.units[-1].last_frame, 79)
        self.assertEqual([u.first_frame for u in HeadBank('instantaneous').units], [])

    def test_fifo_and_stce_curvature(self):
        fifo = HeadBank('instantaneous')
        for i in range(6):
            fifo.append(self.unit(i, i, i))
        self.assertEqual([x.first_frame for x in fifo.units], [2, 3, 4, 5])
        smooth = curvature_importance(self.unit(0, 0, 0), self.unit(1, 1, 1), self.unit(2, 2, 2))
        turn = curvature_importance(self.unit(0, 0, 0), self.unit(1, 1, 1), self.unit(2, 2, 0))
        self.assertAlmostEqual(smooth, 0)
        self.assertGreater(turn, smooth)

    def test_head_routes_and_no_resurrection(self):
        keys = [(layer, head) for layer in range(28) for head in range(4)]
        v2v = {key: i for i, key in enumerate(keys)}
        t2v = dict(v2v)
        routes = classify_heads(v2v, t2v)
        self.assertEqual([sum(x == kind for x in routes.values()) for kind in
                          ('instantaneous', 'short_term', 'episodic')], [11, 45, 56])
        bank = HeadBank('instantaneous')
        first = self.unit(0, 1, 0)
        bank.append(first)
        bank.append(self.unit(2, 3, 2))
        copy = bank.clone()
        bank.append(self.unit(4, 5, 4))
        self.assertNotIn(first.first_frame, [u.first_frame for u in bank.units])
        self.assertIn(first.first_frame, [u.first_frame for u in copy.units])


if __name__ == '__main__':
    unittest.main()
