"""Check TAD attention mass and temporal distance on a tiny causal cache."""
import unittest

import torch

from methods.h2svc.observe import TADObserver


class CacheLayer:
    def __init__(self, keys):
        self.keys = keys

    def get_segment(self, head):
        return self.keys, torch.zeros_like(self.keys)


class Decoder:
    device = 'cpu'
    nq = 2
    nk = 1


class ObservationTest(unittest.TestCase):
    def test_v2v_and_t2v_keep_nonvisual_softmax_mass(self):
        observer = TADObserver(Decoder())
        observer.key_frames = [-1, 0]
        observer.new_frames = [1]
        observer.mode = 'visual'
        q = torch.zeros((1, 2, 1, 1))
        k = torch.zeros((1, 1, 1, 1))
        observer(0, q, k, CacheLayer(torch.zeros((2, 1))), 1.0)
        self.assertAlmostEqual(float(observer.v2v[0, 0]), 1 / 3)
        observer.key_frames = [-1, 0, 1]
        observer.new_frames = [-1]
        observer.mode = 'text'
        observer.final_frame = 1
        observer(0, q, k, CacheLayer(torch.zeros((3, 1))), 1.0)
        self.assertAlmostEqual(float(observer.t2v[0, 0]), 1 / 4)


if __name__ == '__main__':
    unittest.main()
