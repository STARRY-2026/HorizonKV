"""CPU integration check for H2SVC cache assembly and bounded history."""
import unittest

try:
    import torch
    from sparsemm_baselines.cache import RaggedCache
except ImportError:
    torch = None

if torch is not None:
    from methods.h2svc.state import H2State


@unittest.skipIf(torch is None, 'Requires torch and the vendored ragged cache on PYTHONPATH')
class H2SVCStateTests(unittest.TestCase):
    def test_cache_stays_bounded_and_evicted_frames_do_not_return(self):
        class Decoder:
            nl, nk, dim, kind, device, dtype, position_limit = 1, 1, 2, 'llavaov', 'cpu', torch.float32, 4096

            def new_cache(self):
                return RaggedCache(1, 1, 2, 'cpu', torch.float32)

            def rotate(self, tensor, position, layer):
                return tensor

            def forward(self, embeddings, position, cache, capture=False):
                raw = embeddings[:, None]
                cache.layers[0].append_chunk(raw, raw, position)
                return embeddings, [raw] if capture else [], []

        decoder = Decoder()
        prefix_cache = decoder.new_cache()
        prefix = torch.ones(1, 1, 2)
        prefix_cache.layers[0].append_chunk(prefix[:, None], prefix[:, None], torch.zeros(3, 1, dtype=torch.long))
        branch = type('Prefix', (), dict(decoder=decoder, n=1, raw_keys=[prefix[:, None]], cache=prefix_cache))()
        state = H2State(branch, {(0, 0):'episodic'})
        for frame in range(70):
            state.append(torch.tensor([[float(frame), 1.0]]), (1, 1), 1)
            self.assertLessEqual(state.retained_counts()[0, 0], 64)
        cache, start = state.make_cache()
        self.assertEqual(int(cache.layers[0].head_lens[0]), 65)
        self.assertEqual(len(state.metadata), 64)
        self.assertGreater(start, 0)
        self.assertEqual(state.seen_frames, 70)


if __name__ == '__main__':
    unittest.main()
