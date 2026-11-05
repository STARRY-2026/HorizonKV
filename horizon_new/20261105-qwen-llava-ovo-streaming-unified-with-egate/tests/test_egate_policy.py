"""CPU policy checks; available on the inference host where torch is installed."""
import unittest
try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from sparsemm_baselines.cache import RaggedCache
    from methods.horizonkv_egate.policy import Branch, State, append_unit, answer_caches
    from methods.horizonkv_egate.positions import token_ids, frame_of


class FakeDecoder:
    kind = 'llavaov'
    device = 'cpu'
    nl, nk, nq, dim = 2, 2, 4, 4
    position_limit = 1000000

    def new_cache(self):
        return RaggedCache(self.nl, self.nk, self.dim, 'cpu', torch.float32)

    def rotate(self, tensor, pos, layer):
        return tensor

    def forward(self, embeddings, pos, cache, capture=False):
        n = embeddings.shape[1]
        k = embeddings.view(1, n, 2, 4).transpose(1, 2).contiguous()
        keys, probes = [], []
        for i, layer in enumerate(cache.layers):
            layer.append_chunk(k, k + i, pos)
            keys.append(k.clone())
            probes.append(k.repeat_interleave(2, dim=1)[:, :, -32:].clone())
        return embeddings, keys, probes


@unittest.skipIf(torch is None, 'Requires the inference host torch runtime')
class EGatePolicyTests(unittest.TestCase):
    def state(self):
        torch.manual_seed(13)
        return State(Branch(FakeDecoder(), torch.randn(1,4,8)))

    def allocation(self):
        return {(li,h):dict(recent_group='one_frame' if h==0 else 'four_frames',
                  recent_quota=460,history_quota=40,total_quota=500) for li in range(2) for h in range(2)}

    def add(self, state, n):
        for _ in range(n):
            frame = state.seen_frames
            features = torch.randn(128,8)
            ids = token_ids(frame,8,16)
            append_unit(state,self.allocation(),features,ids,{frame:(8,16)},features,ids,{frame:(8,16)},1)

    def test_direct_transfer_is_bounded_and_cannot_resurrect_evicted_kv(self):
        state=self.state()
        self.add(state,5)
        self.assertEqual(state.transfers,1)
        first={(li,h):set(e['uids']) for (li,h),e in state.history.items()}
        self.assertTrue(all(len(x)==40 for x in first.values()))
        self.add(state,4)
        self.assertEqual(state.transfers,2)
        second={(li,h):set(e['uids']) for (li,h),e in state.history.items()}
        for key, ids in second.items():
            older={uid for uid in ids if frame_of(uid)<4}
            self.assertTrue(older <= first[key])
        self.add(state,4)
        self.assertEqual(state.transfers,3)
        for key,e in state.history.items():
            self.assertEqual(len(e['uids']),40)
            self.assertTrue({uid for uid in e['uids'] if frame_of(uid)<8} <= second[key])
        self.assertEqual(state.batch_frames,1)
        self.assertEqual(len(state.full.ids),4+128)
        self.assertTrue(all(frame<12 for frame in state.history_metadata))

    def test_passes_share_selected_recent_and_do_not_mutate_state(self):
        state=self.state(); self.add(state,8)
        before=state.fingerprint()
        caches,start,audit=answer_caches(state,self.allocation())
        self.assertEqual(before,state.fingerprint())
        self.assertEqual(audit['max_visual_per_head'],500)
        self.assertEqual(audit['history_slots'],160)
        n=state.prefix.n
        for li in range(2):
            for h in range(2):
                recent_k,recent_v=caches[0].layers[li].get_segment(h)
                full_k,full_v=caches[1].layers[li].get_segment(h)
                history=len(state.history[li,h]['uids'])
                self.assertTrue(torch.equal(recent_k[:n],full_k[:n]))
                self.assertTrue(torch.equal(recent_k[n:],full_k[n+history:]))
                self.assertTrue(torch.equal(recent_v[n:],full_v[n+history:]))
                self.assertLessEqual(len(full_k)-n,500)
        snapshot=state.clone(); frozen=snapshot.fingerprint()
        self.add(state,1)
        self.assertEqual(snapshot.fingerprint(),frozen)
        self.assertNotEqual(state.fingerprint(),frozen)


if __name__=='__main__':unittest.main()
