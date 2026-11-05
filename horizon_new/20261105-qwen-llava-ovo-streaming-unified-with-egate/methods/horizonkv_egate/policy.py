"""Four-frame recent batches, independent latest frame, irreversible KV transfer.

Selected history is physically separate per layer/head. Only selected raw K,
V and coordinate IDs survive a batch transition; no old visual embeddings
or CPU video feature archive is used to reconstruct history.
"""
import hashlib
import torch
from .positions import positions, retained_metadata, frame_of
from .selection import snap_select


class Branch:
    def __init__(self, decoder, prefix_embeddings):
        self.decoder = decoder
        self.n = prefix_embeddings.shape[1]
        self.ids = list(range(-self.n, 0))
        self.metadata = {}
        self.cache = decoder.new_cache()
        p = positions(self.ids, self.n, {}, decoder.kind, decoder.device)
        _, self.raw_keys, _ = decoder.forward(prefix_embeddings, p, self.cache, capture=True)
        self.probes = None
        self.probe_ids = []

    def clone(self):
        other = object.__new__(Branch)
        other.__dict__ = dict(self.__dict__)
        other.cache = self.decoder.new_cache()
        other.cache.restore(self.cache.snapshot())
        other.raw_keys = list(self.raw_keys)
        other.ids = list(self.ids)
        other.metadata = dict(self.metadata)
        return other

    def append(self, embeddings, ids, metadata):
        if len(set(ids)) != len(ids) or set(ids) & set(self.ids):
            raise ValueError('Repeated visual identifiers')
        self.metadata = {**self.metadata, **metadata}
        p = positions(ids, self.n, self.metadata, self.decoder.kind, self.decoder.device)
        _, raw, self.probes = self.decoder.forward(embeddings[None], p, self.cache, capture=True)
        self.raw_keys = [torch.cat((old, new), dim=2) for old, new in zip(self.raw_keys, raw)]
        self.ids += ids
        self.probe_ids = ids[-32:]

    def queries(self, layer, metadata=None):
        p = positions(self.probe_ids, self.n, self.metadata if metadata is None else metadata,
                      self.decoder.kind, self.decoder.device)
        return self.decoder.rotate(self.probes[layer], p, layer)[0]


class State:
    def __init__(self, prefix):
        self.prefix = prefix
        self.full = prefix.clone()
        self.one = None
        self.history = {}
        self.history_metadata = {}
        self.batch_frames = 0
        self.seen_frames = 0
        self.transfers = 0

    def clone(self):
        other = object.__new__(State)
        other.__dict__ = dict(self.__dict__)
        other.full = self.full.clone()
        other.one = self.one.clone() if self.one is not None else None
        # History updates replace the dictionary and its tensors, never mutate it.
        return other

    def fingerprint(self):
        tensors = [t for b in (self.full, self.one) if b is not None for layer in b.cache.layers
                   for t in (layer.flat_k, layer.flat_v, layer.flat_pos, layer.head_lens)]
        tensors += [t for e in self.history.values() for t in (e['keys'], e['values'])]
        return (self.seen_frames, self.batch_frames, self.transfers, tuple(self.full.ids),
                tuple((t.data_ptr(), t._version if not t.is_inference() else None, tuple(t.shape)) for t in tensors),
                tuple((key, tuple(e['uids'])) for key, e in sorted(self.history.items())))


def recent_selection(state, allocation):
    """Select once, with history capacity reserved identically for both passes."""
    selected = {}
    decoder = state.prefix.decoder
    group = decoder.nq // decoder.nk
    for li in range(decoder.nl):
        for head in range(decoder.nk):
            config = allocation[li, head]
            branch = state.one if config['recent_group'] == 'one_frame' else state.full
            old = state.history.get((li, head))
            old_count = len(old['uids']) if old else 0
            if old_count > config['history_quota']:
                raise RuntimeError('Historical head budget exceeded')
            budget = config['total_quota'] - old_count
            query = branch.queries(li)[head * group:(head + 1) * group]
            key, val = branch.cache.layers[li].get_segment(head)
            keep = snap_select(key, query, None, branch.n, budget)
            selected[li, head] = dict(
                keys=branch.raw_keys[li][0, head].index_select(0, keep).contiguous(),
                values=val.index_select(0, keep).contiguous(),
                uids=[branch.ids[j] for j in keep.tolist()], branch=branch)
    return selected


def transfer(state, allocation):
    if state.batch_frames != 4:
        raise RuntimeError('Only complete nonoverlapping four-frame batches transfer')
    decoder = state.prefix.decoder
    n = state.prefix.n
    selected = recent_selection(state, allocation)
    metadata = {**state.history_metadata, **state.full.metadata}
    heads = {}
    group = decoder.nq // decoder.nk
    for (li, head), recent in selected.items():
        old = state.history.get((li, head))
        keys = torch.cat(([old['keys']] if old else []) + [recent['keys']])
        values = torch.cat(([old['values']] if old else []) + [recent['values']])
        ids = (old['uids'] if old else []) + recent['uids']
        branch = recent['branch']
        raw = torch.cat((branch.raw_keys[li][0, head, :n], keys))
        p = positions(list(range(-n, 0)) + ids, n, metadata, decoder.kind, decoder.device)
        rotated = decoder.rotate(raw[None, None], p, li)[0, 0]
        q = branch.queries(li, metadata)[head * group:(head + 1) * group]
        idx = snap_select(rotated, q, None, n, allocation[li, head]['history_quota'], protect=0) - n
        kept = [ids[j] for j in idx.tolist()]
        if len(kept) > allocation[li, head]['history_quota'] or len(set(kept)) != len(kept):
            raise RuntimeError('History transfer invariant failed')
        heads[li, head] = dict(keys=keys.index_select(0, idx).contiguous(),
                              values=values.index_select(0, idx).contiguous(), uids=kept)
    state.history = heads
    state.history_metadata = retained_metadata(heads, metadata)
    state.full = state.prefix.clone()
    state.one = None
    state.batch_frames = 0
    state.transfers += 1


def append_unit(state, allocation, features, ids, metadata, latest_features,
                latest_ids, latest_metadata, input_frames):
    if state.batch_frames == 4:
        transfer(state, allocation)
    if not 1 <= input_frames <= 2 or state.batch_frames + input_frames > 4:
        raise ValueError('Input crosses an EGate four-frame batch boundary')
    state.full.append(features, ids, metadata)
    one = state.prefix.clone()
    # The independent branch reads prefix + latest frame, never full/history KV.
    one.metadata = dict(state.full.metadata)
    one.metadata.update(latest_metadata)
    one.append(latest_features, latest_ids, latest_metadata)
    state.one = one
    state.batch_frames += input_frames
    state.seen_frames += input_frames


def answer_caches(state, allocation):
    selected = recent_selection(state, allocation)
    decoder, n = state.prefix.decoder, state.prefix.n
    metadata = {**state.history_metadata, **state.full.metadata, **state.one.metadata}
    active = {uid for e in selected.values() for uid in e['uids']}
    active.update(uid for e in state.history.values() for uid in e['uids'])
    # Match the original answer-time compaction: unselected frames do not
    # consume a temporal interval. This union describes coordinates only.
    active_frames = {frame_of(uid) for uid in active}
    metadata = {frame: grid for frame, grid in metadata.items() if frame in active_frames}
    all_pos = positions(sorted(active), n, metadata, decoder.kind, decoder.device)
    start = max(n, int(all_pos.max()) + 1) if active else n
    if start + 256 >= decoder.position_limit:
        raise RuntimeError('Compacted active KV reached model position limit')
    caches = [decoder.new_cache(), decoder.new_cache()]
    digest = hashlib.sha256()
    recent_slots = history_slots = 0
    for li in range(decoder.nl):
        by_pass = [[], []]
        for head in range(decoder.nk):
            e = selected[li, head]
            b = e['branch']
            raw = torch.cat((b.raw_keys[li][0, head, :n], e['keys']))
            _, prefix_v = b.cache.layers[li].get_segment(head)
            value = torch.cat((prefix_v[:n], e['values']))
            ids = list(range(-n, 0)) + e['uids']
            old = state.history.get((li, head))
            digest.update(str((li, head, e['uids'])).encode())
            recent_slots += len(e['uids'])
            history_slots += len(old['uids']) if old else 0
            if len(e['uids']) + (len(old['uids']) if old else 0) > 500:
                raise RuntimeError('Combined per-head visual budget exceeded 500')
            for index, use_history in enumerate((False, True)):
                rr, vv, uu = raw, value, ids
                if use_history and old:
                    rr = torch.cat((raw[:n], old['keys'], raw[n:]))
                    vv = torch.cat((value[:n], old['values'], value[n:]))
                    uu = ids[:n] + old['uids'] + ids[n:]
                p = positions(uu, n, metadata, decoder.kind, decoder.device)
                k = decoder.rotate(rr[None, None], p, li)[0, 0]
                by_pass[index].append((k, vv, p))
        for index in range(2):
            layer = caches[index].layers[li]
            parts = by_pass[index]
            layer.flat_k = torch.cat([e[0] for e in parts]).contiguous()
            layer.flat_v = torch.cat([e[1] for e in parts]).contiguous()
            layer.flat_pos = torch.cat([e[2] for e in parts], dim=1).to(torch.float32).contiguous()
            layer.head_lens = layer.head_lens.new_tensor([len(e[0]) for e in parts])
            layer._refresh()
    return caches, start, dict(recent_selection_sha256=digest.hexdigest(), recent_slots=recent_slots,
                               history_slots=history_slots, suffix_start=start,
                               max_visual_per_head=max(max(layer.head_lens.tolist()) - n for layer in caches[1].layers),
                               history_transfers=state.transfers)
