"""Per-head raw KV state with paper-budget H2SVC memory banks."""
import torch

from methods.horizonkv_egate.positions import positions, token_ids
from .policy import FrameUnit, HeadBank


class H2State:
    def __init__(self, prefix, routes):
        self.prefix = prefix
        self.routes = routes
        decoder = prefix.decoder
        expected = {(layer, head) for layer in range(decoder.nl) for head in range(decoder.nk)}
        if set(routes) != expected:
            raise ValueError('Incomplete H2SVC head map')
        self.banks = {key: HeadBank(routes[key]) for key in expected}
        self.metadata = {}
        self.seen_frames = 0

    def clone(self):
        other = object.__new__(H2State)
        other.prefix = self.prefix
        other.routes = self.routes
        other.banks = {key: bank.clone() for key, bank in self.banks.items()}
        other.metadata = dict(self.metadata)
        other.seen_frames = self.seen_frames
        return other

    def fingerprint(self):
        return (self.seen_frames, tuple((key, tuple((unit.first_frame, unit.last_frame,
                    unit.keys.data_ptr(), unit.values.data_ptr()) for unit in bank.units))
                    for key, bank in sorted(self.banks.items())))

    def _active_metadata(self, extra=None):
        frames = {unit.last_frame for bank in self.banks.values() for unit in bank.units}
        result = {frame: self.metadata[frame] for frame in frames}
        if extra:
            result.update(extra)
        return result

    def make_cache(self):
        decoder, n = self.prefix.decoder, self.prefix.n
        cache = decoder.new_cache()
        metadata = self._active_metadata()
        maximum = n - 1
        for layer in range(decoder.nl):
            parts = []
            for head in range(decoder.nk):
                bank = self.banks[layer, head]
                prefix_keys = self.prefix.raw_keys[layer][0, head, :n]
                _, prefix_values = self.prefix.cache.layers[layer].get_segment(head)
                raw = torch.cat([prefix_keys] + [unit.keys for unit in bank.units])
                values = torch.cat([prefix_values[:n]] + [unit.values for unit in bank.units])
                uids = list(range(-n, 0)) + [uid for unit in bank.units for uid in unit.uids]
                pos = positions(uids, n, metadata, decoder.kind, decoder.device)
                rotated = decoder.rotate(raw[None, None], pos, layer)[0, 0]
                parts.append((rotated, values, pos))
                maximum = max(maximum, int(pos.max()))
            dest = cache.layers[layer]
            dest.flat_k = torch.cat([item[0] for item in parts]).contiguous()
            dest.flat_v = torch.cat([item[1] for item in parts]).contiguous()
            dest.flat_pos = torch.cat([item[2] for item in parts], dim=1).to(torch.float32).contiguous()
            dest.head_lens = dest.head_lens.new_tensor([len(item[0]) for item in parts])
            dest._refresh()
        return cache, maximum + 1

    @torch.inference_mode()
    def append(self, features, grid, input_frames):
        decoder, n = self.prefix.decoder, self.prefix.n
        if not 1 <= input_frames <= (2 if decoder.kind == 'qwen25vl' else 1):
            raise ValueError('Invalid visual input unit width')
        first = self.seen_frames
        last = first + input_frames - 1
        ids = token_ids(last, *grid)
        if len(ids) != len(features):
            raise ValueError('Visual grid and embeddings disagree')
        cache, _ = self.make_cache()
        meta = self._active_metadata({last: grid})
        pos = positions(ids, n, meta, decoder.kind, decoder.device)
        if int(pos.max()) + 1 >= decoder.position_limit:
            raise RuntimeError('H2SVC active visual positions exceed model limit')
        _, raw_keys, _ = decoder.forward(features[None], pos, cache, capture=True)
        new_entries = {}
        for layer in range(decoder.nl):
            for head in range(decoder.nk):
                _, all_values = cache.layers[layer].get_segment(head)
                new_entries[layer, head] = FrameUnit(first, last,
                    raw_keys[layer][0, head].contiguous(),
                    all_values[-len(ids):].contiguous(), tuple(ids))
        for key, entry in new_entries.items():
            self.banks[key].append(entry)
        self.metadata[last] = grid
        self.metadata = self._active_metadata()
        self.seen_frames += input_frames

    def retained_counts(self):
        return {key: bank.retained_frames for key, bank in self.banks.items()}
