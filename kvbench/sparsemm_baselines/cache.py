"""SparseMM head-flattened KV storage, with portable chunk append.

The upstream tiny_api_cuda path appends one token per head. Streaming video
also needs multi-token appends; concatenation here retains the same physical
head-segment layout without a CUDA extension tied to an older torch ABI.
"""

import torch


class RaggedLayer:
    def __init__(self, heads, dim, device, dtype):
        self.flat_k = torch.empty((0, dim), device=device, dtype=dtype)
        self.flat_v = torch.empty_like(self.flat_k)
        self.flat_pos = torch.empty((3, 0), device=device, dtype=torch.float32)
        self.head_lens = torch.zeros(heads, device=device, dtype=torch.int32)
        self._refresh()

    def _refresh(self):
        self.cu_klen = torch.cat((self.head_lens.new_zeros(1), self.head_lens.cumsum(0, dtype=torch.int32)))
        self.max_seqlen_k = int(self.head_lens.max().item())

    def get_segment(self, head):
        start, end = int(self.cu_klen[head]), int(self.cu_klen[head + 1])
        return self.flat_k[start:end], self.flat_v[start:end]

    def get_position_segment(self, head):
        start, end = int(self.cu_klen[head]), int(self.cu_klen[head + 1])
        return self.flat_pos[:, start:end]

    def append_chunk(self, keys, values, pos):
        if keys.ndim != 4 or keys.shape != values.shape or keys.shape[0] != 1:
            raise ValueError("Chunk KV must have matching [1,heads,tokens,dim] shapes")
        if keys.shape[1] != self.head_lens.numel() or pos.shape != (3, keys.shape[2]):
            raise ValueError("KV heads or position shape do not match the cache")
        kp, vp, pp = [], [], []
        for head in range(keys.shape[1]):
            old_k, old_v = self.get_segment(head)
            kp.extend((old_k, keys[0, head]))
            vp.extend((old_v, values[0, head]))
            pp.extend((self.get_position_segment(head), pos.to(self.flat_pos)))
        self.flat_k = torch.cat(kp).contiguous()
        self.flat_v = torch.cat(vp).contiguous()
        self.flat_pos = torch.cat(pp, dim=1).contiguous()
        self.head_lens = self.head_lens + keys.shape[2]
        self._refresh()

    def prune(self, indices):
        if len(indices) != self.head_lens.numel():
            raise ValueError("One keep-index tensor is required per KV head")
        kp, vp, pp, lengths = [], [], [], []
        for head, idx in enumerate(indices):
            if idx.ndim != 1 or idx.dtype != torch.long:
                raise ValueError("Keep indices must be one-dimensional long tensors")
            key, value = self.get_segment(head)
            if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= key.shape[0]):
                raise ValueError("Keep index is outside its head segment")
            if idx.unique().numel() != idx.numel():
                raise ValueError("Repeated keep indices would duplicate KV entries")
            kp.append(key.index_select(0, idx))
            vp.append(value.index_select(0, idx))
            pp.append(self.get_position_segment(head).index_select(1, idx))
            lengths.append(idx.numel())
        self.flat_k = torch.cat(kp).contiguous()
        self.flat_v = torch.cat(vp).contiguous()
        self.flat_pos = torch.cat(pp, dim=1).contiguous()
        self.head_lens = self.head_lens.new_tensor(lengths)
        self._refresh()

    def snapshot(self):
        # Updates replace tensors, so a reference snapshot is sufficient.
        return self.flat_k, self.flat_v, self.flat_pos, self.head_lens

    def restore(self, saved):
        self.flat_k, self.flat_v, self.flat_pos, self.head_lens = saved
        self._refresh()


class RaggedCache:
    def __init__(self, layers, heads, dim, device, dtype):
        self.layers = [RaggedLayer(heads, dim, device, dtype) for _ in range(layers)]

    def snapshot(self):
        return [layer.snapshot() for layer in self.layers]

    def restore(self, saved):
        for layer, state in zip(self.layers, saved):
            layer.restore(state)

    def head_lengths(self):
        return [layer.head_lens.tolist() for layer in self.layers]

    def nbytes(self):
        return sum(t.numel() * t.element_size() for layer in self.layers
                   for t in (layer.flat_k, layer.flat_v))
