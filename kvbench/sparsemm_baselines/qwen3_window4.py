"""Experimental Qwen3-VL vision topology with 8x8-patch windows and four global blocks.

This changes the pretrained model's receptive field. It is only for an explicitly
labelled model-variant experiment, never an ordinary Qwen3-VL calibration.
"""

from types import MethodType

import torch
import torch.nn.functional as F


WINDOW_PATCHES = 8
FULL_BLOCKS = (6, 13, 20, 26)


def window_index_and_lengths(grid_thw, merge_size=2, window_patches=WINDOW_PATCHES):
    """Return a permutation of merged visual units and patch counts per window."""
    if window_patches % merge_size:
        raise ValueError("The patch window must contain whole merge units")
    width = window_patches // merge_size
    pieces, lengths = [], []
    offset = 0
    for temporal, height, columns in grid_thw.tolist():
        if height % merge_size or columns % merge_size:
            raise ValueError("Vision grid must be divisible by the merge size")
        merged_h, merged_w = height // merge_size, columns // merge_size
        for _ in range(temporal):
            merged = torch.arange(merged_h * merged_w, device=grid_thw.device).reshape(merged_h, merged_w)
            for row in range(0, merged_h, width):
                for col in range(0, merged_w, width):
                    unit = merged[row:row + width, col:col + width].reshape(-1)
                    pieces.append(unit + offset)
                    lengths.append(unit.numel() * merge_size * merge_size)
            offset += merged_h * merged_w
    if not pieces:
        raise ValueError("No visual patches")
    index = torch.cat(pieces)
    if index.numel() != offset or not torch.equal(torch.sort(index).values, torch.arange(offset, device=index.device)):
        raise RuntimeError("Window order is not a permutation")
    return index, lengths


def window4_forward(self, hidden_states, grid_thw, **kwargs):
    hidden_states = self.patch_embed(hidden_states)
    hidden_states = hidden_states + self.fast_pos_embed_interpolate(grid_thw)
    rotary = self.rot_pos_emb(grid_thw)
    seq_len = hidden_states.shape[0]
    merge_unit = self.spatial_merge_unit
    index, window_lengths = window_index_and_lengths(grid_thw, self.spatial_merge_size)
    if index.numel() * merge_unit != seq_len:
        raise ValueError("Window permutation does not cover the input patches")
    inverse = torch.argsort(index)
    hidden_states = hidden_states.reshape(-1, merge_unit, hidden_states.shape[-1])[index].reshape(seq_len, -1)
    rotary = rotary.reshape(-1, merge_unit, rotary.shape[-1])[index].reshape(seq_len, -1)
    embedding = torch.cat((rotary, rotary), dim=-1)
    position_embeddings = (embedding.cos(), embedding.sin())

    grid_lengths = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
    cu_full = F.pad(grid_lengths.cumsum(0, dtype=torch.int32), (1, 0), value=0)
    cu_window = F.pad(torch.tensor(window_lengths, dtype=torch.int32, device=hidden_states.device).cumsum(0),
                      (1, 0), value=0)
    features = []
    for layer, block in enumerate(self.blocks):
        hidden_states = block(hidden_states,
                              cu_seqlens=cu_full if layer in FULL_BLOCKS else cu_window,
                              position_embeddings=position_embeddings, **kwargs)
        if layer in self.deepstack_visual_indexes:
            feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer)](hidden_states)
            features.append(feature[inverse])
    merged = self.merger(hidden_states)
    return merged[inverse], features


def linear_patch_forward(self, hidden_states):
    """Express Qwen3's nonoverlapping full-patch Conv3d as one linear map."""
    conv = self.proj
    kernel = (self.temporal_patch_size, self.patch_size, self.patch_size)
    if conv.kernel_size != kernel or conv.stride != kernel or conv.groups != 1 or conv.padding != (0, 0, 0):
        raise ValueError("Patch Conv3d cannot be flattened into an equivalent linear map")
    patches = hidden_states.reshape(-1, self.in_channels * kernel[0] * kernel[1] * kernel[2])
    return F.linear(patches.to(conv.weight.dtype), conv.weight.reshape(self.embed_dim, -1), conv.bias)


def enable_window4_vision(model):
    visual = model.model.visual
    if len(visual.blocks) != 27 or visual.spatial_merge_size != 2 or visual.patch_size != 16:
        raise ValueError("Window4 topology requires the Qwen3-VL-8B vision configuration")
    if getattr(visual, "_kvbench_window4", False):
        return
    visual.patch_embed.forward = MethodType(linear_patch_forward, visual.patch_embed)
    visual.forward = MethodType(window4_forward, visual)
    visual._kvbench_window4 = True
