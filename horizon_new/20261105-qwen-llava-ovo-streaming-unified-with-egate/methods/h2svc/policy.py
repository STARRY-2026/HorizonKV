"""Paper-budget H2SVC banks over indivisible visual input units.

Qwen's temporal merge produces one KV unit from up to two sampled frames. The
unit is retained or evicted in full; its ``span`` records the number of frames it
represents. LLaVA units each represent one frame. No evicted KV is recoverable.
"""
from dataclasses import dataclass, field
import math

import torch
import torch.nn.functional as F

from .routing import BUDGET_FRAMES, classify_heads


@dataclass(frozen=True)
class FrameUnit:
    """The raw per-head KV and immutable coordinates of one visual input unit."""

    first_frame: int
    last_frame: int
    keys: torch.Tensor
    values: torch.Tensor
    uids: tuple[int, ...]
    _signature: torch.Tensor = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if self.first_frame < 0 or self.last_frame < self.first_frame:
            raise ValueError('Invalid sampled-frame range')
        if self.last_frame - self.first_frame + 1 > 2:
            raise ValueError('A visual unit can contain at most two sampled frames')
        if self.keys.shape != self.values.shape or self.keys.ndim != 2:
            raise ValueError('Per-head keys and values must have matching [tokens, dim] shapes')
        if self.keys.shape[0] != len(self.uids):
            raise ValueError('Visual identifiers do not match KV tokens')
        object.__setattr__(self, '_signature', self.values.float().mean(dim=0))

    @property
    def span(self):
        return self.last_frame - self.first_frame + 1

    @property
    def signature(self):
        # Value vectors are used because RoPE introduces positional bias in K.
        return self._signature


def curvature_importance(left: FrameUnit, center: FrameUnit, right: FrameUnit):
    """STCE angular curvature × magnitude gate × log temporal separation."""
    if not left.last_frame < center.first_frame <= center.last_frame < right.first_frame:
        raise ValueError('STCE needs three chronological, nonoverlapping units')
    incoming = center.signature - left.signature
    outgoing = right.signature - center.signature
    norm_in = torch.linalg.vector_norm(incoming)
    norm_out = torch.linalg.vector_norm(outgoing)
    if not bool(norm_in > 0 and norm_out > 0):
        return 0.0
    cosine = F.cosine_similarity(incoming[None], outgoing[None], dim=-1)[0]
    magnitude = torch.minimum(norm_in, norm_out)
    separation = right.last_frame - left.last_frame
    return float(((1 - cosine).clamp_min(0) * magnitude * math.log(max(2, separation))).item())


def episodic_eviction_index(units):
    """Score all interior units on the GPU and transfer only one index to CPU."""
    if len(units) < 3:
        raise ValueError('Episodic eviction needs an interior unit')
    values = torch.stack([unit.signature for unit in units])
    incoming = values[1:-1] - values[:-2]
    outgoing = values[2:] - values[1:-1]
    length_in = torch.linalg.vector_norm(incoming, dim=-1)
    length_out = torch.linalg.vector_norm(outgoing, dim=-1)
    cosine = F.cosine_similarity(incoming, outgoing, dim=-1)
    separation = torch.tensor(
        [right.last_frame - left.last_frame for left, right in zip(units[:-2], units[2:])],
        device=values.device, dtype=values.dtype).clamp_min(2).log()
    scores = (1 - cosine).clamp_min(0) * torch.minimum(length_in, length_out) * separation
    scores = torch.where((length_in > 0) & (length_out > 0), scores, 0)
    return 1 + int(scores.argmin().item())


class HeadBank:
    """One KV head's bounded, chronological sequence of complete frame units."""

    def __init__(self, kind):
        if kind not in BUDGET_FRAMES:
            raise ValueError('Unknown head route: ' + str(kind))
        self.kind = kind
        self.budget_frames = BUDGET_FRAMES[kind]
        self.units = []

    def clone(self):
        other = HeadBank(self.kind)
        other.units = list(self.units)
        return other

    @property
    def retained_frames(self):
        return sum(unit.span for unit in self.units)

    def append(self, unit: FrameUnit):
        if unit.span > self.budget_frames:
            raise ValueError('Visual unit exceeds its head budget')
        if self.units and self.units[-1].last_frame >= unit.first_frame:
            raise ValueError('Visual units must arrive in chronological order')
        self.units.append(unit)
        while self.retained_frames > self.budget_frames:
            if self.kind == 'episodic' and len(self.units) > 2:
                # Preserve temporal endpoints; only the two neighbors of the
                # removed unit change scores. A 64-frame bank is small enough
                # to score in one bounded GPU operation at each eviction.
                self.units.pop(episodic_eviction_index(self.units))
            else:
                self.units.pop(0)
        if self.retained_frames > self.budget_frames:
            raise RuntimeError('H2SVC head budget exceeded')


