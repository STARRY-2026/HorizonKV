"""Bounded ProtoKV state following Algorithms 1--3 of arXiv:2606.26762.

The bank is vectorized over layer/KV-head groups. Tokens are still absorbed in
stream order, with one EMA and one maintenance step per evicted token. The
authors have not released source; unspecified constants are explicit below.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ProtoConfig:
    budget: int = 1000
    modes: int = 8
    subquantizers: int = 8
    codewords: int = 16
    beam: int = 32
    lambda_sp: float = 0.1
    lambda_idle: float = 0.01
    idle_threshold: int = 120
    alpha: float = 0.05
    beta: float = 0.05
    eta: float = 0.05
    gamma: float = 0.05
    merge_key: float = 0.20
    merge_value: float = 0.25
    warmup_residuals: int = 64  # T_warm is unspecified in the paper.
    pq_smoothing: float = 1.0  # epsilon is unspecified in the paper.
    covariance_ridge: float = 1e-4
    kmeans_iterations: int = 10
    kmeans_batch_size: int = 32

    @property
    def capacity(self):
        return max(1, (self.budget - self.budget // 4) // self.modes)

    @property
    def near(self):
        return self.budget - self.capacity * self.modes


class FullProtoBank:
    """One fixed-capacity bank for every layer and KV head."""

    def __init__(self, groups, dim, device, config=ProtoConfig()):
        if (config.budget <= config.modes or config.capacity < 1 or config.near < 1
                or config.warmup_residuals < config.codewords or config.beam < config.modes
                or config.kmeans_batch_size < 1 or config.kmeans_iterations < 1):
            raise ValueError("Invalid ProtoKV budget or PQ configuration")
        if dim % config.subquantizers:
            raise ValueError("KV head dimension must divide PQ subquantizers")
        self.cfg = config
        self.groups = groups
        self.dim = dim
        self.device = device
        p, k, g, c = groups, config.capacity, config.subquantizers, config.codewords
        self._groups = torch.arange(p, device=device)
        self._slots = torch.arange(k, device=device)[None]
        self._subquantizers = torch.arange(g, device=device)[None]
        self._eye = torch.eye(2, device=device)
        self.center_k = torch.zeros(p, k, dim, device=device, dtype=torch.float32)
        self.center_v = torch.zeros_like(self.center_k)
        self.mass = torch.zeros(p, k, device=device, dtype=torch.float32)
        self.mu = torch.zeros(p, k, 2, device=device, dtype=torch.float32)
        eye = self._eye
        self.cov = eye[None, None].expand(p, k, -1, -1).clone()
        self.inv_cov = (eye / (1 + config.covariance_ridge))[None, None].expand(p, k, -1, -1).clone()
        self.last = torch.zeros(p, k, device=device, dtype=torch.long)
        self.anchor = torch.zeros(3, p, k, device=device, dtype=torch.float32)
        self.active = torch.zeros(p, k, device=device, dtype=torch.bool)
        self.merge_pairs = torch.zeros(p, k, k, device=device, dtype=torch.bool)
        self.ever_used = torch.zeros_like(self.active)
        self.hk = torch.zeros(p, k, g, c, device=device, dtype=torch.int32)
        self.hv = torch.zeros_like(self.hk)
        self.n_res = torch.zeros(p, k, device=device, dtype=torch.long)
        t = config.warmup_residuals
        self.warm_k = torch.zeros(p, t, dim, device=device, dtype=torch.float32)
        self.warm_v = torch.zeros_like(self.warm_k)
        self.warm_slot = torch.full((p, t), -1, device=device, dtype=torch.long)
        self.warm_count = torch.zeros(p, device=device, dtype=torch.long)
        dg = dim // g
        self.code_k = torch.zeros(p, g, c, dg, device=device, dtype=torch.float32)
        self.code_v = torch.zeros_like(self.code_k)
        self.code_ready = torch.zeros(p, device=device, dtype=torch.bool)
        self.total_absorbed = 0

    def _init_slots(self, group, slot, key, value, xy, pos, time):
        if group.numel() == 0:
            return
        self.center_k[group, slot] = key.float()
        self.center_v[group, slot] = value.float()
        self.mass[group, slot] = 1
        self.mu[group, slot] = xy.float()
        self.cov[group, slot] = self._eye
        self.inv_cov[group, slot] = self._eye / (1 + self.cfg.covariance_ridge)
        self.last[group, slot] = time
        self.anchor[:, group, slot] = pos[:, None]
        self.active[group, slot] = True
        self.ever_used[group, slot] = True
        self.hk[group, slot] = 0
        self.hv[group, slot] = 0
        self.n_res[group, slot] = 0
        # A recycled slot must not inherit unquantized residuals from its prior life.
        for n in range(group.numel()):
            stale = self.warm_slot[group[n]] == slot[n]
            self.warm_slot[group[n], stale] = -1
        self._refresh_merge_pairs(group, slot)

    def _refresh_merge_pairs(self, group, slot):
        """Refresh only pairs involving centers that have changed."""
        if group.numel() == 0:
            return
        dk = torch.linalg.vector_norm(self.center_k[group, slot, None] - self.center_k[group], dim=-1)
        dv = torch.linalg.vector_norm(self.center_v[group, slot, None] - self.center_v[group], dim=-1)
        pair = ((dk < self.cfg.merge_key) & (dv < self.cfg.merge_value)
                & self.active[group] & self.active[group, slot, None])
        pair[torch.arange(group.numel(), device=self.device), slot] = False
        other = self._slots
        self.merge_pairs[group, slot] = pair & (slot[:, None] < other)
        self.merge_pairs[group, :, slot] = pair & (other < slot[:, None])

    def _quantize(self, group, slot, rk, rv):
        if group.numel() == 0:
            return
        cfg = self.cfg
        dg = self.dim // cfg.subquantizers
        sub_k = rk.float().reshape(-1, cfg.subquantizers, dg)
        sub_v = rv.float().reshape(-1, cfg.subquantizers, dg)
        code_k = ((sub_k[:, :, None] - self.code_k[group]) ** 2).sum(-1).argmin(-1)
        code_v = ((sub_v[:, :, None] - self.code_v[group]) ** 2).sum(-1).argmin(-1)
        gg = self._subquantizers
        flat_k = (((group[:, None] * cfg.capacity + slot[:, None]) * cfg.subquantizers + gg) * cfg.codewords + code_k).reshape(-1)
        flat_v = (((group[:, None] * cfg.capacity + slot[:, None]) * cfg.subquantizers + gg) * cfg.codewords + code_v).reshape(-1)
        ones = torch.ones_like(flat_k, dtype=self.hk.dtype)
        self.hk.view(-1).scatter_add_(0, flat_k, ones)
        self.hv.view(-1).scatter_add_(0, flat_v, ones)
        flat = group * cfg.capacity + slot
        self.n_res.view(-1).scatter_add_(0, flat, torch.ones_like(flat))

    def _fit_codebooks(self, group):
        cfg = self.cfg
        if group.numel() == 0:
            return
        t, g, c = cfg.warmup_residuals, cfg.subquantizers, cfg.codewords
        dg = self.dim // g
        samples = torch.linspace(0, t - 1, c, device=self.device).long()
        for warm, target in ((self.warm_k, self.code_k), (self.warm_v, self.code_v)):
            x = warm[group].reshape(-1, t, g, dg).permute(0, 2, 1, 3).contiguous()
            centers = x[:, :, samples].clone()
            accumulated = torch.ones(centers.shape[:-1], device=self.device)
            batch_size = min(t, cfg.kmeans_batch_size)
            for iteration in range(cfg.kmeans_iterations):
                chosen = (iteration * batch_size + torch.arange(batch_size, device=self.device)) % t
                batch = x[:, :, chosen]
                distance = torch.cdist(batch.reshape(-1, batch_size, dg), centers.reshape(-1, c, dg)).reshape(-1, g, batch_size, c)
                assignment = distance.argmin(-1)
                hot = F.one_hot(assignment, c).float()
                counts = hot.sum(dim=2)
                sums = torch.einsum("agtc,agtd->agcd", hot, batch)
                centers = (accumulated[..., None] * centers + sums) / (accumulated + counts)[..., None]
                accumulated += counts
            target[group] = centers
        self.code_ready[group] = True
        warm_group = group[:, None].expand(-1, t).reshape(-1)
        warm_slot = self.warm_slot[group].reshape(-1)
        valid = warm_slot >= 0
        self._quantize(
            warm_group[valid], warm_slot[valid],
            self.warm_k[group].reshape(-1, self.dim)[valid],
            self.warm_v[group].reshape(-1, self.dim)[valid],
        )

    def _record_residual(self, group, slot, rk, rv):
        if group.numel() == 0:
            return
        ready = self.code_ready[group]
        self._quantize(group[ready], slot[ready], rk[ready], rv[ready])
        pending = ~ready
        if pending.any():
            gp, sp = group[pending], slot[pending]
            index = self.warm_count[gp]
            under = index < self.cfg.warmup_residuals
            gp, sp, index = gp[under], sp[under], index[under]
            self.warm_k[gp, index] = rk[pending][under]
            self.warm_v[gp, index] = rv[pending][under]
            self.warm_slot[gp, index] = sp
            self.warm_count[gp] += 1
            now_ready = gp[self.warm_count[gp] == self.cfg.warmup_residuals]
            self._fit_codebooks(now_ready)

    def _apply_merge(self, group, left, right):
        """Apply the same merge update for global and changed-slot searches."""
        cfg = self.cfg
        m1, m2 = self.mass[group, left].clone(), self.mass[group, right].clone()
        total = (m1 + m2).clamp_min(1)
        for state in (self.center_k, self.center_v, self.mu):
            state[group, left] = (m1[:, None] * state[group, left] + m2[:, None] * state[group, right]) / total[:, None]
        self.cov[group, left] = (m1[:, None, None] * self.cov[group, left] + m2[:, None, None] * self.cov[group, right]) / total[:, None, None]
        self.inv_cov[group, left] = torch.linalg.inv(self.cov[group, left] + cfg.covariance_ridge * self._eye)
        newer = self.last[group, right] > self.last[group, left]
        self.anchor[:, group, left] = torch.where(newer[None, :], self.anchor[:, group, right], self.anchor[:, group, left])
        self.last[group, left] = torch.maximum(self.last[group, left], self.last[group, right])
        self.mass[group, left] = total
        self.hk[group, left] += self.hk[group, right]
        self.hv[group, left] += self.hv[group, right]
        self.n_res[group, left] += self.n_res[group, right]
        self.active[group, right] = False
        self.merge_pairs[group, right] = False
        self.merge_pairs[group, :, right] = False
        self._refresh_merge_pairs(group, left)
        for n in range(group.numel()):
            self.warm_slot[group[n], self.warm_slot[group[n]] == right[n]] = left[n]

    def _merge_once(self):
        cfg = self.cfg
        if cfg.capacity < 2:
            return False
        pairs = self.merge_pairs & self.active[:, :, None] & self.active[:, None, :]
        has = pairs.flatten(1).any(-1)
        if not has.any():
            return False
        group = has.nonzero(as_tuple=True)[0]
        flat = pairs[group].flatten(1).int().argmax(-1)
        left, right = flat // cfg.capacity, flat % cfg.capacity
        self._apply_merge(group, left, right)
        return True

    def _maintenance(self, time, recent_k, recent_v, recent_xy, recent_pos):
        if time > self.cfg.idle_threshold:
            stale = self.active & ((time - self.last) > self.cfg.idle_threshold)
            self.mass[stale] = torch.floor((1 - self.cfg.gamma) * self.mass[stale])
            self.active &= self.mass > 0
        merged = False
        while self._merge_once():
            merged = True
        if not merged and time <= self.cfg.idle_threshold:
            return
        recycled = self.ever_used & ~self.active
        group, slot = recycled.nonzero(as_tuple=True)
        if group.numel():
            self._init_slots(group, slot, recent_k[group], recent_v[group], recent_xy, recent_pos, time)

    @torch.no_grad()
    def absorb(self, raw_k, value, xy, pos, outgoing, time, recent_start=None):
        """Absorb the oldest `outgoing` visual tokens, in their original order.

        `raw_k`, `value`: [layer*KV-head, exact_visual_length, head_dim].
        `xy`: [exact_visual_length,2]; `pos`: [3,exact_visual_length].
        """
        p, n, d = raw_k.shape
        if (p, d) != (self.groups, self.dim) or value.shape != raw_k.shape:
            raise ValueError("KV shape differs from bank configuration")
        if xy.shape != (n, 2) or pos.shape != (3, n) or not 0 <= outgoing <= n:
            raise ValueError("Spatial or position metadata mismatch")
        all_group = self._groups
        for i in range(outgoing):
            key = raw_k[:, i].float()
            val = value[:, i].float()
            # Cost in Algorithm 1: negative cosine, regularized Mahalanobis,
            # and frame-time staleness. Inactive slots are initialized first.
            filled = self.total_absorbed >= self.cfg.capacity
            if not filled:
                free = ~self.active
                has_free = free.any(-1)
                free_slot = free.int().argmax(-1)
                all_free = bool(has_free.all())
            if filled or not all_free:
                cosine = F.cosine_similarity(key[:, None], self.center_k, dim=-1)
                delta = xy[i][None, None] - self.mu
                mahal = torch.sqrt(torch.einsum("pki,pkij,pkj->pk", delta, self.inv_cov, delta).clamp_min(0))
                idle = ((time - self.last) > self.cfg.idle_threshold).float()
                cost = -cosine + self.cfg.lambda_sp * mahal + self.cfg.lambda_idle * idle
                cost = cost.masked_fill(~self.active, float("inf"))
                chosen = cost.argmin(-1)
            if filled:
                update, si = all_group, chosen
            else:
                slot = free_slot if all_free else torch.where(has_free, free_slot, chosen)
                existing = self.active[all_group, slot]
                fresh = all_group[~existing]
                self._init_slots(fresh, slot[~existing], key[~existing], val[~existing], xy[i], pos[:, i], time)
                update = all_group[existing]
                si = slot[existing]
            if update.numel():
                self.center_k[update, si] = (1 - self.cfg.alpha) * self.center_k[update, si] + self.cfg.alpha * key[update]
                self.center_v[update, si] = (1 - self.cfg.beta) * self.center_v[update, si] + self.cfg.beta * val[update]
                self._refresh_merge_pairs(update, si)
                self.mass[update, si] += 1
                self.last[update, si] = time
                self.anchor[:, update, si] = pos[:, i, None]
                self.mu[update, si] = (1 - self.cfg.eta) * self.mu[update, si] + self.cfg.eta * xy[i]
                spatial_delta = xy[i][None] - self.mu[update, si]
                outer = spatial_delta[:, :, None] * spatial_delta[:, None, :]
                self.cov[update, si] = (1 - self.cfg.eta) * self.cov[update, si] + self.cfg.eta * outer
                self.inv_cov[update, si] = torch.linalg.inv(self.cov[update, si] + self.cfg.covariance_ridge * self._eye)
                rk = key[update] - self.center_k[update, si]
                rv = val[update] - self.center_v[update, si]
                if self.total_absorbed >= self.cfg.capacity + self.cfg.warmup_residuals:
                    self._quantize(update, si, rk, rv)
                else:
                    self._record_residual(update, si, rk, rv)
            recent = min(i + (self.cfg.near if recent_start is None else recent_start), n - 1)
            self._maintenance(time, raw_k[:, recent], value[:, recent], xy[recent], pos[:, recent])
            self.total_absorbed += 1

    def _decode(self, hist, codebook):
        cfg = self.cfg
        p, k, g, c = hist.shape
        logp = (hist.float() + cfg.pq_smoothing).log()
        logp -= (hist.sum(-1, keepdim=True).float() + c * cfg.pq_smoothing).log()
        scores = torch.zeros(p, k, 1, device=self.device)
        choices = torch.empty(p, k, 1, 0, device=self.device, dtype=torch.long)
        for sub in range(g):
            candidate = scores[..., None] + logp[:, :, sub, None, :]
            top, indices = candidate.flatten(2).topk(min(cfg.beam, candidate.shape[2] * c), dim=2)
            prev, new = indices // c, indices % c
            choices = torch.cat((choices.gather(2, prev[..., None].expand(-1, -1, -1, choices.shape[-1])), new[..., None]), dim=-1)
            scores = top
        choices = choices[:, :, :cfg.modes]
        parts = []
        for sub in range(g):
            cb = codebook[:, sub]
            idx = choices[..., sub].reshape(p, -1)
            parts.append(cb.gather(1, idx[..., None].expand(-1, -1, cb.shape[-1])).reshape(p, k, cfg.modes, -1))
        return torch.cat(parts, dim=-1).reshape(p, k * cfg.modes, self.dim)

    @torch.no_grad()
    def pseudo(self):
        cfg = self.cfg
        base_k = self.center_k.repeat_interleave(cfg.modes, dim=1)
        base_v = self.center_v.repeat_interleave(cfg.modes, dim=1)
        rk = self._decode(self.hk, self.code_k)
        rv = self._decode(self.hv, self.code_v)
        has_res = (self.code_ready[:, None] & (self.n_res > 0)).repeat_interleave(cfg.modes, dim=1)
        pk = base_k + torch.where(has_res[..., None], rk, 0)
        pv = base_v + torch.where(has_res[..., None], rv, 0)
        active = self.active.repeat_interleave(cfg.modes, dim=1)
        pk = torch.where(active[..., None], pk, 0)
        pv = torch.where(active[..., None], pv, 0)
        anchor = self.anchor.repeat_interleave(cfg.modes, dim=2)
        return pk, pv, anchor, active, self.mass.repeat_interleave(cfg.modes, dim=1)
