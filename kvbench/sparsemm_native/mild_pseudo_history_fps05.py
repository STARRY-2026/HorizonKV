"""Independent recent/history KV on KVBench's native fps=0.5 video input.

Both this arm and the existing SparseMM/DP controls use the same source-frame
sampler, Qwen two-frame video processor, pixel cap, prompt and scorer. Only the
KV policy changes. The fixed pseudo query never reads the real question.
"""

import csv
import hashlib
import time
from pathlib import Path

import torch

from sparsemm_native.decoupled import (
    DecoupledRecent4StreamingQwen, IndependentHistoryCache, VisualUnit, prefill_recent,
    dense_prefix,
)
from sparsemm_native.simplestream_recent_share_pseudo_history_mild500 import (
    PSEUDO_QUERY, _attention_visual_indices, _pseudo_queries_and_keys,
)
from sparsemm_protokv.cache import StreamingCache
from sparsemm_protokv.streaming import StreamingQwen


ALLOCATION_TABLE = (Path(__file__).resolve().parents[1] / 'results' /
                    'ovobench_simplestream_uniform_history_budget500_proposal_20261025' /
                    'per_head_allocation.csv')
ALLOCATION_SHA256 = hashlib.sha256(ALLOCATION_TABLE.read_bytes()).hexdigest()
with ALLOCATION_TABLE.open(newline='') as handle:
    ROWS = {(int(row['layer']), int(row['kv_head'])): row for row in csv.DictReader(handle)}
if (len(ROWS) != 112 or sum(int(row['total_quota']) for row in ROWS.values()) != 56000
        or sum(row['recent_group'] == 'one_frame' for row in ROWS.values()) != 56):
    raise ValueError('Expected 112 mild-budget heads with an exact 56,000-token total')
TOTAL = torch.tensor([[int(ROWS[layer, head]['total_quota']) for head in range(4)]
                      for layer in range(28)], dtype=torch.long)
RECENT = torch.tensor([[int(ROWS[layer, head]['recent_quota']) for head in range(4)]
                       for layer in range(28)], dtype=torch.long)
HISTORY = TOTAL - RECENT


class MildPseudoHistoryCache(IndependentHistoryCache):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, history_weights=HISTORY, **kwargs)
        self.budgets = TOTAL.clone()
        self.recent_budgets = RECENT.clone()
        self.history_budgets = HISTORY.clone()
        self.allocation = dict(method='independent_fixed_pseudo_mild500',
                               source=str(ALLOCATION_TABLE), sha256=ALLOCATION_SHA256,
                               total_visual_slots=56000,
                               history_query=PSEUDO_QUERY)
        self.pseudo_states = None

    def push_unit(self, unit):
        if (unit.features.ndim != 3 or unit.features.shape[0] != 1
                or unit.positions.shape != (3, 1, unit.tokens)
                or not 1 <= unit.real_frames <= 2):
            raise ValueError('Invalid Qwen visual temporal unit')
        if self.units and unit.frame_start <= self.units[-1].frame_start:
            raise ValueError('Temporal unit positions must advance')
        pending = self.units + [unit]
        outgoing = pending[0] if len(pending) > 2 else None
        self.units = pending[-2:]
        self.recent_dense = None
        self.revision += 1
        return outgoing

    @torch.inference_mode()
    def append_layer(self, layer_idx, k, v, raw_k=None, q=None):
        if self.phase != 'video':
            return StreamingCache.append_layer(self, layer_idx, k, v, raw_k=raw_k, q=q)
        if self.pseudo_states is None or layer_idx in self._appended:
            raise RuntimeError('A fixed pseudo probe is required for each historical layer')
        if (not self.units or
                bool((self._positions[0] >= self.units[0].positions[0, 0].min()).any())):
            raise AssertionError('A recent temporal unit leaked into the historical pass')
        layer = self.layers[layer_idx]
        layer.append_chunk(k, v, self._positions)
        before = layer.head_lens.tolist()
        queries, pseudo_keys = self.pseudo_states[layer_idx]
        group = queries.shape[0] // self.num_kv_heads
        selected = []
        for head in range(self.num_kv_heads):
            key = layer.get_segment(head)[0]
            quota = int(self.history_budgets[layer_idx, head])
            visual = _attention_visual_indices(
                key, queries[head * group:(head + 1) * group],
                pseudo_keys[head], self.prefix_tokens, quota, protect=0)
            selected.append(torch.cat((torch.arange(self.prefix_tokens, device=key.device),
                                       visual)))
        layer.prune(selected)
        after = layer.head_lens.tolist()
        if any(n - self.prefix_tokens > int(self.history_budgets[layer_idx, head])
               for head, n in enumerate(after)):
            raise AssertionError('Historical pseudo selector exceeded a per-head quota')
        self._pending_metadata[layer_idx] = dict(
            method='independent_fixed_pseudo_history', layer_idx=layer_idx,
            head_budgets=self.budgets[layer_idx].tolist(),
            history_budgets=self.history_budgets[layer_idx].tolist(),
            before_head_lengths=before, head_lengths=after,
            compressed=any(a < b for a, b in zip(after, before)),
            position_checks=dict(history_quota_respected=True,
                                 no_recent_unit_in_history=True))
        self._appended.add(layer_idx)


def assemble(cache, one, four, one_unit, recent_probes, logical_tokens):
    fused = StreamingCache(cache.config, cache.num_layers, cache.num_kv_heads,
                           cache.head_dim, cache.device, cache.dtype)
    prefix = cache.prefix_tokens
    totals, history_slots, recent_slots = [], 0, 0
    for layer_idx, layer in enumerate(cache.layers):
        k_parts, v_parts, p_parts, lengths, visual_counts = [], [], [], [], []
        queries, pseudo_keys = recent_probes[layer_idx]
        group = queries.shape[0] // cache.num_kv_heads
        for head in range(cache.num_kv_heads):
            row = ROWS[layer_idx, head]
            historical_k, historical_v = layer.get_segment(head)
            historical_p = layer.get_position_segment(head)
            h = int(historical_k.shape[0]) - prefix
            h_quota = int(row['history_quota'])
            if h < 0 or h > h_quota:
                raise AssertionError('Historical head occupancy exceeds allocation')
            chosen = one if row['recent_group'] == 'one_frame' else four
            recent_k, recent_v = chosen[layer_idx]
            recent_positions = torch.cat(
                [unit.positions[:, 0] for unit in
                 ([one_unit] if row['recent_group'] == 'one_frame' else cache.units)], dim=1)
            if recent_k.shape[2] != prefix + recent_positions.shape[1]:
                raise AssertionError('Independent recent prefill has wrong length')
            budget = int(row['recent_quota']) + h_quota - h
            visual = _attention_visual_indices(
                recent_k[0, head], queries[head * group:(head + 1) * group],
                pseudo_keys[head], prefix, budget, protect=32)
            if h and bool((historical_p[0, prefix:] >=
                           recent_positions[0].index_select(0, visual - prefix).min()).any()):
                raise AssertionError('Historical and recent M-RoPE positions overlap')
            rk = recent_k[0, head].index_select(0, visual)
            rv = recent_v[0, head].index_select(0, visual)
            rp = recent_positions.index_select(1, visual - prefix)
            k_parts.append(torch.cat((historical_k, rk)))
            v_parts.append(torch.cat((historical_v, rv)))
            p_parts.append(torch.cat((historical_p, rp), dim=1))
            r = int(visual.numel())
            lengths.append(prefix + h + r)
            visual_counts.append(h + r)
            history_slots += h
            recent_slots += r
        target = fused.layers[layer_idx]
        target.flat_k, target.flat_v = torch.cat(k_parts), torch.cat(v_parts)
        target.flat_pos = torch.cat(p_parts, dim=1)
        target.head_lens = target.head_lens.new_tensor(lengths)
        target._refresh()
        totals.append(visual_counts)
    if history_slots + recent_slots > 56000:
        raise AssertionError('Configured K=500 mean visual budget exceeded')
    fused.prefix_tokens, fused.logical_tokens = prefix, int(logical_tokens)
    fused._initialized, fused._context_done = True, True
    fused.phase = 'video'
    return fused, dict(actual_history_visual_slots=history_slots,
                       actual_recent_visual_slots=recent_slots,
                       actual_visual_slots=history_slots + recent_slots,
                       per_head_visual_slots=totals,
                       configured_visual_slots=56000,
                       allocation_sha256=ALLOCATION_SHA256,
                       history_recent_independent_prefill=True,
                       fixed_pseudo_query=True,
                       history_recent_time_ranges_disjoint=True)


class MildPseudoHistoryQwen(DecoupledRecent4StreamingQwen):
    def clear_cache(self):
        self.kv_cache = MildPseudoHistoryCache(
            self.config, self.num_layers, self.num_kv_heads, self.head_dim,
            self.device, self.dtype)
        self.visual_start_idx = 0
        self._next_position = self._seen_visual_tokens = self.total_processed_frames = 0
        self.last_encoded_frames = self._prefills = self._compressions = 0
        self.conv_history, self.last_metadata = [], []
        self.last_question_metrics = {}
        if not hasattr(self, '_query_budget_audits'):
            self._query_budget_audits = {}

    @torch.inference_mode()
    def encode_video_chunk(self, video_chunk, is_last_chunk=False):
        super().encode_video_chunk(video_chunk, is_last_chunk=is_last_chunk)
        if video_chunk is not None and video_chunk.shape[0]:
            # Store the actual final sampled frame, rather than the final
            # two-frame Qwen temporal patch, for the one-frame head group.
            self.kv_cache.latest_raw_frame = video_chunk[-1:].detach().cpu().clone()

    @torch.inference_mode()
    def _last_frame_unit(self, history):
        frame = history.latest_raw_frame
        if frame.shape[-1] == 3:
            frame = frame.permute(0, 3, 1, 2)
        with self._measure('vision_seconds'):
            processor_kwargs = {}
            if self.video_max_pixels is not None:
                processor_kwargs['size'] = {
                    'shortest_edge': self.processor.video_processor.size['shortest_edge'],
                    'longest_edge': self.video_max_pixels,
                }
            inputs = self.processor(
                text=[''], videos=frame, return_tensors='pt',
                **processor_kwargs).to(self.device, self.dtype)
            grid = tuple(int(value) for value in inputs['video_grid_thw'][0].tolist())
            features = self.model.get_video_features(
                inputs['pixel_values_videos'], inputs['video_grid_thw'])[0].unsqueeze(0)
        if grid[0] != 1:
            raise AssertionError('A single sampled frame must produce one temporal unit')
        vision = self.model_config.vision_config
        h = grid[1] // int(vision.spatial_merge_size)
        w = grid[2] // int(vision.spatial_merge_size)
        if h * w != features.shape[1]:
            raise AssertionError('One-frame visual grid and feature count differ')
        sampled_index = self.total_processed_frames - 1
        temporal = int(sampled_index / self.sample_fps * float(vision.tokens_per_second))
        height = torch.arange(h, device=self.device).view(h, 1).expand(h, w).flatten()
        width = torch.arange(w, device=self.device).view(1, w).expand(h, w).flatten()
        positions = torch.stack((torch.full_like(height, temporal), height, width))
        positions = positions[:, None] + self.visual_start_idx
        return VisualUnit(features.detach(), positions, grid, sampled_index, 1)

    @torch.inference_mode()
    def forward_chunk(self, inputs_embeds, positions, compress=True, **kwargs):
        cache = self.kv_cache
        history_pass = isinstance(cache, MildPseudoHistoryCache) and compress
        if history_pass:
            self._pending_visual_input = dict(
                input_tokens=int(inputs_embeds.shape[1]),
                processor='Qwen2.5-VL native two-frame video pathway',
                video_max_pixels=self.video_max_pixels)
            cache.pseudo_states = _pseudo_queries_and_keys(
                self.model, self.processor.tokenizer, int(positions.max().item()) + 1)
        try:
            result = StreamingQwen.forward_chunk(self, inputs_embeds, positions,
                                                 compress=compress, **kwargs)
        finally:
            if history_pass:
                cache.pseudo_states = None
        if history_pass and compress:
            for layer in cache.last_metadata['layers']:
                self.validation_checks['selector_checks'].append(dict(
                    layer_idx=layer['layer_idx'],
                    head_budgets=layer['head_budgets'], validation_passed=True))
        return result

    @torch.inference_mode()
    def encode_init_prompt(self):
        super().encode_init_prompt()
        self.kv_cache.prefix_dense = dense_prefix(self.kv_cache)

    @torch.inference_mode()
    def question_answering(self, input_text, max_new_tokens=16,
                           temperature=0, repetition_penalty=1.1):
        history = self.kv_cache
        if not history.units or history.prefix_dense is None:
            raise RuntimeError('No recent visual units or shared text prefix')
        start = time.perf_counter()
        one_unit = self._last_frame_unit(history)
        one = prefill_recent(self.language_model, history.prefix_dense, [one_unit])
        four = prefill_recent(self.language_model, history.prefix_dense, history.units)
        # The independently encoded last raw frame can be one sampled-frame
        # timestamp later than the native final two-frame patch's origin.
        # Place probes and question tokens after both recent branches.
        question_position = max(self._next_position, int(one_unit.positions.max()) + 1)
        recent_probes = _pseudo_queries_and_keys(
            self.model, self.processor.tokenizer, question_position)
        fused, audit = assemble(history, one, four, one_unit, recent_probes,
                                self.visual_start_idx + self._seen_visual_tokens)
        # A short video has no outgoing historical unit. Its recent selector
        # still ran, so record that verified selection for the shared runner's
        # audit rather than relying on history-forward records to exist.
        for layer_idx, actual in enumerate(audit['per_head_visual_slots']):
            caps = history.budgets[layer_idx].tolist()
            if any(used > cap for used, cap in zip(actual, caps)):
                raise AssertionError('Recent/history assembly exceeded a head quota')
            self.validation_checks['selector_checks'].append(dict(
                method='fixed_pseudo_recent_history_assembly',
                layer_idx=layer_idx, head_budgets=caps,
                actual_visual_slots=actual, validation_passed=True))
        del one, four
        self.kv_cache = fused
        video_position = self._next_position
        self._next_position = question_position
        try:
            answer = StreamingQwen.question_answering(
                self, input_text, max_new_tokens=max_new_tokens,
                temperature=temperature, repetition_penalty=repetition_penalty)
        finally:
            self.kv_cache = history
            self._next_position = video_position
        audit.update(recent_rebuild_and_assembly_seconds=time.perf_counter() - start,
                     recent_one_frame_heads=56, recent_four_frame_heads=56,
                     question_position=question_position,
                     last_single_frame_position_max=int(one_unit.positions.max()),
                     question_after_both_recent_branches=True,
                     history_units=history.history_units)
        self.last_question_metrics = audit
        query_id = getattr(self, '_active_query_id', None)
        if query_id is not None:
            self._query_budget_audits[query_id] = dict(audit)
        return answer


def load_model(model_path, config, **kwargs):
    from sparsemm_protokv.model_loader import load_pretrained
    from sparsemm_native.assets import asset_metadata
    dtype = kwargs.pop('dtype', 'bfloat16')
    device = kwargs.pop('device', 'cuda')
    model, processor = load_pretrained(model_path, device=device, dtype=dtype)
    return MildPseudoHistoryQwen(model, processor, config,
                                 asset_metadata=asset_metadata(config.sparsemm_score_path or None),
                                 **kwargs)
