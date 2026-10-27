"""Question-blind independent history/recent KV with the mild K=500 allocation."""

import collections
import contextvars
import csv
import hashlib
import sys
import time
import weakref
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import DynamicCache
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb

HERMES_ROOT = Path(__file__).resolve().parents[2] / 'hermes'
RAGGED_SOURCE = HERMES_ROOT / 'head_analysis/hermes_kv_head_ragged.py'
RAGGED_SHA256 = hashlib.sha256(RAGGED_SOURCE.read_bytes()).hexdigest()
_question = contextvars.ContextVar('physical_recent_share_question', default=None)
_history_context = contextvars.ContextVar('pseudo_history_candidates', default=None)
ALLOCATION_TABLE = (Path(__file__).resolve().parents[1] / 'results' /
                    'ovobench_simplestream_uniform_history_budget500_proposal_20261025' /
                    'per_head_allocation.csv')
ALLOCATION_SHA256 = hashlib.sha256(ALLOCATION_TABLE.read_bytes()).hexdigest()
with ALLOCATION_TABLE.open(newline='') as _handle:
    ALLOCATION = {(int(row['layer']), int(row['kv_head'])): row
                  for row in csv.DictReader(_handle)}
if len(ALLOCATION) != 112 or sum(int(row['total_quota']) for row in ALLOCATION.values()) != 56000:
    raise ValueError('Expected 112 heads and 56,000 configured visual KV slots')
if (sum(row['recent_group'] == 'one_frame' for row in ALLOCATION.values()) != 56 or
        sum(row['recent_group'] == 'four_frames' for row in ALLOCATION.values()) != 56 or
        any(int(row['recent_quota']) + int(row['history_quota']) != int(row['total_quota'])
            for row in ALLOCATION.values())):
    raise ValueError('Mild allocation table has unexpected groups or quotas')
ACTIVE_ALLOCATION = ALLOCATION
ACTIVE_ARM = 'mild_history'
PROTECT_WINDOW = 32
POOL_KERNEL = 5
PSEUDO_QUERY = (
    'Describe the current scene in detail, focusing on specific objects, '
    'fine-grained actions, and spatial relationships. Summarize the video '
    'narrative, identifying main characters, key events, timeline changes, '
    'and the overall theme.'
)


def _pseudo_queries_and_keys(model, tokenizer, position):
    """HERMES-style projected probe, without reading the actual question."""
    ids = tokenizer(PSEUDO_QUERY, return_tensors='pt').input_ids.to(
        model.device)
    length = ids.shape[1]
    embeddings = model.get_input_embeddings()(ids)
    positions = torch.arange(position, position + length, device=ids.device)
    positions = positions.view(1, 1, length).expand(3, 1, length)
    cos, sin = model.language_model.rotary_emb(embeddings, positions)
    states = []
    for decoder in model.language_model.layers:
        attn = decoder.self_attn
        hidden = decoder.input_layernorm(embeddings)
        q = attn.q_proj(hidden).view(1, length, -1, attn.head_dim).transpose(1, 2)
        k = attn.k_proj(hidden).view(1, length, -1, attn.head_dim).transpose(1, 2)
        q, k = apply_multimodal_rotary_pos_emb(
            q, k, cos, sin, attn.rope_scaling['mrope_section'])
        states.append((q[0], k[0]))
    return states


def _attention_visual_indices(key, queries, pseudo_keys, visual_start, budget,
                              protect=PROTECT_WINDOW):
    """FP32 pseudo-query QK mass, GQA mean, avgpool-5, top-k + last-32."""
    n = int(key.shape[0])
    visual = n - visual_start
    if visual <= budget:
        return torch.arange(visual_start, n, device=key.device)
    recent = min(protect, visual, budget)
    candidate_end = n - recent
    capacity = min(budget - recent, candidate_end - visual_start)
    q = queries.float()
    all_keys = torch.cat((key, pseudo_keys), dim=0).float()
    logits = torch.matmul(q, all_keys.T) * (key.shape[-1] ** -0.5)
    probe_length = pseudo_keys.shape[0]
    q_position = torch.arange(n, n + probe_length, device=key.device)
    k_position = torch.arange(n + probe_length, device=key.device)
    logits.masked_fill_(k_position[None, None, :] > q_position[None, :, None],
                        torch.finfo(logits.dtype).min)
    mass = torch.softmax(logits, dim=-1).mean(dim=(0, 1))
    score = mass[visual_start:candidate_end]
    score = F.avg_pool1d(score[None, None], kernel_size=POOL_KERNEL,
                         padding=POOL_KERNEL//2, stride=1)[0, 0]
    if not torch.isfinite(score).all():
        raise FloatingPointError('Non-finite pseudo-query attention score')
    ranked = torch.topk(score, capacity).indices.sort().values + visual_start
    protected = torch.arange(candidate_end, n, device=key.device)
    return torch.cat((ranked, protected))


def single_frame_position_indices(visual_end, visual_start, last_frame_start):
    """Indices for text prefix plus the final image only."""
    if not 0 <= visual_start <= last_frame_start < visual_end:
        raise ValueError('Invalid image token boundaries')
    return list(range(visual_start)) + list(range(last_frame_start, visual_end))


def _ragged_helpers():
    if str(HERMES_ROOT) not in sys.path:
        sys.path.insert(0, str(HERMES_ROOT))
    from head_analysis.hermes_kv_head_ragged import (
        HeadRaggedLayerCache, HeadRaggedCache, _ragged_language_decode_step,
    )
    return HeadRaggedLayerCache, HeadRaggedCache, _ragged_language_decode_step


def _combine_prefix_caches(history_ragged, full_dense, one_dense, Layer,
                           visual_start, probes):
    """Join independent older and recent image passes without duplicate frames."""
    layers = []
    actual_history = 0
    actual_recent = 0
    for layer_idx, ((full_k, full_v), (one_k, one_v)) in enumerate(zip(full_dense, one_dense)):
        if full_k.shape[0:2] != (1, 4) or one_k.shape[0:2] != (1, 4):
            raise RuntimeError('Unexpected dense K/V dimensions')
        historical = history_ragged.layers[layer_idx] if history_ragged is not None else None
        historical_offsets = ([int(x) for x in historical.cu_klen.tolist()]
                              if historical is not None else None)
        k_parts = []
        v_parts = []
        lengths = []
        for head in range(4):
            row = ACTIVE_ALLOCATION[layer_idx, head]
            use_one = row['recent_group'] == 'one_frame'
            chosen_k, chosen_v = (one_k, one_v) if use_one else (full_k, full_v)
            queries, pseudo_keys = probes[layer_idx]
            groups = queries.shape[0] // 4
            query = queries[head*groups:(head+1)*groups]
            history_quota = int(row['history_quota'])
            history_count = (historical_offsets[head + 1] - historical_offsets[head]
                             - visual_start) if historical is not None else 0
            if history_count < 0 or history_count > history_quota:
                raise RuntimeError('Historical head quota exceeded')
            recent_budget = int(row['recent_quota']) + history_quota - history_count
            visual = _attention_visual_indices(
                chosen_k[0, head], query, pseudo_keys[head],
                visual_start, recent_budget)
            prefix = torch.arange(visual_start, device=chosen_k.device)
            selected = torch.cat((prefix, visual))
            key = chosen_k[0, head].index_select(0, selected)
            value = chosen_v[0, head].index_select(0, selected)
            if history_count:
                start, end = historical_offsets[head], historical_offsets[head + 1]
                key = torch.cat((key[:visual_start],
                                 historical.flat_k[start + visual_start:end],
                                 key[visual_start:]), dim=0)
                value = torch.cat((value[:visual_start],
                                   historical.flat_v[start + visual_start:end],
                                   value[visual_start:]), dim=0)
            k_parts.append(key)
            v_parts.append(value)
            lengths.append(int(key.shape[0]))
            actual_history += history_count
            actual_recent += int(visual.numel())
        layers.append(Layer(torch.cat(k_parts, dim=0), torch.cat(v_parts, dim=0),
                            torch.tensor(lengths, device=full_k.device, dtype=torch.int32)))
    return layers, actual_history, actual_recent


def _prompt_ids(qa, visual_tokens, question):
    tokenizer = qa.processor.tokenizer
    return ([qa._im_start_id] + tokenizer.encode('user\n', add_special_tokens=False)
            + [qa._vision_start_id] + [qa.image_token_id] * visual_tokens
            + [qa._vision_end_id] + tokenizer.encode('\n', add_special_tokens=False)
            + tokenizer.encode(question, add_special_tokens=False)
            + [qa._im_end_id] + tokenizer.encode('\n', add_special_tokens=False)
            + [qa._im_start_id] + tokenizer.encode('assistant\n', add_special_tokens=False))


def _apply_repetition_penalty(logits, ids, factor):
    if factor == 1.0:
        return logits
    indices = torch.as_tensor(sorted(set(ids)), device=logits.device, dtype=torch.long)
    selected = logits.index_select(0, indices)
    selected = torch.where(selected < 0, selected * factor, selected / factor)
    return logits.scatter(0, indices, selected)


def _prefill_question_suffix(model, embeds, position_ids, ragged):
    """Prefill question tokens against physically pruned per-head visual K/V."""
    from flash_attn import flash_attn_varlen_func

    hidden = embeds
    q_len = hidden.shape[1]
    if hidden.shape[0] != 1 or q_len < 1:
        raise ValueError('Expected a nonempty single-question suffix')
    for layer_idx, decoder_layer in enumerate(model.language_model.layers):
        residual = hidden
        normed = decoder_layer.input_layernorm(hidden)
        attn = decoder_layer.self_attn
        n_kv = attn.num_key_value_heads
        group = attn.num_key_value_groups
        dim = attn.head_dim
        q = attn.q_proj(normed).view(1, q_len, n_kv * group, dim).transpose(1, 2)
        k = attn.k_proj(normed).view(1, q_len, n_kv, dim).transpose(1, 2)
        v = attn.v_proj(normed).view(1, q_len, n_kv, dim).transpose(1, 2)
        cos, sin = model.language_model.rotary_emb(normed, position_ids)
        q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, attn.rope_scaling['mrope_section'])
        layer_cache = ragged.layers[layer_idx]
        layer_cache.append_chunk(k, v)
        query = q.view(1, n_kv, group, q_len, dim)[0].permute(0, 2, 1, 3).reshape(
            n_kv * q_len, group, dim).contiguous()
        cu_q = torch.arange(0, (n_kv + 1) * q_len, q_len, dtype=torch.int32, device=q.device)
        attended = flash_attn_varlen_func(
            query, layer_cache.flat_k.view(-1, 1, dim).contiguous(),
            layer_cache.flat_v.view(-1, 1, dim).contiguous(),
            cu_q, layer_cache.cu_klen, max_seqlen_q=q_len,
            max_seqlen_k=layer_cache.max_seqlen_k, causal=True)
        attended = attended.view(n_kv, q_len, group, dim).permute(1, 0, 2, 3).reshape(
            1, q_len, n_kv * group * dim)
        hidden = residual + attn.o_proj(attended.contiguous())
        residual = hidden
        hidden = decoder_layer.post_attention_layernorm(hidden)
        hidden = residual + decoder_layer.mlp(hidden)
    return model.language_model.norm(hidden)


def _prune_history(ragged, visual_start, probes):
    """Evict each old-frame head using only the fixed pseudo query."""
    for layer_idx, layer in enumerate(ragged.layers):
        offsets = [int(x) for x in layer.cu_klen.tolist()]
        keys, values, lengths = [], [], []
        queries, pseudo_keys = probes[layer_idx]
        group = queries.shape[0] // 4
        for head in range(4):
            start, end = offsets[head], offsets[head + 1]
            key = layer.flat_k[start:end]
            value = layer.flat_v[start:end]
            quota = int(ACTIVE_ALLOCATION[layer_idx, head]['history_quota'])
            if key.shape[0] < visual_start:
                raise RuntimeError('Historical text prefix was evicted')
            if quota == 0:
                selected = torch.arange(visual_start, device=key.device)
            else:
                old = _attention_visual_indices(
                    key, queries[head * group:(head + 1) * group],
                    pseudo_keys[head], visual_start, quota, protect=0)
                selected = torch.cat((torch.arange(visual_start, device=key.device), old))
            keys.append(key.index_select(0, selected))
            values.append(value.index_select(0, selected))
            lengths.append(int(selected.numel()))
        layer.flat_k = torch.cat(keys).contiguous()
        layer.flat_v = torch.cat(values).contiguous()
        layer.head_lens = torch.tensor(lengths, device=layer.flat_k.device, dtype=torch.int32)
        layer._refresh_cu_klen()


def install(output, *, validation_keep_all=False, recent_only_control=False):
    """Use released Qwen image processing with every older fps=1 frame."""
    global ACTIVE_ALLOCATION, ACTIVE_ARM
    if recent_only_control:
        ACTIVE_ALLOCATION = {
            key: {**row, 'recent_quota': '500', 'history_quota': '0', 'total_quota': '500'}
            for key, row in ALLOCATION.items()}
        ACTIVE_ARM = 'recent_only_same_timeline'
    else:
        ACTIVE_ALLOCATION = ALLOCATION
        ACTIVE_ARM = 'mild_history'
    from lib import recent_window_eval_qwen25 as upstream
    from lib.qwen_exact_recent_decoder import (
        _build_sampling_plan, _decode_indices, _resize_like_fetch_video,
    )
    from PIL import Image
    from sparsemm_native.supervise_simplestream_upstream import append

    original_question = upstream.RecentWindowQAModel.generate_with_cached_vision
    original_generate = upstream.RecentWindowQAModel._generate_from_model_inputs
    original_encode = upstream.RecentWindowQAModel.encode_vision
    low_heads = (set() if validation_keep_all else
                 {head for head, row in ACTIVE_ALLOCATION.items()
                  if row['recent_group'] == 'one_frame'})
    Layer, Cache, decode_step = _ragged_helpers()
    vision_cache = collections.OrderedDict()
    vision_cache_bytes = 0
    cache_limit_bytes = 2 * 1024 ** 3
    cache_stats = collections.Counter()

    @torch.inference_mode()
    def encode_reusable(qa, images):
        """Reuse only bit-identical resized frames under the same model/processor."""
        nonlocal vision_cache_bytes
        digest = hashlib.sha256()
        for frame in images:
            digest.update(str((frame.mode, frame.size)).encode())
            digest.update(frame.tobytes())
        key = digest.hexdigest()
        if key in vision_cache:
            embeddings, grid = vision_cache.pop(key)
            vision_cache[key] = (embeddings, grid)
            cache_stats['hits'] += 1
            return embeddings.to(qa.model.device), grid.to(qa.model.device)
        embeddings, grid = original_encode(qa, images)
        cpu_embeds, cpu_grid = embeddings.detach().cpu(), grid.detach().cpu()
        size = cpu_embeds.numel() * cpu_embeds.element_size() + cpu_grid.numel() * cpu_grid.element_size()
        if size <= cache_limit_bytes:
            while vision_cache and vision_cache_bytes + size > cache_limit_bytes:
                _, (old_embeds, old_grid) = vision_cache.popitem(last=False)
                vision_cache_bytes -= (old_embeds.numel() * old_embeds.element_size()
                                       + old_grid.numel() * old_grid.element_size())
                cache_stats['evictions'] += 1
            vision_cache[key] = (cpu_embeds, cpu_grid)
            vision_cache_bytes += size
        cache_stats['misses'] += 1
        return embeddings, grid

    def question(self, cached_embeds, cached_grid_thw, prompt):
        counts = (cached_grid_thw.prod(dim=-1) // self.merge_size ** 2).tolist()
        history_context = _history_context.get()
        if history_context is None:
            raise RuntimeError('History/recent frame partition is missing')
        history_frames = history_context['history_frames']
        recent_counts = counts[history_frames:]
        if not 1 <= len(recent_counts) <= 4 or sum(counts) != int(cached_embeds.shape[0]):
            raise ValueError('Unexpected image-token layout')
        old_start = 1 + len(self.processor.tokenizer.encode('user\n', add_special_tokens=False)) + 1
        last_frame_start = old_start + sum(counts[:-1])
        ids = _prompt_ids(self, sum(counts), prompt)
        token = _question.set(dict(counts=counts, old_start=old_start,
                                   query_id=history_context['query_id'],
                                   history_frames=history_frames,
                                   history_visual_end=old_start + sum(counts[:history_frames]),
                                   recent_counts=recent_counts,
                                   history_sample_indices=history_context['history_sample_indices'],
                                   recent_sample_indices=history_context['recent_sample_indices'],
                                   vision_cache_hits=history_context['vision_cache_hits'],
                                   vision_cache_misses=history_context['vision_cache_misses'],
                                   recent_frames_sha256=history_context['recent_frames_sha256'],
                                   last_frame_start=last_frame_start,
                                   visual_end=old_start + sum(counts), ids=ids,
                                   prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest()))
        try:
            return original_question(self, cached_embeds, cached_grid_thw, prompt)
        finally:
            _question.reset(token)

    @torch.inference_mode()
    def generate(self, prompt_length, **kwargs):
        layout = _question.get()
        if layout is None:
            return original_generate(self, prompt_length, **kwargs)
        if len(layout['ids']) != prompt_length:
            raise RuntimeError('Prompt token construction differs from upstream')
        model = self.model
        full_dense = DynamicCache()
        started = time.perf_counter()
        visual_end = layout['visual_end']
        history_end = layout['history_visual_end']
        chronological_positions = True
        if history_end > layout['old_start']:
            positions = kwargs['position_ids']
            older = positions[:, :, layout['old_start']:history_end]
            recent = positions[:, :, history_end:visual_end]
            chronological_positions = bool((older.amax(dim=-1) < recent.amin(dim=-1)).all())
            if not chronological_positions:
                raise RuntimeError('History and recent M-RoPE positions overlap')
        full_indices = torch.as_tensor(
            list(range(layout['old_start'])) + list(range(history_end, visual_end)),
            device=kwargs['inputs_embeds'].device, dtype=torch.long)
        full_outputs = model(inputs_embeds=kwargs['inputs_embeds'].index_select(1, full_indices),
                             attention_mask=kwargs['attention_mask'].index_select(1, full_indices),
                             position_ids=kwargs['position_ids'].index_select(2, full_indices),
                             past_key_values=full_dense, use_cache=True, return_dict=True)
        del full_outputs
        # Match the completed recent-only K=500 arm for the recent selector.
        recent_probes = _pseudo_queries_and_keys(
            model, self.processor.tokenizer, len(full_indices))
        history_probes = _pseudo_queries_and_keys(
            model, self.processor.tokenizer,
            int(kwargs['position_ids'][:, :, visual_end - 1].max().item()) + 1)
        if len(full_dense) != len(model.language_model.layers):
            raise RuntimeError('Incomplete four-frame prefill cache')
        history_ragged = None
        if layout['history_frames'] and not recent_only_control:
            history_prefix = DynamicCache()
            prefix_result = model(
                inputs_embeds=kwargs['inputs_embeds'][:, :layout['old_start']],
                attention_mask=kwargs['attention_mask'][:, :layout['old_start']],
                position_ids=kwargs['position_ids'][:, :, :layout['old_start']],
                past_key_values=history_prefix, use_cache=True, return_dict=True)
            del prefix_result
            if len(history_prefix) != len(full_dense):
                raise RuntimeError('Incomplete independent historical text prefix')
            history_ragged = Cache([
                Layer(torch.cat([history_prefix[i][0][0, h] for h in range(4)]),
                      torch.cat([history_prefix[i][1][0, h] for h in range(4)]),
                      torch.full((4,), layout['old_start'], device=model.device,
                                 dtype=torch.int32))
                for i in range(len(history_prefix))])
            del history_prefix
            offset = layout['old_start']
            for count in layout['counts'][:layout['history_frames']]:
                next_offset = offset + count
                _prefill_question_suffix(
                    model, kwargs['inputs_embeds'][:, offset:next_offset],
                    kwargs['position_ids'][:, :, offset:next_offset], history_ragged)
                _prune_history(history_ragged, layout['old_start'], history_probes)
                offset = next_offset
            if offset != history_end:
                raise RuntimeError('Historical frame/token partition disagrees')
        one_indices = single_frame_position_indices(
            visual_end, layout['old_start'], layout['last_frame_start'])
        independent_pass = bool(low_heads and len(layout['recent_counts']) > 1)
        if independent_pass:
            index = torch.as_tensor(one_indices, device=kwargs['inputs_embeds'].device,
                                    dtype=torch.long)
            one_dense = DynamicCache()
            one_outputs = model(
                inputs_embeds=kwargs['inputs_embeds'].index_select(1, index),
                attention_mask=kwargs['attention_mask'].index_select(1, index),
                position_ids=kwargs['position_ids'].index_select(2, index),
                past_key_values=one_dense, use_cache=True, return_dict=True)
            del one_outputs
            if len(one_dense) != len(full_dense):
                raise RuntimeError('Incomplete independent one-frame prefill cache')
        else:
            one_dense = full_dense
        if any(full_dense[i][0].shape[2] != len(full_indices) or
               one_dense[i][0].shape[2] != len(one_indices)
               for i in range(len(full_dense))):
            raise RuntimeError('Prefix K/V lengths differ from the requested frames')
        full_key_ref = weakref.ref(full_dense[-1][0])
        one_key_ref = weakref.ref(one_dense[-1][0])
        layers, history_slots, recent_slots = _combine_prefix_caches(
            history_ragged, full_dense, one_dense, Layer,
            layout['old_start'], recent_probes)
        ragged = Cache(layers)
        del full_dense, one_dense, history_ragged
        if full_key_ref() is not None or one_key_ref() is not None:
            raise RuntimeError('Dense visual K/V still live after branch combination')
        hidden = _prefill_question_suffix(
            model, kwargs['inputs_embeds'][:, visual_end:],
            kwargs['position_ids'][:, :, visual_end:], ragged)
        logits = model.lm_head(hidden)[0, -1]
        historical_tokens = sum(layout['counts'][:layout['history_frames']])
        lens = [layer.head_lens.tolist() for layer in ragged.layers]
        suffix_tokens = prompt_length - visual_end
        for layer_idx, row in enumerate(lens):
            for head, length in enumerate(row):
                allocation = ACTIVE_ALLOCATION[layer_idx, head]
                quota = int(allocation['history_quota'])
                historical = min(historical_tokens, quota)
                available = (layout['counts'][-1] if allocation['recent_group'] == 'one_frame'
                             else sum(layout['recent_counts']))
                recent = min(available, int(allocation['recent_quota']) + quota - historical)
                if length != layout['old_start'] + historical + recent + suffix_tokens:
                    raise RuntimeError('Physical head K/V length differs from allocation')
        physical_slots = sum(sum(row) for row in lens)
        dense_slots = len(lens) * 4 * prompt_length
        physical_bytes = sum(layer.flat_k.numel() * layer.flat_k.element_size()
                             + layer.flat_v.numel() * layer.flat_v.element_size()
                             for layer in ragged.layers)

        config = model.generation_config
        penalty = float(config.repetition_penalty or 1.0)
        eos = config.eos_token_id
        stop_ids = set(eos if isinstance(eos, list) else [eos])
        generated = []
        position = int(kwargs['position_ids'].max().item()) + 1
        for step in range(self.max_new_tokens):
            current = _apply_repetition_penalty(logits, layout['ids'] + generated, penalty)
            next_token = int(current.argmax().item())
            if step == 0:
                self._last_ttft_seconds = time.perf_counter() - started
            generated.append(next_token)
            if next_token in stop_ids or step + 1 == self.max_new_tokens:
                break
            positions = torch.full((3, 1, 1), position, dtype=torch.long,
                                   device=kwargs['inputs_embeds'].device)
            hidden = decode_step(model, torch.tensor([[next_token]],
                                 device=positions.device), ragged,
                                 [positions] * len(lens))
            logits = model.lm_head(hidden)[0, -1]
            position += 1
        append(output / 'ragged_query_audit.jsonl', dict(
            query_id=layout['query_id'],
            prompt_sha256=layout['prompt_sha256'], frame_visual_tokens=layout['recent_counts'],
            history_candidate_frame_count=layout['history_frames'],
            history_candidate_visual_tokens=historical_tokens,
            history_sample_indices=layout['history_sample_indices'],
            recent_sample_indices=layout['recent_sample_indices'],
            vision_cache_hits=layout['vision_cache_hits'],
            vision_cache_misses=layout['vision_cache_misses'],
            recent_frames_sha256=layout['recent_frames_sha256'],
            history_recent_positions_separate=chronological_positions,
            pruned_before_question=True, question_suffix_tokens=prompt_length - visual_end,
            dense_visual_kv_released=True,
            independent_single_frame_prefill=independent_pass,
            one_frame_visual_tokens=layout['counts'][-1],
            configured_visual_slots=56000,
            allocation_sha256=ALLOCATION_SHA256,
            arm=ACTIVE_ARM,
            selector=('fixed_pseudo_query_recent_only_same_timeline' if recent_only_control
                      else 'fixed_pseudo_query_all_older_frames_mild_head_quotas'),
            actual_visual_slots=sum(sum(row) for row in lens) - 112 * (prompt_length - sum(layout['counts'])),
            actual_history_visual_slots=history_slots,
            actual_recent_visual_slots=recent_slots,
            low_head_count=len(low_heads), dense_kv_slots=dense_slots,
            physical_kv_slots_at_answer_start=physical_slots,
            physical_kv_bytes_at_answer_start=physical_bytes,
            per_head_visual_slots=[[
                length - layout['old_start'] - suffix_tokens for length in row] for row in lens],
            generated_tokens=len(generated)))
        return self.processor.batch_decode([generated], skip_special_tokens=True,
                    clean_up_tokenization_spaces=False)[0].strip()

    upstream.RecentWindowQAModel.generate_with_cached_vision = question
    upstream.RecentWindowQAModel._generate_from_model_inputs = generate

    def query_recent_window(qa, video_path, prompt, chunk_duration, fps,
                            recent_frames_only, video_start=None, video_end=None):
        """Decode every sampled old frame plus the exact recent four frames."""
        if float(fps) != 1.0 or float(chunk_duration) != 1.0 or int(recent_frames_only) != 4:
            raise ValueError('This matched OVO arm requires fps=1, one-second chunks, recent=4')
        request = {'video': video_path, 'fps': float(fps)}
        if video_start is not None:
            request['video_start'] = float(video_start)
        if video_end is not None:
            request['video_end'] = float(video_end)
        plan = _build_sampling_plan(request, last_nframes=4)
        if not plan.full_indices:
            raise ValueError(f'No sampled frames in {video_path}')
        old_indices = plan.full_indices[:-4]
        history_indices = old_indices
        selected_indices = history_indices + plan.tail_indices
        if len(set(history_indices) & set(plan.tail_indices)):
            raise RuntimeError('Historical and recent raw frame indices overlap')
        buckets = []
        for batch_start in range(0, len(selected_indices), 16):
            batch_indices = selected_indices[batch_start:batch_start + 16]
            frames = _resize_like_fetch_video(
                _decode_indices(video_path, plan.backend, batch_indices), request,
                plan.sampled_nframes_full, image_patch_size=14)
            for batch_rank, (raw_index, frame) in enumerate(zip(batch_indices, frames)):
                rank = batch_start + batch_rank
                timestamp = float(raw_index) / plan.raw_fps
                chunk_index = int(timestamp // chunk_duration)
                image = Image.fromarray(frame.clamp(0, 255).to(torch.uint8)
                                        .permute(1, 2, 0).cpu().numpy())
                kind = 'history' if rank < len(history_indices) else 'recent'
                if buckets and buckets[-1][0] == (kind, chunk_index):
                    buckets[-1][1].append(image)
                else:
                    buckets.append(((kind, chunk_index), [image]))
        if not buckets or sum(kind == 'recent' for (kind, _), _ in buckets) > 4:
            raise RuntimeError('Exact recent-frame grouping changed')
        recent_fingerprint = hashlib.sha256()
        for (kind, _), images in buckets:
            if kind == 'recent':
                for image in images:
                    recent_fingerprint.update(str((image.mode, image.size)).encode())
                    recent_fingerprint.update(image.tobytes())
        encoded = []
        hits_before = cache_stats['hits']
        misses_before = cache_stats['misses']
        for (kind, chunk_index), images in buckets:
            embeds, grid = encode_reusable(qa, images)
            encoded.append((kind, chunk_index, embeds, grid))
        history_bucket_count = sum(kind == 'history' for kind, *_ in encoded)
        history_count = sum(int(row[3].shape[0]) for row in encoded[:history_bucket_count])
        recent = encoded[history_bucket_count:]
        if not 1 <= len(recent) <= 4:
            raise RuntimeError('No recent image window')
        embeds = torch.cat([row[2].to(qa.model.device) for row in encoded], dim=0)
        grid = torch.cat([row[3].to(qa.model.device) for row in encoded], dim=0)
        recent_tokens = sum(int(row[2].shape[0]) for row in recent)
        token = _history_context.set(dict(history_frames=history_count,
                                          query_id=Path(video_path).stem,
                                          history_sample_indices=history_indices,
                                          recent_sample_indices=plan.tail_indices,
                                          vision_cache_hits=cache_stats['hits'] - hits_before,
                                          vision_cache_misses=cache_stats['misses'] - misses_before,
                                          recent_frames_sha256=recent_fingerprint.hexdigest()))
        start = time.perf_counter()
        try:
            answer = qa.generate_with_cached_vision(embeds, grid, prompt)
        finally:
            _history_context.reset(token)
        elapsed = time.perf_counter() - start
        return (upstream.RecentWindowResult(
            answer=answer, final_chunk_ids=[row[1] for row in recent],
            generate_time=elapsed,
            ttft_seconds=getattr(qa, '_last_ttft_seconds', 0.0) or 0.0,
            num_vision_tokens=recent_tokens,
            num_vision_tokens_before=recent_tokens,
            num_vision_tokens_after=recent_tokens,
            num_frames=sum(int(row[3].shape[0]) for row in recent)),
            f'{plan.backend}_exact_full_sample_history')

    upstream.query_recent_window = query_recent_window
