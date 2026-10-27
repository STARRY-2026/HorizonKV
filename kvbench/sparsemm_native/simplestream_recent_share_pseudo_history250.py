"""Question-blind 250 recent + 250 history for one-frame KV heads.

Older candidates span the sampled video before the last four frames. Their
K/V and the recent K/V are computed in independent image-prefix passes, then
joined per head before the unchanged question suffix is prefetched.
"""

import contextvars
import hashlib
import sys
import time
import weakref
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import DynamicCache
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb

from sparsemm_native.simplestream_recent_share_attention import (
    HEADS, SOURCE_SHA256, SOURCE_TABLE,
)


HERMES_ROOT = Path(__file__).resolve().parents[2] / 'hermes'
RAGGED_SOURCE = HERMES_ROOT / 'head_analysis/hermes_kv_head_ragged.py'
RAGGED_SHA256 = hashlib.sha256(RAGGED_SOURCE.read_bytes()).hexdigest()
_question = contextvars.ContextVar('physical_recent_share_question', default=None)
_history_context = contextvars.ContextVar('pseudo_history_candidates', default=None)
VISUAL_BUDGET = 500
LOW_RECENT_BUDGET = 250
LOW_HISTORY_BUDGET = 250
HISTORY_CANDIDATE_FRAMES = 16
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


def _combine_prefix_caches(history_dense, full_dense, one_dense, low_heads,
                           Layer, visual_start, probes):
    """Join independent older and recent image passes without duplicate frames."""
    layers = []
    actual_history = 0
    actual_recent = 0
    for layer_idx, ((full_k, full_v), (one_k, one_v)) in enumerate(zip(full_dense, one_dense)):
        if full_k.shape[0:2] != (1, 4) or one_k.shape[0:2] != (1, 4):
            raise RuntimeError('Unexpected dense K/V dimensions')
        history_k, history_v = history_dense[layer_idx] if history_dense is not None else (None, None)
        k_parts = []
        v_parts = []
        lengths = []
        for head in range(4):
            use_one = (layer_idx, head) in low_heads
            chosen_k, chosen_v = (one_k, one_v) if use_one else (full_k, full_v)
            queries, pseudo_keys = probes[layer_idx]
            groups = queries.shape[0] // 4
            query = queries[head*groups:(head+1)*groups]
            historical_capacity = (min(LOW_HISTORY_BUDGET, int(history_k.shape[2]) - visual_start)
                                   if use_one and history_k is not None else 0)
            recent_budget = VISUAL_BUDGET - historical_capacity if use_one else VISUAL_BUDGET
            visual = _attention_visual_indices(
                chosen_k[0, head], query, pseudo_keys[head],
                visual_start, recent_budget)
            prefix = torch.arange(visual_start, device=chosen_k.device)
            selected = torch.cat((prefix, visual))
            key = chosen_k[0, head].index_select(0, selected)
            value = chosen_v[0, head].index_select(0, selected)
            history_count = 0
            if use_one and history_k is not None:
                older = _attention_visual_indices(
                    history_k[0, head], query, pseudo_keys[head],
                    visual_start, LOW_HISTORY_BUDGET, protect=0)
                history_count = int(older.numel())
                key = torch.cat((key[:visual_start],
                                 history_k[0, head].index_select(0, older),
                                 key[visual_start:]), dim=0)
                value = torch.cat((value[:visual_start],
                                   history_v[0, head].index_select(0, older),
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


def install(output, *, validation_keep_all=False):
    """Use the released prompt and frame resize with a bounded older-frame pool."""
    from lib import recent_window_eval_qwen25 as upstream
    from lib.qwen_exact_recent_decoder import (
        _build_sampling_plan, _decode_indices, _resize_like_fetch_video,
    )
    from PIL import Image
    from sparsemm_native.supervise_simplestream_upstream import append

    original_question = upstream.RecentWindowQAModel.generate_with_cached_vision
    original_generate = upstream.RecentWindowQAModel._generate_from_model_inputs
    low_heads = set() if validation_keep_all else set(HEADS)
    Layer, Cache, decode_step = _ragged_helpers()

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
                                   history_frames=history_frames,
                                   history_visual_end=old_start + sum(counts[:history_frames]),
                                   recent_counts=recent_counts,
                                   history_sample_indices=history_context['history_sample_indices'],
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
        full_probes = _pseudo_queries_and_keys(model, self.processor.tokenizer, visual_end)
        if len(full_dense) != len(model.language_model.layers):
            raise RuntimeError('Incomplete four-frame prefill cache')
        history_dense = None
        if layout['history_frames']:
            history_dense = DynamicCache()
            history_outputs = model(
                inputs_embeds=kwargs['inputs_embeds'][:, :history_end],
                attention_mask=kwargs['attention_mask'][:, :history_end],
                position_ids=kwargs['position_ids'][:, :, :history_end],
                past_key_values=history_dense, use_cache=True, return_dict=True)
            del history_outputs
            if len(history_dense) != len(full_dense):
                raise RuntimeError('Incomplete independent history prefill cache')
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
               (history_dense is not None and history_dense[i][0].shape[2] != history_end) or
               one_dense[i][0].shape[2] != len(one_indices)
               for i in range(len(full_dense))):
            raise RuntimeError('Prefix K/V lengths differ from the requested frames')
        full_key_ref = weakref.ref(full_dense[-1][0])
        one_key_ref = weakref.ref(one_dense[-1][0])
        history_key_ref = weakref.ref(history_dense[-1][0]) if history_dense is not None else None
        layers, history_slots, recent_slots = _combine_prefix_caches(
            history_dense, full_dense, one_dense, low_heads, Layer,
            layout['old_start'], full_probes)
        ragged = Cache(layers)
        del full_dense, one_dense, history_dense
        if (full_key_ref() is not None or one_key_ref() is not None or
                (history_key_ref is not None and history_key_ref() is not None)):
            raise RuntimeError('Dense visual K/V still live after branch combination')
        hidden = _prefill_question_suffix(
            model, kwargs['inputs_embeds'][:, visual_end:],
            kwargs['position_ids'][:, :, visual_end:], ragged)
        logits = model.lm_head(hidden)[0, -1]
        historical_tokens = sum(layout['counts'][:layout['history_frames']])
        historical_capacity = min(historical_tokens, LOW_HISTORY_BUDGET)
        expected_low_visual = (min(layout['counts'][-1], VISUAL_BUDGET - historical_capacity)
                               + historical_capacity)
        expected_low = layout['old_start'] + expected_low_visual + (prompt_length - visual_end)
        expected_high = layout['old_start'] + min(sum(layout['recent_counts']), VISUAL_BUDGET) + (prompt_length - visual_end)
        lens = [layer.head_lens.tolist() for layer in ragged.layers]
        if any(length != (expected_low if (layer_idx, head) in low_heads else expected_high)
               for layer_idx, row in enumerate(lens) for head, length in enumerate(row)):
            raise RuntimeError('Physical head K/V lengths differ from requested windows')
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
            prompt_sha256=layout['prompt_sha256'], frame_visual_tokens=layout['recent_counts'],
            history_candidate_frame_count=layout['history_frames'],
            history_candidate_visual_tokens=historical_tokens,
            history_sample_indices=layout['history_sample_indices'],
            history_recent_positions_separate=chronological_positions,
            pruned_before_question=True, question_suffix_tokens=prompt_length - visual_end,
            dense_visual_kv_released=True,
            independent_single_frame_prefill=independent_pass,
            one_frame_visual_tokens=layout['counts'][-1],
            visual_budget_per_head=VISUAL_BUDGET,
            low_recent_budget=LOW_RECENT_BUDGET,
            low_history_budget=LOW_HISTORY_BUDGET,
            selector='pseudo_history16_recent250_history250_lowheads_recent500_highheads',
            actual_visual_slots=sum(sum(row) for row in lens) - 112 * (prompt_length - sum(layout['counts'])),
            actual_history_visual_slots=history_slots,
            actual_recent_visual_slots=recent_slots,
            low_head_count=len(low_heads), dense_kv_slots=dense_slots,
            physical_kv_slots_at_answer_start=physical_slots,
            physical_kv_bytes_at_answer_start=physical_bytes,
            low_head_tokens=expected_low, high_head_tokens=expected_high,
            generated_tokens=len(generated)))
        return self.processor.batch_decode([generated], skip_special_tokens=True,
                    clean_up_tokenization_spaces=False)[0].strip()

    upstream.RecentWindowQAModel.generate_with_cached_vision = question
    upstream.RecentWindowQAModel._generate_from_model_inputs = generate

    def query_recent_window(qa, video_path, prompt, chunk_duration, fps,
                            recent_frames_only, video_start=None, video_end=None):
        """Decode exact recent frames plus 16 evenly spaced older candidates."""
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
        candidate_count = min(HISTORY_CANDIDATE_FRAMES, len(old_indices))
        selected_positions = (torch.linspace(0, len(old_indices) - 1, candidate_count)
                              .round().long().tolist() if candidate_count else [])
        history_indices = [old_indices[i] for i in selected_positions]
        selected_indices = history_indices + plan.tail_indices
        frames = _resize_like_fetch_video(
            _decode_indices(video_path, plan.backend, selected_indices), request,
            plan.sampled_nframes_full, image_patch_size=14)
        buckets = []
        for rank, (raw_index, frame) in enumerate(zip(selected_indices, frames)):
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
        encoded = []
        for (kind, chunk_index), images in buckets:
            embeds, grid = qa.encode_vision(images)
            encoded.append((kind, chunk_index, embeds, grid))
        history_count = sum(kind == 'history' for kind, *_ in encoded)
        recent = encoded[history_count:]
        if not 1 <= len(recent) <= 4:
            raise RuntimeError('No recent image window')
        embeds = torch.cat([row[2].to(qa.model.device) for row in encoded], dim=0)
        grid = torch.cat([row[3].to(qa.model.device) for row in encoded], dim=0)
        recent_tokens = sum(int(row[2].shape[0]) for row in recent)
        token = _history_context.set(dict(history_frames=history_count,
                                          history_sample_indices=history_indices))
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
            f'{plan.backend}_exact_history16')

    upstream.query_recent_window = query_recent_window
