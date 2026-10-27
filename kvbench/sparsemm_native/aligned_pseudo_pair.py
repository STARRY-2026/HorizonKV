"""Paired old one/four-head decoder on the current fps=.5 video frontend.

The two arms share visual features, independent recent prefills, text wrapping,
absolute video M-RoPE, pseudo probes and the released ragged QA decoder. Only
the per-head recent/history quotas change. Questions never enter the selector.
Existing experiment implementations and their pinned sources are untouched.
"""

import hashlib
import json
import time

import torch

from sparsemm_native.decoupled import dense_prefix, prefill_recent
from sparsemm_native.mild_pseudo_history_fps05 import MildPseudoHistoryQwen, ROWS
from sparsemm_native.simplestream_recent_share_pseudo_history_mild500 import (
    _attention_visual_indices, _pseudo_queries_and_keys,
)
from sparsemm_native.simplestream_recent_share_pseudo_budget500 import (
    _apply_repetition_penalty, _prefill_question_suffix, _ragged_helpers,
)


ARMS = ('recent_only', 'mild_history')


def allocation(arm):
    if arm not in ARMS:
        raise ValueError(arm)
    return {key: ({**row, 'recent_quota': '500', 'history_quota': '0',
                   'total_quota': '500'} if arm == 'recent_only' else dict(row))
            for key, row in ROWS.items()}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def assemble(history, one, four, one_unit, probes, arm):
    """Build a new owned ragged cache: prefix once, old history, fresh recent."""
    rows = allocation(arm)
    Layer, Cache, _ = _ragged_helpers()
    prefix = history.prefix_tokens
    layers, per_head, recent_indices, history_indices = [], [], [], []
    recent_slots = history_slots = recent_candidates = 0
    pruned_heads = 0
    for layer_idx, source in enumerate(history.layers):
        keys, values, lengths, visual_counts = [], [], [], []
        queries, pseudo_keys = probes[layer_idx]
        groups = queries.shape[0] // history.num_kv_heads
        for head in range(history.num_kv_heads):
            row = rows[layer_idx, head]
            historical_k, historical_v = source.get_segment(head)
            historical_p = source.get_position_segment(head)
            h_available = int(historical_k.shape[0]) - prefix
            h_quota = int(row['history_quota'])
            h = h_available if arm == 'mild_history' else 0
            if not 0 <= h <= h_quota:
                raise AssertionError('Historical quota exceeded')
            chosen = one if row['recent_group'] == 'one_frame' else four
            rk, rv = chosen[layer_idx]
            units = [one_unit] if row['recent_group'] == 'one_frame' else history.units
            positions = torch.cat([unit.positions[:, 0] for unit in units], dim=1)
            if rk.shape[2] != prefix + positions.shape[1]:
                raise AssertionError('Wrong independent recent length')
            capacity = int(row['recent_quota']) + h_quota - h
            selected = _attention_visual_indices(
                rk[0, head], queries[head * groups:(head + 1) * groups],
                pseudo_keys[head], prefix, capacity, protect=32)
            if h and bool((historical_p[0, prefix:] >=
                           positions[0].index_select(0, selected - prefix).min()).any()):
                raise AssertionError('Recent and historical time ranges overlap')
            # All concatenations create owned storage. QA on either arm cannot
            # append to or alter the persistent history or shared recent KV.
            keys.append(torch.cat((historical_k[:prefix + h],
                                   rk[0, head].index_select(0, selected))))
            values.append(torch.cat((historical_v[:prefix + h],
                                     rv[0, head].index_select(0, selected))))
            r = int(selected.numel())
            if h + r > int(row['total_quota']):
                raise AssertionError('Per-head visual quota exceeded')
            lengths.append(prefix + h + r)
            visual_counts.append(h + r)
            recent_slots += r
            history_slots += h
            recent_candidates += int(positions.shape[1])
            pruned_heads += r < positions.shape[1]
            recent_indices.append(selected.tolist())
            history_indices.append(historical_p[:, prefix:prefix + h].tolist())
        layers.append(Layer(torch.cat(keys), torch.cat(values),
                            torch.tensor(lengths, device=keys[0].device, dtype=torch.int32)))
        per_head.append(visual_counts)
    if recent_slots + history_slots > 56000:
        raise AssertionError('Mean visual K500 exceeded')
    return Cache(layers), dict(
        actual_recent_visual_slots=recent_slots, actual_history_visual_slots=history_slots,
        actual_visual_slots=recent_slots + history_slots, per_head_visual_slots=per_head,
        configured_visual_slots=56000, recent_candidate_visual_slots=recent_candidates,
        recent_pruned_heads=pruned_heads, recent_selection_sha256=digest(recent_indices),
        selected_history_positions_sha256=digest(history_indices),
        history_recent_independent_prefill=True, history_recent_time_ranges_disjoint=True,
        fixed_pseudo_query=True, budget_scope='visual_KV_before_real_question',
        text_prefix_tokens=prefix, text_and_generation_not_in_visual_budget=True)


class AlignedPseudoPairQwen(MildPseudoHistoryQwen):
    @torch.inference_mode()
    def encode_init_prompt(self):
        # Match the old SimpleStream visual wrapper, using the same frozen
        # prefix for the history branch and both independent recent prefills.
        self.clear_cache()
        tokenizer = self.processor.tokenizer
        ids = tokenizer.encode('<|im_start|>user\n<|vision_start|>', add_special_tokens=False)
        self.init_prompt_ids = torch.tensor([ids], device=self.device, dtype=torch.long)
        count = len(ids)
        positions = torch.arange(count, device=self.device).view(1, 1, -1).expand(3, 1, -1)
        self.forward_chunk(self.model.get_input_embeddings()(self.init_prompt_ids),
                           positions, compress=False)
        self.visual_start_idx = self._next_position = count
        self.kv_cache.finish_init(count)
        self.kv_cache.prefix_dense = dense_prefix(self.kv_cache)

    @torch.inference_mode()
    def decode_arm(self, ragged, suffix_ids, question_position, penalty_prompt_ids,
                   max_new_tokens):
        """Use the old physical ragged prefill, decode step, EOS and penalty."""
        _, _, decode_step = _ragged_helpers()
        ids = torch.tensor([suffix_ids], device=self.device, dtype=torch.long)
        positions = torch.arange(question_position, question_position + len(suffix_ids),
                                 device=self.device).view(1, 1, -1).expand(3, 1, -1)
        hidden = _prefill_question_suffix(
            self.model, self.model.get_input_embeddings()(ids), positions, ragged)
        logits = self.model.lm_head(hidden)[0, -1]
        config = self.model.generation_config
        penalty = float(config.repetition_penalty or 1.0)
        eos = config.eos_token_id
        stop_ids = set(eos if isinstance(eos, list) else [eos])
        generated, cursor = [], question_position + len(suffix_ids)
        for step in range(max_new_tokens):
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError('Non-finite answer logits')
            scores = _apply_repetition_penalty(logits, penalty_prompt_ids + generated, penalty)
            token = int(scores.argmax())
            generated.append(token)
            if token in stop_ids or step + 1 == max_new_tokens:
                break
            pos = torch.full((3, 1, 1), cursor, device=self.device, dtype=torch.long)
            hidden = decode_step(self.model, torch.tensor([[token]], device=self.device),
                                 ragged, [pos] * self.num_layers)
            logits = self.model.lm_head(hidden)[0, -1]
            cursor += 1
        response = self.processor.batch_decode([generated], skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)[0].strip()
        return response, dict(generated_tokens=len(generated),
                              hit_generation_limit=(len(generated) == max_new_tokens
                                                    and generated[-1] not in stop_ids),
                              eos_token_ids=sorted(stop_ids), repetition_penalty=penalty)

    @torch.inference_mode()
    def answer_pair(self, query, max_new_tokens=256):
        history = self.kv_cache
        if not history.units or history.prefix_dense is None:
            raise RuntimeError('Missing recent visual units')
        state = history.state_tensors()
        frozen = {name: (value, value.clone()) for name, value in state.items()}
        started = time.perf_counter()
        one_unit = self._last_frame_unit(history)
        if one_unit.grid[1:] != history.units[-1].grid[1:]:
            raise AssertionError('One/four branch spatial grids differ')
        one = prefill_recent(self.language_model, history.prefix_dense, [one_unit])
        four = prefill_recent(self.language_model, history.prefix_dense, history.units)
        question_position = max(self._next_position, int(one_unit.positions.max()) + 1)
        probes = _pseudo_queries_and_keys(self.model, self.processor.tokenizer, question_position)
        tokenizer = self.processor.tokenizer
        # Encode pieces separately, exactly as in the old wrapper. The middle
        # text is the unchanged official MCQA template, not the old instruction.
        suffix = ([tokenizer.convert_tokens_to_ids('<|vision_end|>')]
                  + tokenizer.encode('\n', add_special_tokens=False)
                  + tokenizer.encode(query['prompt'], add_special_tokens=False)
                  + [tokenizer.convert_tokens_to_ids('<|im_end|>')]
                  + tokenizer.encode('\n', add_special_tokens=False)
                  + [tokenizer.convert_tokens_to_ids('<|im_start|>')]
                  + tokenizer.encode('assistant\n', add_special_tokens=False))
        candidate_count = sum(unit.tokens for unit in history.units)
        penalty_prompt = (self.init_prompt_ids[0].tolist()
                          + [int(self.model_config.image_token_id)] * candidate_count + suffix)
        # Alternate decode order without changing model inputs; arm caches own
        # their tensors and are discarded after the answer.
        order = ARMS if int(query['source_id']) % 2 == 0 else ARMS[::-1]
        results = {}
        for arm in order:
            ragged, audit = assemble(history, one, four, one_unit, probes, arm)
            response, decode_audit = self.decode_arm(
                ragged, suffix, question_position, penalty_prompt, max_new_tokens)
            audit.update(decode_audit, question_suffix_tokens=len(suffix),
                         question_position=question_position,
                         official_prompt_sha256=hashlib.sha256(query['prompt'].encode()).hexdigest(),
                         shared_wrapper_ids_sha256=digest(self.init_prompt_ids[0].tolist() + suffix),
                         common_absolute_video_positions=True,
                         independent_last_raw_frame=True,
                         recent_one_frame_heads=56, recent_four_frame_heads=56)
            results[arm] = dict(response=response, budget_audit=audit)
            del ragged
        current = history.state_tensors()
        if (current.keys() != frozen.keys()
                or any(current[name] is not ref or not torch.equal(current[name], saved)
                       for name, (ref, saved) in frozen.items())):
            raise AssertionError('Answering mutated persistent video/history state')
        for result in results.values():
            result['budget_audit']['persistent_state_unchanged_after_both_answers'] = True
        selector_checks = self.validation_checks['selector_checks']
        if self.validation_checks['failures'] or any(
                check.get('validation_passed') is not True for check in selector_checks):
            raise AssertionError('Historical selector validation failed')
        selector_checks.clear()
        return dict(arms=results, paired_audit=dict(
            shared_visual_features=True, shared_independent_recent_prefills=True,
            shared_question_and_decoder=True, shared_absolute_positions=True,
            recent_selection_identical=(results['recent_only']['budget_audit']['recent_selection_sha256']
                                        == results['mild_history']['budget_audit']['recent_selection_sha256']),
            persistent_state_unchanged=True, decode_order=list(order),
            recent_real_frames=sum(unit.real_frames for unit in history.units),
            recent_visual_unit_tokens=[unit.tokens for unit in history.units],
            recent_visual_grids=[list(unit.grid) for unit in history.units],
            recent_resized_hw=[[unit.grid[1] * int(self.model_config.vision_config.patch_size),
                                unit.grid[2] * int(self.model_config.vision_config.patch_size)]
                               for unit in history.units],
            single_frame_visual_tokens=one_unit.tokens,
            single_frame_visual_grid=list(one_unit.grid),
            video_max_pixels=self.video_max_pixels,
            processor_size_overridden=self.video_max_pixels is not None,
            rebuild_and_two_answers_seconds=time.perf_counter() - started))


def load_model(model_path, config, **kwargs):
    from sparsemm_protokv.model_loader import load_pretrained
    from sparsemm_native.assets import asset_metadata
    device = kwargs.pop('device', 'cuda')
    dtype = kwargs.pop('dtype', 'bfloat16')
    model, processor = load_pretrained(model_path, device=device, dtype=dtype)
    return AlignedPseudoPairQwen(model, processor, config,
                                asset_metadata=asset_metadata(config.sparsemm_score_path or None),
                                **kwargs)
