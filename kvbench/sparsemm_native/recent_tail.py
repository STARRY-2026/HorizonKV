"""Exact sparse-tail input for the existing one/four-frame recent control.

Both modes below discard outgoing features without a history forward. The
full-prefix mode is a conservative single-answer speed control: it encodes
every sampled frame but does not do the paired run's unrelated history work.
The tail mode encodes only the identical final two temporal units. Neither
mode changes head groups, pseudo selection, positions, question or decoder.
"""

import hashlib
import math
import time

import torch

from sparsemm_baselines.run_streamingbench import sampled_indices
from sparsemm_native.aligned_pseudo_pair import AlignedPseudoPairQwen, assemble, digest
from sparsemm_native.decoupled import prefill_recent
from sparsemm_native.mild_pseudo_history_fps05 import MildPseudoHistoryCache
from sparsemm_native.simplestream_recent_share_pseudo_history_mild500 import _pseudo_queries_and_keys
from sparsemm_native.video_reader import BatchScopedVideoReader


def sampling_plan(frame_count, avg_fps, clip_end, sample_fps=.5):
    indices = sampled_indices(frame_count, avg_fps, sample_fps)
    target = min(len(indices), math.ceil(clip_end * sample_fps))
    if target < 1:
        raise ValueError('Question has no sampled frame')
    # Preserve pairing from source ordinal zero. For an odd endpoint the last
    # unit is a single frame plus processor padding, not a newly shifted pair.
    start = max(0, target - (3 if target % 2 else 4))
    if start % 2:
        raise AssertionError('Tail start must preserve the original even pairing')
    return dict(sampled_prefix_frames=target, start_ordinal=start,
                tail_ordinals=list(range(start, target)),
                tail_raw_indices=indices[start:target],
                unit_start_ordinals=list(range(start, target, 2)),
                prefix_indices=indices[:target], rounded_source_fps=round(avg_fps))


def tensor_digest(value):
    # NumPy has no native bfloat16 dtype; hash the exact storage bytes instead.
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def cuda_sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


class RecentOnlyCache(MildPseudoHistoryCache):
    def push_unit(self, unit):
        super().push_unit(unit)
        # The control has no historical visual KV. Old features can be freed;
        # forwarding them through the language model would do unused work.
        return None


class RecentTailQwen(AlignedPseudoPairQwen):
    def clear_cache(self):
        super().clear_cache()
        self.kv_cache = RecentOnlyCache(
            self.config, self.num_layers, self.num_kv_heads, self.head_dim,
            self.device, self.dtype)
        self._tail_rgb = None

    @torch.inference_mode()
    def encode_video_chunk(self, frames, is_last_chunk=False):
        super().encode_video_chunk(frames, is_last_chunk=is_last_chunk)
        if frames is not None and frames.shape[0]:
            previous = self._tail_rgb
            self._tail_rgb = ((torch.cat((previous, frames.cpu()), dim=0)
                               if previous is not None else frames.cpu())[-4:]).clone()

    @torch.inference_mode()
    def answer_recent(self, query, max_new_tokens=256):
        history = self.kv_cache
        if not history.units or history.prefix_dense is None or self._prefills:
            raise RuntimeError('Recent control must have units and zero historical language forwards')
        prefix = history.prefix_tokens
        if any(int(n) != prefix for layer in history.layers for n in layer.head_lens):
            raise AssertionError('Recent-only persistent cache contains historical visual KV')
        state = history.state_tensors()
        frozen = {name: (value, value.clone()) for name, value in state.items()}
        one_unit = self._last_frame_unit(history)
        if one_unit.grid[1:] != history.units[-1].grid[1:]:
            raise AssertionError('Independent last frame has a different spatial grid')
        one = prefill_recent(self.language_model, history.prefix_dense, [one_unit])
        four = prefill_recent(self.language_model, history.prefix_dense, history.units)
        question_position = max(self._next_position, int(one_unit.positions.max()) + 1)
        probes = _pseudo_queries_and_keys(self.model, self.processor.tokenizer, question_position)
        tokenizer = self.processor.tokenizer
        suffix = ([tokenizer.convert_tokens_to_ids('<|vision_end|>')]
                  + tokenizer.encode('\n', add_special_tokens=False)
                  + tokenizer.encode(query['prompt'], add_special_tokens=False)
                  + [tokenizer.convert_tokens_to_ids('<|im_end|>')]
                  + tokenizer.encode('\n', add_special_tokens=False)
                  + [tokenizer.convert_tokens_to_ids('<|im_start|>')]
                  + tokenizer.encode('assistant\n', add_special_tokens=False))
        penalty_prompt = (self.init_prompt_ids[0].tolist()
                          + [int(self.model_config.image_token_id)] * sum(u.tokens for u in history.units)
                          + suffix)
        ragged, audit = assemble(history, one, four, one_unit, probes, 'recent_only')
        # Unlike the shared two-arm experiment, a standalone control can release
        # both temporary dense caches before feeding the real question.
        del one, four
        response, decode_audit = self.decode_arm(
            ragged, suffix, question_position, penalty_prompt, max_new_tokens)
        del ragged
        current = history.state_tensors()
        if (current.keys() != frozen.keys()
                or any(current[name] is not ref or not torch.equal(current[name], saved)
                       for name, (ref, saved) in frozen.items())):
            raise AssertionError('Answer changed persistent recent/prefix state')
        audit.update(decode_audit, question_position=question_position,
                     question_suffix_tokens=len(suffix),
                     official_prompt_sha256=hashlib.sha256(query['prompt'].encode()).hexdigest(),
                     shared_wrapper_ids_sha256=digest(self.init_prompt_ids[0].tolist() + suffix),
                     persistent_state_unchanged_after_both_answers=True,
                     temporary_dense_kv_released_before_question=True,
                     common_absolute_video_positions=True,
                     independent_last_raw_frame=True,
                     recent_one_frame_heads=56, recent_four_frame_heads=56)
        real_frames = sum(unit.real_frames for unit in history.units)
        tail_audit = dict(
            single_frame_visual_tokens=one_unit.tokens,
            recent_visual_unit_tokens=[unit.tokens for unit in history.units],
            recent_visual_grids=[list(unit.grid) for unit in history.units],
            recent_real_frames=real_frames,
            unit_start_ordinals=[unit.frame_start for unit in history.units],
            recent_positions_sha256=tensor_digest(torch.cat([u.positions for u in history.units], dim=2)),
            single_frame_positions_sha256=tensor_digest(one_unit.positions),
            recent_features_sha256=tensor_digest(torch.cat([u.features for u in history.units], dim=1)),
            single_frame_features_sha256=tensor_digest(one_unit.features),
            tail_rgb_sha256=tensor_digest(self._tail_rgb[-real_frames:]),
            question_position=question_position,
            video_max_pixels=self.video_max_pixels,
            total_processed_frames=self.total_processed_frames,
            no_history_language_forwards=self._prefills == 0,
            no_historical_visual_kv=True,
            dense_recent_kv_released=True)
        return dict(response=response, budget_audit=audit, tail_audit=tail_audit)


@torch.inference_mode()
def run_recent_query(bridge, query, mode, sample_fps=.5, max_new_tokens=256):
    if mode not in ('tail_only', 'full_prefix'):
        raise ValueError(mode)
    cuda_sync(bridge.device)
    started = time.perf_counter()
    reader = BatchScopedVideoReader(query['video_path'], num_threads=1)
    plan = sampling_plan(len(reader), reader.get_avg_fps(), query['clip_end'], sample_fps)
    after_metadata = time.perf_counter()
    bridge.encode_init_prompt()
    after_prefix = time.perf_counter()
    decode_seconds = vision_seconds = 0.
    target = plan['sampled_prefix_frames']
    if mode == 'tail_only':
        # Counter starts at its source ordinal; all M-RoPE computations remain
        # the original model helper, with no local-window renumbering.
        bridge.total_processed_frames = plan['start_ordinal']
        ranges = [(plan['start_ordinal'], target)]
    else:
        even = target - target % 2
        ranges = [(start, min(start + 16, even)) for start in range(0, even, 16)]
        if target % 2:
            ranges.append((even, target))
    for start, end in ranges:
        before = time.perf_counter()
        frames = torch.from_numpy(reader.get_batch(plan['prefix_indices'][start:end]).asnumpy())
        decode_seconds += time.perf_counter() - before
        before = time.perf_counter()
        bridge.encode_video_chunk(frames, is_last_chunk=end == target)
        bridge.predict_and_compress()
        cuda_sync(bridge.device)
        vision_seconds += time.perf_counter() - before
    before = time.perf_counter()
    result = bridge.answer_recent(query, max_new_tokens)
    cuda_sync(bridge.device)
    qa_seconds = time.perf_counter() - before
    result['seconds'] = time.perf_counter() - started
    result['timing'] = dict(
        metadata_seconds=after_metadata - started,
        prefix_seconds=after_prefix - after_metadata,
        decode_seconds=decode_seconds, vision_seconds=vision_seconds,
        recent_prefill_selection_qa_seconds=qa_seconds,
        end_to_end_excluding_model_load_seconds=result['seconds'])
    result['tail_audit'].update(
        sampled_prefix_frames=target, tail_raw_indices=plan['tail_raw_indices'],
        tail_ordinals=plan['tail_ordinals'], source_tail_start=plan['start_ordinal'],
        rounded_source_fps=plan['rounded_source_fps'], mode=mode,
        encoded_real_frames=sum(end - start for start, end in ranges),
        decoded_rgb_frames_requested=sum(end - start for start, end in ranges),
        decode_batch_count=len(ranges))
    if (result['tail_audit']['unit_start_ordinals'] != plan['unit_start_ordinals']
            or bridge.total_processed_frames != target):
        raise AssertionError('Tail unit pairing or global frame cursor changed')
    bridge.clear_cache()
    return result


def load_model(model_path, config, **kwargs):
    from sparsemm_protokv.model_loader import load_pretrained
    from sparsemm_native.assets import asset_metadata
    device, dtype = kwargs.pop('device', 'cuda'), kwargs.pop('dtype', 'bfloat16')
    model, processor = load_pretrained(model_path, device=device, dtype=dtype)
    return RecentTailQwen(model, processor, config,
                          asset_metadata=asset_metadata(config.sparsemm_score_path or None), **kwargs)
