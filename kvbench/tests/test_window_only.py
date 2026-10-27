import copy
from unittest.mock import patch

import pytest
import torch
from transformers import Qwen2_5_VLTextConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel

from sparsemm_native.window_only import WindowOnlyCache
from sparsemm_protokv.config import StreamingConfig
from sparsemm_protokv.qwen2_5 import install, uninstall


@pytest.fixture
def native():
    torch.manual_seed(41)
    config = Qwen2_5_VLTextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=4096,
        rope_scaling={'type': 'mrope', 'mrope_section': [1, 1, 2]},
        attn_implementation='sdpa')
    model = Qwen2_5_VLTextModel(config).eval()
    cache = WindowOnlyCache(StreamingConfig(method='sparsemm', budget=500),
                            2, 2, 8, 'cpu', torch.float32)
    install()
    positions = torch.arange(3).reshape(1, 1, -1).expand(3, 1, -1)
    cache.set_context('init', positions[:, 0])
    with torch.inference_mode():
        model(input_ids=torch.tensor([[1, 2, 3]]), position_ids=positions,
              past_key_values=cache, cache_position=torch.arange(3),
              attention_mask={'full_attention': None}, use_cache=True, return_dict=True)
    cache.finish_init(3)
    yield model, cache
    uninstall()


def push(model, cache, start, tokens=8, real=2):
    positions = torch.full((3, 1, tokens), start * 10. + 10.)
    cache.begin_unit(positions[:, 0], start, real)
    cache.set_context('video', positions[:, 0], xy=torch.zeros(tokens, 2), time=start + real)
    logical = cache.get_seq_length()
    with torch.inference_mode():
        model(inputs_embeds=torch.full((1, tokens, 32), start + 1.), position_ids=positions,
              past_key_values=cache, cache_position=torch.arange(logical, logical + tokens),
              attention_mask={'full_attention': None}, use_cache=True, return_dict=True)
    cache.finish_video()


def test_outgoing_kv_is_removed_before_every_layer_attends(native):
    model, cache = native
    push(model, cache, 0)
    push(model, cache, 2)
    observed = []
    hooks = [layer.self_attn.register_forward_pre_hook(
        lambda module, args, kwargs: observed.append(cache.layers[module.layer_idx].head_lens.tolist()),
        with_kwargs=True) for layer in model.layers]
    try:
        push(model, cache, 4)
    finally:
        for hook in hooks:
            hook.remove()
    assert observed == [[3 + 8] * 2] * 2
    audit = cache.audit_window()
    assert audit['recent_frame_starts'] == [2, 4]
    assert audit['recent_visual_tokens_per_head'] == 16
    assert audit['history_visual_tokens'] == 0


def test_retained_unit_is_never_recomputed_and_prefix_survives(native):
    model, cache = native
    push(model, cache, 0)
    push(model, cache, 2)
    old = [(layer.get_segment(0)[0][:3].clone(), layer.get_segment(0)[1][:3].clone(),
            layer.get_segment(0)[0][-8:].clone(), layer.get_segment(0)[1][-8:].clone())
           for layer in cache.layers]
    push(model, cache, 4)
    for layer, (pk, pv, rk, rv) in zip(cache.layers, old):
        key, value = layer.get_segment(0)
        assert torch.equal(key[:3], pk) and torch.equal(value[:3], pv)
        assert torch.equal(key[3:11], rk) and torch.equal(value[3:11], rv)


def test_variable_resolution_uses_unit_lengths_not_a_fixed_token_budget(native):
    model, cache = native
    for start, tokens in [(0, 8), (2, 10), (4, 6)]:
        push(model, cache, start, tokens=tokens)
    audit = cache.audit_window()
    assert audit['recent_unit_tokens'] == [10, 6]
    assert audit['actual_per_head_visual_counts'] == [[16, 16], [16, 16]]
    assert audit['actual_visual_slots'] == 64
    assert cache.bank is None


def test_odd_padded_branch_does_not_change_even_parent(native):
    model, cache = native
    push(model, cache, 0)
    push(model, cache, 2)
    saved = {key: value.clone() for key, value in cache.state_tensors().items()}
    odd = copy.deepcopy(cache)
    push(model, odd, 4, real=1)
    audit = odd.audit_window()
    assert audit['recent_real_frames'] == 3 and audit['recent_padded_frames'] == 1
    assert [unit.frame_start for unit in cache.units] == [0, 2]
    assert all(torch.equal(value, saved[key]) for key, value in cache.state_tensors().items())


def test_question_kv_and_cursors_are_restored(native):
    model, cache = native
    push(model, cache, 0)
    push(model, cache, 2)
    saved = {key: (value, value.clone()) for key, value in cache.state_tensors().items()}
    logical = cache.get_seq_length()
    with cache.query():
        positions = torch.arange(100, 102).reshape(1, 1, -1).expand(3, 1, -1)
        cache.set_context('query', positions[:, 0])
        with torch.inference_mode():
            model(input_ids=torch.tensor([[4, 5]]), position_ids=positions,
                  past_key_values=cache, cache_position=torch.arange(logical, logical + 2),
                  attention_mask={'full_attention': None}, use_cache=True, return_dict=True)
        cache.advance_query(2)
        assert cache.layers[0].max_seqlen_k == 3 + 16 + 2
    assert cache.get_seq_length() == logical
    assert all(value is saved[key][0] and torch.equal(value, saved[key][1])
               for key, value in cache.state_tensors().items())
    cache.audit_window()


def test_no_attention_score_selector_is_called(native):
    model, cache = native
    with patch('sparsemm_protokv.cache.select_layer', side_effect=AssertionError('selector called')):
        for start in (0, 2, 4, 6):
            push(model, cache, start)
    assert cache.audit_window()['dropped_units'] == 2


def test_sharding_never_splits_one_video_and_preserves_all_queries():
    from sparsemm_native.run_ovobench_window_only import schedule, split_videos
    rows = [dict(query_id='a0', video_path='a', clip_end=8),
            dict(query_id='b0', video_path='b', clip_end=5),
            dict(query_id='a1', video_path='a', clip_end=3),
            dict(query_id='c0', video_path='c', clip_end=11)]
    planned = schedule(rows)
    assert [q['query_id'] for q in planned] == ['a1', 'a0', 'b0', 'c0']
    jobs, estimates = split_videos(planned)
    assert {q['query_id'] for job in jobs for q in job} == {q['query_id'] for q in rows}
    assert not ({q['video_path'] for q in jobs[0]} & {q['video_path'] for q in jobs[1]})
    # A needs eight base frames plus one isolated odd-endpoint frame;
    # B and C need their respective five and eleven frames.
    assert sum(estimates) == 9 + 5 + 11
