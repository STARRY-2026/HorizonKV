"""CPU checks for shared pixels and Qwen image/video token accounting."""

import pytest
import torch

from kvbench.fixed_inputs import (
    assert_qwen_grid, fixed_input_metadata, frame_rgb_sha256,
    prepare_fixed_frames,
)


def test_contract_distinguishes_real_video_frames_from_encoding_units():
    metadata = fixed_input_metadata()
    assert metadata["resized_hw"] == [896, 896]
    assert metadata["spatial_grid"] == [64, 64]
    assert metadata["tokens_per_temporal_unit"] == 1024
    assert metadata["image_tokens_per_real_frame"] == 1024
    assert metadata["video_tokens_for_real_frames"] == "ceil(T/2)*1024"
    assert metadata["sampling_changed"] is False


def test_letterbox_preserves_order_color_aspect_ratio_and_source():
    # Wide source: preserve a 2:1 image on the middle 448 rows, without crop.
    frames = torch.stack([torch.full((56, 112, 3), color, dtype=torch.uint8)
                          for color in (23, 217)])
    before = frames.clone()
    result = prepare_fixed_frames(frames)
    assert result.shape == (2, 896, 896, 3)
    assert result.dtype == torch.uint8 and result.device.type == "cpu"
    assert torch.equal(frames, before)
    assert bool((result[:, :224] == 127).all())
    assert bool((result[:, 672:] == 127).all())
    assert bool((result[0, 224:672] == 23).all())
    assert bool((result[1, 224:672] == 217).all())


def test_chw_and_hwc_paths_have_identical_actual_pixels():
    source = torch.arange(2 * 71 * 43 * 3).remainder(256).to(torch.uint8).reshape(2, 71, 43, 3)
    before = source.clone()
    hwc = prepare_fixed_frames(source)
    chw = prepare_fixed_frames(source.permute(0, 3, 1, 2))
    assert chw.shape == (2, 3, 896, 896)
    assert torch.equal(hwc, chw.permute(0, 2, 3, 1))
    assert frame_rgb_sha256(hwc) == frame_rgb_sha256(chw)
    assert torch.equal(source, before)


def test_odd_padding_extra_pixel_is_at_bottom():
    source = torch.zeros(1, 55, 112, 3, dtype=torch.uint8)
    # At canvas28 the content is 14x28; at canvas84 it is 41x84, leaving43.
    result = prepare_fixed_frames(source, canvas_size=84)
    assert bool((result[:, :21] == 127).all())
    assert bool((result[:, 21:62] == 0).all())
    assert bool((result[:, 62:] == 127).all())


def test_half_pixel_resize_rounds_up_and_downsampling_uses_antialiasing():
    # 9/24*28 = 10.5 -> 11 content rows (rather than Python's ties-to-even 10).
    result = prepare_fixed_frames(torch.zeros(1, 9, 24, 3, dtype=torch.uint8),
                                  canvas_size=28)
    assert bool((result[:, :8] == 127).all())
    assert bool((result[:, 8:19] == 0).all())
    assert bool((result[:, 19:] == 127).all())
    # A high-frequency checkerboard must average to gray when reduced 4x;
    # nearest-neighbor sampling would silently lose that visual information.
    board = ((torch.arange(112)[:, None] + torch.arange(224)[None, :]) % 2 * 255)
    source = board.to(torch.uint8)[None, :, :, None].expand(1, -1, -1, 3)
    reduced = prepare_fixed_frames(source, canvas_size=56)
    center = reduced[:, 17:39, 4:-4]
    assert int(center.min()) >= 120 and int(center.max()) <= 135


def test_disabled_profile_is_no_op_and_square_inputs_are_not_aliased():
    source = torch.full((1, 28, 28, 3), 42, dtype=torch.uint8)
    assert prepare_fixed_frames(source, canvas_size=None) is source
    assert fixed_input_metadata(None) == {"enabled": False}
    result = prepare_fixed_frames(source, canvas_size=28)
    assert torch.equal(result, source) and result.data_ptr() != source.data_ptr()


@pytest.mark.parametrize("raw_frames", [1, 2, 3, 4, 15, 16, 17])
def test_video_grid_checks_entire_chunk_including_odd_padding(raw_frames):
    temporal = (raw_frames + 1) // 2
    audit = assert_qwen_grid(torch.tensor([[temporal, 64, 64]]),
                             temporal * 1024, raw_frames)
    assert audit["visual_tokens"] == temporal * 1024


def test_video_chunk_cannot_be_mislabeled_as_1024_tokens():
    with pytest.raises(AssertionError, match="8192"):
        assert_qwen_grid([8, 64, 64], 1024, raw_frames=16)
    with pytest.raises(AssertionError, match="ceil"):
        assert_qwen_grid([2, 64, 64], 2048, raw_frames=2)
    with pytest.raises(AssertionError, match="spatial"):
        assert_qwen_grid([8, 32, 128], 8192, raw_frames=16)


def test_images_keep_one_full_unit_per_real_frame():
    audit = assert_qwen_grid([[1, 64, 64]] * 4, 4096, raw_frames=4, kind="image")
    assert audit["visual_tokens"] == 4096
    with pytest.raises(AssertionError, match="T=1"):
        assert_qwen_grid([2, 64, 64], 2048, raw_frames=2, kind="image")


@pytest.mark.parametrize("kwargs", [{"canvas_size": 895}, {"canvas_size": 0},
                                    {"canvas_size": True}, {"pad_value": -1},
                                    {"pad_value": 256}])
def test_invalid_profiles_are_rejected(kwargs):
    with pytest.raises(ValueError):
        fixed_input_metadata(**kwargs)


@pytest.mark.parametrize("source", [torch.zeros(1, 12, 20, 3),
                                    torch.zeros(0, 12, 20, 3, dtype=torch.uint8),
                                    torch.zeros(1, 3, 12, 3, dtype=torch.uint8)])
def test_invalid_decoded_frames_are_rejected(source):
    with pytest.raises((TypeError, ValueError)):
        prepare_fixed_frames(source)
