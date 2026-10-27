"""Position-only EPM probes for the recent-only video frontend.

This module does not load a model or encode any frames.  ``position_variant``
keeps the exact feature tensors supplied by the caller and returns replacement
``VisualUnit`` objects, leaving the persistent units untouched.

The legacy variant reconstructs the *position layout* of the old SimpleStream
continuous-image prefix.  That serialization has one vision-start token for
all image grids.  Qwen's get_rope_index consequently gives only the first grid
3-D image coordinates; the remaining visual tokens receive consecutive 1-D
coordinates.  Here a current two-frame-fused video unit stands in for each
image grid.  This is not the old per-image visual encoder or a full restoration
of the old four-image input.  The independently encoded one-frame candidate
uses the coordinate slice of the final video unit; its features are unchanged.
"""

from dataclasses import replace

import torch


VARIANTS = ('native', 'simplestream_contiguous')


def _validate_units(one_unit, units, merge_size):
    if not 1 <= len(units) <= 2:
        raise ValueError('A recent-only probe requires one or two video units')
    for unit in [one_unit, *units]:
        t, h, w = map(int, unit.grid)
        if (t != 1 or h % merge_size or w % merge_size
                or h * w // merge_size ** 2 != unit.tokens
                or unit.positions.shape != (3, 1, unit.tokens)):
            raise ValueError('Probe units must each contain one complete spatial grid')
    if (one_unit.tokens != units[-1].tokens
            or tuple(one_unit.grid) != tuple(units[-1].grid)):
        raise ValueError('One-frame candidates must match the final video unit grid')


def position_variant(bridge, one_unit, units, variant):
    """Return ``(one_unit_copy, unit_copies, question_position)``.

    ``native`` reproduces the production cursor exactly, including any larger
    historical absolute cursor stored in ``bridge._next_position``.  The other
    variant calls the model's actual ``get_rope_index`` on the old single-start
    serialization, with one image-grid row per current video unit.  The returned
    cursor is the position of vision-end, i.e. the first suffix token.  Use that
    *same* cursor for the actual question suffix and the fixed pseudoquery if
    testing the matching-coordinate probe variant.

    The caller must use the returned unit list both for temporary prefills and
    position metadata during cache assembly.  It should not assign these copies
    back to a persistent cache.  This function changes no model, tokenizer,
    features, RGB, question text, allocation, or selection implementation.
    """
    if variant not in VARIANTS:
        raise ValueError(f'Unknown position variant: {variant}')
    units = list(units)
    merge_size = int(bridge.model_config.vision_config.spatial_merge_size)
    _validate_units(one_unit, units, merge_size)
    if variant == 'native':
        return (replace(one_unit, positions=one_unit.positions.clone()),
                [replace(unit, positions=unit.positions.clone()) for unit in units],
                max(bridge._next_position, int(one_unit.positions.max()) + 1))

    prefix_ids = bridge.init_prompt_ids
    prefix = int(bridge.visual_start_idx)
    if prefix_ids.ndim != 2 or prefix_ids.shape != (1, prefix):
        raise ValueError('A single unchanged common text prefix is required')
    config = bridge.model_config
    start_id = int(config.vision_start_token_id)
    image_id = int(config.image_token_id)
    end_id = int(config.vision_end_token_id)
    if (int(prefix_ids[0, -1]) != start_id
            or int((prefix_ids == start_id).sum()) != 1):
        raise ValueError('Legacy serialization requires exactly one final vision-start')
    total = sum(unit.tokens for unit in units)
    input_ids = torch.cat((
        prefix_ids,
        torch.full((1, total), image_id, device=prefix_ids.device, dtype=prefix_ids.dtype),
        torch.tensor([[end_id]], device=prefix_ids.device, dtype=prefix_ids.dtype),
    ), dim=1)
    grids = torch.tensor([unit.grid for unit in units],
                         device=prefix_ids.device, dtype=torch.long)
    multimodal = bridge.model.model
    positions, _ = multimodal.get_rope_index(
        input_ids=input_ids, image_grid_thw=grids, video_grid_thw=None,
        attention_mask=torch.ones_like(input_ids))
    if positions.shape != (3, 1, prefix + total + 1):
        raise AssertionError('Unexpected legacy RoPE result shape')
    expected_prefix = (torch.arange(prefix, device=positions.device)
                       .view(1, 1, -1).expand(3, 1, -1))
    if not torch.equal(positions[:, :, :prefix], expected_prefix):
        raise AssertionError('Legacy layout changed the immutable text prefix')
    cursor = positions[:, 0, -1]
    if not bool((cursor == cursor[0]).all()):
        raise AssertionError('The first suffix token must have a shared 1-D position')
    start = prefix
    replacements = []
    for unit in units:
        end = start + unit.tokens
        replacements.append(replace(unit, positions=positions[:, :, start:end].clone()))
        start = end
    # Old one/four prefills share coordinates: select the final candidate's
    # region from the full layout, rather than restarting the one-frame branch.
    one = replace(one_unit, positions=replacements[-1].positions.clone())
    return one, replacements, int(cursor[0])
