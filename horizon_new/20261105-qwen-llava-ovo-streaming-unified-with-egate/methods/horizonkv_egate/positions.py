"""Stable frame/row/column identifiers independent of visual token count.

Only coordinates of currently retained KV and the current <=4-frame batch
are kept. A layout is metadata, never a union of physically stored head KV.
"""
AXIS = 1024
FRAME_STRIDE = AXIS * AXIS


def token_ids(frame, rows, cols):
    if frame < 0 or not (0 < rows <= AXIS and 0 < cols <= AXIS):
        raise ValueError('Invalid visual grid')
    return [frame * FRAME_STRIDE + r * AXIS + c for r in range(rows) for c in range(cols)]


def frame_of(uid):
    return uid // FRAME_STRIDE


def position_values(ids, prefix_length, metadata, model):
    bases = {}
    cursor = prefix_length
    for frame, (rows, cols) in sorted(metadata.items()):
        bases[frame] = cursor
        cursor += max(rows, cols) if model == 'qwen25vl' else rows * cols
    output = [[], [], []]
    for uid in ids:
        if uid < 0:
            value = uid + prefix_length
            if not 0 <= value < prefix_length:
                raise ValueError('Invalid prefix identifier')
            xyz = [value] * 3
        else:
            frame, patch = divmod(uid, FRAME_STRIDE)
            row, col = divmod(patch, AXIS)
            rows, cols = metadata[frame]
            if row >= rows or col >= cols:
                raise ValueError('Identifier does not fit its frame grid')
            base = bases[frame]
            xyz = [base, base + row, base + col] if model == 'qwen25vl' else [base + row * cols + col] * 3
        for axis in range(3):
            output[axis].append(xyz[axis])
    return output


def positions(ids, prefix_length, metadata, model, device):
    import torch
    return torch.tensor(position_values(ids, prefix_length, metadata, model), device=device, dtype=torch.long)


def retained_metadata(heads, current_metadata):
    frames = {frame_of(uid) for entry in heads.values() for uid in entry['uids']}
    return {frame: current_metadata[frame] for frame in sorted(frames)}
