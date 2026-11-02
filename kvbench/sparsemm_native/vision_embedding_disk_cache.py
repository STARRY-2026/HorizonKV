"""Exact, reusable on-disk cache for fixed-input Qwen visual embeddings.

The cache key contains the model/preprocessing namespace and the actual RGB
frames passed to the visual encoder. Files are written atomically and loaded
as tensors only, so a partial write cannot become a cache hit.
"""

import atexit
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch


def install(qa, *, cache_root, namespace, stats_path=None):
    cache_root = Path(cache_root).resolve() / namespace
    cache_root.mkdir(parents=True, exist_ok=True)
    original = qa.encode_vision
    stats = Counter()

    def persist_stats():
        if stats_path is None:
            return
        path = Path(stats_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(namespace=namespace, cache_root=str(cache_root),
                                        **stats), indent=2) + '\n')

    atexit.register(persist_stats)

    def encode(frames):
        digest = hashlib.sha256()
        digest.update(namespace.encode())
        for frame in frames:
            rgb = frame.convert('RGB')
            digest.update(str(rgb.size).encode())
            digest.update(rgb.tobytes())
        key = digest.hexdigest()
        path = cache_root / key[:2] / f'{key}.pt'
        if path.exists():
            try:
                payload = torch.load(path, map_location='cpu', weights_only=True)
                embeddings, grid = payload['embeddings'], payload['grid']
                if (payload['key'] != key or not isinstance(embeddings, torch.Tensor)
                        or not isinstance(grid, torch.Tensor)
                        or embeddings.ndim != 2 or grid.ndim != 2
                        or int(grid.shape[0]) != len(frames)):
                    raise ValueError('Invalid visual cache entry')
                stats['hits'] += 1
                return embeddings.to(qa.model.device), grid.to(qa.model.device)
            except (OSError, RuntimeError, KeyError, TypeError, ValueError):
                stats['invalid_entries'] += 1
        embeddings, grid = original(frames)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f'.{key}.', suffix='.tmp', dir=path.parent)
        os.close(fd)
        try:
            torch.save(dict(key=key, embeddings=embeddings.detach().cpu(),
                            grid=grid.detach().cpu()), temporary)
            os.replace(temporary, path)
            stats['writes'] += 1
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        stats['misses'] += 1
        return embeddings, grid

    qa.encode_vision = encode
    return stats
