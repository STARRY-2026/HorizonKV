"""Atomic, reusable Qwen image-embedding cache for fixed-896 OVO inputs."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from safetensors.torch import load_file, save_file


VERSION = 'qwen25_fixed896_image_embeddings_v1'
MIN_FREE_BYTES = 1024 ** 4  # Keep at least 1 TiB available for other work.


def frame_digest(images):
    digest = hashlib.sha256()
    for frame in images:
        digest.update(str((frame.mode, frame.size)).encode())
        digest.update(frame.tobytes())
    return digest.hexdigest()


def identity(model_path, image_processor, fixed_input, upstream_source):
    model_path = Path(model_path)
    index = model_path / 'model.safetensors.index.json'
    shards = sorted(model_path.glob('model-*.safetensors'))
    if not index.exists() or not shards:
        raise FileNotFoundError('Model weight index and shards are required for cache identity')
    return dict(
        version=VERSION,
        model_config_sha256=hashlib.sha256((model_path / 'config.json').read_bytes()).hexdigest(),
        model_index_sha256=hashlib.sha256(index.read_bytes()).hexdigest(),
        # JSON stores arrays, so use lists here too. Otherwise a later process
        # hashes into the same namespace but rejects its identity.json pin.
        model_shards=[[p.name, p.stat().st_size, p.stat().st_mtime_ns] for p in shards],
        image_processor=image_processor.to_dict(),
        fixed_input=fixed_input,
        upstream_encoder_sha256=hashlib.sha256(Path(upstream_source).read_bytes()).hexdigest(),
    )


class PersistentVisionCache:
    def __init__(self, root, cache_identity):
        self.root = Path(root).expanduser().resolve()
        document = json.dumps(cache_identity, sort_keys=True, ensure_ascii=False).encode()
        self.namespace = hashlib.sha256(document).hexdigest()
        self.directory = self.root / self.namespace
        self.directory.mkdir(parents=True, exist_ok=True)
        pin = self.directory / 'identity.json'
        if pin.exists():
            if json.loads(pin.read_text()) != cache_identity:
                raise ValueError('Persistent vision-cache identity mismatch')
        else:
            temporary = self.directory / f'identity.{os.getpid()}.tmp'
            temporary.write_text(json.dumps(cache_identity, ensure_ascii=False, indent=2) + '\n')
            os.replace(temporary, pin)
        self.hits = self.writes = self.bytes_written = 0

    def path(self, frame_digest):
        if len(frame_digest) != 64:
            raise ValueError('Expected SHA-256 frame digest')
        return self.directory / frame_digest[:2] / (frame_digest + '.safetensors')

    def get(self, frame_digest):
        path = self.path(frame_digest)
        if not path.exists():
            return None
        tensors = load_file(str(path), device='cpu')
        embeddings, grid = tensors['embeddings'], tensors['grid']
        if (embeddings.ndim != 2 or embeddings.shape[1] != 3584
                or grid.ndim != 2 or grid.shape[1] != 3
                or embeddings.dtype != __import__('torch').bfloat16):
            raise ValueError(f'Invalid persistent visual embedding: {path}')
        self.hits += 1
        return embeddings, grid

    def put(self, frame_digest, embeddings, grid):
        path = self.path(frame_digest)
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        estimated = embeddings.numel() * embeddings.element_size() + grid.numel() * grid.element_size()
        if shutil.disk_usage(path.parent).free < MIN_FREE_BYTES + estimated:
            raise OSError(f'Vision-cache disk reserve below 1 TiB: {path.parent}')
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.writing-', suffix='.safetensors',
                                         delete=False) as handle:
            temporary = Path(handle.name)
        try:
            save_file({'embeddings': embeddings.detach().cpu().contiguous(),
                       'grid': grid.detach().cpu().contiguous()}, str(temporary))
            with temporary.open('rb') as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        self.writes += 1
        self.bytes_written += path.stat().st_size
