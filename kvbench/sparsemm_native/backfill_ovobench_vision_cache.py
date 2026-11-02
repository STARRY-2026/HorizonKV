"""Fill missing fixed-896 visual embeddings for an official OVO manifest.

This performs visual encoding only, with the exact sampling, one-second
bucket grouping, image preprocessing and frame fingerprint used by the online
history runner. It does not answer questions or alter scored results.
"""

import argparse
import json
from pathlib import Path
import sys

from sparsemm_native import run_simplestream_k500_fixed896 as fixed
from sparsemm_native import run_simplestream_k500_input_only as old
from sparsemm_native.run_simplestream_dscache_history64 import check_upstream_pin
from sparsemm_native.gpu_slots import gpu_lease
from sparsemm_native.persistent_vision_cache import (
    PersistentVisionCache, frame_digest, identity,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    check_upstream_pin()
    manifest, _ = old.validate_manifest(args.manifest.resolve())
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    progress = output / 'completed.jsonl'
    done = [json.loads(line) for line in progress.open() if line.strip()] if progress.exists() else []
    if [row['query_id'] for row in done] != [q['query_id'] for q in manifest[:len(done)]]:
        raise ValueError('Backfill progress is not a prefix of the frozen manifest')
    if len(done) == len(manifest):
        return
    sys.path.insert(0, str(old.base.UPSTREAM))
    from lib import recent_window_eval_qwen25 as upstream
    from lib.qwen_exact_recent_decoder import _build_sampling_plan, _decode_indices
    from PIL import Image
    import torch
    with gpu_lease(output):
        qa = upstream.RecentWindowQAModel(str(old.base.MODEL), device='cuda',
                                          max_new_tokens=old.MAX_NEW,
                                          attn_implementation='flash_attention_2')
        fixed.install_fixed_encoder(qa)
        source = old.base.UPSTREAM / 'lib/recent_window_eval_qwen25.py'
        cache = PersistentVisionCache(args.cache_root, identity(
            qa.model.config._name_or_path, qa.processor.image_processor,
            fixed.fixed_input_metadata(fixed.SIZE, fixed.PAD), source))
        for index, query in enumerate(manifest[len(done):], start=len(done)):
            clip = old.base.DATA / 'chunked_videos' / f"{query['source_id']}.mp4"
            plan = _build_sampling_plan({'video': str(clip), 'fps': fixed.FPS}, last_nframes=4)
            selected = plan.full_indices[:-4] + plan.tail_indices
            if not selected:
                raise RuntimeError(f'No sampled frames: {clip}')
            start_hits, start_writes = cache.hits, cache.writes
            for batch_start in range(0, len(selected), 16):
                raw_indices = selected[batch_start:batch_start + 16]
                frames = _decode_indices(str(clip), plan.backend, raw_indices)
                for raw_index, frame in zip(raw_indices, frames):
                    image = Image.fromarray(frame.clamp(0, 255).to(torch.uint8)
                                            .permute(1, 2, 0).cpu().numpy())
                    key = frame_digest([image])
                    if cache.path(key).is_file():
                        cache.hits += 1
                    else:
                        cache.put(key, *qa.encode_vision([image]))
            record = dict(query_id=query['query_id'], source_id=query['source_id'],
                          sampled_frames=len(selected), disk_hits=cache.hits-start_hits,
                          disk_writes=cache.writes-start_writes,
                          namespace=cache.namespace)
            old.base.append(progress, record)
            done.append(record)
            old.base.write(output / 'status.json', dict(done=len(done), total=len(manifest),
                                                        writes=cache.writes, hits=cache.hits,
                                                        bytes_written=cache.bytes_written,
                                                        namespace=cache.namespace))
            print(f'{len(done)}/{len(manifest)} {query["query_id"]} '
                  f'write={record["disk_writes"]} hit={record["disk_hits"]}', flush=True)


if __name__ == '__main__':
    main()
