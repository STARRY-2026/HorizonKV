"""Run a separate SimpleStream ablation with exact visual tokens per frame.

The released SimpleStream files and the completed original-setting run remain
untouched. This wrapper changes only the image preparation in worker processes.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from sparsemm_native import supervise_simplestream_upstream as original

ROOT = Path('/home/yujun/program/kvbench')
SOURCE_FILES = (
    ROOT / 'sparsemm_native/fixed_visual_tokens.py',
    ROOT / 'sparsemm_native/supervise_simplestream_fixed_frames.py',
)
DEFAULT_TOKENS = 512
original_prepare = original.prepare
original_worker = original.worker


def frame_tokens() -> int:
    value = os.environ.get('OVO_FIXED_FRAME_TOKENS')
    if value is None:
        # A manually restarted worker can recover the configured value.
        arguments = sys.argv
        if '--output' in arguments:
            output = Path(arguments[arguments.index('--output') + 1])
            config = output.parent.parent / 'fixed_frame_config.json'
            if config.exists():
                value = str(json.loads(config.read_text())['visual_tokens_per_frame'])
    tokens = int(value) if value is not None else DEFAULT_TOKENS
    from sparsemm_native.fixed_visual_tokens import canvas_shape
    canvas_shape(tokens, 16, 9)
    return tokens


class SubprocessProxy:
    def __getattr__(self, name):
        return getattr(subprocess, name)

    @staticmethod
    def Popen(command, *args, **kwargs):
        if isinstance(command, (list, tuple)) and '-m' in command:
            index = command.index('-m') + 1
            command = list(command)
            if command[index] == 'sparsemm_native.download_ovobench_chunks':
                command[index] = 'sparsemm_native.download_ovobench_chunks_resumable'
            elif command[index] == 'sparsemm_native.supervise_simplestream_upstream' and '--worker' in command:
                command[index] = 'sparsemm_native.supervise_simplestream_fixed_frames'
        return subprocess.Popen(command, *args, **kwargs)


def prepare_fixed(output: Path):
    tokens = frame_tokens()
    config_path = output / 'fixed_frame_config.json'
    if config_path.exists():
        config = json.loads(config_path.read_text())
        if config['visual_tokens_per_frame'] != tokens:
            raise RuntimeError('Existing fixed-frame token configuration differs')
    else:
        output.mkdir(parents=True, exist_ok=True)
        original.write(config_path, dict(visual_tokens_per_frame=tokens, recent_frames=4,
                    expected_vision_tokens_per_query=4 * tokens, pixel_grid=28,
                    resize='preserve_aspect_ratio_then_gray_letterbox'))

    protocol_path = output / 'protocol.json'
    existing = protocol_path.exists()
    protocol = original_prepare(output)
    if existing:
        if protocol.get('fixed_vision_tokens_per_frame') != tokens:
            raise RuntimeError('Existing protocol is not the requested fixed-frame variant')
        return protocol

    protocol.update(
        version='simplestream_released_qwen25_fixed_frame_vision_v1',
        fixed_vision_tokens_per_frame=tokens,
        expected_vision_tokens_per_query=4 * tokens,
        image_processing=f'original qwen_vl_utils decode; aspect-preserving gray letterbox to exact {tokens} tokens/frame before original AutoProcessor',
        comparison_caveat='The fixed-frame letterbox changes the original SimpleStream visual input; compare as a separate ablation, not as the unmodified upstream result.',
    )
    protocol['experimental_change'] = 'exact per-frame visual-token control in a separate ablation'
    for source in SOURCE_FILES:
        protocol['code_sha256'][str(source)] = original.sha(source)
        snapshot = output / 'source_snapshot' / ROOT.name / source.relative_to(ROOT)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, snapshot)
    original.write(protocol_path, protocol)
    return protocol


def worker_fixed(args):
    from sparsemm_native.fixed_visual_tokens import install
    sys.path.insert(0, str(original.UPSTREAM))
    install(frame_tokens())
    return original_worker(args)


def audit_completed(output: Path, tokens: int):
    if not (output / 'summary.json').exists():
        return
    summary = json.loads((output / 'summary.json').read_text())
    if not summary.get('complete'):
        return
    counts = []
    for shard in range(2):
        path = output / f'shards/shard{shard}/query_observations.jsonl'
        for line in path.read_text().splitlines():
            record = json.loads(line)
            counts.append((record['result']['num_frames'], record['result']['num_vision_tokens']))
    expected = (4, 4 * tokens)
    if len(counts) != 3035 or any(value != expected for value in counts):
        raise RuntimeError('Fixed visual-token audit failed')
    original.write(output / 'fixed_frame_audit.json',
                   dict(passed=True, queries=len(counts), visual_tokens_per_frame=tokens,
                        vision_tokens_per_query=4 * tokens))


if __name__ == '__main__':
    tokens = frame_tokens()
    original.subprocess = SubprocessProxy()
    original.prepare = prepare_fixed
    original.worker = worker_fixed
    original.main()
    if '--worker' not in sys.argv and '--output' in sys.argv:
        audit_completed(Path(sys.argv[sys.argv.index('--output') + 1]), tokens)
