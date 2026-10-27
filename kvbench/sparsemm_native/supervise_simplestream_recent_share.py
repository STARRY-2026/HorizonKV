"""Run SimpleStream with 56 heads viewing two frames and 56 viewing four."""

from pathlib import Path
import shutil
import subprocess
import sys

from sparsemm_native import supervise_simplestream_upstream as base


ROOT = base.ROOT
SOURCE_FILES = (
    ROOT / 'sparsemm_native/simplestream_headmix_attention.py',
    ROOT / 'sparsemm_native/simplestream_recent_share_attention.py',
    ROOT / 'sparsemm_native/supervise_simplestream_recent_share.py',
)
original_prepare = base.prepare
original_worker = base.worker


class SubprocessProxy:
    def __getattr__(self, name):
        return getattr(subprocess, name)

    @staticmethod
    def Popen(command, *args, **kwargs):
        if isinstance(command, (list, tuple)) and '-m' in command:
            command = list(command)
            index = command.index('-m') + 1
            if command[index] == 'sparsemm_native.supervise_simplestream_upstream' and '--worker' in command:
                command[index] = 'sparsemm_native.supervise_simplestream_recent_share'
        return subprocess.Popen(command, *args, **kwargs)


def prepare(output):
    from sparsemm_native.simplestream_recent_share_attention import (
        HEADS, MEDIAN_BRACKET, SOURCE_SHA256, SOURCE_TABLE)

    existed = (output / 'protocol.json').exists()
    protocol = original_prepare(output)
    if existed:
        if (protocol.get('version') != 'simplestream_qwen25_recent_share_56_56_v1'
                or protocol.get('two_frame_heads') != [list(pair) for pair in HEADS]):
            raise RuntimeError('Existing protocol has a different recent-share assignment')
        return protocol
    protocol.update(
        version='simplestream_qwen25_recent_share_56_56_v1',
        experimental_change='56 lower recent-share KV heads cannot directly read oldest two image frames',
        two_frame_heads=[list(pair) for pair in HEADS],
        four_frame_head_count=56, two_frame_head_count=56,
        selection_rule='four-frame K1000 DP recent_r/capacity_n ascending; first 56 use two frames; layer/head tie break',
        median_bracket=list(MEDIAN_BRACKET),
        selection_source=str(SOURCE_TABLE), selection_source_sha256=SOURCE_SHA256,
        per_head_frame_visibility='low-share 56 heads last 2; high-share 56 heads last 4; no historical KV',
        physical_visual_kv='all four input frames remain in the standard KV cache; old two keys are masked in lower-share heads',
        image_processing='unchanged released dynamic resize and AutoProcessor',
        attention_implementation='released FlashAttention plus selected-head SDPA replacement',
        comparison_caveat='Direct attention visibility only; physical KV occupancy and earlier residual-stream mixing still include four frames.',
    )
    protocol['input_sha256'][str(SOURCE_TABLE)] = SOURCE_SHA256
    for source in SOURCE_FILES:
        protocol['code_sha256'][str(source)] = base.sha(source)
        snapshot = output / 'source_snapshot' / ROOT.name / source.relative_to(ROOT)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, snapshot)
    base.write(output / 'protocol.json', protocol)
    return protocol


def worker(args):
    from sparsemm_native.simplestream_recent_share_attention import install, install_question_wrapper

    sys.path.insert(0, str(base.UPSTREAM))
    install()
    install_question_wrapper(Path(args.output))
    return original_worker(args)


if __name__ == '__main__':
    base.subprocess = SubprocessProxy()
    base.prepare = prepare
    base.worker = worker
    base.main()
