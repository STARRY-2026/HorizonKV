"""Durable OVO run with the reverse independent one/four-frame head split."""

from pathlib import Path
import shutil
import subprocess
import sys

from sparsemm_native import supervise_simplestream_upstream as base


ROOT = base.ROOT
SOURCE_FILES = (
    ROOT / 'sparsemm_native/simplestream_recent_share_attention.py',
    ROOT / 'sparsemm_native/simplestream_recent_share_independent_last1_reverse.py',
    ROOT / 'sparsemm_native/supervise_simplestream_recent_share_independent_last1_reverse.py',
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
                command[index] = 'sparsemm_native.supervise_simplestream_recent_share_independent_last1_reverse'
        return subprocess.Popen(command, *args, **kwargs)


def prepare(output):
    from sparsemm_native.simplestream_recent_share_independent_last1_reverse import (
        HEADS, SOURCE_TABLE, SOURCE_SHA256, RAGGED_SOURCE, RAGGED_SHA256)
    existed = (output / 'protocol.json').exists()
    protocol = original_prepare(output)
    if existed:
        if (protocol.get('version') != 'simplestream_recent_share_independent_last1_reverse_v1'
                or protocol.get('one_frame_heads') != [list(pair) for pair in HEADS]):
            raise RuntimeError('Existing protocol differs from the requested ragged run')
        return protocol
    protocol.update(
        version='simplestream_recent_share_independent_last1_reverse_v1',
        experimental_change='56 highest recent-share KV heads use independently recomputed last-one-image prefix; remaining 56 use four-image prefix',
        one_frame_heads=[list(pair) for pair in HEADS], one_frame_head_count=56,
        four_frame_head_count=56,
        selection_rule='K1000 recent4 DP recent_r/capacity_n ascending; highest 56 one-frame, lowest 56 four-frame',
        selection_source=str(SOURCE_TABLE), selection_source_sha256=SOURCE_SHA256,
        physical_visual_kv='one-frame head K/V comes from an independent one-image prefill; four-frame head K/V comes from four-image prefill; combined before question prefill',
        generation='ragged FlashAttention for question prefill and manual greedy decode with original repetition_penalty/eos',
        prefill_caveat='The one-image branch sees no older image embeddings, but the shared question suffix fuses information from one-image and four-image heads.',
        position_alignment='single-image prefix uses positions sliced from the original four-image prompt to align with the shared question and four-image branch',
        dense_visual_kv_reference_audit='release and weakref-verify both dense prefix caches after ragged combination',
        primary_scorer='unmodified_scoring/score_ovo_bench.py',
        comparison_caveat='Same SimpleStream videos, fps, resize and prompt; additional one-image prefill raises compute cost and is a separate intervention from cache pruning.',
    )
    protocol['input_sha256'][str(SOURCE_TABLE)] = SOURCE_SHA256
    protocol['code_sha256'][str(RAGGED_SOURCE)] = RAGGED_SHA256
    for source in SOURCE_FILES:
        protocol['code_sha256'][str(source)] = base.sha(source)
    for source in (*SOURCE_FILES, RAGGED_SOURCE):
        snapshot = output / 'source_snapshot' / source.parent.parent.name / source.relative_to(source.parent.parent)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, snapshot)
    base.write(output / 'protocol.json', protocol)
    return protocol


def worker(args):
    from sparsemm_native.simplestream_recent_share_independent_last1_reverse import install
    sys.path.insert(0, str(base.UPSTREAM))
    install(Path(args.output))
    return original_worker(args)


if __name__ == '__main__':
    base.subprocess = SubprocessProxy()
    base.prepare = prepare
    base.worker = worker
    base.main()
