"""Durable official OVO run with actual two/four-frame head-ragged KV."""

from pathlib import Path
import shutil
import subprocess
import sys

from sparsemm_native import supervise_simplestream_upstream as base


ROOT = base.ROOT
SOURCE_FILES = (
    ROOT / 'sparsemm_native/simplestream_recent_share_attention.py',
    ROOT / 'sparsemm_native/simplestream_recent_share_ragged_gt70.py',
    ROOT / 'sparsemm_native/supervise_simplestream_recent_share_ragged_gt70.py',
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
                command[index] = 'sparsemm_native.supervise_simplestream_recent_share_ragged_gt70'
        return subprocess.Popen(command, *args, **kwargs)


def prepare(output):
    from sparsemm_native.simplestream_recent_share_ragged_gt70 import (
        HEADS, SOURCE_TABLE, SOURCE_SHA256, RAGGED_SOURCE, RAGGED_SHA256)
    existed = (output / 'protocol.json').exists()
    protocol = original_prepare(output)
    if existed:
        if (protocol.get('version') != 'simplestream_recent_share_physical_ragged_gt70_v1'
                or protocol.get('two_frame_heads') != [list(pair) for pair in HEADS]):
            raise RuntimeError('Existing protocol differs from the requested ragged run')
        return protocol
    protocol.update(
        version='simplestream_recent_share_physical_ragged_gt70_v1',
        experimental_change='87 KV heads with recent share at most 70% physically retain last two images before question prefill; remaining 25 retain all four',
        two_frame_heads=[list(pair) for pair in HEADS], two_frame_head_count=87,
        four_frame_head_count=25,
        selection_rule='K1000 recent4 DP: recent_r/capacity_n > 70% keeps four frames; otherwise two frames',
        selection_source=str(SOURCE_TABLE), selection_source_sha256=SOURCE_SHA256,
        physical_visual_kv='ragged K/V physically pruned per head after visual-prefix prefill and before question prefill',
        generation='ragged FlashAttention for question prefill and manual greedy decode with original repetition_penalty/eos',
        prefill_caveat='All four images participate in shared visual-prefix hidden-state prefill; their K/V is pruned before the question text.',
        dense_visual_kv_reference_audit='explicitly release and weakref-verify dense K/V after ragged conversion',
        primary_scorer='unmodified_scoring/score_ovo_bench.py',
        comparison_caveat='Same SimpleStream videos, fps, resize and prompt; custom decode must be checked against unpruned original before score interpretation.',
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
    from sparsemm_native.simplestream_recent_share_ragged_gt70 import install
    sys.path.insert(0, str(base.UPSTREAM))
    install(Path(args.output))
    return original_worker(args)


if __name__ == '__main__':
    base.subprocess = SubprocessProxy()
    base.prepare = prepare
    base.worker = worker
    base.main()
