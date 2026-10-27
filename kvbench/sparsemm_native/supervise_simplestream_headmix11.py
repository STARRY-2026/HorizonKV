"""Run SimpleStream dynamic-resolution recent-four with 11 two-frame KV heads."""

import json
from pathlib import Path
import shutil
import subprocess
import sys

from sparsemm_native import supervise_simplestream_upstream as base


ROOT = base.ROOT
SOURCE_FILES = (
    ROOT / 'sparsemm_native/simplestream_headmix_attention.py',
    ROOT / 'sparsemm_native/supervise_simplestream_headmix11.py',
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
                command[index] = 'sparsemm_native.supervise_simplestream_headmix11'
        return subprocess.Popen(command, *args, **kwargs)


def prepare(output):
    from sparsemm_native.simplestream_headmix_attention import HEADS, SOURCE_SHA256, SOURCE_TABLE

    existed = (output / 'protocol.json').exists()
    protocol = original_prepare(output)
    if existed:
        if (protocol.get('version') != 'simplestream_qwen25_dynamic_recent4_headmix11_v1'
                or protocol.get('selected_heads') != [list(pair) for pair in HEADS]):
            raise RuntimeError('Existing protocol is not the 11-head dynamic-resolution variant')
        return protocol
    protocol.update(
        version='simplestream_qwen25_dynamic_recent4_headmix11_v1',
        experimental_change='eleven KV heads block direct attention to oldest two of four image frames',
        selected_heads=[list(pair) for pair in HEADS], selected_head_count=11,
        selection_rule='old_DP_recent32_candidates_then_largest_history_quota_layer_head_ties',
        selection_source=str(SOURCE_TABLE), selection_source_sha256=SOURCE_SHA256,
        per_head_frame_visibility='11 heads last 2, remaining 101 heads last 4; no historical KV',
        image_processing='unchanged released dynamic resize and AutoProcessor',
        token_budget='unchanged four-frame visual input; old two-frame keys masked only in 11 heads',
        attention_implementation='released FlashAttention for all heads plus selected-head SDPA replacement',
        comparison_caveat='This is a per-head direct-attention ablation against the completed dynamic-resolution four-frame run. Old visual information can still flow through residual streams and other heads.',
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
    from sparsemm_native.simplestream_headmix_attention import install, install_question_wrapper

    sys.path.insert(0, str(base.UPSTREAM))
    install()
    install_question_wrapper(Path(args.output))
    return original_worker(args)


if __name__ == '__main__':
    base.subprocess = SubprocessProxy()
    base.prepare = prepare
    base.worker = worker
    base.main()
