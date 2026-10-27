"""Two/four-frame direct visibility, split by the K=1000 recent share.

This is an attention-visibility experiment. All four image frames are still
physically present in the standard Qwen KV cache during fresh prefill.
"""

import csv
import hashlib
from pathlib import Path

from sparsemm_native import simplestream_headmix_attention as base


SOURCE_TABLE = Path(__file__).resolve().parents[2] / (
    'hermes/results/myobservations/lukv_samecalib_two_bank_k1000_20261022/'
    'per_head_allocations.csv')
SOURCE_SHA256 = hashlib.sha256(SOURCE_TABLE.read_bytes()).hexdigest()


def select_heads():
    with SOURCE_TABLE.open(newline='') as handle:
        rows = [row for row in csv.DictReader(handle) if int(row['frames']) == 4]
    if len(rows) != 112 or len({(int(r['layer']), int(r['kv_head'])) for r in rows}) != 112:
        raise ValueError('Expected 112 distinct four-frame KV heads')
    ranked = sorted(rows, key=lambda row: (
        int(row['recent_r']) / int(row['capacity_n']), int(row['layer']), int(row['kv_head'])))
    return tuple((int(r['layer']), int(r['kv_head'])) for r in ranked[:56]), (
        int(ranked[55]['recent_r']) / int(ranked[55]['capacity_n']),
        int(ranked[56]['recent_r']) / int(ranked[56]['capacity_n']))


HEADS, MEDIAN_BRACKET = select_heads()
BY_LAYER = {layer: tuple(head for l, head in HEADS if l == layer)
            for layer in sorted({l for l, _ in HEADS})}


def install():
    """Use the existing audited head mask with this experiment's fixed head set."""
    base.HEADS = HEADS
    base.BY_LAYER = BY_LAYER
    base.install()


def install_question_wrapper(output):
    base.install_question_wrapper(output)
