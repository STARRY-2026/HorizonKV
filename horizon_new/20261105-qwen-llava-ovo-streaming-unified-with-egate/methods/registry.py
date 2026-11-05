"""Local method paths. Never resolve method implementations from old releases."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHODS_ROOT = PROJECT_ROOT / 'methods'
SHARED_KVBENCH = METHODS_ROOT / '_shared' / 'kvbench'
LLAVA_BACKBONE = PROJECT_ROOT / 'backbones' / 'llava_next'
SPARSEMM_SCORES = METHODS_ROOT / 'sparsemm' / 'assets' / 'head_scores.csv'
LUKV_PROFILE = METHODS_ROOT / 'lukv' / 'assets' / 'attention_native50v_20261023' / 'profile.json'
METHOD_NAMES = ('adakv', 'pyramidkv', 'sparsemm', 'lukv', 'hermes', 'fullkv')
ALL_METHOD_NAMES = METHOD_NAMES + ('horizonkv-egate', 'h2svc')
EGATE_ROOT = METHODS_ROOT / 'horizonkv_egate'
H2SVC_ROOT = METHODS_ROOT / 'h2svc'


def method_root(method):
    if method not in ALL_METHOD_NAMES:
        raise ValueError('Unsupported local method: ' + method)
    return EGATE_ROOT if method == 'horizonkv-egate' else H2SVC_ROOT if method == 'h2svc' else METHODS_ROOT / method


def llava_impl(method):
    if method not in METHOD_NAMES:
        raise ValueError('Unsupported local method: ' + method)
    return METHODS_ROOT / method / 'llava'


def qwen_impl(method):
    if method not in ('hermes', 'fullkv'):
        raise ValueError('Dense Qwen implementation requested for ' + method)
    return METHODS_ROOT / method / 'qwen'


def local_python_paths():
    return [PROJECT_ROOT, LLAVA_BACKBONE, SHARED_KVBENCH,
            METHODS_ROOT / 'lukv', METHODS_ROOT / 'sparsemm', qwen_impl('hermes')]
