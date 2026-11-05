"""Capabilities verified in the existing adapters, not claims of numerical parity."""
BASELINE_METHODS = ('adakv', 'pyramidkv', 'sparsemm', 'lukv', 'hermes', 'fullkv')
METHODS = BASELINE_METHODS + ('horizonkv-egate', 'h2svc')
MODELS = ('qwen25vl', 'llavaov')
BENCHMARKS = ('ovobench', 'streamingbench')
BACKENDS = ('qwen', 'qwen_full2495', 'llava', 'egate', 'h2svc')
LLAVA_EQUIVALENT_CHOICES = frozenset(('sparsemm', 'lukv', 'hermes', 'fullkv', 'horizonkv-egate', 'h2svc'))
INPUT_PROFILES = {
    'qwen25vl': dict(name='dscache448-video', fps=1, ordinary_target_pixels=448**2,
                      latest_target_pixels=896**2, chunk_frames=2,
                      note='Experimental latest-frame replay; DSCache temporal pairing parity is not established.'),
    'llavaov': dict(name='dscache', fps=1, frame_size=[384, 384],
                   ordinary_visual_tokens=196, latest_visual_tokens=729),
}
