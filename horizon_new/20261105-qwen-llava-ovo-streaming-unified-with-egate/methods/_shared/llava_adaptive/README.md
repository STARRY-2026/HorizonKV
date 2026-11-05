# LLaVA-OneVision AdaKV / PyramidKV prototype

Experimental one-sample OVO-Bench smoke; not the full benchmark runner. Uses a manually bridged Qwen2 decoder and the selectors and ragged cache from the existing pinned 20261022 OVO baseline release. Both methods use the same 0.5 fps, FP16, visual K=500 average and a 32-token recent window. This experiment uses SigLIP, the checkpoint's bilinear pooling (stride 2), and 13 video frames as one chunk (maximum 16 frames); FlashAttention variable-length is used when available. The 13-frame sample tests eviction, but does not test multiple subsequent online updates. The hand-built video feature concatenation and prompt have not been proven equivalent to the original LLaVA video template; do not treat these as comparable benchmark scores without further validation. The scripts currently use a one-chunk smoke, not a complete OVO-Bench runner.

The environment requires the local LLaVA-NeXT checkout, baseline package, pinned Transformers 4.40.2 overlay, and a downloaded SigLIP vision encoder. The /tmp installs are ephemeral and need restoration when the container restarts. Results and logs are stored in experiments/horizonkv/llava-adakv-pyramidkv-smoke-20261101.

## Full evaluation update

On 2026-11-01 evaluate.py and launch.py added multi-chunk OVO evaluation; two-chunk 16+3-frame smoke passed for both methods. Full 1468-question runs were launched with four shards per method. Video feature assembly remains a streaming adaptation and has not been proven identical to the original LLaVA video template.
