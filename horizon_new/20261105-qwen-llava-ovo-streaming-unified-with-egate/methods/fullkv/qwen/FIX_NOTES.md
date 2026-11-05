# PPU FlashAttention 2 M-RoPE compatibility fix

Base release: `20261014-ppu-dp500-v5-8way`.

Qwen2.5-VL passes 3D M-RoPE position IDs as `[3, batch, sequence]`. With the
installed Transformers 4.57 runtime, `_is_packed_sequence` interprets these
IDs as packed examples. During incremental decoding with an existing KV cache,
FlashAttention 2 then uses the wrong attention span and produces empty or
nonsensical answers. This release changes only the packed-sequence predicate
for 3D position IDs; 2D behavior and HERMES selection/compression are retained.

Evidence on 2026-10-15:
- Original full 498-video run: 1/2495 correct, 1672 empty answers.
- Text-only cached QA before fix: empty answer; after fix: `B) 4`.
- One-video smoke after fix: valid `D)` answer, correct, 4 compression events.
- Five-video / 25-question smoke after fix: 25/25 valid A-D choices,
  20/25 correct, 48 compression events.

Set `HORIZONKV_ATTENTION_BACKEND=flash_attention_2` for this release.
The prior eager-QA workaround is not used: eager attention produces NaN logits
on this PPU runtime even in standard model generation.
