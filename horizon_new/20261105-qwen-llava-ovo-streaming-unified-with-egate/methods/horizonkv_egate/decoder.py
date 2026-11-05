"""Manual decoder using the same ragged GQA operation as bundled baselines.

Raw keys are saved at projection time, avoiding inverse-RoPE roundoff.
No query, question text, or answer is retained in the persistent video state.
"""
import torch
from sparsemm_baselines.cache import RaggedCache
from .attention import ragged_attention, flash_attn_varlen_func


class Decoder:
    def __init__(self, model, kind):
        self.model, self.kind = model, kind
        self.lm = model.language_model if kind == 'qwen25vl' else model.get_model()
        self.cfg = self.lm.config
        self.nq = self.cfg.num_attention_heads
        self.nk = self.cfg.num_key_value_heads
        self.dim = self.cfg.hidden_size // self.nq
        self.nl = len(self.lm.layers)
        self.device = model.device
        self.dtype = next(model.parameters()).dtype
        self.position_limit = self.cfg.max_position_embeddings
        if (self.nl, self.nk) != (28, 4):
            raise ValueError('Frozen allocation requires exactly 28 layers x 4 KV heads')
        if flash_attn_varlen_func is None:
            raise RuntimeError('EGate GPU worker requires FlashAttention varlen')
        if kind == 'qwen25vl':
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb
            self.apply_rope = apply_multimodal_rotary_pos_emb
        else:
            from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
            self.apply_rope = apply_rotary_pos_emb

    def new_cache(self):
        return RaggedCache(self.nl, self.nk, self.dim, self.device, self.dtype)

    def rotate(self, tensor, pos, layer_index):
        if pos.shape != (3, tensor.shape[2]):
            raise ValueError('Rotation shape mismatch')
        if not tensor.shape[2]:
            return tensor
        if self.kind == 'qwen25vl':
            dummy = tensor.new_empty((1, tensor.shape[2], self.cfg.hidden_size))
            cos, sin = self.lm.rotary_emb(dummy, pos[:, None])
            return self.apply_rope(tensor, tensor, cos, sin, self.cfg.rope_scaling['mrope_section'])[0]
        att = self.lm.layers[layer_index].self_attn
        cos, sin = att.rotary_emb(tensor, seq_len=int(pos.max()) + 1)
        return self.apply_rope(tensor, tensor, cos, sin, pos[:1])[0]

    @torch.inference_mode()
    def forward(self, embeddings, pos, cache, capture=False, observer=None):
        hidden = embeddings
        n = hidden.shape[1]
        if pos.shape != (3, n):
            raise ValueError('Decoder position shape mismatch')
        raw_keys, probes = [], []
        for i, layer in enumerate(self.lm.layers):
            residual = hidden
            z = layer.input_layernorm(hidden)
            att = layer.self_attn
            q = att.q_proj(z).view(1, n, self.nq, self.dim).transpose(1, 2)
            k = att.k_proj(z).view(1, n, self.nk, self.dim).transpose(1, 2)
            v = att.v_proj(z).view(1, n, self.nk, self.dim).transpose(1, 2)
            if capture:
                raw_keys.append(k.detach().clone())
                probes.append(q[:, :, -32:].detach().clone())
            q, k = self.rotate(q, pos, i), self.rotate(k, pos, i)
            old = cache.layers[i]
            if observer is not None:
                observer(i, q, k, old, self.dim**-0.5)
            attended = ragged_attention(q, k, v, old, self.dim**-0.5)
            old.append_chunk(k, v, pos)
            hidden = residual + att.o_proj(attended.contiguous())
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        return self.lm.norm(hidden), raw_keys, probes

    @torch.inference_mode()
    def parity_check(self, tokenizer):
        ids = tokenizer('A short decoder parity check.', return_tensors='pt', add_special_tokens=False).input_ids.to(self.device)
        embeddings = self.model.get_input_embeddings()(ids)
        seq = torch.arange(ids.shape[1], device=self.device)[None].expand(3, -1)
        if self.kind == 'qwen25vl':
            expected = self.lm(inputs_embeds=embeddings, position_ids=seq[:, None], use_cache=False, return_dict=True).last_hidden_state
        else:
            expected = self.lm(inputs_embeds=embeddings, position_ids=seq[:1], use_cache=False, return_dict=True).last_hidden_state
        actual, _, _ = self.forward(embeddings, seq, self.new_cache())
        delta = (actual.float() - expected.float()).abs()
        report = dict(max_abs=float(delta.max()), mean_abs=float(delta.mean()), flash_attention=True)
        if not torch.allclose(actual.float(), expected.float(), atol=0.1, rtol=0.05):
            raise RuntimeError('EGate decoder parity failed: ' + str(report))
        print('DECODER_PARITY', report, flush=True)
        return report
