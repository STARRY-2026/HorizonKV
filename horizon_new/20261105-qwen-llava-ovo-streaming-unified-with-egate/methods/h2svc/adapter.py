"""H2SVC model adapter using the project's fixed visual input protocol.

The visual processor, ragged decoder, and compact position mapping are shared
with the unified EGate adapter. Only the head-classified KV policy is new.
"""
import torch

from methods.horizonkv_egate.adapter import Adapter as VisualInputAdapter, choose_letter
from methods.horizonkv_egate.decoder import Decoder
from methods.horizonkv_egate.policy import Branch
from methods.horizonkv_egate.protocol import format_mc_question
from .state import H2State


class Adapter:
    _features = VisualInputAdapter._features

    def __init__(self, model, tokenizer, processor, kind, routes):
        self.model, self.tokenizer, self.processor, self.kind = model, tokenizer, processor, kind
        self.decoder = Decoder(model, kind)
        self.routes = routes
        self.parity = self.decoder.parity_check(tokenizer)
        self.high_processor = None
        self.clear_cache()

    @torch.inference_mode()
    def clear_cache(self):
        prefix = ('<|im_start|>user\n<|vision_start|>' if self.kind == 'qwen25vl' else
                  '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n')
        ids = self.tokenizer(prefix, return_tensors='pt', add_special_tokens=False).input_ids.to(self.decoder.device)
        branch = Branch(self.decoder, self.model.get_input_embeddings()(ids))
        self.state = H2State(branch, self.routes)
        self.latest_pre_state = None
        self.latest_frames = None
        self.latest_high_features = None
        self.input_audit = []
        self.last_audit = None

    @property
    def max_input_frames(self):
        return 2 if self.kind == 'qwen25vl' else 1

    @torch.inference_mode()
    def encode_video_chunk(self, frames):
        if not 1 <= len(frames) <= self.max_input_frames:
            raise ValueError('Invalid visual input unit')
        self.latest_pre_state = self.state.clone()
        self.latest_frames = frames.detach().to(self.decoder.device).clone()
        features, grid, high_features = self._features(frames)
        self.state.append(features, grid, len(frames))
        self.latest_high_features = high_features

    @torch.inference_mode()
    def question_answering(self, question):
        if self.latest_pre_state is None:
            raise RuntimeError('No visible frames')
        saved = self.state.fingerprint()
        temporary = self.latest_pre_state.clone()
        frames = self.latest_frames
        if len(frames) == 2:
            first, grid, _ = self._features(frames[:1])
            temporary.append(first, grid, 1)
        if self.kind == 'llavaov':
            features, grid = self.latest_high_features, (27, 27)
        else:
            features, grid, _ = self._features(frames[-1:], high=True)
        temporary.append(features, grid, 1)
        cache, start = temporary.make_cache()
        text, _ = format_mc_question(question['question'], question['choices'])
        suffix = ('<|vision_end|>\n' if self.kind == 'qwen25vl' else '\n') + text
        suffix += '<|im_end|>\n<|im_start|>assistant\n'
        ids = self.tokenizer(suffix, return_tensors='pt', add_special_tokens=False).input_ids.to(self.decoder.device)
        if start + ids.shape[1] >= self.decoder.position_limit:
            raise RuntimeError('Question reached model position limit')
        pos = torch.arange(start, start + ids.shape[1], device=self.decoder.device)[None].expand(3, -1)
        hidden, _, _ = self.decoder.forward(self.model.get_input_embeddings()(ids), pos, cache)
        letters = 'ABCDEFGH'[:len(question['choices'])]
        answer, fallback = choose_letter(self.model.lm_head(hidden[:, -1])[0], self.tokenizer, letters)
        if saved != self.state.fingerprint():
            raise RuntimeError('Question mutated persistent H2SVC video KV')
        counts = temporary.retained_counts()
        self.last_audit = dict(sampled_frames=self.state.seen_frames,
                               max_retained_frames_per_head=max(counts.values()),
                               visual_slots=sum(sum(len(unit.uids) for unit in bank.units)
                                                for bank in temporary.banks.values()),
                               max_visual_per_head=max(sum(len(unit.uids) for unit in bank.units)
                                                       for bank in temporary.banks.values()),
                               constrained_choice_fallback=fallback,
                               persistent_state_unchanged=True)
        self.input_audit = self.input_audit[-8:]
        return answer
