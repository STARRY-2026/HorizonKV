"""Ordinary visual inputs with HorizonKV-EGate's own memory policy.

The frozen budget table is transferred, not recalibrated. Qwen's two-frame
temporal merge is attributed to the newer frame ID; its independent latest
branch is separately encoded as one frame. The memory batch remains four
sampled frames. This adapter does not claim numerical parity with old fixed
896 inputs, native DSCache caching, or the other methods' positional policy.
"""
import math
import torch
from .decoder import Decoder
from .positions import positions, token_ids
from .policy import Branch, State, append_unit, answer_caches
from .protocol import load_allocation, format_mc_question, route_answer


def choose_letter(logits, tokenizer, letters):
    tokens = {}
    for letter in letters:
        for spelling in (letter, ' ' + letter):
            ids = tokenizer.encode(spelling, add_special_tokens=False)
            if len(ids) == 1 and tokenizer.decode(ids).strip() == letter:
                tokens[ids[0]] = letter
    if set(tokens.values()) != set(letters):
        raise RuntimeError('Tokenizer cannot encode all answer letters')
    first = int(logits.argmax())
    if first in tokens:
        return tokens[first], False
    best = max(tokens, key=lambda token: float(logits[token]))
    return tokens[best], True


class Adapter:
    def __init__(self, model, tokenizer, processor, kind):
        self.model, self.tokenizer, self.processor, self.kind = model, tokenizer, processor, kind
        self.decoder = Decoder(model, kind)
        self.allocation, self.allocation_sha256 = load_allocation()
        self.parity = self.decoder.parity_check(tokenizer)
        self.clear_cache()

    @torch.inference_mode()
    def clear_cache(self):
        prefix = ('<|im_start|>user\n<|vision_start|>' if self.kind == 'qwen25vl' else
                  '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n')
        ids = self.tokenizer(prefix, return_tensors='pt', add_special_tokens=False).input_ids.to(self.decoder.device)
        branch = Branch(self.decoder, self.model.get_input_embeddings()(ids))
        self.state = State(branch)
        self.input_audit = []
        self.last_audit = None

    @property
    def max_input_frames(self):
        room = 4 - self.state.batch_frames if self.state.batch_frames < 4 else 4
        return min(room, 2 if self.kind == 'qwen25vl' else 1)

    def _features(self, frames):
        if self.kind == 'qwen25vl':
            from torchvision.transforms.functional import resize
            from torchvision.transforms import InterpolationMode
            from qwen_vl_utils.vision_process import smart_resize
            x = frames
            if x.shape[-1] == 3:
                x = x.permute(0, 3, 1, 2)
            h, w = smart_resize(int(x.shape[-2]), int(x.shape[-1]), factor=28,
                                min_pixels=448**2, max_pixels=448**2)
            x = resize(x, [h, w], interpolation=InterpolationMode.BICUBIC, antialias=True).float()
            data = self.processor(text=[''], videos=x, return_tensors='pt').to(self.decoder.device, self.decoder.dtype)
            grid = data['video_grid_thw']
            t, h, w = map(int, grid[0].tolist())
            if t != 1:
                raise RuntimeError('EGate ordinary unit must represent <=2 raw frames in one temporal grid')
            h //= 2
            w //= 2
            features = self.model.get_video_features(data['pixel_values_videos'], grid)[0]
            if features.ndim == 3:
                features = features.reshape(-1, features.shape[-1])
            if len(features) != h * w:
                raise RuntimeError('Qwen grid/feature mismatch')
            self.input_audit.append(dict(frames=len(frames), grid=grid[0].tolist(), tokens=len(features)))
            return features, (h, w), None
        pix = self.processor.preprocess(frames.detach().cpu().numpy(), return_tensors='pt')['pixel_values']
        pix = pix.to(self.decoder.device, self.decoder.dtype)
        if tuple(pix.shape[-2:]) != (384, 384) or len(frames) != 1:
            raise RuntimeError('Expected one 384 x 384 LLaVA frame')
        features = self.model.encode_images(pix)
        low = self.model.get_2dPool(features, stride=2)[0]
        if len(low) != 196:
            raise RuntimeError('LLaVA token counts differ from input profile')
        side = math.isqrt(len(low))
        self.input_audit.append(dict(frames=1, grid=[side, side], tokens=len(low)))
        return low, (side, side), None

    def _append(self, state, frames):
        count = len(frames)
        frame = state.seen_frames + count - 1
        feat, grid, _ = self._features(frames)
        ids = token_ids(frame, *grid)
        metadata = {frame: grid}
        if count == 1:
            one_feat, one_grid = feat, grid
        else:
            one_feat, one_grid, _ = self._features(frames[-1:])
        append_unit(state, self.allocation, feat, ids, metadata, one_feat,
                    token_ids(frame, *one_grid), {frame: one_grid}, count)

    @torch.inference_mode()
    def encode_video_chunk(self, frames):
        if not 1 <= len(frames) <= self.max_input_frames:
            raise ValueError('Input must respect the visual unit and four-frame memory batch')
        self._append(self.state, frames)

    @torch.inference_mode()
    def question_answering(self, question):
        if not self.state.seen_frames:
            raise RuntimeError('No visible frames')
        saved = self.state.fingerprint()
        caches, start, audit = answer_caches(self.state, self.allocation)
        answers, fallbacks = [], []
        for i, history in enumerate((False, True)):
            text, _ = format_mc_question(question['question'], question['choices'], recent_gate=not history)
            suffix = ('<|vision_end|>\n' if self.kind == 'qwen25vl' else '\n') + text
            suffix += '<|im_end|>\n<|im_start|>assistant\n'
            ids = self.tokenizer(suffix, return_tensors='pt', add_special_tokens=False).input_ids.to(self.decoder.device)
            if start + ids.shape[1] >= self.decoder.position_limit:
                raise RuntimeError('Question suffix reached model position limit')
            pos = torch.arange(start, start + ids.shape[1], device=self.decoder.device)[None].expand(3, -1)
            hidden, _, _ = self.decoder.forward(self.model.get_input_embeddings()(ids), pos, caches[i])
            letters = 'ABCDEFGHI'[:len(question['choices']) + int(not history)]
            answer, fallback = choose_letter(self.model.lm_head(hidden[:, -1])[0], self.tokenizer, letters)
            answers.append(answer)
            fallbacks.append(fallback)
        if saved != self.state.fingerprint():
            raise RuntimeError('Two-pass answering mutated persistent video KV')
        result, source = route_answer(answers[0], answers[1], len(question['choices']))
        audit.update(recent_answer=answers[0], history_answer=answers[1], gate_source=source,
                     constrained_choice_fallbacks=fallbacks, persistent_state_unchanged=True,
                     sampled_frames=self.state.seen_frames, persistent_history_transfers=self.state.transfers)
        self.last_audit = audit
        # Diagnostics retain only the most recent small input summaries.
        self.input_audit = self.input_audit[-8:]
        return result
