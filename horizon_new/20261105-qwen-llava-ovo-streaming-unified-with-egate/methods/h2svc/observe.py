"""Full-context offline TAD observation for H2SVC's 50 calibration videos.

No online cache eviction is applied. The video input and positional convention
match the unified H2SVC adapter. Each Qwen merged pair is one attention unit;
its tokens use the later sampled-frame index, as in online inference.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from methods.registry import LUKV_PROFILE
from methods.horizonkv_egate.adapter import Adapter as InputAdapter
from methods.horizonkv_egate.decoder import Decoder
from methods.horizonkv_egate.positions import positions, token_ids
from .routing import map_from_tad, read_head_map

DESCRIPTION = 'Please provide a detailed description of the video, focusing on the main subjects, their actions, the background scenes.'
HEADS = [(layer, head) for layer in range(28) for head in range(4)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TADObserver:
    """Accumulate paper equations (1) and (2), including nonvisual softmax mass."""

    def __init__(self, decoder):
        self.decoder = decoder
        self.key_frames = [-1] * 0
        self.new_frames = []
        self.mode = None
        self.final_frame = -1
        self.v2v = torch.zeros((28, 4), device=decoder.device, dtype=torch.float64)
        self.t2v = torch.zeros_like(self.v2v)
        self.visual_queries = 0
        self.visual_units = 0
        self.text_queries = 0

    @torch.inference_mode()
    def __call__(self, layer, queries, keys, old_cache, scale):
        q_len = queries.shape[2]
        if len(self.new_frames) != q_len:
            raise ValueError('Query/frame alignment differs')
        old_len = old_cache.get_segment(0)[0].shape[0]
        if old_len != len(self.key_frames):
            raise ValueError('KV/frame alignment differs')
        all_frames = torch.tensor(self.key_frames + self.new_frames, device=queries.device)
        qframes = torch.tensor(self.new_frames, device=queries.device)
        visual = all_frames >= 0
        if self.mode == 'visual':
            distance = (qframes[:, None] - all_frames[None, :]).clamp_min(0)
            distance = distance * visual[None]
        elif self.mode == 'text':
            distance = ((self.final_frame - all_frames).clamp_min(0) * visual).expand(q_len, -1)
        else:
            return
        causal = torch.arange(len(all_frames), device=queries.device)[None] <= (
            old_len + torch.arange(q_len, device=queries.device)[:, None])
        distance = distance.float()
        group = self.decoder.nq // self.decoder.nk
        for head in range(self.decoder.nk):
            old_keys = old_cache.get_segment(head)[0]
            all_keys = torch.cat((old_keys, keys[0, head]))
            group_q = queries[0, head * group:(head + 1) * group]
            # Softmax includes prefix and text keys; their distance is zero.
            logits = torch.matmul(group_q.float(), all_keys.float().T) * scale
            probs = F.softmax(logits.masked_fill(~causal[None], -torch.inf), dim=-1)
            per_query = (probs.mean(0) * distance).sum(-1)
            value = per_query.sum().double()
            if self.mode == 'visual':
                self.v2v[layer, head] += value / q_len
            else:
                self.t2v[layer, head] += value

    def step(self, embeddings, pos, cache, new_frames, mode):
        self.new_frames = list(new_frames)
        self.mode = mode
        hidden, _, _ = self.decoder.forward(embeddings, pos, cache, observer=self)
        self.key_frames.extend(self.new_frames)
        if mode == 'visual':
            self.visual_queries += len(self.new_frames)
            self.visual_units += 1
        elif mode == 'text':
            self.text_queries += len(self.new_frames)
        return hidden


def load_model(kind, model_path, siglip=None):
    if kind == 'qwen25vl':
        from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
        processor = Qwen2_5_VLProcessor.from_pretrained(model_path, local_files_only=True,
                    min_pixels=448**2, max_pixels=448**2)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path,
                    torch_dtype=torch.float16, attn_implementation='sdpa',
                    local_files_only=True).to('cuda').eval()
        return model, processor.tokenizer, processor
    import llava.model.llava_arch as arch
    from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower
    arch.build_vision_tower = lambda cfg, **kw: SigLipVisionTower(
        cfg.mm_vision_tower, vision_tower_cfg=cfg, **kw)
    from llava.model.builder import load_pretrained_model
    tokenizer, model, processor, _ = load_pretrained_model(
        str(model_path), None, 'llava_qwen', device_map='cuda:0',
        attn_implementation='sdpa', multimodal=True,
        overwrite_config={'mm_vision_tower': str(siglip)})
    return model.eval(), tokenizer, processor


@torch.inference_mode()
def observe_video(video_path, kind, model, tokenizer, processor, max_frames, max_new_tokens):
    from decord import VideoReader, cpu
    decoder = Decoder(model, kind)
    observer = TADObserver(decoder)
    prefix = ('<|im_start|>user\n<|vision_start|>' if kind == 'qwen25vl' else
              '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n')
    prefix_ids = tokenizer(prefix, return_tensors='pt', add_special_tokens=False).input_ids.to(decoder.device)
    cache = decoder.new_cache()
    n = prefix_ids.shape[1]
    p = torch.arange(n, device=decoder.device)[None].expand(3, -1)
    decoder.forward(model.get_input_embeddings()(prefix_ids), p, cache)
    observer.key_frames = [-1] * n
    # Reuse exactly the ordinary visual encoder, without constructing H2State.
    encoder = object.__new__(InputAdapter)
    encoder.model, encoder.tokenizer, encoder.processor, encoder.kind = model, tokenizer, processor, kind
    encoder.decoder, encoder.high_processor, encoder.input_audit = decoder, None, []
    reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    fps = round(reader.get_avg_fps())
    if fps < 1:
        raise ValueError('Invalid video fps')
    indices = list(range(0, len(reader), fps))[:max_frames]
    if len(indices) < 2:
        raise ValueError('Profiling needs at least two sampled frames')
    metadata = {}
    cursor = 0
    while cursor < len(indices):
        width = min(2 if kind == 'qwen25vl' else 1, len(indices) - cursor)
        frame = cursor + width - 1
        raw = torch.from_numpy(reader.get_batch(indices[cursor:cursor + width]).asnumpy())
        features, grid, _ = encoder._features(raw)
        metadata[frame] = grid
        ids = token_ids(frame, *grid)
        pos = positions(ids, n, metadata, kind, decoder.device)
        observer.step(features[None], pos, cache, [frame] * len(ids), 'visual')
        cursor += width
        del raw, features
    observer.final_frame = len(indices) - 1
    suffix = ('<|vision_end|>\n' if kind == 'qwen25vl' else '\n') + DESCRIPTION
    suffix += '<|im_end|>\n<|im_start|>assistant\n'
    ids = tokenizer(suffix, return_tensors='pt', add_special_tokens=False).input_ids.to(decoder.device)
    start = max(int(cache.layers[0].flat_pos.max()) + 1, n)
    if start + ids.shape[1] + max_new_tokens >= decoder.position_limit:
        raise RuntimeError('Profiling text exceeds position limit')
    pos = torch.arange(start, start + ids.shape[1], device=decoder.device)[None].expand(3, -1)
    hidden = observer.step(model.get_input_embeddings()(ids), pos, cache, [-1] * ids.shape[1], 'text')
    current = start + ids.shape[1]
    eos = tokenizer.eos_token_id
    for _ in range(max_new_tokens):
        next_id = int(model.lm_head(hidden[:, -1])[0].argmax())
        if next_id == eos:
            break
        token = torch.tensor([[next_id]], device=decoder.device)
        pos = torch.full((3, 1), current, device=decoder.device, dtype=torch.long)
        hidden = observer.step(model.get_input_embeddings()(token), pos, cache, [-1], 'text')
        current += 1
    # Paper takes a per-frame mean, then a per-video mean. Qwen's merged unit
    # contributes one query group per pair, matching its actual KV representation.
    visual_units = observer.visual_units
    return dict(sampled_frames=len(indices), visual_units=visual_units,
                visual_queries=observer.visual_queries, text_queries=observer.text_queries,
                tad_v2v=(observer.v2v / visual_units).cpu().tolist(),
                tad_t2v=(observer.t2v / observer.text_queries).cpu().tolist(),
                peak_gpu_gib=torch.cuda.max_memory_allocated() / 1024**3)


def main():
    os.environ.setdefault('DECORD_EOF_RETRY_MAX', '40960')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['qwen25vl', 'llavaov'], required=True)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--siglip', type=Path)
    parser.add_argument('--annotation', type=Path, default=Path('/home/yujun/program/hermes/data/streamingbench/streamingbench_realtime_local.json'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-frames', type=int, default=64)
    parser.add_argument('--max-new-tokens', type=int, default=32)
    parser.add_argument('--max-videos', type=int, default=50)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    if not 2 <= args.max_frames <= 64 or not 1 <= args.max_videos <= 50 or args.max_new_tokens < 0:
        parser.error('Expected 2..64 frames, 1..50 videos, nonnegative generation limit')
    profile = json.loads(LUKV_PROFILE.read_text())['calibration']['selected_videos']
    local = json.loads(args.annotation.read_text())
    by_name = {Path(v['video_path']).name: v for v in local}
    selected = []
    for original in profile[:args.max_videos]:
        name = Path(original['video_path']).name
        if name not in by_name or not Path(by_name[name]['video_path']).is_file():
            raise FileNotFoundError('Calibration video unavailable: ' + name)
        selected.append(by_name[name])
    output = args.output_dir.resolve()
    measurement = output / 'measurements.json'
    plan = dict(model=args.model, model_path=str(args.model_path.resolve()),
                annotation=str(args.annotation.resolve()), annotation_sha256=sha(args.annotation),
                calibration_profile_sha256=sha(LUKV_PROFILE),
                calibration_videos=[v['video_path'] for v in selected],
                max_frames=args.max_frames, max_new_tokens=args.max_new_tokens,
                description=DESCRIPTION, fps=1, qwen_pair_representation='later sampled frame',
                normal_pixel_area=448**2 if args.model == 'qwen25vl' else None,
                llava_resolution=[384, 384] if args.model == 'llavaov' else None)
    print(json.dumps(plan, indent=2), flush=True)
    if args.plan_only:
        return
    if not args.model_path.is_dir() or (args.model == 'llavaov' and (args.siglip is None or not args.siglip.is_dir())):
        raise FileNotFoundError('Model or SigLIP path missing')
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / 'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('Resume plan differs from existing observation')
    plan_path.write_text(json.dumps(plan, indent=2) + '\n')
    rows_path = output / 'per_video.jsonl'
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()] if rows_path.exists() else []
    if any(row['video_path'] != selected[index]['video_path'] for index, row in enumerate(rows)):
        raise ValueError('Resume video order differs')
    if len(rows) == len(selected):
        print('OBSERVATION_ALREADY_COMPLETE', flush=True)
    else:
        model, tokenizer, processor = load_model(args.model, args.model_path, args.siglip)
        for index in range(len(rows), len(selected)):
            video = selected[index]
            torch.cuda.reset_peak_memory_stats()
            result = observe_video(video['video_path'], args.model, model, tokenizer, processor,
                                   args.max_frames, args.max_new_tokens)
            result.update(video_id=video['video_id'], video_path=video['video_path'])
            with rows_path.open('a') as stream:
                stream.write(json.dumps(result) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            rows.append(result)
            print('OBSERVED', index + 1, '/', len(selected), video['video_id'], flush=True)
    heads = []
    for layer, head in HEADS:
        heads.append(dict(layer=layer, head=head,
                          tad_v2v=sum(row['tad_v2v'][layer][head] for row in rows) / len(rows),
                          tad_t2v=sum(row['tad_t2v'][layer][head] for row in rows) / len(rows)))
    complete = len(rows) == 50 and args.max_frames == 64 and args.max_new_tokens == 32
    measured = dict(schema_version=1, model=args.model, source_sha256=sha(rows_path),
                    calibration_videos=plan['calibration_videos'], heads=heads,
                    observation_plan_sha256=sha(plan_path), complete=complete)
    measurement.write_text(json.dumps(measured, indent=2) + '\n')
    if complete:
        frozen = map_from_tad(measured)
        head_path = output / f'{args.model}_head_map.json'
        if head_path.exists():
            if json.loads(head_path.read_text()) != frozen:
                raise ValueError('Existing frozen map differs')
        else:
            head_path.write_text(json.dumps(frozen, indent=2) + '\n')
        read_head_map(head_path)
        print('FROZEN_HEAD_MAP', head_path, flush=True)


if __name__ == '__main__':
    main()
