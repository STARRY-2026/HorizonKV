"""H2SVC inference worker with per-question durable results and resume."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import resource
import time

from .data import accuracy, correct_letter, read_rows, task_of, validate_rows, write_rows
from .routing import read_head_map


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['qwen25vl', 'llavaov'], required=True)
    parser.add_argument('--annotation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--head-map', type=Path, required=True)
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--accept-equivalent-choices', action='store_true')
    args = parser.parse_args()
    routes, head_map = read_head_map(args.head_map)
    if head_map['model'] != args.model:
        raise ValueError('Head map model differs from inference model')
    data = json.loads(args.annotation.read_text())
    destination = args.output / 'results.csv'
    rows = read_rows(destination)
    expected = validate_rows(rows, data, args.accept_equivalent_choices)
    completed = len(rows)
    print('RESUME', completed, '/', len(expected), flush=True)
    if completed == len(expected):
        return
    import torch
    from decord import VideoReader, cpu
    from .adapter import Adapter
    warehouse = Path(os.environ.get('HORIZON_WAREHOUSE', '/mnt/aifordata2026-p/common/aifordata/541707'))
    if args.model == 'qwen25vl':
        from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
        model_path = warehouse / 'model/Qwen2.5-VL-7B-Instruct'
        processor = Qwen2_5_VLProcessor.from_pretrained(model_path, local_files_only=True,
                    min_pixels=448**2, max_pixels=448**2)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path,
                    torch_dtype=torch.float16, attn_implementation='sdpa',
                    local_files_only=True).to('cuda').eval()
        tokenizer = processor.tokenizer
    else:
        import llava.model.llava_arch as arch
        from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower
        arch.build_vision_tower = lambda cfg, **kw: SigLipVisionTower(
            cfg.mm_vision_tower, vision_tower_cfg=cfg, **kw)
        from llava.model.builder import load_pretrained_model
        tokenizer, model, processor, _ = load_pretrained_model(
            str(warehouse / 'model/llava-onevision-qwen2-7b-ov'), None, 'llava_qwen',
            device_map='cuda:0', attn_implementation='sdpa', multimodal=True,
            overwrite_config={'mm_vision_tower': '/tmp/siglip-so400m-patch14-384'})
        model.eval()
    if any(parameter.is_meta for parameter in model.parameters()):
        raise RuntimeError('Unmaterialized model parameters')
    with torch.inference_mode():
        adapter = Adapter(model, tokenizer, processor, args.model, routes)
        print('MODEL_READY', flush=True)
        cursor = 0
        for video in data:
            count = len(video['conversations'])
            if cursor + count <= completed:
                cursor += count
                continue
            adapter.clear_cache()
            reader = VideoReader(video['video_path'], ctx=cpu(0), num_threads=1)
            try:
                fps = round(reader.get_avg_fps())
                if fps < 1:
                    raise ValueError('Invalid measured video FPS')
                clip = video.get('clip')
                begin = max(0, int(clip[0]*fps)) if clip else 0
                end = min(len(reader), int(clip[1]*fps)+1) if clip else len(reader)
                indices = list(range(begin, end, fps))
                processed = 0
                for index, question in enumerate(video['conversations']):
                    started = time.monotonic()
                    target = min(len(indices), math.ceil(float(question['end_time'])))
                    if target < 1 or target < processed:
                        raise ValueError('Invalid visible-frame target')
                    while processed < target:
                        width = min(adapter.max_input_frames, target-processed)
                        frames = torch.from_numpy(reader.get_batch(
                            indices[processed:processed+width]).asnumpy())
                        adapter.encode_video_chunk(frames)
                        processed += width
                        del frames
                    if cursor < completed:
                        cursor += 1
                        continue
                    answer = adapter.question_answering(question)
                    audit = adapter.last_audit
                    row = dict(video_id=str(video['video_id']), question_index=index,
                               question=question['question'], choices=question['choices'],
                               answer=question['answer'], correct_choice=correct_letter(question),
                               pred_answer=answer, pred_choice=answer,
                               qa_acc=accuracy(question, answer, args.accept_equivalent_choices),
                               task=task_of(video, question),
                               duration_category=video.get('duration_category', ''),
                               sampled_frames=target, seconds=round(time.monotonic()-started, 3),
                               visual_slots=audit['visual_slots'],
                               max_visual_per_head=audit['max_visual_per_head'],
                               max_retained_frames_per_head=audit['max_retained_frames_per_head'])
                    rows.append(row)
                    write_rows(destination, rows)
                    cursor += 1
                    status = dict(completed=len(rows), expected=len(expected), last_question=audit,
                                  last_input_units=adapter.input_audit, decoder_parity=adapter.parity,
                                  peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
                                  max_gpu_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
                    temporary = args.output / 'worker_status.json.tmp'
                    temporary.write_text(json.dumps(status, indent=2)+'\n')
                    temporary.replace(args.output/'worker_status.json')
                    print('DONE', len(rows), '/', len(expected), 'video', video['video_id'],
                          'question', index, 'prediction', answer, flush=True)
            finally:
                del reader
                adapter.clear_cache()
                gc.collect()
        validate_rows(rows, data, args.accept_equivalent_choices, complete=True)
        print('COMPLETE', len(rows), flush=True)


if __name__ == '__main__':
    main()
