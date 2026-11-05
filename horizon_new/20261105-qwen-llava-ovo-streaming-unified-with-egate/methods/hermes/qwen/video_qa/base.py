import csv
import warnings
import random
import json
import os
import math
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image
from decord import VideoReader
from transformers import (
    logging,
)

import logzero
from logzero import logger

from inference.llavaov_hermes import load_model as llavaov_hermes_load_model


def qwenvl_hermes_load_model(*args, **kwargs):
    try:
        from inference.qwenvl_hermes import load_model as _load_model
    except Exception as exc:
        raise ImportError(
            "Failed to import inference.qwenvl_hermes. "
            "Qwen models require a newer transformers version. "
            "Please use llava models on old transformers, or upgrade for qwen."
        ) from exc
    return _load_model(*args, **kwargs)

MODELS = {
    'llava_ov_0.5b': {
        'load_func': llavaov_hermes_load_model,
        'model_path': 'models/llava-onevision-qwen2-0.5b-ov-hf',
    },
    'llava_ov_7b': {
        'load_func': llavaov_hermes_load_model,
        'model_path': 'models/llava-onevision-qwen2-7b-ov-hf',
    },
    'llava_ov_72b': {
        'load_func': llavaov_hermes_load_model,
        'model_path': 'models/llava-onevision-qwen2-72b-ov-hf',
    },
    'qwen2.5_vl_3b': {
        'load_func': qwenvl_hermes_load_model,
        'model_path': 'models/Qwen2.5-VL-3B-Instruct',
    },
    'qwen2.5_vl_7b': {
        'load_func': qwenvl_hermes_load_model,
        'model_path': 'models/Qwen2.5-VL-7B-Instruct',
    },
    'qwen2.5_vl_32b': {
        'load_func': qwenvl_hermes_load_model,
        'model_path': 'models/Qwen2.5-VL-32B-Instruct',
    },
}

class BaseVQA:
    def __init__(self, anno, save_dir, sample_fps,
                 qa_model, qa_processor=None,
                 num_chunks=None, chunk_idx=None, encode_chunk_size=16) -> None:
        self.sample_fps = sample_fps
        self.qa_model = qa_model
        self.qa_processor = qa_processor

        self.num_chunks = num_chunks
        self.chunk_idx = chunk_idx
        self.encode_chunk_size = int(encode_chunk_size)
        if num_chunks is not None:
            anno = self.get_chunk(anno, num_chunks, chunk_idx)
        self.anno = anno

        self.save_dir = save_dir
        self.choice_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        self.record = []

    def split_list(self, lst, n):
        """Split a list into n (roughly) equal-sized chunks"""
        chunk_size = math.ceil(len(lst) / n)
        return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]

    def get_chunk(self, lst, n, k):
        chunks = self.split_list(lst, n)
        return chunks[k]
    
    def load_video(self, video_path, clip=None):
        """
        Load video from file.
        
        Args:
            video_path: Path to the video file (.npy or regular video)
            clip: Optional [start_time, end_time] in seconds to extract a specific segment
            
        Returns:
            For .npy files: numpy array of frames
            For regular videos: (numpy array of frames, resized_height, resized_width)
        """
        if video_path.endswith('.npy'):
            video = np.load(video_path)
            num_frames = len(video)
            frame_idx = np.linspace(0, num_frames-1, int(num_frames*self.sample_fps), dtype=int).tolist()
            video = video[frame_idx]
            return video
        else:
            vr = VideoReader(video_path, num_threads=1)
            fps = round(vr.get_avg_fps())
            total_frames = len(vr)
            
            if clip is not None:
                # Calculate frame range based on clip times
                start_frame = max(0, int(clip[0] * fps))
                end_frame = min(total_frames, int(clip[1] * fps) + 1)
                print(f"start_frame: {start_frame}")
                print(f"end_frame: {end_frame}")
            else:
                start_frame = 0
                end_frame = total_frames
            
            # Sample frames at target fps within the clip range
            sample_step = int(fps / self.sample_fps)
            frame_idx = [i for i in range(start_frame, end_frame, sample_step)]
            video = vr.get_batch(frame_idx).asnumpy()
            return video
    
    def load_video_frames(self, video_path, video_fps, clip=None):
        """
        Load video from a directory of image frames (for OVBench image-based videos).
        
        Args:
            video_path: Path to the directory containing image frames
            video_fps: Original FPS of the video (from annotation)
            clip: Optional [start_time, end_time] to extract a specific segment
            
        Returns:
            video: numpy array of frames
        """
        # Get sorted list of image files
        img_files = sorted(os.listdir(video_path))
        # Filter only image files
        img_files = [f for f in img_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        num_frames = len(img_files)
        
        # Calculate frame indices based on clip times
        if clip is not None:
            start_time, end_time = clip
            start_frame = max(0, int(start_time * video_fps))
            end_frame = min(num_frames - 1, int(end_time * video_fps))
            print(f"start_frame: {start_frame}")
            print(f"end_frame: {end_frame}")
        else:
            start_frame = 0
            end_frame = num_frames - 1
        
        # Generate sampled frame indices based on sample_fps
        sample_step = max(1, int(video_fps / self.sample_fps))
        frame_idx = list(range(start_frame, end_frame + 1, sample_step))
        
        # Load images
        frames = []
        for i in frame_idx:
            if i < len(img_files):
                img_path = os.path.join(video_path, img_files[i])
                img = Image.open(img_path).convert('RGB')
                frames.append(np.array(img))
        
        video = np.stack(frames, axis=0)
        return video
    
    def format_mcqa_prompt(self, question, candidates):
        assert len(question) > 0, f"Q: {question}"

        formatted_choices = "\n".join(["(" + self.choice_letters[i] + ") " + candidate for i, candidate in enumerate(candidates)])
        formatted_question = f"Question: {question}\nOptions:\n{formatted_choices}\nOnly give the best option."

        return {
            "question": f"{question}",
            "formatted_question": formatted_question,
            "prompt": self.qa_model.get_prompt(formatted_question, mc=True)
        }

    def extract_characters_regex(self, s):
        s = s.strip()
        if ")" in s:
            index = s.index(")")
            pred = s[index - 1 : index]
            return pred
        else:
            try:
                return s[0]
            except:
                return s

    def video_open_qa(self, question, max_new_tokens=1024, retrieved_indices=None):
        input_text = {
            "question": question,
            "prompt": self.qa_model.get_prompt(question)
        }
        pred_answer = self.qa_model.question_answering(input_text, max_new_tokens=max_new_tokens)
        return {
            'pred_answer': pred_answer.replace('\n', ''),
        }

    def video_close_qa(self, question, candidates, correct_choice, retrieved_indices=None):
        input_text = self.format_mcqa_prompt(question, candidates)
        pred_answer = self.qa_model.question_answering(input_text, max_new_tokens=16)
        pred_letter = self.extract_characters_regex(pred_answer)
        return {
            'pred_answer': pred_answer.replace('\n', ''),
            'pred_choice': pred_letter,
            'acc': float(pred_letter == correct_choice),
        }

    def pseudo_qa(self, prediction_prompt=None):
        if prediction_prompt is None:
            prediction_prompt = "<|im_end|><|im_start|>assistant\n"
        input_text = {
            "question": prediction_prompt,
            "prompt": self.qa_model.get_prompt(prediction_prompt)
        }
        self.qa_model.pseudo_forward(input_text)

    @torch.inference_mode()
    def analyze_a_video(self, video_sample):
        pass

    def analyze(self, debug=False):
        video_annos = self.anno[:1] if debug else self.anno
        results_path = f'{self.save_dir}/{self.num_chunks}_{self.chunk_idx}.csv'
        processed_ids = set()
        if os.path.exists(results_path):
            try:
                existing_df = pd.read_csv(results_path)
                processed_ids = set(existing_df['video_id'].astype(str).values)
                self.record = existing_df.to_dict('records')
                logger.info(f'Resumed from {results_path}: {len(processed_ids)} videos already done')
            except Exception:
                pass

        for video_sample in tqdm(video_annos):
            vid = str(video_sample['video_id'])
            if vid in processed_ids:
                continue
            logger.debug(f'video_id: {vid}')
            self.analyze_a_video(video_sample)
            final_df = pd.DataFrame(self.record)
            final_df.to_csv(results_path, index=False, quoting=csv.QUOTE_NONNUMERIC)


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1', 'yes'):
        return True
    elif value.lower() in ('false', '0', 'no'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def work(QA_CLASS):
    logging.set_verbosity_error()

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_fps", type=float, default=1)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--save_dir", "--output_dir", dest="save_dir", type=str, required=True,
                        help="Directory for this worker's result CSV.")
    parser.add_argument("--anno_path", type=str, required=True)
    parser.add_argument("--model", type=str, default="llava_ov_7b", choices=list(MODELS))
    parser.add_argument("--model_path", type=str, default=None,
                        help="Override the model directory or Hugging Face model ID.")
    parser.add_argument("--debug", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--kv_size", type=int)
    parser.add_argument("--compress_mode", type=str, default="hermes", choices=["hermes", "streamingvlm"],
                        help="KV cache compression strategy. 'hermes' (default): attention-guided; 'streamingvlm': simple sliding window.")
    parser.add_argument("--head_budget_scores", type=str, default=None,
                        help="Enable HERMES per-head voting budgets. Use 'pseudo', 'sparsemm', or a .npz/.json score file.")
    parser.add_argument("--layer_budget_scores", type=str, default=None,
                        help="Enable HERMES per-layer budgets. Use 'pseudo', 'sparsemm', or a .npz/.json score file.")
    parser.add_argument("--layer_budget_variable", type=str2bool, nargs='?', const=True, default=False,
                        help="If true, allow real per-layer KV cache lengths and install per-layer attention masks. Experimental for Qwen.")
    parser.add_argument("--kv_head_budget_scores", type=str, default=None,
                        help="Enable per-KV-head logical eviction. Use 'pseudo', 'sparsemm_qwen25', or a .npz/.json score file.")
    parser.add_argument("--horizonkv_policy", type=str, default=None,
                        help="Frozen 112-row HorizonKV CSV; enables fixed Recent FIFO plus per-head Memory.")
    parser.add_argument("--horizonkv_recent_capacity", type=int, default=2691,
                        help="Fixed visual-token FIFO capacity shared by all HorizonKV heads.")
    parser.add_argument("--horizonkv_ema_gamma", type=float, default=0.9,
                        help="Attention EMA decay used by HorizonKV Memory eviction.")
    parser.add_argument("--horizonkv_apply_age_penalty", type=str2bool, nargs='?', const=True, default=True,
                        help="Apply the additional chunk-age denominator in HorizonKV Memory scores.")
    parser.add_argument("--horizonkv_temporal_coverage", type=str, default="cold", choices=["cold", "all"],
                        help="Distribute anchor quota over cold chunks only or all Memory chunks.")
    parser.add_argument("--horizonkv_probe_style", type=str, default="original",
                        choices=["original", "hermes_routed"],
                        help="Fixed Horizon probes or static HERMES Local/Mixed/Global probes with layer routing.")
    parser.add_argument("--horizonkv_score_blend", type=str, default="head_type",
                        choices=["head_type", "hermes_layer"],
                        help="Head-type recency weights or HERMES layer-wise recency/attention mixing.")
    parser.add_argument("--timestamp_v8_policy", type=str, default=None,
                        help="Original v8 CSV enabling timestamp-gated, role-aware ragged eviction.")
    parser.add_argument("--timestamp_v8_grace_frames", type=int, default=1,
                        help="Number of newest raw-frame timestamps protected before historical eviction.")
    parser.add_argument("--ragged_frame_trace_path", type=str, default=None,
                        help="Optional JSONL path recording per-head frame retention at question boundaries.")
    parser.add_argument("--ragged_readout_trace_path", type=str, default=None,
                        help="Optional JSONL path recording true-question attention over retained frames.")
    parser.add_argument("--ragged_readout_trace_last_n", type=int, default=1,
                        help="Number of trailing question tokens pooled by the readout trace; 0 averages the full prompt.")
    parser.add_argument("--ragged_token_trace_path", type=str, default=None,
                        help="Optional JSONL path recording immutable visual-token IDs and "
                             "per-token true-question attention.")
    parser.add_argument("--ragged_redundancy_trace_path", type=str, default=None,
                        help="Optional JSONL path recording sampled cross-frame Value redundancy "
                             "for selected historical tokens at question boundaries.")
    parser.add_argument("--ragged_redundancy_trace_max_tokens", type=int, default=128,
                        help="Maximum selected historical tokens sampled per KV head for redundancy tracing.")
    parser.add_argument("--answer_history_max_age_seconds", type=float, default=0.0,
                        help="Diagnostic only: during real-question answering, temporarily hide visual KV "
                             "older than this many seconds, then restore the original cache. 0 disables it.")
    parser.add_argument("--prospective_lifecycle_trace_path", type=str, default=None,
                        help="Optional JSONL path tracking source-frame admission and future readout.")
    parser.add_argument("--prospective_lifecycle_trace_last_n", type=int, default=1,
                        help="Number of trailing probe/question tokens pooled in prospective traces.")
    parser.add_argument("--prospective_lifecycle_top_fraction", type=float, default=0.2,
                        help="Top fraction of within-frame admission scores used as birth salience.")
    parser.add_argument("--prospective_lifecycle_visual_queries", type=int, default=16,
                        help="Uniformly sampled new visual queries used to measure future reads; 0 disables them.")
    parser.add_argument("--prospective_lifecycle_visual_top_readers", type=int, default=16,
                        help="Strongest sampled GQA query reads averaged for each cached token.")
    parser.add_argument("--prospective_lifecycle_content_bins", type=int, default=4,
                        help="Within-frame admission-rank bins written to prospective traces.")
    parser.add_argument("--encode_chunk_size", type=int, default=16,
                        help="Number of sampled video frames encoded per streaming update.")
    parser.add_argument("--kv_head_budget_union_cap_ratio", type=float, default=1.0,
                        help="Dense union cap relative to kv_size for per-KV-head logical eviction.")
    parser.add_argument("--kv_head_budget_max_mask_q_len", type=int, default=128,
                        help="Only apply per-head masks when q_len is at most this value, to avoid OOM on large video chunks.")
    parser.add_argument(
        "--kv_head_budget_scheme",
        type=str,
        default="relative",
        choices=[
            "relative",
            "sparsemm",
            "sparsemm_layer_total",
            "sparsemm_layer",
            "sparsemm_total",
            "sparsemm_per_layer_total",
            "sparsemm_layer_exact",
            "sparsemm_per_layer",
        ],
        help=(
            "Per-KV-head budget allocation. 'relative': old per-layer min/max ratios; "
            "'sparsemm': SparseMM min-cache + global visual-score allocation with "
            "num_keep average per layer-KV-head; 'sparsemm_layer_total': SparseMM "
            "allocation with total budget num_keep*num_layers; "
            "'sparsemm_per_layer_total': SparseMM allocation with each layer summing "
            "to num_keep."
        ),
    )
    parser.add_argument("--kv_head_budget_sparsemm_ratio", type=float, default=0.1,
                        help="SparseMM floor ratio for each KV head when --kv_head_budget_scheme=sparsemm.")
    parser.add_argument("--kv_head_budget_sparsemm_window_size", type=int, default=32,
                        help="SparseMM recent/window budget added back after score allocation.")
    parser.add_argument("--kv_head_selector_policy", type=str, default="deployed",
                        choices=["deployed", "uniform_joint"],
                        help="Historical-token selector policy for physical KV heads.")
    parser.add_argument("--kv_head_budget_floor_ratio", type=float, default=0.0,
                        help="Conservative per-KV-head budget floor relative to kv_size; preserves total budget.")
    parser.add_argument("--kv_head_budget_memory_current_split", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Enable memory/current budget split: each KV head's budget is divided "
                             "into memory (pre-chunk) and current (latest-chunk) portions.")
    parser.add_argument("--kv_head_budget_memory_scores", type=str, default=None,
                        help="Path to per-KV-head memory orientation scores CSV for split ratio "
                             "computation. Falls back to --kv_head_budget_scores if not set.")
    parser.add_argument("--kv_head_budget_memory_ratio", type=float, default=0.7,
                        help="Base memory budget fraction (0.0-1.0) when --kv_head_budget_memory_current_split "
                             "is enabled.")
    parser.add_argument("--kv_head_budget_split_dynamic_scale", type=float, default=0.3,
                        help="How much per-head score deviation can shift the memory ratio "
                             "from the base value (range: base +/- scale, clipped to [0.1, 0.9]).")
    parser.add_argument("--kv_head_policy_roles", type=str, default=None,
                        help="CSV with layer,kv_head,long_score,short_score for role-specific "
                             "ragged token selection. Default disabled.")
    parser.add_argument("--kv_head_policy_memory_quantile", type=float, default=0.75,
                        help="Quantile threshold on long_score for memory-role KV heads.")
    parser.add_argument("--kv_head_policy_current_quantile", type=float, default=0.75,
                        help="Quantile threshold on short_score for current-role KV heads.")
    parser.add_argument("--kv_head_policy_memory_bonus", type=float, default=0.1,
                        help="Soft score bonus for previous visual tokens in memory-role KV heads.")
    parser.add_argument("--kv_head_policy_current_bonus", type=float, default=0.1,
                        help="Soft score bonus for latest-chunk visual tokens in current-role KV heads.")
    parser.add_argument("--kv_head_recency_scores", type=str, default=None,
                        help="CSV with layer,kv_head and a temporal-orientation metric used to adjust HERMES recency alpha per KV head.")
    parser.add_argument("--kv_head_recency_metric", type=str, default="b_log_per_token_ratio",
                        help="Column in --kv_head_recency_scores; larger values mean more current-oriented heads.")
    parser.add_argument("--kv_head_recency_delta", type=float, default=0.2,
                        help="Maximum signed residual added to the HERMES layer recency alpha.")
    parser.add_argument("--kv_head_recency_tau", type=float, default=1.0,
                        help="Temperature of the tanh mapping from layer-centered head score to alpha residual.")
    parser.add_argument("--kv_head_recency_alpha_min", type=float, default=0.0)
    parser.add_argument("--kv_head_recency_alpha_max", type=float, default=1.0)
    parser.add_argument("--kv_head_recency_k_delta", type=float, default=0.0,
                        help="Maximum signed residual added to the HERMES per-layer recency decay k.")
    parser.add_argument("--kv_head_recency_k_min", type=float, default=0.0)
    parser.add_argument("--kv_head_recency_k_max", type=float, default=20.0)
    parser.add_argument("--kv_head_layerwise_last_frame", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Enable layerwise last-frame protection for ragged KV-head selection. "
                             "Shallow/mid/deep layers keep different ratios of the latest frame.")
    parser.add_argument("--kv_head_layerwise_shallow_end", type=int, default=10,
                        help="Exclusive layer end for shallow last-frame protection, e.g. 10 means layers 0-9.")
    parser.add_argument("--kv_head_layerwise_mid_end", type=int, default=20,
                        help="Exclusive layer end for mid last-frame protection; deeper layers start here.")
    parser.add_argument("--kv_head_layerwise_shallow_ratio", type=float, default=1.0,
                        help="Latest-frame token keep ratio for shallow layers. 1.0 keeps the full latest frame.")
    parser.add_argument("--kv_head_layerwise_mid_ratio", type=float, default=0.5,
                        help="Latest-frame token keep ratio for mid layers.")
    parser.add_argument("--kv_head_layerwise_deep_ratio", type=float, default=0.0,
                        help="Latest-frame token keep ratio for deep layers. 0 disables extra protection.")
    parser.add_argument("--kv_head_role_recent_scores", type=str, default=None,
                        help="CSV score file used to split KV heads into memory/mixed/current roles for recent-token retention.")
    parser.add_argument("--kv_head_role_dual_scores", type=str, default=None,
                        help="CSV with independent memory/current scores, such as long_score and short_score. "
                             "This is mutually exclusive with --kv_head_role_recent_scores.")
    parser.add_argument("--kv_head_role_recent_memory_quantile", type=float, default=0.75,
                        help="Heads with score >= this quantile are memory heads.")
    parser.add_argument("--kv_head_role_recent_current_quantile", type=float, default=0.25,
                        help="Heads with score <= this quantile are current heads.")
    parser.add_argument("--kv_head_role_recent_memory_count", type=int, default=-1,
                        help="If >=0, use exactly this many top-scoring KV heads as memory heads, overriding memory quantile.")
    parser.add_argument("--kv_head_role_recent_current_count", type=int, default=-1,
                        help="If >=0, use exactly this many bottom-scoring KV heads as current heads, overriding current quantile.")
    parser.add_argument("--kv_head_role_recent_memory_tokens", type=int, default=0,
                        help="Recent visual tokens protected for memory heads.")
    parser.add_argument("--kv_head_role_recent_mixed_tokens", type=int, default=2000,
                        help="Recent visual tokens protected for mixed heads.")
    parser.add_argument("--kv_head_role_recent_current_tokens", type=int, default=4000,
                        help="Recent visual tokens protected for current heads.")
    parser.add_argument("--kv_head_role_recent_window_mode", type=str, default="override",
                        choices=["override", "max"],
                        help="override: role windows replace sparsemm recent window; max: take max(role, sparsemm window).")
    parser.add_argument("--kv_head_role_recent_uniform_budget", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="If true, force every KV head budget to kv_size in the ragged selector.")
    parser.add_argument("--kv_head_role_budget_enable", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="If true, override per-KV-head budgets by role after role_recent splits heads.")
    parser.add_argument("--kv_head_role_memory_budget", type=int, default=8000,
                        help="KV budget for memory-role heads when --kv_head_role_budget_enable is true.")
    parser.add_argument("--kv_head_role_mixed_budget", type=int, default=6000,
                        help="KV budget for mixed-role heads when --kv_head_role_budget_enable is true.")
    parser.add_argument("--kv_head_role_current_budget", type=int, default=4000,
                        help="KV budget for current-role heads when --kv_head_role_budget_enable is true.")
    parser.add_argument("--kv_head_role_budget_auto_mixed", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="If true, compute mixed-role budget to keep the global average at kv_size.")
    parser.add_argument("--kv_head_role_history_uniform_memory_ratio", type=float, default=0.0,
                        help="Fraction of each memory head's history budget selected uniformly across time. "
                             "The fraction stays inside the head budget; 0 disables it.")
    parser.add_argument("--kv_head_role_history_uniform_mixed_ratio", type=float, default=0.0,
                        help="Fraction of each mixed head's history budget selected uniformly across time. "
                             "The fraction stays inside the head budget; 0 disables it.")
    parser.add_argument("--kv_head_role_history_uniform_bins", type=int, default=32,
                        help="Maximum number of temporal bins used by role history-uniform selection.")
    parser.add_argument("--kv_head_role_last_frame_by_role", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="If true, apply last-frame protection per head role instead of per layer.")
    parser.add_argument("--kv_head_role_last_frame_ratio_current", type=float, default=1.0,
                        help="Last-frame retention ratio for 'current' role heads.")
    parser.add_argument("--kv_head_role_last_frame_ratio_memory", type=float, default=0.0,
                        help="Last-frame retention ratio for 'memory' role heads.")
    parser.add_argument("--kv_head_role_last_frame_ratio_mixed", type=float, default=0.0,
                        help="Last-frame retention ratio for 'mixed' role heads.")
    parser.add_argument("--short_term_frames", type=int, default=0,
                        help="Number of recent frames to fully protect from per-head top-k eviction. "
                             "Tokens from the last N frames bypass budget competition and are always "
                             "retained. 0 (default) = disabled.")
    parser.add_argument("--multi_scale", type=str2bool, nargs='?', const=True, default=False,
                        help="Enable multi-scale encoding: encode the last frame at both normal "
                             "and high resolution, giving the model both global (color/brightness) "
                             "and detailed (texture/text) views. Only active when --last_frame_stride > 1.0.")
    parser.add_argument("--layered_resolution", type=str2bool, nargs='?', const=True, default=False,
                        help="Layered-resolution encoding: shallow/mid layers receive high-res last-frame "
                             "tokens, deep layers receive only normal-res tokens to preserve color/attribute "
                             "semantics. Only active when --last_frame_stride > 1.0.")
    parser.add_argument("--fixed_base_resolution", type=str2bool, nargs='?', const=True, default=False,
                        help="Lock normal-frame processor to min_pixels=max_pixels (448x448), matching "
                             "DSCache's fixed-resolution strategy. All non-last frames use this fixed "
                             "low resolution to preserve color/temporal semantics.")
    parser.add_argument("--fixed_frame_canvas_size", type=int, default=0,
                        help="If >0, letterbox/resize every video frame to a fixed square canvas before "
                             "Qwen processing, making per-frame visual token count stable. 0 = disabled.")
    parser.add_argument("--fixed_frame_pad_mode", type=str, default="letterbox",
                        choices=["letterbox", "resize"],
                        help="Frame normalization for --fixed_frame_canvas_size. 'letterbox' preserves "
                             "aspect ratio with padding; 'resize' stretches to square.")
    parser.add_argument("--fixed_frame_pad_value", type=float, default=0.0,
                        help="Padding pixel value for letterbox fixed-frame canvas.")
    parser.add_argument("--transient_highres", type=str2bool, nargs='?', const=True, default=False,
                        help="Use high-res last frame only for current question; replace with "
                             "normal-res after answering to prevent cache pollution.")
    parser.add_argument("--feature_buffer_frames", type=int, default=0,
                        help="DSCache-style feature buffer: hold last N raw frames, encode as "
                             "independent Instant Cache at query time. 0 (default) = disabled.")
    parser.add_argument("--defer_last_frame_compress", type=str2bool, nargs='?', const=True, default=False,
                        help="Encode the final frame before answering without pre-answer compression; "
                             "compress it into cumulative KV after answering. Default disabled.")
    parser.add_argument("--defer_last_frame_budget_mode", type=str, default="overflow",
                        choices=["overflow", "strict"],
                        help="Budget handling for --defer_last_frame_compress. 'overflow' allows "
                             "temporary pre-answer KV overflow; 'strict' pre-compresses history to "
                             "reserve estimated final-frame visual tokens.")
    parser.add_argument("--defer_last_frame_keep_ratio", type=float, default=1.0,
                        help="Fraction of the final frame retained before answering in strict defer mode. "
                             "1.0 preserves legacy deferlast; 0.0 uses the normal non-deferred path.")
    parser.add_argument("--defer_last_frame_head_budget_threshold", type=int, default=0,
                        help="If positive, strict partial defer reserves a within-budget latest-frame "
                             "bank only for KV heads whose LU budget exceeds this threshold.")
    parser.add_argument("--kv_head_ragged_decode", type=str2bool, nargs='?', const=True, default=False,
                        help="Use experimental physical per-KV-head ragged cache for answer decode.")
    parser.add_argument("--kv_head_ragged_prefill", type=str2bool, nargs='?', const=True, default=False,
                        help="Use experimental physical per-KV-head ragged cache for video/text prefill and decode.")
    parser.add_argument("--rolekv_ragged", type=str2bool, nargs='?', const=True, default=False,
                        help="Use RoleKV-v2 role-aware physical per-KV-head ragged retention.")
    parser.add_argument("--rolekv_head_classes", type=str,
                        default="results/observations/head_classes_prev_current/head_classes.json",
                        help="Head-class JSON built from offline previous/current eager profiling.")
    parser.add_argument(
        "--rolekv_ragged_mode",
        type=str,
        default="rolekv_quota",
        choices=["baseline", "uniform", "norole", "rolekv", "rolekv_quota", "random", "random_quota", "inverted", "inverted_quota"],
        help="RoleKV-v2 role assignment/retention mode for ragged compression.",
    )
    parser.add_argument("--rolekv_quota_ratio", type=float, default=0.7,
                        help="Fraction of a role KV-head budget reserved for its target region.")
    parser.add_argument("--rolekv_lambda_memory", type=float, default=0.2,
                        help="Soft score bonus added to previous-memory tokens for memory-oriented KV heads.")
    parser.add_argument("--rolekv_lambda_current", type=float, default=0.2,
                        help="Soft score bonus added to latest-chunk tokens for current-sensitive KV heads.")
    parser.add_argument("--rolekv_seed", type=int, default=0,
                        help="Seed for random/inverted RoleKV control modes.")
    parser.add_argument("--rolekv_role_min_votes", type=int, default=1,
                        help="Minimum query-head votes needed to assign a KV head to a non-mixed role.")
    parser.add_argument("--answer_head_router", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--answer_head_profile", type=str, default=None)
    parser.add_argument("--answer_head_router_mode", type=str, default="question",
                        choices=["question", "inverted", "static_memory", "baseline"])
    parser.add_argument("--answer_head_router_strength", type=float, default=0.2)
    parser.add_argument("--answer_head_min_gate", type=float, default=0.8)
    parser.add_argument("--answer_head_max_gate", type=float, default=1.2)
    parser.add_argument("--answer_head_start_layer", type=int, default=8)
    parser.add_argument("--streaming", type=str2bool, nargs='?', const=True, default=False,
                        help="Streaming (online) mode. If False (default), uses offline mode where should_compact is always True.")
    parser.add_argument("--last_frame_stride", type=float, default=1.0,
                        help="Stride for last-frame spatial resolution. >1.0 gives the last frame "
                             "before a question higher resolution (more visual tokens per frame). "
                             "1.0 (default) = disabled, same resolution for all frames.")
    parser.add_argument("--partition_position_frames", type=int, default=0,
                        help="Number of recent frames to protect with high position IDs during "
                             "position compaction. When >0, recent-frame tokens get higher position "
                             "values while older tokens are compacted to low values. "
                             "0 (default) = disabled (global uniform compaction).")
    args = parser.parse_args()

    from head_analysis.horizonkv_policy import validate_horizonkv_args
    validate_horizonkv_args(args)

    if args.kv_head_recency_scores:
        if not args.kv_head_budget_scores:
            raise ValueError(
                "--kv_head_recency_scores requires --kv_head_budget_scores so capacity "
                "and eviction preference remain separate axes."
            )
        if args.kv_head_policy_roles:
            raise ValueError(
                "--kv_head_recency_scores and --kv_head_policy_roles are mutually "
                "exclusive eviction policies."
            )
        if not os.path.isfile(args.kv_head_recency_scores):
            raise FileNotFoundError(
                f"KV-head recency profile not found: {args.kv_head_recency_scores}"
            )
        if args.kv_head_recency_delta < 0.0:
            raise ValueError("--kv_head_recency_delta must be non-negative.")
        if args.kv_head_recency_tau <= 0.0:
            raise ValueError("--kv_head_recency_tau must be positive.")
        if not (
            0.0 <= args.kv_head_recency_alpha_min
            < args.kv_head_recency_alpha_max <= 1.0
        ):
            raise ValueError(
                "KV-head recency alpha bounds must satisfy 0 <= min < max <= 1."
            )
        if args.kv_head_recency_k_delta < 0.0:
            raise ValueError("--kv_head_recency_k_delta must be non-negative.")
        if not (
            0.0 <= args.kv_head_recency_k_min < args.kv_head_recency_k_max
        ):
            raise ValueError(
                "KV-head recency k bounds must satisfy 0 <= min < max."
            )
        args.kv_head_ragged_prefill = True

    if args.timestamp_v8_policy:
        if not os.path.isfile(args.timestamp_v8_policy):
            raise FileNotFoundError(
                f"Timestamp-v8 profile not found: {args.timestamp_v8_policy}"
            )
        if args.timestamp_v8_grace_frames < 0:
            raise ValueError("--timestamp_v8_grace_frames must be non-negative")
        conflicts = {
            "--horizonkv_policy": args.horizonkv_policy,
            "--kv_head_policy_roles": args.kv_head_policy_roles,
            "--kv_head_recency_scores": args.kv_head_recency_scores,
            "--kv_head_role_recent_scores": args.kv_head_role_recent_scores,
            "--kv_head_role_dual_scores": args.kv_head_role_dual_scores,
            "--rolekv_ragged": args.rolekv_ragged,
        }
        enabled = [name for name, value in conflicts.items() if value]
        if enabled:
            raise ValueError(
                "--timestamp_v8_policy is mutually exclusive with " + ", ".join(enabled)
            )
        args.kv_head_ragged_prefill = True

    budget_modes = [
        bool(args.head_budget_scores),
        bool(args.layer_budget_scores),
        bool(args.kv_head_budget_scores),
    ]
    if sum(budget_modes) > 1:
        raise ValueError(
            "Use only one of --head_budget_scores, --layer_budget_scores, "
            "or --kv_head_budget_scores."
        )

    if not args.debug:
        logzero.loglevel(logging.INFO)
        warnings.filterwarnings('ignore')

    os.makedirs(args.save_dir, exist_ok=True)

    # fix random seed
    random.seed(2024)
    logger.info('seed: 2024')

    # VideoQA model
    model_path = args.model_path or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        MODELS[args.model]['model_path'],
    )
    model_path = os.path.expanduser(model_path)
    load_func = MODELS[args.model]['load_func']
    logger.info(f"Loading VideoQA model: {model_path}")
    videoqa_model, videoqa_processor = load_func(
        model_path=model_path,
        kv_size=args.kv_size,
        streaming=args.streaming,
        sample_fps=args.sample_fps,
        compress_mode=args.compress_mode,
        last_frame_stride=args.last_frame_stride,
        partition_position_frames=args.partition_position_frames,
        multi_scale=args.multi_scale,
        layered_resolution=args.layered_resolution,
        fixed_base_resolution=args.fixed_base_resolution,
        transient_highres=args.transient_highres,
        feature_buffer_frames=args.feature_buffer_frames,
        fixed_frame_canvas_size=args.fixed_frame_canvas_size,
        fixed_frame_pad_mode=args.fixed_frame_pad_mode,
        fixed_frame_pad_value=args.fixed_frame_pad_value,
    )
    if args.head_budget_scores:
        from head_analysis.hermes_head_budget import apply_head_budget

        language_model = getattr(videoqa_model, "language_model", None)
        language_config = getattr(language_model, "config", None)
        num_heads = getattr(language_config, "num_attention_heads", None)
        if num_heads is None and hasattr(language_model, "model"):
            num_heads = getattr(language_model.model.config, "num_attention_heads", None)
        if num_heads is None:
            num_heads = 28

        videoqa_model = apply_head_budget(
            videoqa_model,
            head_scores_path=args.head_budget_scores,
            num_layers=videoqa_model.num_layers,
            num_heads=num_heads,
        )

    if args.layer_budget_scores:
        from head_analysis.hermes_layer_budget import apply_layer_budget

        language_model = getattr(videoqa_model, "language_model", None)
        language_config = getattr(language_model, "config", None)
        num_heads = getattr(language_config, "num_attention_heads", None)
        if num_heads is None and hasattr(language_model, "model"):
            num_heads = getattr(language_model.model.config, "num_attention_heads", None)
        if num_heads is None:
            num_heads = 28

        videoqa_model = apply_layer_budget(
            videoqa_model,
            layer_scores_path=args.layer_budget_scores,
            num_layers=videoqa_model.num_layers,
            num_heads=num_heads,
            variable_lengths=args.layer_budget_variable,
        )

    if args.kv_head_budget_scores:
        from head_analysis.hermes_kv_head_budget import apply_kv_head_budget

        language_model = getattr(videoqa_model, "language_model", None)
        language_config = getattr(language_model, "config", None)
        num_heads = getattr(language_config, "num_attention_heads", None)
        num_kv_heads = getattr(language_config, "num_key_value_heads", None)
        if num_heads is None and hasattr(language_model, "model"):
            num_heads = getattr(language_model.model.config, "num_attention_heads", None)
        if num_kv_heads is None and hasattr(language_model, "model"):
            num_kv_heads = getattr(language_model.model.config, "num_key_value_heads", None)
        if num_heads is None:
            num_heads = 28
        if num_kv_heads is None:
            num_kv_heads = 4

        videoqa_model = apply_kv_head_budget(
            videoqa_model,
            kv_head_scores_path=args.kv_head_budget_scores,
            num_layers=videoqa_model.num_layers,
            num_query_heads=num_heads,
            num_kv_heads=num_kv_heads,
            budget_scheme=args.kv_head_budget_scheme,
            sparsemm_ratio=args.kv_head_budget_sparsemm_ratio,
            sparsemm_window_size=args.kv_head_budget_sparsemm_window_size,
            floor_ratio=args.kv_head_budget_floor_ratio,
            union_cap_ratio=args.kv_head_budget_union_cap_ratio,
            max_mask_q_len=args.kv_head_budget_max_mask_q_len,
            memory_current_split=args.kv_head_budget_memory_current_split,
            memory_scores_path=args.kv_head_budget_memory_scores,
            base_memory_ratio=args.kv_head_budget_memory_ratio,
            split_dynamic_scale=args.kv_head_budget_split_dynamic_scale,
            short_term_frames=args.short_term_frames,
            layerwise_last_frame=args.kv_head_layerwise_last_frame,
            layerwise_shallow_end=args.kv_head_layerwise_shallow_end,
            layerwise_mid_end=args.kv_head_layerwise_mid_end,
            layerwise_shallow_ratio=args.kv_head_layerwise_shallow_ratio,
            layerwise_mid_ratio=args.kv_head_layerwise_mid_ratio,
            layerwise_deep_ratio=args.kv_head_layerwise_deep_ratio,
        )
        videoqa_model._kv_head_selector_policy = args.kv_head_selector_policy

    videoqa_model.defer_last_frame_compress = args.defer_last_frame_compress
    videoqa_model.defer_last_frame_budget_mode = args.defer_last_frame_budget_mode
    videoqa_model.defer_last_frame_keep_ratio = args.defer_last_frame_keep_ratio
    videoqa_model.defer_last_frame_head_budget_threshold = max(
        0, int(args.defer_last_frame_head_budget_threshold)
    )
    videoqa_model._answer_history_max_age_seconds = max(
        0.0, float(args.answer_history_max_age_seconds)
    )

    if not 0.0 <= args.defer_last_frame_keep_ratio <= 1.0:
        raise ValueError("--defer_last_frame_keep_ratio must be in [0, 1].")
    partial_defer = (
        args.defer_last_frame_compress
        and 0.0 < args.defer_last_frame_keep_ratio < 1.0
    )
    if partial_defer and args.defer_last_frame_budget_mode != "strict":
        raise ValueError(
            "Partial defer requires --defer_last_frame_budget_mode strict."
        )
    if partial_defer and args.kv_head_role_budget_enable:
        raise ValueError(
            "Partial defer is not supported with --kv_head_role_budget_enable; "
            "use the v8 per-head budget table without role overrides."
        )
    if partial_defer and args.kv_head_layerwise_last_frame:
        raise ValueError(
            "Partial defer and --kv_head_layerwise_last_frame are mutually "
            "exclusive; use only --defer_last_frame_keep_ratio."
        )
    if args.defer_last_frame_head_budget_threshold > 0:
        if not partial_defer:
            raise ValueError(
                "--defer_last_frame_head_budget_threshold requires strict partial defer "
                "with --defer_last_frame_keep_ratio in (0, 1)."
            )
        if not args.kv_head_ragged_prefill:
            logger.warning(
                "Head-budget defer requires physical ragged prefill; enabling "
                "--kv_head_ragged_prefill."
            )
            args.kv_head_ragged_prefill = True
    if partial_defer and not args.kv_head_ragged_prefill:
        logger.warning(
            "Partial defer requires physical ragged prefill; enabling "
            "--kv_head_ragged_prefill."
        )
        args.kv_head_ragged_prefill = True

    if args.defer_last_frame_compress and args.feature_buffer_frames > 0:
        logger.warning(
            "--defer_last_frame_compress is ignored when --feature_buffer_frames > 0; "
            "feature buffer already keeps final raw frames for query-time Instant Cache."
        )

    if args.kv_head_role_recent_scores and args.kv_head_role_dual_scores:
        raise ValueError(
            "Use only one role source: --kv_head_role_recent_scores or "
            "--kv_head_role_dual_scores."
        )
    role_score_path = args.kv_head_role_dual_scores or args.kv_head_role_recent_scores
    for name, value in (
        ("--kv_head_role_history_uniform_memory_ratio", args.kv_head_role_history_uniform_memory_ratio),
        ("--kv_head_role_history_uniform_mixed_ratio", args.kv_head_role_history_uniform_mixed_ratio),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1].")
    if args.kv_head_role_history_uniform_bins <= 0:
        raise ValueError("--kv_head_role_history_uniform_bins must be positive.")
    if args.kv_head_role_budget_enable and not role_score_path:
        raise ValueError(
            "--kv_head_role_budget_enable requires --kv_head_role_recent_scores "
            "or --kv_head_role_dual_scores to define head roles."
        )
    uniform_history_enabled = (
        args.kv_head_role_history_uniform_memory_ratio > 0.0
        or args.kv_head_role_history_uniform_mixed_ratio > 0.0
    )
    if uniform_history_enabled and not role_score_path:
        raise ValueError(
            "Role history-uniform selection requires --kv_head_role_recent_scores "
            "or --kv_head_role_dual_scores."
        )

    if role_score_path and not args.kv_head_ragged_prefill:
        logger.warning("KV-head role selection requires physical ragged prefill; enabling --kv_head_ragged_prefill.")
        args.kv_head_ragged_prefill = True

    if args.rolekv_ragged and not args.kv_head_ragged_prefill:
        logger.warning("--rolekv_ragged requires physical ragged prefill; enabling --kv_head_ragged_prefill.")
        args.kv_head_ragged_prefill = True
    if args.answer_head_router:
        if not args.answer_head_profile:
            raise ValueError("--answer_head_router requires --answer_head_profile")
        if not os.path.isfile(args.answer_head_profile):
            raise FileNotFoundError(
                f"Answer-head profile not found: {args.answer_head_profile}"
            )
        if not args.kv_head_ragged_prefill:
            logger.warning(
                "--answer_head_router requires physical ragged prefill; "
                "enabling --kv_head_ragged_prefill."
            )
            args.kv_head_ragged_prefill = True

    if args.kv_head_ragged_prefill:
        from head_analysis.hermes_kv_head_ragged_prefill import apply_kv_head_ragged_prefill

        videoqa_model = apply_kv_head_ragged_prefill(videoqa_model)
        if args.timestamp_v8_policy:
            from head_analysis.timestamp_v8_policy import install_timestamp_v8_policy

            videoqa_model = install_timestamp_v8_policy(
                videoqa_model,
                csv_path=args.timestamp_v8_policy,
                grace_frames=args.timestamp_v8_grace_frames,
                trace_path=args.ragged_frame_trace_path,
            )
        elif args.ragged_frame_trace_path:
            from head_analysis.timestamp_v8_policy import install_ragged_frame_trace

            videoqa_model = install_ragged_frame_trace(
                videoqa_model, args.ragged_frame_trace_path
            )
        if args.ragged_readout_trace_path:
            from head_analysis.hermes_kv_head_ragged_prefill import install_ragged_readout_trace

            videoqa_model = install_ragged_readout_trace(
                videoqa_model,
                args.ragged_readout_trace_path,
                score_last_n=args.ragged_readout_trace_last_n,
            )
        if args.ragged_token_trace_path:
            from head_analysis.hermes_kv_head_ragged_prefill import install_ragged_token_trace

            videoqa_model = install_ragged_token_trace(
                videoqa_model,
                args.ragged_token_trace_path,
                score_last_n=args.ragged_readout_trace_last_n,
            )
        if args.ragged_redundancy_trace_path:
            from head_analysis.hermes_kv_head_ragged_prefill import (
                install_ragged_redundancy_trace,
            )

            videoqa_model = install_ragged_redundancy_trace(
                videoqa_model,
                args.ragged_redundancy_trace_path,
                max_tokens=args.ragged_redundancy_trace_max_tokens,
            )
        if args.prospective_lifecycle_trace_path:
            from head_analysis.hermes_kv_head_ragged_prefill import (
                install_prospective_lifecycle_trace,
            )

            videoqa_model = install_prospective_lifecycle_trace(
                videoqa_model,
                args.prospective_lifecycle_trace_path,
                score_last_n=args.prospective_lifecycle_trace_last_n,
                top_fraction=args.prospective_lifecycle_top_fraction,
                visual_query_samples=args.prospective_lifecycle_visual_queries,
                visual_top_readers=args.prospective_lifecycle_visual_top_readers,
                content_bins=args.prospective_lifecycle_content_bins,
            )
        if args.kv_head_recency_scores:
            from head_analysis.hermes_kv_head_ragged_prefill import install_kv_head_recency_profile

            videoqa_model = install_kv_head_recency_profile(
                videoqa_model,
                csv_path=args.kv_head_recency_scores,
                metric=args.kv_head_recency_metric,
                delta=args.kv_head_recency_delta,
                tau=args.kv_head_recency_tau,
                alpha_min=args.kv_head_recency_alpha_min,
                alpha_max=args.kv_head_recency_alpha_max,
                k_delta=args.kv_head_recency_k_delta,
                k_min=args.kv_head_recency_k_min,
                k_max=args.kv_head_recency_k_max,
            )
        if args.horizonkv_policy:
            from head_analysis.horizonkv_policy import install_horizonkv_policy

            videoqa_model = install_horizonkv_policy(
                videoqa_model,
                csv_path=args.horizonkv_policy,
                recent_capacity=args.horizonkv_recent_capacity,
                ema_gamma=args.horizonkv_ema_gamma,
                apply_age_penalty=args.horizonkv_apply_age_penalty,
                temporal_coverage=args.horizonkv_temporal_coverage,
                probe_style=args.horizonkv_probe_style,
                score_blend=args.horizonkv_score_blend,
            )
        if args.kv_head_policy_roles:
            from head_analysis.hermes_kv_head_ragged_prefill import install_kv_head_policy_roles

            videoqa_model = install_kv_head_policy_roles(
                videoqa_model,
                csv_path=args.kv_head_policy_roles,
                memory_quantile=args.kv_head_policy_memory_quantile,
                current_quantile=args.kv_head_policy_current_quantile,
                memory_bonus=args.kv_head_policy_memory_bonus,
                current_bonus=args.kv_head_policy_current_bonus,
            )
        if role_score_path:
            from head_analysis.hermes_kv_head_ragged_prefill import install_kv_head_role_recent_retention

            videoqa_model = install_kv_head_role_recent_retention(
                videoqa_model,
                csv_path=args.kv_head_role_recent_scores,
                dual_scores_path=args.kv_head_role_dual_scores,
                memory_quantile=args.kv_head_role_recent_memory_quantile,
                current_quantile=args.kv_head_role_recent_current_quantile,
                memory_count=args.kv_head_role_recent_memory_count,
                current_count=args.kv_head_role_recent_current_count,
                memory_recent_tokens=args.kv_head_role_recent_memory_tokens,
                mixed_recent_tokens=args.kv_head_role_recent_mixed_tokens,
                current_recent_tokens=args.kv_head_role_recent_current_tokens,
                window_mode=args.kv_head_role_recent_window_mode,
                uniform_budget=args.kv_head_role_recent_uniform_budget,
                role_budget_enable=args.kv_head_role_budget_enable,
                memory_budget=args.kv_head_role_memory_budget,
                mixed_budget=args.kv_head_role_mixed_budget,
                current_budget=args.kv_head_role_current_budget,
                budget_auto_mixed=args.kv_head_role_budget_auto_mixed,
                history_uniform_memory_ratio=args.kv_head_role_history_uniform_memory_ratio,
                history_uniform_mixed_ratio=args.kv_head_role_history_uniform_mixed_ratio,
                history_uniform_bins=args.kv_head_role_history_uniform_bins,
                last_frame_by_role=args.kv_head_role_last_frame_by_role,
                last_frame_ratio_current=args.kv_head_role_last_frame_ratio_current,
                last_frame_ratio_memory=args.kv_head_role_last_frame_ratio_memory,
                last_frame_ratio_mixed=args.kv_head_role_last_frame_ratio_mixed,
            )
        if args.rolekv_ragged:
            from head_analysis.rolekv_ragged_policy import apply_rolekv_ragged_policy

            videoqa_model = apply_rolekv_ragged_policy(
                videoqa_model,
                head_classes_path=args.rolekv_head_classes,
                mode=args.rolekv_ragged_mode,
                quota_ratio=args.rolekv_quota_ratio,
                lambda_memory=args.rolekv_lambda_memory,
                lambda_current=args.rolekv_lambda_current,
                seed=args.rolekv_seed,
                role_min_votes=args.rolekv_role_min_votes,
            )
        if args.answer_head_router:
            from head_analysis.answer_head_router import install_answer_head_router

            videoqa_model = install_answer_head_router(
                videoqa_model,
                profile_path=args.answer_head_profile,
                mode=args.answer_head_router_mode,
                strength=args.answer_head_router_strength,
                min_gate=args.answer_head_min_gate,
                max_gate=args.answer_head_max_gate,
                start_layer=args.answer_head_start_layer,
            )
    elif args.kv_head_ragged_decode:
        from head_analysis.hermes_kv_head_ragged import apply_kv_head_ragged_decode

        videoqa_model = apply_kv_head_ragged_decode(videoqa_model)

    # Load ground truth file
    anno = json.load(open(args.anno_path))

    analyzer = QA_CLASS(
        anno=anno,
        sample_fps=args.sample_fps,
        qa_model=videoqa_model,
        qa_processor=videoqa_processor,
        num_chunks=args.num_chunks,
        chunk_idx=args.chunk_idx,
        save_dir=args.save_dir,
        encode_chunk_size=args.encode_chunk_size,
    )

    analyzer.analyze(debug=args.debug)
