import math
import os
import torch
from logzero import logger
from tqdm import tqdm

from video_qa.base import BaseVQA, work


class HermesVQA(BaseVQA):
    """
    Unified VQA class for both offline and streaming benchmarks.
    
    Streaming mode is auto-detected per conversation sample:
    if a sample has 'end_time', frames are encoded up to that timestamp;
    otherwise all frames are encoded before answering.
    """


    def _estimate_frame_token_count(self, frame):
        """Estimate one Qwen-VL frame's LLM visual token count from processor grid."""
        processor = getattr(self.qa_model, "processor", None)
        if (
            getattr(self.qa_model, "last_frame_stride", 1.0) != 1.0
            and getattr(self.qa_model, "processor_last", None) is not None
        ):
            processor = self.qa_model.processor_last
        vision_config = getattr(getattr(self.qa_model, "config", None), "vision_config", None)
        if processor is None or vision_config is None:
            return None

        frame_for_processor = frame
        if (
            int(getattr(self.qa_model, "fixed_frame_canvas_size", 0) or 0) > 0
            and hasattr(self.qa_model, "_apply_fixed_frame_canvas")
        ):
            frame_for_processor = self.qa_model._apply_fixed_frame_canvas(frame_for_processor)
        elif len(frame_for_processor.shape) == 4 and frame_for_processor.shape[-1] == 3:
            frame_for_processor = frame_for_processor.permute(0, 3, 1, 2)
        try:
            video_input = processor(text=[""], videos=frame_for_processor, return_tensors="pt")
            grid = video_input["video_grid_thw"][0].tolist()
            spatial_merge_size = int(getattr(vision_config, "spatial_merge_size", 2))
            return int(grid[0]) * int(grid[1] // spatial_merge_size) * int(grid[2] // spatial_merge_size)
        except Exception as exc:
            logger.warning(f"[defer_last_frame] Failed to estimate last-frame token count: {exc}")
            return None

    def _compress_with_temporary_kv_size(self, kv_size):
        original_kv_size = self.qa_model.kv_size
        try:
            self.qa_model.kv_size = max(1, int(kv_size))
            self.qa_model.predict_and_compress()
        finally:
            self.qa_model.kv_size = original_kv_size

    def _role_defer_history_budgets(self):
        cfg = getattr(self.qa_model, "_kv_head_role_recent_retention", None)
        if not cfg or not bool(cfg.get("role_budget_enable", False)):
            return None

        mixed_budget = cfg.get("mixed_budget", getattr(self.qa_model, "kv_size", 0))
        if bool(cfg.get("budget_auto_mixed", False)):
            mixed_budget = cfg.get("resolved_mixed_budget", mixed_budget)
        role_budgets = {
            "memory": int(cfg.get("memory_budget", getattr(self.qa_model, "kv_size", 0))),
            "mixed": int(mixed_budget),
            "current": int(cfg.get("current_budget", getattr(self.qa_model, "kv_size", 0))),
        }
        recent_tokens = {
            "memory": int(cfg.get("memory_recent_tokens", 0)),
            "mixed": int(cfg.get("mixed_recent_tokens", 0)),
            "current": int(cfg.get("current_recent_tokens", 0)),
        }
        return {
            role: max(0, int(role_budgets[role]) - int(recent_tokens[role]))
            for role in ("memory", "mixed", "current")
        }

    def _compress_with_temporary_role_history_budgets(self, history_budgets):
        cfg = getattr(self.qa_model, "_kv_head_role_recent_retention", None)
        if not cfg:
            self.qa_model.predict_and_compress()
            return

        original_kv_size = self.qa_model.kv_size
        original_values = {
            "memory_budget": cfg.get("memory_budget"),
            "mixed_budget": cfg.get("mixed_budget"),
            "current_budget": cfg.get("current_budget"),
            "budget_auto_mixed": cfg.get("budget_auto_mixed"),
            "memory_recent_tokens": cfg.get("memory_recent_tokens"),
            "mixed_recent_tokens": cfg.get("mixed_recent_tokens"),
            "current_recent_tokens": cfg.get("current_recent_tokens"),
            "window_mode": cfg.get("window_mode"),
        }
        trigger_budget = min(max(1, int(value)) for value in history_budgets.values())
        try:
            cfg["memory_budget"] = int(history_budgets["memory"])
            cfg["mixed_budget"] = int(history_budgets["mixed"])
            cfg["current_budget"] = int(history_budgets["current"])
            cfg["budget_auto_mixed"] = False
            cfg["memory_recent_tokens"] = 0
            cfg["mixed_recent_tokens"] = 0
            cfg["current_recent_tokens"] = 0
            cfg["window_mode"] = "override"
            self.qa_model.kv_size = max(1, int(trigger_budget))
            self.qa_model.predict_and_compress()
        finally:
            self.qa_model.kv_size = original_kv_size
            for key, value in original_values.items():
                cfg[key] = value

    def _encode_deferred_last_frame(self, video_chunk):
        mode = getattr(self.qa_model, "defer_last_frame_budget_mode", "overflow")
        keep_ratio = float(
            getattr(self.qa_model, "defer_last_frame_keep_ratio", 1.0)
        )
        head_budget_threshold = int(getattr(
            self.qa_model, "defer_last_frame_head_budget_threshold", 0
        ) or 0)
        head_budget_defer = (
            mode == "strict"
            and head_budget_threshold > 0
            and 0.0 < keep_ratio < 1.0
        )
        if keep_ratio <= 0.0:
            logger.info(
                "[partial_defer] keep_ratio=0; using the normal non-deferred path"
            )
            self.qa_model.encode_video_chunk(video_chunk, is_last_chunk=True)
            self.qa_model.predict_and_compress()
            return False

        compress_after_answer = True
        role_history_budgets = None
        head_budget_table = None
        head_reserve_table = None
        if video_chunk.shape[0] > 1:
            history_chunk = video_chunk[:-1]
            self.qa_model.encode_video_chunk(history_chunk, is_last_chunk=False)
            logger.info("[defer_last_frame] Compressing non-last frames before deferring last frame")
            self.qa_model.predict_and_compress()

        last_frame = video_chunk[-1:]
        if mode == "strict":
            if head_budget_defer:
                logger.info(
                    "[head_budget_defer] Compressing history with strict "
                    f"per-head reserve (threshold={head_budget_threshold}, "
                    f"ratio={keep_ratio:g})"
                )
                head_budget_table, head_reserve_table = (
                    self.qa_model.prepare_head_budget_deferred_history(
                        keep_ratio, head_budget_threshold
                    )
                )
            else:
                role_history_budgets = self._role_defer_history_budgets()
                if role_history_budgets is not None:
                    logger.info(
                        "[defer_last_frame] Strict role-budget mode: compressing history "
                        "with role budgets after reserving role recent windows "
                        f"(memory={role_history_budgets['memory']}, "
                        f"mixed={role_history_budgets['mixed']}, "
                        f"current={role_history_budgets['current']})"
                    )
                    self._compress_with_temporary_role_history_budgets(role_history_budgets)
                else:
                    last_tokens = self._estimate_frame_token_count(last_frame)
                    if last_tokens is not None:
                        reserved_tokens = int(math.ceil(int(last_tokens) * keep_ratio))
                        reserved_kv_size = max(
                            1, int(self.qa_model.kv_size) - reserved_tokens
                        )
                        logger.info(
                            "[defer_last_frame] Strict mode: compressing history to "
                            f"{reserved_kv_size} before appending last frame "
                            f"(frame_tokens={last_tokens}, reserved={reserved_tokens}, "
                            f"keep_ratio={keep_ratio:g}, kv_size={self.qa_model.kv_size})"
                        )
                        self._compress_with_temporary_kv_size(reserved_kv_size)
                    else:
                        if 0.0 < keep_ratio < 1.0:
                            raise RuntimeError(
                                "Partial defer could not estimate final-frame token count; "
                                "refusing to fall back to an over-budget result."
                            )
                        logger.warning("[defer_last_frame] Falling back to overflow mode for this sample")

        history_lens = None
        if 0.0 < keep_ratio < 1.0:
            if not hasattr(self.qa_model.kv_cache, "snapshot_head_lens"):
                raise RuntimeError(
                    "Partial defer requires --kv_head_ragged_prefill True"
                )
            history_lens = self.qa_model.kv_cache.snapshot_head_lens()

        self.qa_model.encode_video_chunk(last_frame, is_last_chunk=True)
        if head_budget_defer:
            self.qa_model.prune_head_budget_deferred_last_frame(
                history_lens, head_budget_table, head_reserve_table
            )
            logger.info(
                "[head_budget_defer] Latest frame independently compressed "
                "and concatenated for answering"
            )
        elif 0.0 < keep_ratio < 1.0:
            self.qa_model.prune_partial_deferred_last_frame(
                history_lens, keep_ratio
            )
            logger.info(
                "[partial_defer] Last frame partially retained before answering "
                f"(keep_ratio={keep_ratio:g})"
            )
        if mode == "strict" and role_history_budgets is not None:
            logger.info(
                "[defer_last_frame] Compressing last frame into role recent windows "
                "before answering"
            )
            self.qa_model.predict_and_compress()
            compress_after_answer = False
            logger.info("[defer_last_frame] Last frame encoded and compressed before answering")
        else:
            logger.info("[defer_last_frame] Last frame encoded without pre-answer compression")
        return compress_after_answer

    @torch.inference_mode()
    def analyze_a_video(self, video_sample, encode_chunk_size=None):
        if encode_chunk_size is None:
            encode_chunk_size = self.encode_chunk_size
        video_path = video_sample['video_path']
        self.qa_model._active_video_id = video_sample.get('video_id')

        video_fps = video_sample.get('fps', None)
        clip = video_sample.get('clip', None)

        if video_path.endswith('.npy'):
            video = self.load_video(video_path, clip=clip)
            video_tensor = torch.from_numpy(video)
        elif os.path.isdir(video_path):
            if video_fps is None:
                raise ValueError(f"video_fps must be provided for image-based video: {video_path}")
            video = self.load_video_frames(video_path, video_fps, clip=clip)
            video_tensor = torch.from_numpy(video)
        else:
            video = self.load_video(video_path, clip=clip)
            video_tensor = torch.from_numpy(video)

        self.qa_model.clear_cache()
        self.qa_model.encode_init_prompt()

        current_frame_idx = 0
        feature_buffer_raw = []
        fb_frames = getattr(self.qa_model, 'feature_buffer_frames', 0) or 0
        deferred_last_frame_pending = False

        for question_index, sample in enumerate(tqdm(video_sample['conversations'])):
            logger.debug(f'sample: {sample}')
            question = sample['question']
            answer = sample['answer']
            self.qa_model._active_question_index = question_index
            self.qa_model._active_question = question
            self.qa_model._active_question_end_time = sample.get('end_time')

            if 'end_time' in sample:
                end_frame_idx = math.ceil(sample['end_time'] * self.sample_fps)
            else:
                end_frame_idx = len(video_tensor)

            while current_frame_idx < end_frame_idx:
                next_encode_end = min(current_frame_idx + encode_chunk_size, end_frame_idx)
                if next_encode_end > current_frame_idx:
                    print(f"Encoding frames {current_frame_idx} to {next_encode_end-1}")
                    video_chunk = video_tensor[current_frame_idx:next_encode_end]
                    is_last_chunk = (next_encode_end == end_frame_idx)

                    defer_last_frame = (
                        getattr(self.qa_model, 'defer_last_frame_compress', False)
                        and fb_frames <= 0
                        and is_last_chunk
                    )

                    if fb_frames > 0 and is_last_chunk:
                        # Phase 1: store last chunk in buffer, not in Cumulative
                        for i in range(video_chunk.shape[0]):
                            feature_buffer_raw.append(video_chunk[i:i+1].clone())
                        print(f"[FB] Buffer size: {len(feature_buffer_raw)}")
                    elif defer_last_frame:
                        deferred_last_frame_pending = self._encode_deferred_last_frame(video_chunk)
                    else:
                        self.qa_model.encode_video_chunk(video_chunk, is_last_chunk=is_last_chunk)
                        logger.info(f"Triggering question prediction and KV compression")
                        self.qa_model.predict_and_compress()

                    current_frame_idx = next_encode_end

                # Drain buffer overflow into Cumulative
                while len(feature_buffer_raw) > fb_frames:
                    oldest = feature_buffer_raw.pop(0)
                    self.qa_model.encode_video_chunk(oldest, is_last_chunk=False)
                    self.qa_model.predict_and_compress()
                    print(f"[FB] Drained 1 frame to Cumulative, buffer={len(feature_buffer_raw)}")

            # ---- Build Instant Cache before answering ----
            if feature_buffer_raw:
                instant_kv = self.qa_model.encode_instant_cache(feature_buffer_raw)
                self.qa_model._assemble_instant_dense(instant_kv)

            if 'choices' in sample:
                choices = sample['choices']
                if answer is None:
                    answer = choices[0]
                correct_choice = self.choice_letters[choices.index(answer)]
                qa_results = self.video_close_qa(question, choices, correct_choice)
                print("Pred Answer: ", qa_results['pred_answer'])

                record_entry = {
                    'video_id': video_sample['video_id'],
                    'question': question,
                    'choices': choices,
                    'answer': answer,
                    'correct_choice': correct_choice,
                    'pred_answer': qa_results['pred_answer'],
                    'pred_choice': qa_results['pred_choice'],
                    'qa_acc': qa_results['acc'] * 100,
                }
            else:
                qa_results = self.video_open_qa(question, max_new_tokens=256)
                print("Pred Answer: ", qa_results['pred_answer'])

                record_entry = {
                    'video_id': video_sample['video_id'],
                    'question': question,
                    'answer': answer,
                    'pred_answer': qa_results['pred_answer'],
                }

            task = sample.get('task', sample.get('question_type', video_sample.get('task', None)))
            if task is not None:
                record_entry['task'] = task

            duration_category = video_sample.get('duration_category', None)
            if duration_category is not None:
                record_entry['duration_category'] = duration_category

            # Transient high-res: replace with normal-res after answering
            if getattr(self.qa_model, 'transient_highres', False):
                if hasattr(self.qa_model, 'replace_highres_with_normal_ragged'):
                    self.qa_model.replace_highres_with_normal_ragged()
                elif hasattr(self.qa_model, 'replace_highres_with_normal'):
                    self.qa_model.replace_highres_with_normal()

            if deferred_last_frame_pending:
                logger.info("[defer_last_frame] Compressing deferred last frame after answering")
                self.qa_model.predict_and_compress()
                deferred_last_frame_pending = False

            # Phase 1: absorb Instant Cache into Cumulative after answering
            if feature_buffer_raw:
                self.qa_model._disassemble_instant_dense()
                for frame in feature_buffer_raw:
                    self.qa_model.encode_video_chunk(frame, is_last_chunk=False)
                self.qa_model.predict_and_compress()
                print(f"[FB] Absorbed {len(feature_buffer_raw)} frames into Cumulative")
                feature_buffer_raw = []

            self.record.append(record_entry)


if __name__ == "__main__":
    work(HermesVQA)
