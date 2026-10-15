"""Offline StreamingBench task: one dataset document is one complete video."""

import hashlib
import json
import os
from pathlib import Path

from datasets import Dataset, DatasetDict
from lmms_eval.api.task import ConfigurableTask

from sparsemm_baselines.run_streamingbench import LETTERS, extract_choice, prepare_annotations


DEFAULT_ANNOTATIONS = (
    Path(__file__).resolve().parents[2]
    / "hermes/data/streamingbench/streamingbench_realtime_half_userpaths.json"
)


def video_from_doc(doc):
    return json.loads(doc["video_json"])


def doc_to_visual(doc):
    # The model reads the original encoded video incrementally, through video_json.
    return []


def doc_to_text(doc):
    video = video_from_doc(doc)
    # No ground truth enters model context; include source identity in response-cache keys.
    return json.dumps({
        "source_sha256": doc["source_sha256"], "video_id": video["video_id"],
        "questions": [
            {key: question[key] for key in ("question", "choices", "end_time")}
            for question in video["conversations"]
        ],
    }, sort_keys=True, ensure_ascii=False)


def doc_to_target(doc):
    return json.dumps([
        LETTERS[question["choices"].index(question["answer"])]
        for question in video_from_doc(doc)["conversations"]
    ])


def process_results(doc, results):
    if len(results) != 1:
        raise ValueError("Expected one video-level response")
    try:
        answers = json.loads(results[0])
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("StreamingBench response must be a JSON answer array") from exc
    questions = video_from_doc(doc)["conversations"]
    if not isinstance(answers, list) or len(answers) != len(questions) or not all(isinstance(a, str) for a in answers):
        raise ValueError("StreamingBench response must contain exactly one string for each question")
    correct = sum(
        extract_choice(answer) == LETTERS[question["choices"].index(question["answer"])]
        for answer, question in zip(answers, questions)
    )
    return {"mcqa_accuracy": (correct, len(questions))}


def aggregate_accuracy(items):
    correct = sum(item[0] for item in items)
    total = sum(item[1] for item in items)
    if total < 1:
        raise ValueError("Cannot score an empty StreamingBench selection")
    return 100.0 * correct / total


class StreamingBenchLocal(ConfigurableTask):
    VERSION = 1

    def __init__(self, config=None):
        settings = {
            "task": "streamingbench_local", "dataset_path": "local_streamingbench_json",
            "test_split": "test", "output_type": "generate_until", "num_fewshot": 0,
            "doc_to_visual": doc_to_visual, "doc_to_text": doc_to_text,
            "doc_to_target": doc_to_target, "process_results": process_results,
            "generation_kwargs": {"max_new_tokens": 16, "temperature": 0, "num_beams": 1,
                                  "do_sample": False, "until": None},
            "metric_list": [{"metric": "mcqa_accuracy", "aggregation": aggregate_accuracy,
                             "higher_is_better": True}],
            "metadata": {"version": 1, "document_unit": "complete_video", "metric_unit": "question"},
        }
        if config:
            settings.update({key: value for key, value in config.items() if key != "class"})
        super().__init__(config=settings)

    def download(self, dataset_kwargs=None):
        # Avoid datasets.load_dataset and Hugging Face Hub entirely.
        source = Path(os.environ.get("STREAMINGBENCH_ANNOTATIONS", str(DEFAULT_ANNOTATIONS))).expanduser().resolve()
        videos = prepare_annotations(source, max_videos=None)
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        records = [
            {"video_id": str(video["video_id"]), "source_sha256": source_hash,
             "video_json": json.dumps(video, ensure_ascii=False)}
            for video in videos
        ]
        self.annotation_path = str(source)
        self.dataset = DatasetDict({"test": Dataset.from_list(records)})
        self.dataset_no_image = self.dataset.copy()
