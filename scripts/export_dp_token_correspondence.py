#!/usr/bin/env python3
"""Export exact retained visual UID -> source frames and spatial patch mappings.

This consumes the source manifest captured by trace_dp_token_frames.py and the
112-head runtime token trace. It performs no model inference or image decoding.
All encoded UIDs observed by each question are exported, including UIDs that
both policies discarded. A UID denotes one merged Qwen spatiotemporal token;
its presence in several physical KV heads is counted separately as copies.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY = ROOT / "results/myobservations/dp_token_frame_study_20261010"
HEADS = 112
BANDS = {"early": (0, 8), "mid": (8, 19), "late": (19, 28)}
BAND_NAMES = tuple(BANDS)
MASKS = {name: ((1 << (hi * 4)) - 1) ^ ((1 << (lo * 4)) - 1)
         for name, (lo, hi) in BANDS.items()}
MASKS = {"all": (1 << HEADS) - 1, **MASKS}


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at {path}:{line_no}") from error


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def ratio(numerator: float, denominator: float):
    return numerator / denominator if denominator else None


@dataclass(frozen=True)
class Group:
    video_id: str
    encode_call: int
    encode_question_index: int
    temporal_index: int
    ordinal: int
    uid_start: int
    uid_stop: int
    rows: int
    cols: int
    patch_span: int
    resized_h: int
    resized_w: int
    source_h: int
    source_w: int
    samples: tuple[dict, dict]

    @property
    def group_id(self) -> str:
        return f"v{self.video_id}_e{self.encode_call}_t{self.temporal_index}"

    @property
    def latest_frame_id(self) -> int:
        return int(self.samples[1]["sample_ordinal"])

    def metadata(self) -> dict:
        result = dict(temporal_group=self.group_id, temporal_group_ordinal=self.ordinal,
                      encode_call=self.encode_call,
                      encode_question_index=self.encode_question_index,
                      temporal_index_in_encode=self.temporal_index,
                      llm_grid_rows=self.rows, llm_grid_cols=self.cols,
                      uid_start=self.uid_start, uid_stop_exclusive=self.uid_stop,
                      source_height=self.source_h, source_width=self.source_w,
                      processor_height=self.resized_h, processor_width=self.resized_w,
                      duplicate_tail_sample=int(self.samples[0] == self.samples[1]))
        for index, sample in enumerate(self.samples):
            pts = sample.get("decoder_pts")
            require(pts is None or len(pts) == 2, "Decoder PTS must be [start,end]")
            result.update({f"sample_ordinal_{index}": int(sample["sample_ordinal"]),
                           f"source_frame_index_{index}": int(sample["source_frame_index"]),
                           f"nominal_time_s_{index}": sample.get("nominal_time_s"),
                           f"decoder_pts_start_s_{index}": None if pts is None else pts[0],
                           f"decoder_pts_end_s_{index}": None if pts is None else pts[1]})
        return result

    def patch(self, uid: int) -> dict:
        require(self.uid_start <= uid < self.uid_stop, "UID outside temporal group")
        row, col = divmod(uid - self.uid_start, self.cols)
        x0, y0, x1, y1 = col / self.cols, row / self.rows, (col + 1) / self.cols, (row + 1) / self.rows
        return dict(row=row, col=col, x0_norm=x0, y0_norm=y0, x1_norm=x1, y1_norm=y1,
                    source_x0=x0 * self.source_w, source_y0=y0 * self.source_h,
                    source_x1=x1 * self.source_w, source_y1=y1 * self.source_h,
                    processor_x0=col * self.patch_span, processor_y0=row * self.patch_span,
                    processor_x1=(col + 1) * self.patch_span,
                    processor_y1=(row + 1) * self.patch_span)


def load_manifest(path: Path, annotation: list[dict]) -> dict[str, list[Group]]:
    videos, encodes = {}, {}
    expected_videos = {str(video["video_id"]) for video in annotation}
    for record in read_jsonl(path):
        kind = record.get("record_type", record.get("type"))
        if kind == "run":
            continue
        video_id = str(record["video_id"])
        require(video_id in expected_videos, f"Unexpected video {video_id} in {path}")
        if kind == "video":
            require(video_id not in videos, f"Duplicate video manifest for {video_id}")
            videos[video_id] = record
        elif kind == "encode":
            encodes.setdefault(video_id, []).append(record)
        else:
            raise ValueError(f"Unknown source-manifest record type: {kind}")
    require(set(videos) == expected_videos, f"Incomplete video manifest: {path}")
    output = {}
    for video in annotation:
        video_id = str(video["video_id"])
        source = videos[video_id]
        require(Path(source["video_path"]).resolve() == Path(video["video_path"]).resolve(),
                f"Source video differs from annotation: {video_id}")
        require(math.isclose(float(source["sample_fps"]), 0.5), "Expected 0.5-fps study")
        source_h, source_w = map(int, source["loaded_shape"][1:3])
        groups, next_uid, next_sample, previous_question = [], 0, 0, -1
        records = encodes.get(video_id, [])
        require(bool(records), f"No encoded sources for video {video_id}")
        for call, record in enumerate(records):
            require(int(record.get("fixed_frame_canvas_size", 0)) == 0,
                    "Fixed-frame-canvas inputs require an explicit inverse padding transform; refusing native bbox export")
            require(int(record["encode_call"]) == call, "Encode-call sequence is not contiguous")
            question = int(record["question_index"])
            require(previous_question <= question < len(video["conversations"]), "Invalid encode question index")
            previous_question = question
            temporal, height, width = map(int, record["video_grid_thw"])
            merge, patch = int(record["spatial_merge_size"]), int(record["patch_size"])
            raw_count = int(record["source_sample_count"])
            require(int(record["temporal_patch_size"]) == 2, "Only observed two-sample temporal patches are supported")
            require(raw_count > 0 and temporal == (raw_count + 1) // 2, "Temporal grid does not match two-sample packing")
            require(height % merge == width % merge == 0, "Spatial merge does not divide encoder grid")
            rows, cols = height // merge, width // merge
            require(list(record["llm_grid_thw"]) == [temporal, rows, cols], "LLM grid mismatch")
            require(list(record["processor_resized_hw"]) == [height * patch, width * patch], "Processor resize mismatch")
            require(int(record["q_len"]) == temporal * rows * cols, "Visual feature length mismatch")
            require(int(record["uid_start"]) == next_uid, "UID ranges are not contiguous")
            require(int(record["source_sample_start"]) == next_sample, "Source sample ranges are not contiguous")
            samples = record["samples"]
            require(len(samples) == raw_count, "Source sample count mismatch")
            require(samples == source["samples"][next_sample:next_sample + raw_count], "Encode samples differ from original loader samples")
            require([int(sample["sample_ordinal"]) for sample in samples] == list(range(next_sample, next_sample + raw_count)), "Sample ordinals are not contiguous")
            require(list(record["source_loaded_shape"]) == list(source["loaded_shape"]), "Loaded source shape mismatch")
            for index in range(temporal):
                first = min(2 * index, raw_count - 1)
                second = min(2 * index + 1, raw_count - 1)
                # This is also the core's build_source_frame_ids convention.
                require(second == ((index + 1) * raw_count - 1) // temporal, "Trace frame-id convention differs from temporal packing")
                begin = next_uid + index * rows * cols
                groups.append(Group(video_id, call, question, index, len(groups), begin,
                                    begin + rows * cols, rows, cols, merge * patch,
                                    height * patch, width * patch, source_h, source_w,
                                    (samples[first], samples[second])))
            next_uid += temporal * rows * cols
            next_sample += raw_count
            require(int(record["uid_stop_exclusive"]) == next_uid, "UID stop mismatch")
        output[video_id] = groups
    return output


@dataclass
class Token:
    mask: int = 0
    attention: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    exclusive_copy_attention: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


def check_question(record: dict, video_id: str, index: int, question: dict) -> None:
    require(str(record["video_id"]) == video_id and int(record["question_index"]) == index,
            f"Token trace order differs from annotation: expected {video_id}/{index}")
    require(record["question"] == question["question"], "Token trace question text differs")
    require(math.isclose(float(record["end_time"]), float(question["end_time"]), abs_tol=1e-6), "Token trace question time differs")
    require(int(record["score_last_n"]) == 0, "Expected average attention over all real question prompt tokens")
    require(math.isclose(float(record["sample_fps"]), 0.5), "Unexpected token trace sampling rate")


def parse_tokens(record: dict, uid_groups: list[Group]) -> dict[int, Token]:
    heads = record["heads"]
    require(len(heads) == HEADS, "Each question must contain all 112 physical KV heads")
    seen_heads, tokens = set(), {}
    for head in heads:
        layer, kv_head = int(head["layer"]), int(head["kv_head"])
        require(0 <= layer < 28 and 0 <= kv_head < 4, "Invalid physical KV head")
        head_id = layer * 4 + kv_head
        require(head_id not in seen_heads, "Duplicate physical KV head")
        seen_heads.add(head_id)
        uids, frames, scores = head["visual_token_uids"], head["visual_frame_ids"], head["visual_attention"]
        require(len(uids) == len(frames) == len(scores), "Token UID/frame/attention lengths differ")
        require(len(set(uids)) == len(uids), "A head contains duplicate immutable visual UIDs")
        band = 0 if layer < 8 else (1 if layer < 19 else 2)
        for uid, frame, score in zip(uids, frames, scores):
            require(isinstance(uid, int) and 0 <= uid < len(uid_groups), "Retained UID was not encoded by this question")
            require(int(frame) == uid_groups[uid].latest_frame_id, "Retained token frame id differs from exact source pair")
            require(math.isfinite(score) and score >= 0, "Invalid visual attention score")
            token = tokens.setdefault(uid, Token())
            token.mask |= 1 << head_id
            token.attention[band] += float(score)
    return tokens


def assign_exclusive_copy_attention(record: dict, own: dict[int, Token], other: dict[int, Token]) -> None:
    for head in record["heads"]:
        layer = int(head["layer"])
        bit = 1 << (layer * 4 + int(head["kv_head"]))
        band = 0 if layer < 8 else (1 if layer < 19 else 2)
        for uid, score in zip(head["visual_token_uids"], head["visual_attention"]):
            if not (other.get(uid, EMPTY_TOKEN).mask & bit):
                own[uid].exclusive_copy_attention[band] += float(score)


EMPTY_TOKEN = Token()


def band_attention(values: list[float], stratum: str) -> float:
    return sum(values) if stratum == "all" else values[BAND_NAMES.index(stratum)]


def metrics(uids: range, dp: dict[int, Token], hermes: dict[int, Token], stratum: str) -> dict:
    mask = MASKS[stratum]
    result = dict(observed_unique_uids=len(uids), observed_physical_copies=len(uids) * mask.bit_count(),
                  dp_token_copies=0, hermes_token_copies=0, dp_unique_uids=0, hermes_unique_uids=0,
                  intersection_unique_uids=0, union_unique_uids=0, intersection_token_copies=0,
                  union_token_copies=0, dp_only_unique_uids=0, hermes_only_unique_uids=0,
                  dp_attention_sum=0.0, hermes_attention_sum=0.0,
                  dp_only_uid_attention_sum=0.0, hermes_only_uid_attention_sum=0.0,
                  dp_only_copy_attention_sum=0.0, hermes_only_copy_attention_sum=0.0)
    for uid in uids:
        d, h = dp.get(uid, EMPTY_TOKEN), hermes.get(uid, EMPTY_TOKEN)
        dm, hm = d.mask & mask, h.mask & mask
        da, ha = band_attention(d.attention, stratum), band_attention(h.attention, stratum)
        result["dp_token_copies"] += dm.bit_count()
        result["hermes_token_copies"] += hm.bit_count()
        result["dp_unique_uids"] += bool(dm)
        result["hermes_unique_uids"] += bool(hm)
        result["intersection_unique_uids"] += bool(dm and hm)
        result["union_unique_uids"] += bool(dm or hm)
        result["intersection_token_copies"] += (dm & hm).bit_count()
        result["union_token_copies"] += (dm | hm).bit_count()
        result["dp_only_unique_uids"] += bool(dm and not hm)
        result["hermes_only_unique_uids"] += bool(hm and not dm)
        result["dp_attention_sum"] += da
        result["hermes_attention_sum"] += ha
        result["dp_only_uid_attention_sum"] += da if not hm else 0.0
        result["hermes_only_uid_attention_sum"] += ha if not dm else 0.0
        result["dp_only_copy_attention_sum"] += band_attention(d.exclusive_copy_attention, stratum)
        result["hermes_only_copy_attention_sum"] += band_attention(h.exclusive_copy_attention, stratum)
    result["uid_jaccard"] = ratio(result["intersection_unique_uids"], result["union_unique_uids"])
    result["copy_jaccard"] = ratio(result["intersection_token_copies"], result["union_token_copies"])
    for method in ("dp", "hermes"):
        result[f"{method}_unique_uid_retention_fraction"] = ratio(result[f"{method}_unique_uids"], len(uids))
        result[f"{method}_physical_copy_retention_fraction"] = ratio(result[f"{method}_token_copies"], result["observed_physical_copies"])
        for kind in ("uid", "copy"):
            result[f"{method}_only_{kind}_attention_fraction"] = ratio(result[f"{method}_only_{kind}_attention_sum"], result[f"{method}_attention_sum"])
    return result


def token_row(uid: int, group: Group, dp: dict[int, Token], hermes: dict[int, Token]) -> dict:
    result = dict(uid=uid, **group.metadata(), **group.patch(uid))
    for method, tokens in (("dp", dp), ("hermes", hermes)):
        token = tokens.get(uid, EMPTY_TOKEN)
        result[f"{method}_heads_kept"] = token.mask.bit_count()
        result[f"{method}_head_mask_hex"] = f"0x{token.mask:028x}"
        result[f"{method}_attention_sum"] = sum(token.attention)
        for band in BAND_NAMES:
            result[f"{method}_{band}_heads_kept"] = (token.mask & MASKS[band]).bit_count()
            result[f"{method}_{band}_attention_sum"] = band_attention(token.attention, band)
    return result


def export(study: Path) -> dict:
    annotation_path = study / "annotation.json"
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    require(isinstance(annotation, list) and bool(annotation), "Annotation must contain videos")
    paths = [annotation_path] + [study / method / name for method in ("dp", "hermes")
                                for name in ("source_manifest.jsonl", "tokens.jsonl")]
    snapshots = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in paths}
    manifests = {method: load_manifest(study / method / "source_manifest.jsonl", annotation)
                 for method in ("dp", "hermes")}
    require(manifests["dp"] == manifests["hermes"], "DP and Hermes encoded different source grids or frame samples; UID identities cannot be shared")
    readers = {method: iter(read_jsonl(study / method / "tokens.jsonl")) for method in ("dp", "hermes")}
    summary = dict(schema_version=1, study=str(study), question_count=0, questions=[],
                   definitions={
                       "uid": "Immutable merged visual-token identity, reset per video; identified globally by (video_id, uid).",
                       "spatial_order": "Within each encode: temporal group, merged-grid row, merged-grid column (row-major).",
                       "frame_pair": "Each temporal token represents two actual sampled frames; the final sample is repeated for an odd encode chunk.",
                       "bbox": "Half-open patch support [x0,y0,x1,y1], not an object box or exclusive semantic receptive field. Native direct resize is inverted independently along x/y; coordinates are float pixels.",
                       "mask": "112-bit hexadecimal: bit layer*4+kv_head; layer and head indices are zero-based.",
                       "physical_copy": "A selected (UID, physical KV head) pair. Summing head token counts counts copies, not unique image patches.",
                       "observed_denominator": "All encoded source UIDs up to this question. observed_physical_copies is UID count times the stratum head count, not current or simultaneous cache occupancy.",
                       "frame_coverage": "A group/sample/source frame is covered if at least one of its source UIDs is retained in the stratum; this does not imply complete frame coverage or preserved semantic content.",
                       "attention": "Actual retained-cache true-question readout, averaged over real question prompt tokens and pooled query heads as emitted by runtime; sums here add physical KV heads. Includes visual keys only, excluding text/sink attention.",
                       "attention_fraction": "Per method and stratum, normalized by all retained visual attention for that question, not by all text+visual attention.",
                       "exclusive_mass": "UID-exclusive means absent from every head in the other policy's stratum; copy-exclusive means absent from the same physical head. These are observed readout masses, not counterfactual evidence importance or causal attribution.",
                       "strata": "early=layers 0..7 (32 KV heads), mid=8..18 (44), late=19..27 (36); all=112.",
                       "observation_filter": "Only source encode records whose question_index <= current question; no time-based interpolation or future sources. Original earlier questions and prefixes are retained.",
                   })
    output_names = ("token_correspondence.csv.gz", "frame_groups.csv", "retention_summary.json")
    temporary = {name: study / (name + f".tmp-{os.getpid()}") for name in output_names}
    token_writer = group_writer = None
    try:
        with gzip.open(temporary[output_names[0]], "wt", newline="", encoding="utf-8") as token_stream, temporary[output_names[1]].open("w", newline="", encoding="utf-8") as group_stream:
            for video in annotation:
                video_id = str(video["video_id"])
                for index, question in enumerate(video["conversations"]):
                    records = {}
                    for method in readers:
                        record = next(readers[method], None)
                        require(record is not None, f"Incomplete {method} trace: missing {video_id}/{index}")
                        check_question(record, video_id, index, question)
                        records[method] = record
                    groups = [group for group in manifests["dp"][video_id] if group.encode_question_index <= index]
                    require(bool(groups), "Question has no observed encoded sources")
                    uid_groups = [group for group in groups for _ in range(group.uid_stop - group.uid_start)]
                    require(groups[0].uid_start == 0 and groups[-1].uid_stop == len(uid_groups), "Observed UID mapping is not a contiguous prefix")
                    tokens = {method: parse_tokens(record, uid_groups) for method, record in records.items()}
                    for method, other in (("dp", "hermes"), ("hermes", "dp")):
                        assign_exclusive_copy_attention(records[method], tokens[method], tokens[other])
                    dp, hermes = tokens["dp"], tokens["hermes"]
                    common = dict(video_id=video_id, question_index=index, question=question["question"], end_time=float(question["end_time"]))
                    qsummary = dict(common, task=question["task"], observed_temporal_groups=len(groups), strata={})
                    totals = {stratum: metrics(range(len(uid_groups)), dp, hermes, stratum) for stratum in MASKS}
                    for uid, group in enumerate(uid_groups):
                        row = dict(common, **token_row(uid, group, dp, hermes))
                        if token_writer is None:
                            token_writer = csv.DictWriter(token_stream, fieldnames=list(row))
                            token_writer.writeheader()
                        token_writer.writerow(row)
                    coverage = {stratum: {method: {"groups": set(), "samples": set(), "source_frames": set()}
                                          for method in ("dp", "hermes")} for stratum in MASKS}
                    for group in groups:
                        for stratum in MASKS:
                            value = metrics(range(group.uid_start, group.uid_stop), dp, hermes, stratum)
                            row = dict(common, stratum=stratum, physical_head_count=MASKS[stratum].bit_count(), **group.metadata(), **value)
                            for method in ("dp", "hermes"):
                                row[f"{method}_attention_fraction_of_visual_total"] = ratio(value[f"{method}_attention_sum"], totals[stratum][f"{method}_attention_sum"])
                                if value[f"{method}_unique_uids"]:
                                    seen = coverage[stratum][method]
                                    seen["groups"].add(group.group_id)
                                    seen["samples"].update(int(sample["sample_ordinal"]) for sample in group.samples)
                                    seen["source_frames"].update(int(sample["source_frame_index"]) for sample in group.samples)
                            if group_writer is None:
                                group_writer = csv.DictWriter(group_stream, fieldnames=list(row))
                                group_writer.writeheader()
                            group_writer.writerow(row)
                    observed_samples = {int(sample["sample_ordinal"]) for group in groups for sample in group.samples}
                    observed_frames = {int(sample["source_frame_index"]) for group in groups for sample in group.samples}
                    qsummary.update(observed_unique_samples=len(observed_samples), observed_unique_source_frames=len(observed_frames))
                    for stratum, total in totals.items():
                        for method in ("dp", "hermes"):
                            seen = coverage[stratum][method]
                            total[f"{method}_covered_temporal_groups"] = len(seen["groups"])
                            total[f"{method}_covered_unique_samples"] = len(seen["samples"])
                            total[f"{method}_covered_unique_source_frames"] = len(seen["source_frames"])
                        qsummary["strata"][stratum] = total
                    summary["questions"].append(qsummary)
                    summary["question_count"] += 1
                    print(f"Exported video {video_id} q{index}: {len(uid_groups):,} observed UIDs", flush=True)
            for method, reader in readers.items():
                require(next(reader, None) is None, f"Unexpected extra question in {method} token trace")
        require(summary["question_count"] == sum(len(video["conversations"]) for video in annotation), "Not all annotation questions were exported")
        summary["source_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        require(all((path.stat().st_size, path.stat().st_mtime_ns) == snapshots[path] for path in paths), "Input trace changed during export; wait for both runs to finish")
        summary["output_rows"] = dict(token_correspondence=sum(q["strata"]["all"]["observed_unique_uids"] for q in summary["questions"]), frame_groups=sum(q["observed_temporal_groups"] * len(MASKS) for q in summary["questions"]))
        temporary[output_names[2]].write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for name in output_names:
            temporary[name].replace(study / name)
        return summary
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    args = parser.parse_args()
    summary = export(args.study.resolve())
    print(json.dumps({"questions": summary["question_count"], "output_rows": summary["output_rows"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
