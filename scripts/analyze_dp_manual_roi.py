#!/usr/bin/env python3
"""Audit v398's human marked name/number ROIs in the actual first tubelet.

The box coordinates refer to sample ordinal 1 (2 seconds). Each visual token
in this tubelet is anchored to the same spatial cell across samples 0 and 1;
the box does not establish semantic contents or exclusive receptive fields.
"""
import argparse
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "results/myobservations/dp_token_frame_study_20261010"
SEGMENTS = {"all_0_27": (0, 27), "layers_0_7": (0, 7),
            "layers_8_18": (8, 18), "layers_19_27": (19, 27)}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT)
    args = parser.parse_args()
    study = args.study
    reviews = json.loads((study / "manual_review.json").read_text())
    review = next(c for c in reviews["cases"] if c["video_id"] == 398 and c["question_index"] == 0)
    rois = review["human_review"]["manual_rois"]
    output = {"video_id": 398, "question_index": 0, "roi_reference_sample_ordinal": 1,
              "tubelet_sample_ordinals": [0, 1], "roi_provenance": "manual_review.json",
              "definitions": {"any_intersection": "source-grid cell intersects human box with positive area",
                              "center_in_box": "source-grid cell center lies within human box",
                              "unique": "distinct visual output UIDs retained in any included layer/head",
                              "copies": "sum of retained UID copies over included layers and KV heads",
                              "mean_head_attention_mass": "sum of ROI true-question-prompt mean attention divided by included layer/head count"},
              "interpretation_limit": "Geometric source position is not semantic proof. Human box from the frame at 2s is applied to the temporal tubelet spanning 0s and 2s. The two frames are not two independent tokens. Vision and language contextualization mix information.",
              "policies": {}}
    for policy in ("dp", "hermes"):
        directory = study / policy
        tokens = next(r for r in read_jsonl(directory / "tokens.jsonl")
                      if str(r["video_id"]) == "398" and r["question_index"] == 0)
        calls = [r for r in read_jsonl(directory / "source_manifest.jsonl")
                 if r.get("type") == "encode" and str(r["video_id"]) == "398"]
        call = calls[0]
        assert call["first_sample_ordinal"] == 0 and call["temporal_patch_size"] == 2
        assert call["fixed_frame_canvas_size"] == 0, "ROI audit currently assumes native resize"
        _, h, w = call["grid_thw"]
        H, W = h // call["merge_size"], w // call["merge_size"]
        start, stop = call["uid_start"], call["uid_start"] + H * W
        policy_out = {}
        for roi in rois:
            x0, y0, x1, y1 = roi["xyxy"]
            selected = {"any_intersection": set(), "center_in_box": set()}
            for row in range(H):
                for col in range(W):
                    uid = start + row * W + col
                    if col / W < x1 and (col + 1) / W > x0 and row / H < y1 and (row + 1) / H > y0:
                        selected["any_intersection"].add(uid)
                    if x0 <= (col + .5) / W <= x1 and y0 <= (row + .5) / H <= y1:
                        selected["center_in_box"].add(uid)
            roi_out = {"normalized_xyxy": roi["xyxy"], "selection_rules": {}}
            for rule, candidates in selected.items():
                segments = {}
                for segment, (first_layer, last_layer) in SEGMENTS.items():
                    heads = [head for head in tokens["heads"] if first_layer <= head["layer"] <= last_layer]
                    union = set()
                    copies = 0
                    masses = []
                    per_head_counts = []
                    for head in heads:
                        kept = set(head["visual_token_uids"]) & candidates
                        union.update(kept)
                        copies += len(kept)
                        per_head_counts.append(len(kept))
                        masses.append(sum(float(a) for u, a in zip(head["visual_token_uids"], head["visual_attention"])
                                          if u in candidates))
                    segments[segment] = {"included_heads": len(heads), "roi_candidate_uids": len(candidates),
                                         "unique_retained_uids": len(union), "retained_kv_copies": copies,
                                         "unique_coverage_fraction": len(union) / max(1, len(candidates)),
                                         "copy_retention_fraction": copies / max(1, len(candidates) * len(heads)),
                                         "mean_head_attention_mass": statistics.mean(masses),
                                         "sum_head_attention_mass": sum(masses),
                                         "heads_with_no_roi_token": per_head_counts.count(0),
                                         "min_roi_tokens_per_head": min(per_head_counts),
                                         "median_roi_tokens_per_head": statistics.median(per_head_counts),
                                         "max_roi_tokens_per_head": max(per_head_counts),
                                         "unique_retained_uid_list": sorted(union)}
                roi_out["selection_rules"][rule] = segments
            policy_out[roi["label"]] = roi_out
        output["policies"][policy] = policy_out
    output["source_pair"] = {"grid_hw": [H, W], "uid_interval": [start, stop],
                              "samples": call["samples"][:2]}
    destination = study / "v398_manual_roi_retention.json"
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    compact = {policy: {label: {rule: {segment: {k: v for k, v in stats.items() if k != "unique_retained_uid_list"}
                                              for segment, stats in rule_stats.items()}
                                     for rule, rule_stats in roi["selection_rules"].items()}
                       for label, roi in rois.items()}
               for policy, rois in output["policies"].items()}
    print(json.dumps({"output": str(destination), "policies": compact}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
