#!/usr/bin/env python3
"""Export three v398 manual-ROI token examples classified per physical copy."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "results/myobservations/dp_token_frame_study_20261010"


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT)
    args = parser.parse_args()
    study = args.study
    manual = json.loads((study / "manual_review.json").read_text())
    case = next(c for c in manual["cases"] if c["video_id"] == 398 and c["question_index"] == 0)
    roi = next(r for r in case["human_review"]["manual_rois"] if r["label"] == "name-number pair")
    call = next(r for r in rows(study / "dp/source_manifest.jsonl")
                if r.get("type") == "encode" and str(r["video_id"]) == "398")
    assert call["fixed_frame_canvas_size"] == 0 and call["first_sample_ordinal"] == 0
    _, raw_h, raw_w = call["grid_thw"]
    height, width = raw_h // call["merge_size"], raw_w // call["merge_size"]
    source_h, source_w = call["source_loaded_shape"][1:3]
    candidates = set()
    x0, y0, x1, y1 = roi["xyxy"]
    for row in range(height):
        for col in range(width):
            if col / width < x1 and (col + 1) / width > x0 and row / height < y1 and (row + 1) / height > y0:
                candidates.add(call["uid_start"] + row * width + col)
    traces = {}
    for policy in ("dp", "hermes"):
        record = next(r for r in rows(study / policy / "tokens.jsonl")
                      if str(r["video_id"]) == "398" and r["question_index"] == 0)
        traces[policy] = {(int(h["layer"]), int(h["kv_head"])):
                         {int(u): float(a) for u, a in zip(h["visual_token_uids"], h["visual_attention"])}
                         for h in record["heads"]}
    pools = {"hermes_only_in_this_layer_head": [], "shared_in_this_layer_head": [],
             "dp_only_in_this_layer_head": []}
    for head_key, hermes in traces["hermes"].items():
        dp = traces["dp"][head_key]
        for uid in candidates & (hermes.keys() | dp.keys()):
            h_attn, d_attn = hermes.get(uid), dp.get(uid)
            if h_attn is not None and d_attn is None:
                kind, score = "hermes_only_in_this_layer_head", h_attn
            elif h_attn is not None and d_attn is not None:
                kind, score = "shared_in_this_layer_head", h_attn
            else:
                kind, score = "dp_only_in_this_layer_head", d_attn
            pools[kind].append((score, uid, head_key, h_attn, d_attn))
    examples = []
    used_uids = set()
    for kind, pool in pools.items():
        pool.sort(reverse=True)
        if not pool:
            examples.append({"physical_copy_status": kind, "available": False})
            continue
        chosen = next((entry for entry in pool if entry[1] not in used_uids), pool[0])
        score, uid, (layer, head), h_attn, d_attn = chosen
        used_uids.add(uid)
        local = uid - call["uid_start"]
        temporal_index, spatial = divmod(local, height * width)
        row, col = divmod(spatial, width)
        first = temporal_index * call["temporal_patch_size"]
        samples = [call["samples"][min(first + i, call["raw_frame_count"] - 1)]
                   for i in range(call["temporal_patch_size"])]
        global_heads = {policy: [{"layer": l, "kv_head": h} for (l, h), mapping in t.items() if uid in mapping]
                        for policy, t in traces.items()}
        examples.append({"available": True, "physical_copy_status": kind,
                         "selection_rule": "highest HERMES attention for H-only/shared; highest DP attention for DP-only; distinct example UIDs when available",
                         "video_id": 398, "question_index": 0, "uid": uid, "layer": layer, "kv_head": head,
                         "hermes_retained_this_copy": h_attn is not None, "dp_retained_this_copy": d_attn is not None,
                         "hermes_attention_at_this_copy": h_attn, "dp_attention_at_this_copy": d_attn,
                         "temporal_index_in_encode": temporal_index, "source_samples": samples,
                         "grid_row": row, "grid_col": col, "grid_hw": [height, width],
                         "processor_xyxy": [col * 28, row * 28, (col + 1) * 28, (row + 1) * 28],
                         "source_xyxy": [col * source_w / width, row * source_h / height,
                                         (col + 1) * source_w / width, (row + 1) * source_h / height],
                         "source_normalized_xyxy": [col / width, row / height, (col + 1) / width, (row + 1) / height],
                         "global_uid_copy_count": {p: len(h) for p, h in global_heads.items()},
                         "global_uid_retained_heads": global_heads})
    output = {"video_id": 398, "question_index": 0, "roi_label": roi["label"],
              "roi_normalized_xyxy": roi["xyxy"], "roi_rule": "positive area cell-box intersection in first temporal tubelet",
              "attention_definition": "true multiple-choice question prompt average, and average over the seven query heads sharing this KV head; probabilities include text/prompt keys in denominator",
              "absence_value": "null means this physical copy was not retained; attention is unobserved rather than measured zero",
              "interpretation_limit": "A HERMES-only or DP-only physical copy does not imply the source UID was globally discarded. A source cell is an origin anchor, not an exclusive semantic receptive field or proof of causal answer evidence.",
              "examples": examples}
    destination = study / "representative_token_examples.json"
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(destination), "examples": [{k: e.get(k) for k in
                      ("physical_copy_status", "uid", "layer", "kv_head", "source_xyxy", "hermes_attention_at_this_copy", "dp_attention_at_this_copy", "global_uid_copy_count")} for e in examples]}, indent=2))


if __name__ == "__main__":
    main()
