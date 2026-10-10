#!/usr/bin/env python3
"""Locate evidence retention/readout loss in protected-policy regressions."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METHODS = ("hermes", "protected")
EPS = 1e-12


def load_jsonl(path):
    records = {}
    with Path(path).open() as handle:
        for line in handle:
            record = json.loads(line)
            records[(int(record["video_id"]), int(record["question_index"]))] = record
    return records


def by_head(record):
    return {
        (int(head["layer"]), int(head["kv_head"])): head
        for head in record["heads"]
    }


def aggregate_frames(record):
    masses = defaultdict(float)
    tokens = defaultdict(int)
    for head in record["heads"]:
        for frame_id, value in head["frame_mass"].items():
            masses[int(frame_id)] += float(value)
        for frame_id, value in head["frame_tokens"].items():
            tokens[int(frame_id)] += int(value)
    return dict(masses), dict(tokens)


def select_evidence_frames(record, count=3):
    masses, _ = aggregate_frames(record)
    fps = float(record["sample_fps"])
    # The first sampled patch is a positional attention sink, not answer evidence.
    masses = {frame: mass for frame, mass in masses.items() if frame / fps > 4.0}
    return [frame for frame, _ in sorted(masses.items(), key=lambda item: item[1], reverse=True)[:count]]


def load_policy(path):
    policy = {}
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            policy[(int(row["layer"]), int(row["kv_head"]))] = row
    return policy


def temporal_bin(age):
    if age == 0:
        return "age=0"
    if age < 4:
        return "0-4s"
    if age < 8:
        return "4-8s"
    if age < 16:
        return "8-16s"
    if age < 64:
        return "16-64s"
    return "64s+"


def attention_weighted_temporal_coverage(hermes, protected, policy):
    protected_heads = by_head(protected)
    fps = float(hermes["sample_fps"])
    latest = max(int(head["latest_frame_id"]) for head in hermes["heads"])
    totals = defaultdict(lambda: {"demand": 0.0, "covered": 0.0, "h_tokens": 0, "p_tokens": 0})
    for h_head in hermes["heads"]:
        key = (int(h_head["layer"]), int(h_head["kv_head"]))
        p_head = protected_heads[key]
        protected_flag = int(policy[key]["protect_latest"])
        for frame_text, mass_value in h_head["frame_mass"].items():
            frame = int(frame_text)
            if frame / fps <= 4.0:
                continue
            age_bin = temporal_bin((latest - frame) / fps)
            h_tokens = int(h_head["frame_tokens"].get(frame_text, 0))
            p_tokens = int(p_head["frame_tokens"].get(frame_text, 0))
            mass = float(mass_value)
            item = totals[(protected_flag, age_bin)]
            item["demand"] += mass
            item["covered"] += mass * min(1.0, p_tokens / max(h_tokens, 1))
            item["h_tokens"] += h_tokens
            item["p_tokens"] += p_tokens
    return totals


def head_evidence(head, frames):
    mass = sum(float(head["frame_mass"].get(str(frame), 0.0)) for frame in frames)
    tokens = sum(int(head["frame_tokens"].get(str(frame), 0)) for frame in frames)
    return mass, tokens


def method_stats(record, frames, focused_heads):
    heads = by_head(record)
    evidence_mass = 0.0
    evidence_tokens = 0
    visual_mass = 0.0
    visual_tokens = 0
    covered_heads = 0
    focused_mass = 0.0
    focused_tokens = 0
    role_tokens = defaultdict(int)
    role_mass = defaultdict(float)
    for key, head in heads.items():
        mass, tokens = head_evidence(head, frames)
        evidence_mass += mass
        evidence_tokens += tokens
        visual_mass += float(head["visual_mass"])
        visual_tokens += int(head["visual_tokens"])
        covered_heads += int(tokens > 0)
        if key in focused_heads:
            focused_mass += mass
            focused_tokens += tokens
        role = head.get("role", "unassigned")
        role_tokens[role] += tokens
        role_mass[role] += mass
    return {
        "evidence_tokens": evidence_tokens,
        "evidence_token_share": evidence_tokens / max(visual_tokens, 1),
        "evidence_readout_mass": evidence_mass,
        "evidence_readout_share": evidence_mass / max(visual_mass, 1e-12),
        "covered_heads": covered_heads,
        "focused_tokens": focused_tokens,
        "focused_mass": focused_mass,
        "role_tokens": dict(role_tokens),
        "role_mass": dict(role_mass),
    }


def ratio(numerator, denominator):
    return numerator / max(denominator, 1e-12)


def mean(rows, path):
    keys = path.split(".")
    values = []
    for row in rows:
        value = row
        for key in keys:
            value = value[key]
        values.append(value)
    return float(np.mean(values)) if values else float("nan")


def pct(value):
    return f"{100.0 * value:.2f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    manifest = json.loads((root / "manifest.json").read_text())
    targets = {
        (int(target["video_id"]), int(target["question_index"])): target
        for target in manifest["targets"]
    }
    traces = {
        method: load_jsonl(root / f"{method}_readout_trace.jsonl")
        for method in METHODS
    }
    policy = load_policy(args.policy)

    rows = []
    for key, target in targets.items():
        missing = [method for method in METHODS if key not in traces[method]]
        if missing:
            raise ValueError(f"Missing target {key} in {missing}")
        hermes = traces["hermes"][key]
        frames = select_evidence_frames(hermes)
        hermes_heads = by_head(hermes)
        density = []
        for head_key, head in hermes_heads.items():
            mass, tokens = head_evidence(head, frames)
            density.append((mass / max(tokens, 1), head_key))
        focused_heads = {head_key for _, head_key in sorted(density, reverse=True)[:14]}
        stats = {
            method: method_stats(traces[method][key], frames, focused_heads)
            for method in METHODS
        }
        fps = float(hermes["sample_fps"])
        latest = max(int(head["latest_frame_id"]) for head in hermes["heads"])
        top_frame = frames[0]
        age = (latest - top_frame) / fps
        token_ratio = ratio(
            stats["protected"]["evidence_tokens"], stats["hermes"]["evidence_tokens"]
        )
        focused_token_ratio = ratio(
            stats["protected"]["focused_tokens"], stats["hermes"]["focused_tokens"]
        )
        readout_ratio = ratio(
            stats["protected"]["evidence_readout_share"],
            stats["hermes"]["evidence_readout_share"],
        )
        if token_ratio < 0.8 or focused_token_ratio < 0.8:
            diagnosis = "cache-retention loss"
        elif readout_ratio < 0.8:
            diagnosis = "readout shift despite retention"
        else:
            diagnosis = "no large top-patch loss"
        rows.append(
            {
                **target,
                "evidence_frames": frames,
                "top_frame_sec": top_frame / fps,
                "top_frame_age_sec": age,
                "token_ratio": token_ratio,
                "focused_token_ratio": focused_token_ratio,
                "readout_ratio": readout_ratio,
                "diagnosis": diagnosis,
                "methods": stats,
            }
        )

    lines = [
        "# Protected temporal-zone regression diagnosis",
        "",
        "Candidate evidence is the three temporal patches receiving the most aggregate true-question attention when HERMES answers correctly. Focused heads are the 14 physical KV heads with the highest HERMES attention density on those patches.",
        "",
        "| Target | Ability | Top age | H/P tokens | H/P focused tokens | H/P readout | Ratios token/focused/readout | Diagnostic |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        h = row["methods"]["hermes"]
        p = row["methods"]["protected"]
        lines.append(
            f"| v{row['video_id']}/q{row['question_index']} {row['task']} | {row['ability']} | "
            f"{row['top_frame_age_sec']:.1f}s | {h['evidence_tokens']}/{p['evidence_tokens']} | "
            f"{h['focused_tokens']}/{p['focused_tokens']} | "
            f"{pct(h['evidence_readout_share'])}/{pct(p['evidence_readout_share'])} | "
            f"{row['token_ratio']:.2f}/{row['focused_token_ratio']:.2f}/{row['readout_ratio']:.2f} | "
            f"{row['diagnosis']} |"
        )

    lines.extend(["", "## Aggregate", ""])
    for diagnosis in (
        "cache-retention loss",
        "readout shift despite retention",
        "no large top-patch loss",
    ):
        lines.append(f"- {diagnosis}: {sum(row['diagnosis'] == diagnosis for row in rows)}/{len(rows)}")
    for ability in ("working memory", "episodic memory"):
        selected = [row for row in rows if row.get("ability") == ability]
        lines.append(
            f"- {ability}: token ratio={mean(selected, 'token_ratio'):.2f}, "
            f"focused-token ratio={mean(selected, 'focused_token_ratio'):.2f}, "
            f"readout ratio={mean(selected, 'readout_ratio'):.2f}."
        )
    lines.append(
        f"- all: token ratio={mean(rows, 'token_ratio'):.2f}, "
        f"focused-token ratio={mean(rows, 'focused_token_ratio'):.2f}, "
        f"readout ratio={mean(rows, 'readout_ratio'):.2f}."
    )

    temporal_totals = defaultdict(
        lambda: {"demand": 0.0, "covered": 0.0, "h_tokens": 0, "p_tokens": 0}
    )
    for key in targets:
        per_question = attention_weighted_temporal_coverage(
            traces["hermes"][key], traces["protected"][key], policy
        )
        for group, values in per_question.items():
            for field, value in values.items():
                temporal_totals[group][field] += value

    temporal_summary = {}
    lines.extend([
        "",
        "## Attention-weighted temporal coverage (attention sink excluded)",
        "",
        "Coverage weights every frame/head cache location by the attention HERMES placed on it while answering correctly. It is a frame-level upper bound because the trace records retained patch counts, not exact spatial token identity.",
        "",
        "| Age | Head group | Coverage | Raw token ratio P/H |",
        "|---|---|---:|---:|",
    ])
    for age_bin in ("age=0", "0-4s", "4-8s", "8-16s", "16-64s", "64s+"):
        for protected_flag, group_name in ((0, "unprotected 98"), (1, "protected 14")):
            values = temporal_totals[(protected_flag, age_bin)]
            coverage = values["covered"] / max(values["demand"], EPS)
            token_ratio = values["p_tokens"] / max(values["h_tokens"], 1)
            temporal_summary[f"{protected_flag}:{age_bin}"] = {
                **values,
                "coverage": coverage,
                "raw_token_ratio": token_ratio,
            }
            lines.append(
                f"| {age_bin} | {group_name} | {pct(coverage)} | {token_ratio:.2f} |"
            )

    output = Path(args.output)
    output.write_text("\n".join(lines) + "\n")
    (root / "analysis.json").write_text(json.dumps(rows, indent=2) + "\n")
    (root / "temporal_coverage_summary.json").write_text(
        json.dumps(temporal_summary, indent=2) + "\n"
    )
    print(f"wrote {output} and {root / 'analysis.json'}")


if __name__ == "__main__":
    main()
