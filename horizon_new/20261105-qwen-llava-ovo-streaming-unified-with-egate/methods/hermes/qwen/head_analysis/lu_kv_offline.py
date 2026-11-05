"""Small, full-prefix LU-style video calibration. No online policy changes.

Physical GQA utility = max over answer prediction positions of sum over shared
query heads of attention * ||V W_o(head)||_2. No per-head normalization.
HERMES selection uses the project's actual ragged scoring and selector.
This is a fixed-capacity, raw-projected-utility adaptation, not a paper replica.
"""

import argparse
import csv
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def integer_average(tables, total, lower):
    raw = np.mean(tables, axis=0).reshape(-1)
    out = np.floor(raw).astype(np.int64)
    missing = int(total - out.sum())
    if not 0 <= missing <= out.size or np.any(out < lower):
        raise ValueError("Infeasible averaged allocation")
    order = np.argsort(-(raw - out), kind="stable")
    out[order[:missing]] += 1
    assert out.sum() == total
    return out.reshape(np.asarray(tables[0]).shape)


def concave_blocks(gains):
    """PAVA of marginal gains: upper concave envelope of prefix utility."""
    stack = []
    for gain in gains:
        stack.append([float(gain), 1])
        while len(stack) > 1:
            a, b = stack[-2:]
            if a[0] / a[1] >= b[0] / b[1]:
                break
            stack[-2:] = [[a[0] + b[0], a[1] + b[1]]]
    return stack


def solve_curves(curves, total, lower):
    curves = np.asarray(curves, dtype=np.float64)
    heads, size = curves.shape
    upper = size - 1
    if not heads * lower <= total <= heads * upper:
        raise ValueError((heads, lower, total, upper))
    blocks = [concave_blocks(np.diff(row[lower:])) for row in curves]
    budget = np.full(heads, lower, dtype=np.int64)
    heap = []
    for h, seq in enumerate(blocks):
        if seq:
            gain, count = seq[0]
            heapq.heappush(heap, (-gain / count, h, 0))
    remaining = total - heads * lower
    relaxed = float(curves[:, lower].sum())
    while remaining:
        neg_gain, h, i = heapq.heappop(heap)
        count = blocks[h][i][1]
        take = min(count, remaining)
        budget[h] += take
        remaining -= take
        relaxed -= neg_gain * take
        if take == count and i + 1 < len(blocks[h]):
            gain, n = blocks[h][i + 1]
            heapq.heappush(heap, (-gain / n, h, i + 1))
    actual = float(curves[np.arange(heads), budget].sum())
    assert budget.sum() == total
    assert relaxed + 1e-7 >= actual
    return budget, {"actual_utility": actual, "relaxed_upper_bound": relaxed,
                    "relaxation_gap": max(0.0, relaxed - actual)}


def choose_manifest(args, out):
    source_paths = [Path(p) for p in args.calibration]
    pool = sum([json.loads(p.read_text()) for p in source_paths], [])
    heldout = json.loads(Path(args.exclude_annotation).read_text())
    excluded = {str(v["video_id"]) for v in heldout}
    excluded_paths = {Path(v["video_path"]).name for v in heldout}
    candidates = []
    seen = set()
    for v in pool:
        vid = str(v["video_id"])
        if vid in seen:
            raise ValueError("Duplicate calibration video " + vid)
        seen.add(vid)
        if vid in excluded or Path(v["video_path"]).name in excluded_paths:
            raise ValueError("Calibration overlaps held-out evaluation: " + vid)
        qs = [{**q, "source_question_index": i} for i, q in enumerate(v["conversations"])
              if args.min_time <= float(q.get("end_time", -1)) <= args.max_time]
        qs.sort(key=lambda q: float(q["end_time"]))
        if qs:
            candidates.append({**v, "conversations": qs[-args.questions_per_video:]})
    random.Random(args.seed).shuffle(candidates)
    selected = candidates[:args.num_videos]
    if len(selected) != args.num_videos:
        raise ValueError("Insufficient eligible videos")
    for v in selected:
        if not Path(v["video_path"]).is_file():
            raise FileNotFoundError(v["video_path"])
    manifest = {
        "config": {k: v for k, v in vars(args).items() if k not in ("smoke", "analyze_only")},
        "videos": selected, "evaluation_overlap": 0,
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in source_paths + [Path(args.exclude_annotation)]},
        "scope": "single compression, full prefix; question-free selector; no prior QA history",
        "utility": "max_answer_position sum_shared_Q attention * L2(V @ Wo_Q); raw scale",
        "adaptations": ["fixed absolute K", "physical GQA group aggregation",
                        "no intra-layer value normalization", "16 answer tokens as MCQA runtime"],
    }
    path = out / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Existing manifest differs; choose a new output directory")
    else:
        save_json(path, manifest)
    return selected


def cache_digest(model):
    import torch
    digest = hashlib.sha256()
    for k, v in model.kv_cache:
        for tensor in (k, v):
            digest.update(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def collect_selector(model, processor, k_values, recent):
    import torch
    from head_analysis.hermes_kv_head_ragged import HeadRaggedCache
    from head_analysis.hermes_kv_head_ragged_prefill import (
        _ragged_query_scores, _default_hermes_scores_for_trace, _select_ragged_keep_indices,
    )
    dense = model.kv_cache
    offsets = model._get_next_global_offset_per_layer()
    local, global_ = model.predict_next_question()
    model.kv_cache = HeadRaggedCache.from_dense_cache(model, dense)
    model._kv_head_budget_config = {"budget_scheme": "sparsemm", "sparsemm_window_size": recent,
                                   "sparsemm_ratio": 0.2}
    try:
        scores = []
        for prompt in (local, global_, local + "; " + global_):
            ids = torch.tensor([processor.tokenizer(prompt).input_ids], device=model.device)
            scores.append(_ragged_query_scores(model, ids, offsets))
        combined = _default_hermes_scores_for_trace(model, *scores)
        start = int(model.visual_start_idx)
        n = dense[0][0].shape[2] - start
        ranking = []
        for layer in combined:
            for score in layer:
                ranking.append(torch.argsort(score[start:n + start - recent], descending=True,
                                             stable=True).cpu().numpy())
        # Check true selector at deployed budgets. Near-tied topk order may differ;
        # record set differences rather than claim bit-identical tie handling.
        validation = []
        raw = np.stack([s[start:].float().cpu().numpy() for layer in combined for s in layer])
        for k in k_values:
            keeps, table, _ = _select_ragged_keep_indices(model, *scores, num_keep=k)
            for l, layer in enumerate(keeps):
                for g, keep in enumerate(layer):
                    h = l * len(layer) + g
                    b = min(int(table[l, g]), n)
                    expected = np.concatenate([ranking[h][:max(0, b - recent)], np.arange(n - recent, n)])
                    actual = keep.cpu().numpy()
                    actual = actual[actual >= start] - start
                    mismatch = len(set(expected.tolist()) ^ set(actual.tolist()))
                    delta = abs(float(raw[h, expected].sum(dtype=np.float64)) -
                                float(raw[h, actual].sum(dtype=np.float64)))
                    if len(expected) != len(actual) or delta > 1e-4:
                        raise AssertionError((l, g, k, mismatch, delta))
                    validation.append({"K": k, "layer": l, "kv_head": g,
                                       "tie_set_difference": mismatch, "score_sum_delta": delta})
        return np.stack(ranking), raw, validation, {"local": local, "global": global_}
    finally:
        model.kv_cache = dense
        model._layer_position_ids.clear()


def projected_norms(model):
    """Exact L2 via Gram matrix, avoiding materializing T x hidden_size."""
    import torch
    result = []
    start = int(model.visual_start_idx)
    for l, layer in enumerate(model.language_model.layers):
        values = model.kv_cache[l][1][0, :, start:].float()
        groups, n, d = values.shape
        heads = int(model.language_model.config.num_attention_heads)
        sharing = heads // groups
        w = layer.self_attn.o_proj.weight.float()
        norms = []
        for h in range(heads):
            wh = w[:, h * d:(h + 1) * d]
            gram = wh.T @ wh
            v = values[h // sharing]
            norms.append(((v @ gram) * v).sum(-1).clamp_min(0).sqrt())
        result.append(torch.stack(norms))
    return result


def collect_utility(model, input_text, norms):
    import torch
    start = int(model.visual_start_idx)
    n = norms[0].shape[-1]
    groups = int(model.language_model.config.num_key_value_heads)
    heads = int(model.language_model.config.num_attention_heads)
    sharing = heads // groups
    utilities = [torch.zeros((groups, n), device=model.device) for _ in norms]
    counts = [0] * len(norms)
    handles = []
    configs = [model.config, model.language_model.config]
    old = [c._attn_implementation for c in configs]
    positions = [p.clone() if p is not None else None for p in model._position_ids_cache]
    before = cache_digest(model)
    history = list(model.conv_history)

    def make_hook(l):
        def hook(module, inputs, outputs):
            counts[l] += 1
            # Legacy QA performs one final unused forward when it hits the
            # generation limit. Do not use that unconsumed prediction position.
            if counts[l] > 16:
                return
            a = outputs[1]
            if a is None:
                raise RuntimeError("Missing eager attention")
            # Last prefill position predicts the FIRST answer token; later calls
            # predict subsequent tokens. Earlier question positions are excluded.
            a = a[0, :, -1, start:start + n].float()
            value = (a * norms[l]).reshape(groups, sharing, n).sum(1)
            utilities[l].copy_(torch.maximum(utilities[l], value))
        return hook

    try:
        for c in configs:
            c._attn_implementation = "eager"
        for l, layer in enumerate(model.language_model.layers):
            handles.append(layer.self_attn.register_forward_hook(make_hook(l)))
        # Existing MCQA uses 16 tokens, temperature=0 and repetition_penalty=1.1.
        answer = model.question_answering(input_text, max_new_tokens=16)
    finally:
        for handle in handles:
            handle.remove()
        for c, impl in zip(configs, old):
            c._attn_implementation = impl
        model.conv_history = history
        # The legacy dense observer tracks visual positions without initial
        # text positions. Its QA truncation uses KV lengths (including text),
        # so restore the branch snapshot explicitly, not that truncation alone.
        model._position_ids_cache = positions
        model._layer_position_ids.clear()
    after = cache_digest(model)
    if before != after:
        raise AssertionError("Answer branch changed the visual-prefix KV")
    for a, b in zip(positions, model._position_ids_cache):
        if a is not None and not torch.equal(a, b):
            raise AssertionError("Answer branch changed prefix positions")
    if len(set(counts)) != 1 or counts[0] < 1:
        raise AssertionError(counts)
    return torch.stack(utilities).flatten(0, 1).cpu().numpy(), answer, counts[0], before


def make_curves(ranking, utility, recent):
    heads, n = utility.shape
    curves = np.zeros((heads, n + 1), dtype=np.float64)
    curves[:, recent] = utility[:, n - recent:].sum(1, dtype=np.float64)
    ordered = np.take_along_axis(utility, ranking, axis=1)
    curves[:, recent + 1:] = curves[:, recent, None] + np.cumsum(ordered, axis=1, dtype=np.float64)
    return curves


def analyze(out, args):
    from head_analysis.hermes_head_budget import load_head_scores
    from head_analysis.hermes_kv_head_budget import build_kv_head_budget_table
    manifests = json.loads((out / "manifest.json").read_text())["videos"]
    records = []
    for v in manifests:
        for q in v["conversations"]:
            stem = f"v{v['video_id']}_q{q['source_question_index']}"
            path = out / (stem + ".npz")
            if not path.exists():
                continue
            with np.load(path) as z:
                curves = make_curves(z["ranking"], z["utility"], args.recent)
            records.append((str(v["video_id"]), stem, curves))
    by_video = sorted({r[0] for r in records})
    if len(by_video) < 2:
        return {"status": "smoke_only", "observations": len(records)}
    score = load_head_scores(args.v8_scores, 28, 28)
    reports = {}
    for k in args.budgets:
        lower = int((k - args.recent) * 0.2) + args.recent
        eligible = [r for r in records if r[2].shape[1] - 1 >= k]
        videos = sorted({r[0] for r in eligible})
        if len(videos) < 2:
            continue
        solved, gaps = {}, {}
        for vid, stem, curves in eligible:
            solved[stem], gaps[stem] = solve_curves(curves, 112 * k, lower)
        means = {v: np.mean([solved[s] for vid, s, _ in eligible if vid == v], axis=0)
                 for v in videos}
        global_budget = integer_average(list(means.values()), 112 * k, lower)
        np.savetxt(out / f"budget_K{k}.csv", np.column_stack([
            np.repeat(np.arange(28), 4), np.tile(np.arange(4), 28), global_budget]),
            delimiter=",", header="layer,kv_head,budget", comments="", fmt="%d")
        # For the existing sparsemm runtime, non-negative weights equal to the
        # capacity above the common floor reproduce this exact table because
        # their sum is precisely the globally distributable capacity.
        runtime_score = global_budget - lower
        np.savetxt(out / f"runtime_score_K{k}.csv", np.column_stack([
            np.repeat(np.arange(28), 4), np.tile(np.arange(4), 28), runtime_score]),
            delimiter=",", header="layer,kv_head,score", comments="", fmt="%d")
        v8_raw = build_kv_head_budget_table(score, k, scheme="sparsemm",
                                            sparsemm_ratio=0.2, sparsemm_window_size=args.recent).flatten()
        v8 = v8_raw.copy()
        # Deterministic exact-total correction; disclose difference from runtime.
        while v8.sum() != 112 * k:
            sign = 1 if v8.sum() < 112 * k else -1
            i = int(np.argmax(v8)) if sign < 0 else int(np.argmin(v8))
            v8[i] += sign
        rows = []
        for vid, stem, curves in eligible:
            train = [means[v] for v in videos if v != vid]
            lu = integer_average(train, 112 * k, lower)
            policies = {"uniform": np.full(112, k), "v8_exact": v8, "lu_loov": lu,
                        "per_observation_oracle_allocation_diagnostic": solved[stem]}
            total_utility = float(curves[:, -1].sum())
            result = {"video_id": vid, "observation": stem, "total_utility": total_utility,
                      "visual_tokens": curves.shape[1] - 1, "policies": {}}
            for name, budget in policies.items():
                effective = np.minimum(budget, curves.shape[1] - 1)
                retained = float(curves[np.arange(112), effective].sum())
                result["policies"][name] = {"retention": retained / total_utility,
                    "actual_slots": int(effective.sum()), "capacity_slots": int(budget.sum())}
            result["lu_budget"] = lu.reshape(28, 4).tolist()
            rows.append(result)
        summary = {}
        for name in rows[0]["policies"]:
            values = [float(np.mean([r["policies"][name]["retention"] for r in rows
                                     if r["video_id"] == v])) for v in videos]
            summary[name] = {"video_mean_retention": float(np.mean(values)), "per_video": values}
        diff = np.asarray(summary["lu_loov"]["per_video"]) - np.asarray(summary["uniform"]["per_video"])
        rng = np.random.default_rng(args.seed)
        boot = rng.choice(diff, size=(10000, len(diff)), replace=True).mean(1)
        split = list(videos)
        random.Random(args.seed).shuffle(split)
        split_a, split_b = split[:len(split)//2], split[len(split)//2:]
        a = integer_average([means[v] for v in split_a], 112 * k, lower)
        b = integer_average([means[v] for v in split_b], 112 * k, lower)
        stability = {"split_a": split_a, "split_b": split_b,
            "budget_pearson": float(np.corrcoef(a, b)[0, 1]),
            "budget_total_variation": float(np.abs(a-b).sum() / (2 * 112 * k)),
            "top10_overlap": len(set(np.argsort(a)[-10:]) & set(np.argsort(b)[-10:]))}
        reports[str(k)] = {"videos": videos, "observations": len(rows), "lower_budget": lower,
            "summary": summary, "lu_minus_uniform_mean": float(diff.mean()),
            "lu_minus_uniform_video_bootstrap95": np.quantile(boot, [.025, .975]).tolist(),
            "lu_minus_uniform_positive_videos": int((diff > 0).sum()),
            "global_budget": global_budget.reshape(28, 4).tolist(),
            "global_budget_range": [int(global_budget.min()), int(global_budget.max())],
            "v8_original_sum": int(v8_raw.sum()), "v8_corrected_sum": int(v8.sum()),
            "split_stability": stability, "solver_diagnostics": gaps, "rows": rows}
    report = {"status": "complete" if len(by_video) == args.num_videos else "partial",
              "num_videos": len(by_video), "num_observations": len(records),
              "metric": "raw projected future-answer utility retained; NOT compressed QA accuracy",
              "reports": reports}
    save_json(out / "report.json", report)
    return report


def run(args):
    import torch
    from head_analysis.profile_horizonkv_pilot import build_pilot_model, token_times_for_chunk
    from head_analysis.profile_effective_memory_readout import load_video_tensor
    from video_qa.base import BaseVQA
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    selected = choose_manifest(args, out)
    if args.analyze_only:
        print(json.dumps(analyze(out, args), indent=2))
        return
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    model_args = SimpleNamespace(model="qwen2.5_vl_7b", device=0, fixed_frame_canvas_size=448,
                                 kv_size=1000000, sample_fps=0.5, compress_mode="hermes")
    model, processor = build_pilot_model(model_args, "models/Qwen2.5-VL-7B-Instruct")
    import transformers
    save_json(out / "runtime.json", {"torch": torch.__version__, "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(), "dtype": str(model.dtype),
        "processor": type(processor.image_processor).__name__,
        "attention_implementation": model.language_model.config._attn_implementation,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "utility_version": "answer_prediction_positions_at_most_16_v1"})
    analyzer = BaseVQA(anno=selected, save_dir=str(out / "unused_qa_output"), sample_fps=0.5,
                       qa_model=model, qa_processor=processor, num_chunks=None, chunk_idx=None)
    completed = 0
    for video in selected:
        if args.smoke and completed:
            break
        started = time.time()
        print(f"LU_VIDEO_START id={video['video_id']}", flush=True)
        # Decode only the required prefix; sampled indices before the cutoff
        # are identical to the project's full-video loader.
        clip_start = float(video.get("clip", [0, 0])[0]) if video.get("clip") else 0.0
        load_sample = {**video, "clip": [clip_start, clip_start + max(
            float(q["end_time"]) for q in video["conversations"])]}
        video_tensor = load_video_tensor(analyzer, load_sample)
        model.clear_cache()
        model._position_ids_cache = [None] * model.num_layers
        model._layer_position_ids.clear()
        model.total_processed_frames = 0
        model.encode_init_prompt()
        current = 0
        source_times, source_chunks = [], []
        chunk_index = 0
        for q in video["conversations"]:
            end = min(len(video_tensor), math.ceil(float(q["end_time"]) * .5))
            while current < end:
                stop = min(current + 16, end)
                chunk = video_tensor[current:stop]
                pre = model.kv_cache[0][0].shape[2]
                model.encode_video_chunk(chunk)
                added = model.kv_cache[0][0].shape[2] - pre
                times, chunks, _ = token_times_for_chunk(model, processor, chunk, current,
                    chunk_index, .5, new_visual_tokens=added)
                assert len(times) == added and len(chunks) == added, "Token provenance mismatch"
                source_times.extend(times)
                source_chunks.extend(chunks)
                current = stop
                chunk_index += 1
            stem = f"v{video['video_id']}_q{q['source_question_index']}"
            if (out / (stem + ".npz")).exists():
                print("LU_RESUME_SKIP " + stem, flush=True)
                continue
            before = cache_digest(model)
            ranking, raw, checks, prompts = collect_selector(model, processor, args.budgets, args.recent)
            assert before == cache_digest(model), "Selector mutated full cache"
            norms = projected_norms(model)
            input_text = analyzer.format_mcqa_prompt(q["question"], q["choices"])
            utility, answer, count, digest = collect_utility(model, input_text, norms)
            assert len(source_times) == utility.shape[1], "Prefix provenance mismatch"
            if not np.isfinite(utility).all() or np.any(utility < 0) or utility.sum() <= 0:
                raise AssertionError("Invalid utility")
            expected_letter = chr(65 + q["choices"].index(q["answer"]))
            predicted_letter = analyzer.extract_characters_regex(answer)
            metadata = {"video_id": video["video_id"], "question": q,
                "answer": answer, "expected_letter": expected_letter,
                "predicted_letter": predicted_letter, "full_cache_correct": predicted_letter == expected_letter,
                "attention_forward_calls": count, "visual_tokens": utility.shape[1],
                "utility_prediction_positions": min(count, 16), "hit_generation_limit": count > 16,
                "prefix_sha256": digest, "selector_validation": checks,
                "pseudo_prompts": prompts, "full_cache": True,
                "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
                "elapsed_video_seconds": time.time() - started,
                "norm_per_layer_mean": [float(x.mean()) for x in norms]}
            save_json(out / (stem + ".json"), metadata)
            np.savez_compressed(out / (stem + ".npz"), ranking=ranking.astype(np.int32),
                utility=utility, hermes_score=raw,
                source_times=np.asarray(source_times), source_chunks=np.asarray(source_chunks),
                source_index=np.arange(utility.shape[1], dtype=np.int32))
            del norms
            print(f"LU_OBSERVATION_DONE {stem} tokens={utility.shape[1]} answer={answer!r} "
                  f"correct={metadata['full_cache_correct']} calls={count}", flush=True)
        completed += 1
        print(f"LU_VIDEO_DONE id={video['video_id']} elapsed={time.time()-started:.1f}s", flush=True)
        del video_tensor
    del model
    torch.cuda.empty_cache()
    report = analyze(out, args)
    print("LU_RUN_DONE " + json.dumps({k: v for k, v in report.items() if k != "reports"}), flush=True)


def main():
    p = argparse.ArgumentParser()
    root = "results/myobservations/current_ourtop_k1000_fixed448_g0"
    p.add_argument("--calibration", nargs="+", default=[f"{root}/calibration_complement_25_gpu{i}.json" for i in (0, 1)])
    p.add_argument("--exclude-annotation", default="data/streamingbench/streamingbench_realtime_half_userpaths.json")
    p.add_argument("--output", default="results/myobservations/lu_kv_offline_10v_20260913")
    p.add_argument("--num-videos", type=int, default=10)
    p.add_argument("--questions-per-video", type=int, default=2)
    p.add_argument("--min-time", type=float, default=120)
    p.add_argument("--max-time", type=float, default=240)
    p.add_argument("--seed", type=int, default=20260913)
    p.add_argument("--recent", type=int, default=32)
    p.add_argument("--budgets", type=int, nargs="+", default=[1000, 6000])
    p.add_argument("--v8-scores", default="results/myobservations/relocatev3/v8_blend3_scores.csv")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--analyze-only", action="store_true")
    args = p.parse_args()
    # Operational flags are excluded from immutable experiment metadata.
    import torch
    with torch.inference_mode():
        run(args)


if __name__ == "__main__":
    main()
