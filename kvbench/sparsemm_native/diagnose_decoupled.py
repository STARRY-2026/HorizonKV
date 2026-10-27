"""Two-question factorial substitution of coupled/independent H and R KV.

This is a diagnostic on the two pilot regressions, not an accuracy benchmark.
All four cases have identical prompts, positions and physical per-head slots.
The all-coupled/all-independent endpoints must reproduce the frozen pilot.
"""

import argparse
import copy
import json
import math
from pathlib import Path
import signal

from sparsemm_native import run_ovobench as official
from sparsemm_native import run_ovobench_recent4 as common
from sparsemm_native.supervise_ovobench_dual import now, write_json


ROOT = official.ROOT
PILOT = ROOT / "results/ovobench_decoupled_pilot24_k500_fps05_official_20261021"


def old_parts(coupled, independent_history):
    """Copy old history and old recent tensors without recomputing either."""
    import torch
    history = copy.deepcopy(independent_history)
    history.recent_dense = None
    p = coupled.prefix_tokens
    r = int(coupled.recent_budgets[0, 0])
    recent = []
    for source, target in zip(coupled.layers, history.layers):
        keys, values, positions, lengths, rk, rv = [], [], [], [], [], []
        for head in range(coupled.num_kv_heads):
            k, v = source.get_segment(head)
            pos = source.get_position_segment(head)
            split = int(k.shape[0]) - r
            if split < p:
                raise AssertionError("Coupled state does not contain the entire recent window")
            keys.append(k[:split].clone())
            values.append(v[:split].clone())
            positions.append(pos[:, :split].clone())
            lengths.append(split)
            rk.append(torch.cat((k[:p], k[-r:])).clone())
            rv.append(torch.cat((v[:p], v[-r:])).clone())
        target.flat_k, target.flat_v = torch.cat(keys), torch.cat(values)
        target.flat_pos = torch.cat(positions, dim=1)
        target.head_lens = target.head_lens.new_tensor(lengths)
        target._refresh()
        recent.append((torch.stack(rk)[None], torch.stack(rv)[None]))
    return history, recent


def state_differences(left, right, *, prefix):
    """RMS differences and overlap of retained historical spatial/time IDs."""
    import torch
    aggregate = dict(shared_history_positions=0, union_history_positions=0)
    for a, b in zip(left.layers, right.layers):
        for head in range(left.num_kv_heads):
            pa = a.get_position_segment(head)[:, prefix:].T.cpu().tolist()
            pb = b.get_position_segment(head)[:, prefix:].T.cpu().tolist()
            sa, sb = set(map(tuple, pa)), set(map(tuple, pb))
            aggregate["shared_history_positions"] += len(sa & sb)
            aggregate["union_history_positions"] += len(sa | sb)
    total = aggregate["union_history_positions"]
    aggregate["history_position_jaccard"] = aggregate["shared_history_positions"] / total if total else 1.
    return aggregate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    queries = {q["query_id"]: q for q in official.queries()}
    selected = [queries[qid] for qid in ("1372", "1534_6")]
    references = {arm: {row["query_id"]: row for row in map(json.loads, (PILOT / arm / "queries.jsonl").read_text().splitlines())}
                  for arm in ("dp_recent4", "dp_decoupled")}
    pins = json.loads((PILOT / "protocol.json").read_text())["code_sha256"]
    if any(official.sha256(ROOT / name) != checksum for name, checksum in pins.items()):
        raise ValueError("Pilot source changed")
    status = dict(state="starting", started_at=now(), questions=[q["query_id"] for q in selected])
    write_json(root / "status.json", status)
    try:
        import torch
        from sparsemm_native.decoupled import DecoupledRecent4StreamingQwen, prefill_recent, concatenate_query_cache
        from sparsemm_native.recent4 import Recent4StreamingQwen
        from sparsemm_native.gpu_slots import gpu_lease
        from sparsemm_native.assets import asset_metadata
        from sparsemm_protokv.model_loader import load_pretrained
        from sparsemm_protokv.config import StreamingConfig
        from sparsemm_protokv.streaming import StreamingQwen
        from sparsemm_protokv.models.qwen2_5_vl_stream import BatchScopedVideoReader
        from sparsemm_baselines.run_streamingbench import sampled_indices
        config = StreamingConfig(method="sparsemm", budget=500, budget_scope="visual",
            sparsemm_budget_path=str(official.DP_DUAL_BUDGET), sparsemm_cache_policy="two_bank",
            sparsemm_two_bank_gate="legacy_global")
        results = []
        with gpu_lease(root):
            model, processor = load_pretrained(official.MODEL, dtype="bfloat16")
            old = Recent4StreamingQwen(model, processor, config, sample_fps=.5,
                video_max_pixels=151200, validate_state=True, asset_metadata=asset_metadata())
            new = DecoupledRecent4StreamingQwen(model, processor, config, sample_fps=.5,
                video_max_pixels=151200, validate_state=True, asset_metadata=asset_metadata())
            native_recent = Recent4StreamingQwen(model, processor, config, sample_fps=.5,
                video_max_pixels=151200, validate_state=True, asset_metadata=asset_metadata())
            with torch.inference_mode():
                for query in selected:
                    status.update(state="encoding", query_id=query["query_id"], updated_at=now())
                    write_json(root / "status.json", status)
                    reader = BatchScopedVideoReader(query["video_path"], num_threads=1)
                    indices = sampled_indices(len(reader), reader.get_avg_fps(), .5)
                    target = min(len(indices), math.ceil(query["clip_end"]*.5))
                    batches = [torch.from_numpy(reader.get_batch(indices[start:min(start+16,target)]).asnumpy())
                               for start in range(0,target,16)]
                    for bridge in (old,new):
                        bridge._active_video_id = query["query_id"]
                        bridge.encode_init_prompt()
                        for index, frames in enumerate(batches):
                            bridge.encode_video_chunk(frames,is_last_chunk=index==len(batches)-1)
                            bridge.predict_and_compress()
                    if old._next_position != new._next_position or old._seen_visual_tokens != new._seen_visual_tokens:
                        raise AssertionError("Factorial endpoints have different positions/visual inputs")
                    new_history = new.kv_cache
                    new_recent = prefill_recent(new.language_model,new_history.prefix_dense,new_history.units)
                    old_history, old_recent = old_parts(old.kv_cache,new_history)
                    # Rebuild the same isolated recent features using the old
                    # native ragged attention and per-unit forward path. This
                    # separates context removal from the dense-kernel change.
                    native_recent.encode_init_prompt()
                    for unit in new_history.units:
                        native_recent.forward_chunk(unit.features,unit.positions,compress=True,
                            time=unit.frame_start+unit.real_frames)
                    _, isolated_native_recent = old_parts(native_recent.kv_cache,new_history)
                    row = dict(query_id=query["query_id"],task=query["task"],target=query["target"],
                               prompt=query["prompt"],frames=target,cases={},
                               history_selection=state_differences(old_history,new_history,prefix=new_history.prefix_tokens))
                    expected_slots=None
                    for name,history,recent in (("old_H_old_R",old_history,old_recent),
                            ("old_H_new_R",old_history,new_recent),
                            ("new_H_old_R",new_history,old_recent),
                            ("new_H_new_R",new_history,new_recent),
                            ("old_H_new_R_old_kernel",old_history,isolated_native_recent),
                            ("new_H_new_R_old_kernel",new_history,isolated_native_recent)):
                        fused,audit=concatenate_query_cache(history,recent,new.visual_start_idx+new._seen_visual_tokens)
                        if expected_slots is None:expected_slots=audit["actual_visual_slots"]
                        if audit["actual_visual_slots"]!=expected_slots:raise AssertionError("Factorial slot budgets differ")
                        new.kv_cache=fused
                        try:
                            response=StreamingQwen.question_answering(new,
                                {"question":query["prompt"],"formatted_question":query["prompt"],
                                 "prompt":new.get_prompt(query["prompt"],mc=False)},max_new_tokens=16)
                        finally:
                            new.kv_cache=new_history
                            new.conv_history.clear()
                            new.validation_checks["qa_checks"].clear()
                        row["cases"][name]=dict(response=response,
                            correct=official.official_correct(query["task"],response,query["target"]),
                            actual_visual_slots=audit["actual_visual_slots"])
                        print(query["query_id"],name,response,flush=True)
                    if (row["cases"]["old_H_old_R"]["response"] != references["dp_recent4"][query["query_id"]]["response"]
                            or row["cases"]["new_H_new_R"]["response"] != references["dp_decoupled"][query["query_id"]]["response"]):
                        raise AssertionError("Factorial endpoints did not reproduce the scored pilot")
                    row["pilot_endpoints_reproduced"]=True
                    row["isolated_recent_kernel_prediction_agreement"] = (
                        row["cases"]["old_H_new_R"]["response"] == row["cases"]["old_H_new_R_old_kernel"]["response"]
                        and row["cases"]["new_H_new_R"]["response"] == row["cases"]["new_H_new_R_old_kernel"]["response"])
                    results.append(row)
                    write_json(root/"results.json",dict(completed=len(results),results=results,
                        diagnostic_source_sha256=official.sha256(Path(__file__)),pilot_code_sha256=pins,
                        note="Selected pilot regressions only; factor substitutions hold budget/prompt/positions fixed. "
                        "Additional recent rebuild uses the original ragged kernel/per-unit path."))
        status.update(state="complete",updated_at=now())
        write_json(root/"status.json",status)
    except BaseException as error:
        status.update(state="failed",updated_at=now(),error=f"{type(error).__name__}: {error}")
        write_json(root/"status.json",status)
        raise


if __name__=="__main__":
    main()
