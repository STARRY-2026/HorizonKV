import os
import sys
import argparse
import glob
import subprocess
import multiprocessing
import csv
import shlex
import tempfile


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def exec(cmd, sub=False, device=None):
    """Run one command and propagate failures to its caller/process."""
    if isinstance(cmd, str):
        cmd = shlex.split(cmd)
    print(f'exec: {shlex.join(cmd)}', flush=True)
    my_env = os.environ.copy()
    existing_pythonpath = my_env.get("PYTHONPATH")
    my_env["PYTHONPATH"] = os.pathsep.join(
        [PROJECT_ROOT] + ([existing_pythonpath] if existing_pythonpath else [])
    )
    if sub:
        my_env["CUDA_VISIBLE_DEVICES"] = str(device)
        cuda_lib_dirs = glob.glob(
            os.path.join(
                sys.prefix,
                "lib",
                "python*",
                "site-packages",
                "nvidia",
                "cu[0-9]*",
                "lib",
            )
        )
        if cuda_lib_dirs:
            existing_path = my_env.get("LD_LIBRARY_PATH")
            path_entries = cuda_lib_dirs + ([existing_path] if existing_path else [])
            my_env["LD_LIBRARY_PATH"] = os.pathsep.join(path_entries)
    return subprocess.run(cmd, env=my_env, check=True)


def merge_chunk_results(chunk_files, results_path):
    """Publish a complete CSV only after all chunk headers validate."""
    output_dir = os.path.dirname(os.path.abspath(results_path))
    os.makedirs(output_dir, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8", dir=output_dir,
            prefix=".results-", suffix=".csv", delete=False,
        ) as output:
            temporary_path = output.name
            writer = csv.writer(output)
            expected_header = None
            row_count = 0
            for chunk_file in chunk_files:
                with open(chunk_file, newline="", encoding="utf-8") as source:
                    reader = csv.reader(source)
                    header = next(reader, None)
                    if not header:
                        raise ValueError(f"Empty result chunk: {chunk_file}")
                    if expected_header is None:
                        expected_header = header
                        writer.writerow(header)
                    elif header != expected_header:
                        raise ValueError(f"Inconsistent result columns: {chunk_file}")
                    for row in reader:
                        if row:
                            writer.writerow(row)
                            row_count += 1
            if row_count == 0:
                raise ValueError("Inference produced no result rows.")
        os.replace(temporary_path, results_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            os.unlink(temporary_path)
    for chunk_file in chunk_files:
        os.remove(chunk_file)


BENCHMARK_CONFIGS = {
    "videomme": {
        "streaming": False,
        "anno_path": "data/videomme/videomme.json",
        "eval_cmds": [
            "{python} {project_root}/eval/eval_multiple_choice.py general --results_path {results_path}",
            "{python} {project_root}/eval/eval_multiple_choice.py videomme --results_path {results_path}",
        ],
    },
    "mvbench": {
        "streaming": False,
        "anno_path": "data/mvbench/mvbench.json",
        "eval_cmds": [
            "{python} {project_root}/eval/eval_multiple_choice.py general --results_path {results_path}",
        ],
    },
    "egoschema": {
        "streaming": False,
        "anno_path": "data/egoschema/egoschema.json",
        "eval_cmds": [
            "{python} {project_root}/eval/eval_multiple_choice.py egoschema --results_path {results_path}",
        ],
    },
    "rvs_ego": {
        "streaming": True,
        "anno_path": "data/rvs/ego/ego4d_oe.json",
        "eval_cmds": [
            "{python} {project_root}/eval/eval_open_ended.py --pred_path {results_path} --output_dir {save_dir}/tmp --output_json {save_dir}/results.json",
        ],
    },
    "rvs_movie": {
        "streaming": True,
        "anno_path": "data/rvs/movie/movienet_oe.json",
        "eval_cmds": [
            "{python} {project_root}/eval/eval_open_ended.py --pred_path {results_path} --output_dir {save_dir}/tmp --output_json {save_dir}/results.json",
        ],
    },
    "ovobench": {
        "streaming": True,
        "anno_path": "data/ovobench/ovobench_realtime_backeward.json",
        "eval_cmds": [
            "{python} {project_root}/eval/eval_multiple_choice.py general --results_path {results_path}",
        ],
    },
    "streamingbench": {
        "streaming": True,
        "anno_path": "data/streamingbench/streamingbench_realtime_half.json",
        "eval_cmds": [
            "{python} {project_root}/eval/eval_multiple_choice.py general --results_path {results_path}",
        ],
    },
}


def run_eval(args, config):
    num_chunks = args.num_chunks
    if num_chunks < 1:
        raise ValueError("--num_chunks must be positive.")
    anno_path = args.anno_path or os.path.join(PROJECT_ROOT, config["anno_path"])
    anno_path = os.path.abspath(os.path.expanduser(anno_path))
    if not os.path.isfile(anno_path):
        raise FileNotFoundError(f"Annotation file not found: {anno_path}")
    from head_analysis.horizonkv_policy import validate_horizonkv_args
    validate_horizonkv_args(args)
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
    if (
        args.kv_head_role_history_uniform_memory_ratio > 0.0
        or args.kv_head_role_history_uniform_mixed_ratio > 0.0
    ) and not role_score_path:
        raise ValueError(
            "Role history-uniform selection requires --kv_head_role_recent_scores "
            "or --kv_head_role_dual_scores."
        )
    if role_score_path and not args.kv_head_ragged_prefill:
        args.kv_head_ragged_prefill = True
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
    if args.rolekv_ragged and not args.kv_head_ragged_prefill:
        args.kv_head_ragged_prefill = True
    if args.answer_head_router:
        if not args.answer_head_profile:
            raise ValueError("--answer_head_router requires --answer_head_profile")
        if not os.path.isfile(args.answer_head_profile):
            raise FileNotFoundError(
                f"Answer-head profile not found: {args.answer_head_profile}"
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

    head_budget_suffix = ""
    if args.head_budget_scores:
        score_tag = os.path.splitext(os.path.basename(args.head_budget_scores))[0]
        if not score_tag:
            score_tag = args.head_budget_scores.replace("/", "_")
        head_budget_suffix = f"-headbudget-{score_tag}"
    layer_budget_suffix = ""
    if args.layer_budget_scores:
        score_tag = os.path.splitext(os.path.basename(args.layer_budget_scores))[0]
        if not score_tag:
            score_tag = args.layer_budget_scores.replace("/", "_")
        layer_budget_suffix = f"-layerbudget-{score_tag}"
        if args.layer_budget_variable:
            layer_budget_suffix += "-varlen"
    kv_head_budget_suffix = ""
    if args.kv_head_budget_scores:
        score_tag = os.path.splitext(os.path.basename(args.kv_head_budget_scores))[0]
        if not score_tag:
            score_tag = args.kv_head_budget_scores.replace("/", "_")
        kv_head_budget_suffix = (
            f"-kvheadbudget-{score_tag}"
            f"-{args.kv_head_budget_scheme}"
            f"-union{args.kv_head_budget_union_cap_ratio:g}"
        )
        if args.kv_head_budget_scheme.startswith("sparsemm"):
            kv_head_budget_suffix += (
                f"-r{args.kv_head_budget_sparsemm_ratio:g}"
                f"-w{args.kv_head_budget_sparsemm_window_size}"
            )
        if args.kv_head_budget_floor_ratio > 0:
            kv_head_budget_suffix += f"-floor{args.kv_head_budget_floor_ratio:g}"
        if args.kv_head_ragged_prefill:
            kv_head_budget_suffix += "-raggedprefill"
        elif args.kv_head_ragged_decode:
            kv_head_budget_suffix += "-raggeddecode"
        if args.kv_head_budget_memory_current_split:
            mem_tag = os.path.splitext(os.path.basename(
                args.kv_head_budget_memory_scores or args.kv_head_budget_scores or ""
            ))[0]
            if not mem_tag:
                mem_tag = (args.kv_head_budget_memory_scores or args.kv_head_budget_scores or "").replace("/", "_")
            kv_head_budget_suffix += (
                f"-memsplit-{mem_tag}"
                f"-mr{args.kv_head_budget_memory_ratio:g}"
                f"-ds{args.kv_head_budget_split_dynamic_scale:g}"
            )
        if args.kv_head_policy_roles:
            role_tag = os.path.splitext(os.path.basename(args.kv_head_policy_roles))[0]
            if not role_tag:
                role_tag = args.kv_head_policy_roles.replace("/", "_")
            kv_head_budget_suffix += (
                f"-kvpolicy-{role_tag}"
                f"-mq{args.kv_head_policy_memory_quantile:g}"
                f"-cq{args.kv_head_policy_current_quantile:g}"
                f"-mb{args.kv_head_policy_memory_bonus:g}"
                f"-cb{args.kv_head_policy_current_bonus:g}"
            )
        if args.kv_head_recency_scores:
            recency_tag = os.path.splitext(os.path.basename(args.kv_head_recency_scores))[0]
            if not recency_tag:
                recency_tag = args.kv_head_recency_scores.replace("/", "_")
            kv_head_budget_suffix += (
                f"-headalpha-{recency_tag}"
                f"-d{args.kv_head_recency_delta:g}"
                f"-t{args.kv_head_recency_tau:g}"
                f"-a{args.kv_head_recency_alpha_min:g}-{args.kv_head_recency_alpha_max:g}"
                f"-kd{args.kv_head_recency_k_delta:g}"
            )
        if args.kv_head_layerwise_last_frame:
            kv_head_budget_suffix += (
                f"-lwlf-s{args.kv_head_layerwise_shallow_end}"
                f"-m{args.kv_head_layerwise_mid_end}"
                f"-sr{args.kv_head_layerwise_shallow_ratio:g}"
                f"-mr{args.kv_head_layerwise_mid_ratio:g}"
                f"-dr{args.kv_head_layerwise_deep_ratio:g}"
            )
        if role_score_path:
            role_tag = os.path.splitext(os.path.basename(role_score_path))[0]
            if not role_tag:
                role_tag = role_score_path.replace("/", "_")
            if args.kv_head_role_recent_memory_count >= 0 or args.kv_head_role_recent_current_count >= 0:
                split_tag = (
                    f"-cc{args.kv_head_role_recent_current_count}"
                    f"-mc{args.kv_head_role_recent_memory_count}"
                )
            else:
                split_tag = (
                    f"-cq{args.kv_head_role_recent_current_quantile:g}"
                    f"-mq{args.kv_head_role_recent_memory_quantile:g}"
                )
            kv_head_budget_suffix += (
                f"-{'roledual' if args.kv_head_role_dual_scores else 'rolerecent'}-{role_tag}"
                f"{split_tag}"
                f"-mem{args.kv_head_role_recent_memory_tokens}"
                f"-mix{args.kv_head_role_recent_mixed_tokens}"
                f"-cur{args.kv_head_role_recent_current_tokens}"
                f"-{args.kv_head_role_recent_window_mode}"
            )
            if args.kv_head_role_recent_uniform_budget:
                kv_head_budget_suffix += "-ub"
            if args.kv_head_role_budget_enable:
                mixed_budget_tag = "auto" if args.kv_head_role_budget_auto_mixed else str(args.kv_head_role_mixed_budget)
                kv_head_budget_suffix += (
                    f"-rb-m{args.kv_head_role_memory_budget}"
                    f"-x{mixed_budget_tag}"
                    f"-c{args.kv_head_role_current_budget}"
                )
            if (
                args.kv_head_role_history_uniform_memory_ratio > 0.0
                or args.kv_head_role_history_uniform_mixed_ratio > 0.0
            ):
                kv_head_budget_suffix += (
                    f"-ufm{args.kv_head_role_history_uniform_memory_ratio:g}"
                    f"-ufx{args.kv_head_role_history_uniform_mixed_ratio:g}"
                    f"-ufb{args.kv_head_role_history_uniform_bins}"
                )
    horizonkv_suffix = ""
    if args.horizonkv_policy:
        policy_tag = os.path.splitext(os.path.basename(args.horizonkv_policy))[0]
        if not policy_tag:
            policy_tag = args.horizonkv_policy.replace("/", "_")
        horizonkv_suffix = (
            f"-horizonkv-{policy_tag}"
            f"-r{args.horizonkv_recent_capacity}"
            f"-g{args.horizonkv_ema_gamma:g}"
            f"-age{int(args.horizonkv_apply_age_penalty)}"
            f"-cov{args.horizonkv_temporal_coverage}"
            f"-probe{args.horizonkv_probe_style}"
            f"-blend{args.horizonkv_score_blend}"
        )
    timestamp_v8_suffix = ""
    if args.timestamp_v8_policy:
        policy_tag = os.path.splitext(
            os.path.basename(args.timestamp_v8_policy)
        )[0]
        timestamp_v8_suffix = (
            f"-timestampv8-{policy_tag}-g{args.timestamp_v8_grace_frames}"
        )
    dscache_suffix = ""
    if args.defer_last_frame_compress:
        dscache_suffix += f"-deferlast-{args.defer_last_frame_budget_mode}"
        if args.defer_last_frame_keep_ratio != 1.0:
            dscache_suffix += f"-r{args.defer_last_frame_keep_ratio:g}"
    if args.last_frame_stride != 1.0:
        dscache_suffix += f"-laststride{args.last_frame_stride:g}"
    if args.partition_position_frames > 0:
        dscache_suffix += f"-partpos{args.partition_position_frames}"
    if args.short_term_frames > 0:
        dscache_suffix += f"-stf{args.short_term_frames}"
    if args.multi_scale:
        dscache_suffix += "-ms"
    if args.layered_resolution:
        dscache_suffix += "-lr"
    if args.fixed_base_resolution:
        dscache_suffix += "-fbr"
    if args.fixed_frame_canvas_size > 0:
        dscache_suffix += f"-ffc{args.fixed_frame_canvas_size}-{args.fixed_frame_pad_mode}"
    if args.transient_highres:
        dscache_suffix += "-thr"
    if args.feature_buffer_frames > 0:
        dscache_suffix += f"-fb{args.feature_buffer_frames}"
    rolekv_suffix = ""
    if args.rolekv_ragged:
        class_tag = os.path.splitext(os.path.basename(args.rolekv_head_classes))[0]
        if not class_tag:
            class_tag = args.rolekv_head_classes.replace("/", "_")
        rolekv_suffix = (
            f"-rolekvragged-{class_tag}"
            f"-{args.rolekv_ragged_mode}"
            f"-q{args.rolekv_quota_ratio:g}"
            f"-lm{args.rolekv_lambda_memory:g}"
            f"-lc{args.rolekv_lambda_current:g}"
        )
    output_suffix = f"-{args.output_tag}" if args.output_tag else ""
    save_dir = (
        f"{PROJECT_ROOT}/results/{args.model}/{args.dataset}/"
        f"fps{args.sample_fps}-kv{args.kv_size}-{args.compress_mode}"
        f"{head_budget_suffix}{layer_budget_suffix}{kv_head_budget_suffix}{horizonkv_suffix}{timestamp_v8_suffix}{rolekv_suffix}{dscache_suffix}"
        f"{output_suffix}"
    )
    if args.output_dir:
        save_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    streaming = config["streaming"]
    results_path = f"{save_dir}/results.csv"

    if not args.only_eval:
        devices = args.devices[:num_chunks] if args.devices else [str(idx) for idx in range(num_chunks)]
        if len(devices) < num_chunks:
            raise ValueError(
                f"num_chunks={num_chunks} requires {num_chunks} devices, "
                f"but got {len(devices)} from --devices."
            )
        processes = []
        for idx in range(num_chunks):
            cmd = [
                sys.executable, "-m", "video_qa.hermes_vqa",
                "--model", args.model,
                "--sample_fps", str(args.sample_fps),
                "--save_dir", save_dir,
                "--anno_path", anno_path,
                "--debug", args.debug,
                "--num_chunks", str(num_chunks),
                "--chunk_idx", str(idx),
                "--streaming", str(streaming),
                "--compress_mode", args.compress_mode,
            ]
            if args.kv_size is not None:
                cmd.extend(["--kv_size", str(args.kv_size)])
            if args.model_path:
                cmd.extend(["--model_path", args.model_path])
            if args.head_budget_scores:
                cmd.extend(["--head_budget_scores", args.head_budget_scores])
            if args.layer_budget_scores:
                cmd.extend(["--layer_budget_scores", args.layer_budget_scores])
                cmd.extend(["--layer_budget_variable", str(args.layer_budget_variable)])
            if args.kv_head_budget_scores:
                cmd.extend(["--kv_head_budget_scores", args.kv_head_budget_scores])
                cmd.extend([
                    "--kv_head_budget_union_cap_ratio",
                    str(args.kv_head_budget_union_cap_ratio),
                ])
                cmd.extend([
                    "--kv_head_budget_max_mask_q_len",
                    str(args.kv_head_budget_max_mask_q_len),
                ])
                cmd.extend(["--kv_head_budget_scheme", args.kv_head_budget_scheme])
                cmd.extend([
                    "--kv_head_budget_sparsemm_ratio",
                    str(args.kv_head_budget_sparsemm_ratio),
                ])
                cmd.extend([
                    "--kv_head_budget_sparsemm_window_size",
                    str(args.kv_head_budget_sparsemm_window_size),
                ])
                cmd.extend(["--kv_head_selector_policy", args.kv_head_selector_policy])
                cmd.extend([
                    "--kv_head_budget_floor_ratio",
                    str(args.kv_head_budget_floor_ratio),
                ])
                if args.kv_head_budget_memory_current_split:
                    cmd.extend(["--kv_head_budget_memory_current_split",
                                str(args.kv_head_budget_memory_current_split)])
                    if args.kv_head_budget_memory_scores:
                        cmd.extend(["--kv_head_budget_memory_scores",
                                    args.kv_head_budget_memory_scores])
                    cmd.extend(["--kv_head_budget_memory_ratio",
                                str(args.kv_head_budget_memory_ratio)])
                    cmd.extend(["--kv_head_budget_split_dynamic_scale",
                                str(args.kv_head_budget_split_dynamic_scale)])
                if args.kv_head_policy_roles:
                    cmd.extend(["--kv_head_policy_roles", args.kv_head_policy_roles])
                    cmd.extend(["--kv_head_policy_memory_quantile",
                                str(args.kv_head_policy_memory_quantile)])
                    cmd.extend(["--kv_head_policy_current_quantile",
                                str(args.kv_head_policy_current_quantile)])
                    cmd.extend(["--kv_head_policy_memory_bonus",
                                str(args.kv_head_policy_memory_bonus)])
                    cmd.extend(["--kv_head_policy_current_bonus",
                                str(args.kv_head_policy_current_bonus)])
                if args.kv_head_recency_scores:
                    cmd.extend(["--kv_head_recency_scores", args.kv_head_recency_scores])
                    cmd.extend(["--kv_head_recency_metric", args.kv_head_recency_metric])
                    cmd.extend(["--kv_head_recency_delta", str(args.kv_head_recency_delta)])
                    cmd.extend(["--kv_head_recency_tau", str(args.kv_head_recency_tau)])
                    cmd.extend(["--kv_head_recency_alpha_min", str(args.kv_head_recency_alpha_min)])
                    cmd.extend(["--kv_head_recency_alpha_max", str(args.kv_head_recency_alpha_max)])
                    cmd.extend(["--kv_head_recency_k_delta", str(args.kv_head_recency_k_delta)])
                    cmd.extend(["--kv_head_recency_k_min", str(args.kv_head_recency_k_min)])
                    cmd.extend(["--kv_head_recency_k_max", str(args.kv_head_recency_k_max)])
                if args.kv_head_layerwise_last_frame:
                    cmd.extend(["--kv_head_layerwise_last_frame", str(args.kv_head_layerwise_last_frame)])
                    cmd.extend(["--kv_head_layerwise_shallow_end", str(args.kv_head_layerwise_shallow_end)])
                    cmd.extend(["--kv_head_layerwise_mid_end", str(args.kv_head_layerwise_mid_end)])
                    cmd.extend(["--kv_head_layerwise_shallow_ratio", str(args.kv_head_layerwise_shallow_ratio)])
                    cmd.extend(["--kv_head_layerwise_mid_ratio", str(args.kv_head_layerwise_mid_ratio)])
                    cmd.extend(["--kv_head_layerwise_deep_ratio", str(args.kv_head_layerwise_deep_ratio)])
                if role_score_path:
                    if args.kv_head_role_dual_scores:
                        cmd.extend(["--kv_head_role_dual_scores", args.kv_head_role_dual_scores])
                    else:
                        cmd.extend(["--kv_head_role_recent_scores", args.kv_head_role_recent_scores])
                    cmd.extend(["--kv_head_role_recent_memory_quantile", str(args.kv_head_role_recent_memory_quantile)])
                    cmd.extend(["--kv_head_role_recent_current_quantile", str(args.kv_head_role_recent_current_quantile)])
                    cmd.extend(["--kv_head_role_recent_memory_count", str(args.kv_head_role_recent_memory_count)])
                    cmd.extend(["--kv_head_role_recent_current_count", str(args.kv_head_role_recent_current_count)])
                    cmd.extend(["--kv_head_role_recent_memory_tokens", str(args.kv_head_role_recent_memory_tokens)])
                    cmd.extend(["--kv_head_role_recent_mixed_tokens", str(args.kv_head_role_recent_mixed_tokens)])
                    cmd.extend(["--kv_head_role_recent_current_tokens", str(args.kv_head_role_recent_current_tokens)])
                    cmd.extend(["--kv_head_role_recent_window_mode", args.kv_head_role_recent_window_mode])
                    cmd.extend(["--kv_head_role_recent_uniform_budget", str(args.kv_head_role_recent_uniform_budget)])
                    cmd.extend(["--kv_head_role_budget_enable", str(args.kv_head_role_budget_enable)])
                    cmd.extend(["--kv_head_role_memory_budget", str(args.kv_head_role_memory_budget)])
                    cmd.extend(["--kv_head_role_mixed_budget", str(args.kv_head_role_mixed_budget)])
                    cmd.extend(["--kv_head_role_current_budget", str(args.kv_head_role_current_budget)])
                    cmd.extend(["--kv_head_role_budget_auto_mixed", str(args.kv_head_role_budget_auto_mixed)])
                    cmd.extend(["--kv_head_role_history_uniform_memory_ratio", str(args.kv_head_role_history_uniform_memory_ratio)])
                    cmd.extend(["--kv_head_role_history_uniform_mixed_ratio", str(args.kv_head_role_history_uniform_mixed_ratio)])
                    cmd.extend(["--kv_head_role_history_uniform_bins", str(args.kv_head_role_history_uniform_bins)])
                    cmd.extend(["--kv_head_role_last_frame_by_role", str(args.kv_head_role_last_frame_by_role)])
                    cmd.extend(["--kv_head_role_last_frame_ratio_current", str(args.kv_head_role_last_frame_ratio_current)])
                    cmd.extend(["--kv_head_role_last_frame_ratio_memory", str(args.kv_head_role_last_frame_ratio_memory)])
                    cmd.extend(["--kv_head_role_last_frame_ratio_mixed", str(args.kv_head_role_last_frame_ratio_mixed)])
            if args.horizonkv_policy:
                cmd.extend(["--horizonkv_policy", args.horizonkv_policy])
                cmd.extend(["--horizonkv_recent_capacity", str(args.horizonkv_recent_capacity)])
                cmd.extend(["--horizonkv_ema_gamma", str(args.horizonkv_ema_gamma)])
                cmd.extend(["--horizonkv_apply_age_penalty", str(args.horizonkv_apply_age_penalty)])
                cmd.extend(["--horizonkv_temporal_coverage", args.horizonkv_temporal_coverage])
                cmd.extend(["--horizonkv_probe_style", args.horizonkv_probe_style])
                cmd.extend(["--horizonkv_score_blend", args.horizonkv_score_blend])
            if args.timestamp_v8_policy:
                cmd.extend(["--timestamp_v8_policy", args.timestamp_v8_policy])
                cmd.extend([
                    "--timestamp_v8_grace_frames",
                    str(args.timestamp_v8_grace_frames),
                ])
            if args.ragged_frame_trace_path:
                cmd.extend(["--ragged_frame_trace_path", args.ragged_frame_trace_path])
            if args.ragged_readout_trace_path:
                cmd.extend(["--ragged_readout_trace_path", args.ragged_readout_trace_path])
                cmd.extend(["--ragged_readout_trace_last_n", str(args.ragged_readout_trace_last_n)])
            if args.ragged_token_trace_path:
                cmd.extend(["--ragged_token_trace_path", args.ragged_token_trace_path])
            if args.ragged_redundancy_trace_path:
                cmd.extend(["--ragged_redundancy_trace_path", args.ragged_redundancy_trace_path])
                cmd.extend([
                    "--ragged_redundancy_trace_max_tokens",
                    str(args.ragged_redundancy_trace_max_tokens),
                ])
            if args.answer_history_max_age_seconds > 0:
                cmd.extend([
                    "--answer_history_max_age_seconds",
                    str(args.answer_history_max_age_seconds),
                ])
            if args.prospective_lifecycle_trace_path:
                trace_path = args.prospective_lifecycle_trace_path.format(chunk_idx=idx)
                cmd.extend(["--prospective_lifecycle_trace_path", trace_path])
                cmd.extend([
                    "--prospective_lifecycle_trace_last_n",
                    str(args.prospective_lifecycle_trace_last_n),
                    "--prospective_lifecycle_top_fraction",
                    str(args.prospective_lifecycle_top_fraction),
                    "--prospective_lifecycle_visual_queries",
                    str(args.prospective_lifecycle_visual_queries),
                    "--prospective_lifecycle_visual_top_readers",
                    str(args.prospective_lifecycle_visual_top_readers),
                    "--prospective_lifecycle_content_bins",
                    str(args.prospective_lifecycle_content_bins),
                ])
            if args.encode_chunk_size != 16:
                cmd.extend(["--encode_chunk_size", str(args.encode_chunk_size)])
            if args.kv_head_ragged_decode:
                cmd.extend(["--kv_head_ragged_decode", str(args.kv_head_ragged_decode)])
            if args.kv_head_ragged_prefill:
                cmd.extend(["--kv_head_ragged_prefill", str(args.kv_head_ragged_prefill)])
            if args.last_frame_stride != 1.0:
                cmd.extend(["--last_frame_stride", str(args.last_frame_stride)])
            if args.partition_position_frames > 0:
                cmd.extend(["--partition_position_frames", str(args.partition_position_frames)])
            if args.short_term_frames > 0:
                cmd.extend(["--short_term_frames", str(args.short_term_frames)])
            if args.multi_scale:
                cmd.extend(["--multi_scale", str(args.multi_scale)])
            if args.layered_resolution:
                cmd.extend(["--layered_resolution", str(args.layered_resolution)])
            if args.fixed_base_resolution:
                cmd.extend(["--fixed_base_resolution", str(args.fixed_base_resolution)])
            if args.fixed_frame_canvas_size > 0:
                cmd.extend(["--fixed_frame_canvas_size", str(args.fixed_frame_canvas_size)])
                cmd.extend(["--fixed_frame_pad_mode", args.fixed_frame_pad_mode])
                cmd.extend(["--fixed_frame_pad_value", str(args.fixed_frame_pad_value)])
            if args.transient_highres:
                cmd.extend(["--transient_highres", str(args.transient_highres)])
            if args.feature_buffer_frames > 0:
                cmd.extend(["--feature_buffer_frames", str(args.feature_buffer_frames)])
            if args.defer_last_frame_compress:
                cmd.extend(["--defer_last_frame_compress", str(args.defer_last_frame_compress)])
                cmd.extend(["--defer_last_frame_budget_mode", args.defer_last_frame_budget_mode])
                cmd.extend(["--defer_last_frame_keep_ratio", str(args.defer_last_frame_keep_ratio)])
                if args.defer_last_frame_head_budget_threshold > 0:
                    cmd.extend([
                        "--defer_last_frame_head_budget_threshold",
                        str(args.defer_last_frame_head_budget_threshold),
                    ])
            if args.rolekv_ragged:
                cmd.extend(["--rolekv_ragged", str(args.rolekv_ragged)])
                cmd.extend(["--rolekv_head_classes", args.rolekv_head_classes])
                cmd.extend(["--rolekv_ragged_mode", args.rolekv_ragged_mode])
                cmd.extend(["--rolekv_quota_ratio", str(args.rolekv_quota_ratio)])
                cmd.extend(["--rolekv_lambda_memory", str(args.rolekv_lambda_memory)])
                cmd.extend(["--rolekv_lambda_current", str(args.rolekv_lambda_current)])
                cmd.extend(["--rolekv_seed", str(args.rolekv_seed)])
                cmd.extend(["--rolekv_role_min_votes", str(args.rolekv_role_min_votes)])
            if args.answer_head_router:
                cmd.extend(["--answer_head_router", str(args.answer_head_router)])
                cmd.extend(["--answer_head_profile", args.answer_head_profile])
                cmd.extend(["--answer_head_router_mode", args.answer_head_router_mode])
                cmd.extend(["--answer_head_router_strength", str(args.answer_head_router_strength)])
                cmd.extend(["--answer_head_min_gate", str(args.answer_head_min_gate)])
                cmd.extend(["--answer_head_max_gate", str(args.answer_head_max_gate)])
                cmd.extend(["--answer_head_start_layer", str(args.answer_head_start_layer)])
            p = multiprocessing.Process(target=exec, args=(cmd, True, devices[idx]))
            processes.append(p)
            p.start()

        failed_chunks = []
        chunk_files = []
        for idx, p in enumerate(processes):
            p.join()
            chunk_file = os.path.join(save_dir, f"{num_chunks}_{idx}.csv")
            chunk_files.append(chunk_file)
            if p.exitcode != 0 or not os.path.isfile(chunk_file):
                failed_chunks.append(idx)
                print(
                    f"Chunk {idx} failed: exitcode={p.exitcode}, "
                    f"result_exists={os.path.isfile(chunk_file)}"
                )

        if failed_chunks:
            raise RuntimeError(
                f"The following chunks failed: {failed_chunks}. "
                "Chunk files were preserved; evaluation was not run."
            )

        merge_chunk_results(chunk_files, results_path)

    if not os.path.isfile(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")
    fmt = {
        "python": shlex.quote(sys.executable),
        "project_root": shlex.quote(PROJECT_ROOT),
        "results_path": shlex.quote(results_path),
        "save_dir": shlex.quote(save_dir),
        "anno_path": shlex.quote(anno_path),
    }
    for cmd_template in config["eval_cmds"]:
        exec(cmd_template.format(**fmt))


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava_ov_7b", choices=['llava_ov_0.5b', 'llava_ov_7b', 'llava_ov_72b', 'qwen2.5_vl_3b', 'qwen2.5_vl_7b', 'qwen2.5_vl_32b'])
    parser.add_argument("--model_path", type=str, default=None,
                        help="Override the model directory or Hugging Face model ID.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Exact result directory; overrides the generated results path.")
    parser.add_argument("--dataset", type=str, required=True, choices=list(BENCHMARK_CONFIGS.keys()))
    parser.add_argument("--anno_path", type=str, default=None,
                        help="Override the benchmark annotation path.")
    parser.add_argument("--output_tag", type=str, default=None,
                        help="Append a tag to the result directory to isolate this run.")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--only_eval", action="store_true")
    parser.add_argument("--sample_fps", type=float, default=1)
    parser.add_argument("--debug", type=str, default='false')
    parser.add_argument("--kv_size", type=int)
    parser.add_argument("--compress_mode", type=str, default="hermes", choices=["hermes", "streamingvlm"])
    parser.add_argument("--head_budget_scores", type=str, default=None,
                        help="Enable HERMES per-head voting budgets. Use 'pseudo', 'sparsemm', or a .npz/.json score file.")
    parser.add_argument("--layer_budget_scores", type=str, default=None,
                        help="Enable HERMES per-layer budgets. Use 'pseudo', 'sparsemm', or a .npz/.json score file.")
    parser.add_argument("--layer_budget_variable", type=str2bool, nargs='?', const=True, default=False,
                        help="Use real per-layer KV cache lengths with per-layer Qwen attention masks. Experimental.")
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
                        help="Optional JSONL path for immutable visual-token IDs and per-token readout.")
    parser.add_argument("--ragged_redundancy_trace_path", type=str, default=None,
                        help="Optional JSONL path for selected-history content-redundancy diagnostics.")
    parser.add_argument("--ragged_redundancy_trace_max_tokens", type=int, default=128,
                        help="Maximum selected history tokens sampled per KV head.")
    parser.add_argument("--answer_history_max_age_seconds", type=float, default=0.0,
                        help="Diagnostic only: temporarily hide older visual KV while answering.")
    parser.add_argument("--prospective_lifecycle_trace_path", type=str, default=None,
                        help="Optional JSONL path; use {chunk_idx} for multi-process runs.")
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
                        help="CSV with layer,kv_head,long_score,short_score for role-specific ragged selection.")
    parser.add_argument("--kv_head_policy_memory_quantile", type=float, default=0.75)
    parser.add_argument("--kv_head_policy_current_quantile", type=float, default=0.75)
    parser.add_argument("--kv_head_policy_memory_bonus", type=float, default=0.1)
    parser.add_argument("--kv_head_policy_current_bonus", type=float, default=0.1)
    parser.add_argument("--kv_head_recency_scores", type=str, default=None,
                        help="CSV temporal profile used for per-KV-head HERMES recency alpha.")
    parser.add_argument("--kv_head_recency_metric", type=str, default="b_log_per_token_ratio")
    parser.add_argument("--kv_head_recency_delta", type=float, default=0.2)
    parser.add_argument("--kv_head_recency_tau", type=float, default=1.0)
    parser.add_argument("--kv_head_recency_alpha_min", type=float, default=0.0)
    parser.add_argument("--kv_head_recency_alpha_max", type=float, default=1.0)
    parser.add_argument("--kv_head_recency_k_delta", type=float, default=0.0)
    parser.add_argument("--kv_head_recency_k_min", type=float, default=0.0)
    parser.add_argument("--kv_head_recency_k_max", type=float, default=20.0)
    parser.add_argument("--kv_head_layerwise_last_frame", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Enable layerwise last-frame protection for ragged KV-head selection.")
    parser.add_argument("--kv_head_layerwise_shallow_end", type=int, default=10,
                        help="Exclusive shallow layer end, e.g. 10 means layers 0-9.")
    parser.add_argument("--kv_head_layerwise_mid_end", type=int, default=20,
                        help="Exclusive mid layer end; deep layers start here.")
    parser.add_argument("--kv_head_layerwise_shallow_ratio", type=float, default=1.0,
                        help="Latest-frame keep ratio for shallow layers.")
    parser.add_argument("--kv_head_layerwise_mid_ratio", type=float, default=0.5,
                        help="Latest-frame keep ratio for middle layers.")
    parser.add_argument("--kv_head_layerwise_deep_ratio", type=float, default=0.0,
                        help="Latest-frame keep ratio for deep layers.")
    parser.add_argument("--kv_head_role_recent_scores", type=str, default=None,
                        help="CSV score file used to split KV heads into memory/mixed/current roles for recent retention.")
    parser.add_argument("--kv_head_role_dual_scores", type=str, default=None,
                        help="CSV with independent memory/current scores, such as long_score and short_score. "
                             "Mutually exclusive with --kv_head_role_recent_scores.")
    parser.add_argument("--kv_head_role_recent_memory_quantile", type=float, default=0.75,
                        help="Heads with score >= this quantile are memory heads.")
    parser.add_argument("--kv_head_role_recent_current_quantile", type=float, default=0.25,
                        help="Heads with score <= this quantile are current heads.")
    parser.add_argument("--kv_head_role_recent_memory_count", type=int, default=-1,
                        help="If >=0, use exactly this many top-scoring KV heads as memory heads.")
    parser.add_argument("--kv_head_role_recent_current_count", type=int, default=-1,
                        help="If >=0, use exactly this many bottom-scoring KV heads as current heads.")
    parser.add_argument("--kv_head_role_recent_memory_tokens", type=int, default=0,
                        help="Recent visual tokens protected for memory heads.")
    parser.add_argument("--kv_head_role_recent_mixed_tokens", type=int, default=2000,
                        help="Recent visual tokens protected for mixed heads.")
    parser.add_argument("--kv_head_role_recent_current_tokens", type=int, default=4000,
                        help="Recent visual tokens protected for current heads.")
    parser.add_argument("--kv_head_role_recent_window_mode", type=str, default="override",
                        choices=["override", "max"],
                        help="override replaces sparsemm recent window; max keeps the larger window.")
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
                        help="Fraction of each memory head's history budget selected uniformly across time.")
    parser.add_argument("--kv_head_role_history_uniform_mixed_ratio", type=float, default=0.0,
                        help="Fraction of each mixed head's history budget selected uniformly across time.")
    parser.add_argument("--kv_head_role_history_uniform_bins", type=int, default=32,
                        help="Maximum temporal bins for role history-uniform selection.")
    parser.add_argument("--kv_head_role_last_frame_by_role", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Apply last-frame protection per head role instead of per layer.")
    parser.add_argument("--kv_head_role_last_frame_ratio_current", type=float, default=1.0,
                        help="Last-frame retention ratio for 'current' role heads.")
    parser.add_argument("--kv_head_role_last_frame_ratio_memory", type=float, default=0.0,
                        help="Last-frame retention ratio for 'memory' role heads.")
    parser.add_argument("--kv_head_role_last_frame_ratio_mixed", type=float, default=0.0,
                        help="Last-frame retention ratio for 'mixed' role heads.")
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
    parser.add_argument("--answer_head_router", type=str2bool, nargs='?', const=True, default=False,
                        help="Gate query-head outputs only while answering a real question.")
    parser.add_argument("--answer_head_profile", type=str, default=None,
                        help="CSV with layer, kv_head, far_share, and recent_share columns.")
    parser.add_argument("--answer_head_router_mode", type=str, default="question",
                        choices=["question", "inverted", "static_memory", "baseline"])
    parser.add_argument("--answer_head_router_strength", type=float, default=0.2)
    parser.add_argument("--answer_head_min_gate", type=float, default=0.8)
    parser.add_argument("--answer_head_max_gate", type=float, default=1.2)
    parser.add_argument("--answer_head_start_layer", type=int, default=8)
    parser.add_argument("--last_frame_stride", type=float, default=1.0,
                        help="Stride for last-frame spatial resolution. >1.0 gives the last frame "
                             "before a question higher resolution (more visual tokens per frame).")
    parser.add_argument("--partition_position_frames", type=int, default=0,
                        help="Number of recent frames to protect with high position IDs during "
                             "position compaction. 0 = disabled (global uniform compaction).")
    parser.add_argument("--short_term_frames", type=int, default=0,
                        help="Number of recent frames to fully protect from per-head top-k eviction. "
                             "0 (default) = disabled.")
    parser.add_argument("--multi_scale", type=str2bool, nargs='?', const=True, default=False,
                        help="Enable multi-scale encoding for the last frame.")
    parser.add_argument("--layered_resolution", type=str2bool, nargs='?', const=True, default=False,
                        help="Layered-resolution: deep layers get normal-res only.")
    parser.add_argument("--fixed_base_resolution", type=str2bool, nargs='?', const=True, default=False,
                        help="Lock normal-frame processor to fixed min=max pixels (DSCache-style).")
    parser.add_argument("--fixed_frame_canvas_size", type=int, default=0,
                        help="If >0, letterbox/resize every video frame to a fixed square canvas before Qwen processing.")
    parser.add_argument("--fixed_frame_pad_mode", type=str, default="letterbox",
                        choices=["letterbox", "resize"],
                        help="Frame normalization for --fixed_frame_canvas_size.")
    parser.add_argument("--fixed_frame_pad_value", type=float, default=0.0,
                        help="Padding pixel value for letterbox fixed-frame canvas.")
    parser.add_argument("--transient_highres", type=str2bool, nargs='?', const=True, default=False,
                        help="High-res last frame only for current question, replaced after answer.")
    parser.add_argument("--feature_buffer_frames", type=int, default=0,
                        help="DSCache-style feature buffer: hold last N frames as raw pixels, "
                             "encode as independent Instant Cache at query time. 0 = disabled.")
    parser.add_argument("--defer_last_frame_compress", type=str2bool, nargs='?', const=True, default=False,
                        help="Encode final frame before answering and defer its compression until after answering.")
    parser.add_argument("--defer_last_frame_budget_mode", type=str, default="overflow",
                        choices=["overflow", "strict"],
                        help="'overflow' allows temporary pre-answer KV overflow; 'strict' reserves estimated final-frame tokens.")
    parser.add_argument("--defer_last_frame_keep_ratio", type=float, default=1.0,
                        help="Final-frame fraction retained before answering in strict defer mode. "
                             "1.0 is legacy deferlast; 0.0 is normal non-deferred compression.")
    parser.add_argument("--defer_last_frame_head_budget_threshold", type=int, default=0,
                        help="Apply strict latest-frame reservation only to heads above this budget.")
    parser.add_argument("--devices", type=str, default=None, help="Comma-separated GPU IDs, e.g. '0,1,2'. Overrides auto-assignment.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

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
            "Partial defer is not supported with --kv_head_role_budget_enable."
        )
    if partial_defer and args.kv_head_layerwise_last_frame:
        raise ValueError(
            "Partial defer and --kv_head_layerwise_last_frame are mutually exclusive."
        )
    if args.defer_last_frame_head_budget_threshold > 0 and not partial_defer:
        raise ValueError(
            "--defer_last_frame_head_budget_threshold requires strict partial defer."
        )

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

    if args.devices is not None:
        args.devices = [device.strip() for device in args.devices.split(",")]
        if not all(args.devices):
            raise ValueError("--devices must contain nonempty GPU IDs.")
    else:
        args.devices = []

    if args.dataset in BENCHMARK_CONFIGS:
        print(f'Execute {args.dataset} evaluation')
        run_eval(args, BENCHMARK_CONFIGS[args.dataset])


if __name__ == "__main__":
    main()
