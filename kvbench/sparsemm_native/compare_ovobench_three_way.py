"""Validate and compare matched OVO questions for native, DP, and dual-bank DP.

Run separately for HERMES nine-task and official forward-three protocols.  The
subtotal is question-weighted coverage, never a mixed-protocol official score.
"""

import argparse
import collections
import hashlib
import json
from pathlib import Path

from sparsemm_native.run_ovobench import (
    forward_queries, hermes_correct, hermes_queries, official_correct, queries,
)


ARMS = ("native", "dp", "dp_dual")
IDENTITY = ("query_id", "source_id", "probe_id", "task", "group", "timestamp", "clip_end", "target")
PROTOCOL_FIELDS = (
    "source_sha256", "model_config_sha256", "budget", "fps", "dtype", "chunk_size",
    "max_new_tokens", "video_max_pixels", "score_mode", "video_mode",
)


def _digest(items):
    return hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()


def _json(path):
    if not path.is_file():
        raise ValueError(f"Missing required file: {path}")
    return json.loads(path.read_text())


def _read_rows(path):
    if not path.exists():
        return []
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed result at {path}:{number}") from error
    return rows


def _matrix(value, name, minimum=0):
    if (not isinstance(value, list) or len(value) != 28
            or any(not isinstance(row, list) or len(row) != 4 for row in value)
            or any(type(n) is not int or n < minimum for row in value for n in row)):
        raise ValueError(f"{name} must be a 28 x 4 integer table with values >= {minimum}")
    return value


def _validate_slots(allocation, label):
    if not isinstance(allocation, dict):
        raise ValueError(f"{label}: missing allocation metadata")
    capacity = _matrix(allocation.get("head_budgets"), f"{label} head_budgets", 32)
    if (sum(map(sum, capacity)) != 56000 or allocation.get("allocated_slots") != 56000
            or allocation.get("target_slots") != 56000):
        raise ValueError(f"{label}: expected exactly 56000 allocated/target visual slots")
    if allocation.get("window_size") != 32:
        raise ValueError(f"{label}: protect/attention window must be 32")
    return capacity


def _native_allocation(protocol, directory):
    # Completed OVO protocols store native allocation in the trace, while the
    # DP allocation is in protocol.json. Read only through the first prefill.
    if isinstance(protocol.get("allocation"), dict):
        return protocol["allocation"]
    trace = directory / "cache_trace.jsonl"
    if trace.is_file():
        with trace.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                event = json.loads(line)
                if event.get("event") == "video_prefill":
                    layers = event.get("layers", [])
                    if layers and isinstance(layers[0].get("allocation"), dict):
                        return layers[0]["allocation"]
                    break
    raise ValueError("Native baseline allocation is missing from protocol/trace")


def _selected_queries(protocol, expected):
    count = protocol.get("query_count")
    if type(count) is not int or not 0 < count <= len(expected):
        raise ValueError("Dual protocol has an invalid query_count")
    ids = protocol.get("query_ids")
    if ids is None:
        if count != len(expected):
            raise ValueError("Subset dual protocol must record its complete query_ids")
        selected = expected
    else:
        if (not isinstance(ids, list) or any(not isinstance(i, str) for i in ids)
                or len(ids) != len(set(ids)) or len(ids) != count):
            raise ValueError("Dual protocol has invalid or duplicate query_ids")
        requested = set(ids)
        selected = [query for query in expected if query["query_id"] in requested]
        if [query["query_id"] for query in selected] != ids:
            raise ValueError("Dual query_ids are not an ordered subset of the expected query list")
    if protocol.get("query_list_sha256") != _digest(selected):
        raise ValueError("Dual selected query identity/prompt hash differs from current query builders")
    if protocol.get("full_query_list_sha256") != _digest(expected):
        raise ValueError("Dual full expected query identity/prompt hash differs from baseline")
    return selected


def _validate_rows(rows, expected, label, dataset, require_complete=False):
    if require_complete and len(rows) != len(expected):
        raise ValueError(f"{label}: baseline must contain the full expected query list")
    seen = set()
    for index, row in enumerate(rows):
        query_id = row.get("query_id")
        if query_id in seen:
            raise ValueError(f"{label}: duplicate query_id {query_id}")
        seen.add(query_id)
        if index >= len(expected):
            raise ValueError(f"{label}: too many result rows")
        query = expected[index]
        for key in IDENTITY:
            if key not in row or row[key] != query[key]:
                raise ValueError(f"{label}: query identity mismatch at {query_id}: {key}")
        if type(row.get("frames")) is not int or row["frames"] < 1:
            raise ValueError(f"{label}: invalid sampled-frame count at {query_id}")
        if type(row.get("correct")) is not bool or not isinstance(row.get("response"), str):
            raise ValueError(f"{label}: invalid answer/correctness at {query_id}")
        correct = (hermes_correct(row["response"], row["target"]) if dataset == "hermes9" else
                   official_correct(row["task"], row["response"], row["target"]))
        if row["correct"] != correct:
            raise ValueError(f"{label}: recorded correctness disagrees with pinned scorer at {query_id}")
    return {row["query_id"]: row for row in rows}


def _metrics(matched):
    total = len(matched)
    counts = {arm: sum(int(row[arm]["correct"]) for row in matched) for arm in ARMS}
    result = {"matched_queries": total}
    for arm in ARMS:
        result[f"{arm}_correct"] = counts[arm]
        result[f"{arm}_accuracy"] = counts[arm] / total if total else None
    for control in ("native", "dp"):
        result[f"dp_dual_minus_{control}_percentage_points"] = (
            100 * (counts["dp_dual"] - counts[control]) / total if total else None)
        result[f"dual_only_vs_{control}"] = sum(
            row["dp_dual"]["correct"] and not row[control]["correct"] for row in matched)
        result[f"{control}_only_vs_dual"] = sum(
            row[control]["correct"] and not row["dp_dual"]["correct"] for row in matched)
    result["dp_minus_native_percentage_points"] = (
        100 * (counts["dp"] - counts["native"]) / total if total else None)
    return result


def compare(baseline_root, dual_root):
    baseline_root, dual_root = Path(baseline_root).resolve(), Path(dual_root).resolve()
    directories = {"native": baseline_root / "native", "dp": baseline_root / "dp",
                   "dp_dual": dual_root / "dp_dual"}
    protocols = {arm: _json(directory / "protocol.json") for arm, directory in directories.items()}
    reference = protocols["dp"]
    dataset = reference.get("dataset", "official_full")
    if dataset not in ("hermes9", "forward3", "official_full"):
        raise ValueError(f"Unsupported dataset protocol: {dataset}")
    expected = (hermes_queries() if dataset == "hermes9" else
                forward_queries() if dataset == "forward3" else queries())
    full_hash = _digest(expected)
    for arm, protocol in protocols.items():
        if protocol.get("arm") != arm:
            raise ValueError(f"{arm}: protocol arm disagrees with output directory")
        if protocol.get("dataset", "official_full") != dataset:
            raise ValueError(f"{arm}: dataset differs from baseline; mixed protocols cannot be pooled")
        for field in PROTOCOL_FIELDS:
            if field not in protocol or field not in reference or protocol[field] != reference[field]:
                raise ValueError(f"{arm}: protocol mismatch: {field}")
        if (not protocol.get("model_path") or not reference.get("model_path")
                or Path(protocol["model_path"]).resolve() != Path(reference["model_path"]).resolve()):
            raise ValueError(f"{arm}: protocol mismatch: model_path")
        if arm != "dp_dual" and (protocol.get("query_count") != len(expected)
                                 or protocol.get("query_list_sha256") != full_hash):
            raise ValueError(f"{arm}: full expected query identity/prompt hash differs from current builders")
    if reference["budget"] != 500:
        raise ValueError("This comparison requires the completed K=500 baseline")
    dual_protocol = protocols["dp_dual"]
    dual_contract = {"cache_policy": "two_bank", "protect_window": 32,
                     "attention_observation_window": 32,
                     "protect_window_included_in_recent": True,
                     "debug_explain": False, "decoding": "greedy", "repetition_penalty": 1.1}
    for field, value in dual_contract.items():
        if field not in dual_protocol or dual_protocol[field] != value:
            raise ValueError(f"Dual DP protocol mismatch: {field}")
    if dual_protocol.get("reference_dp_budget_sha256") != reference.get("dp_budget_sha256"):
        raise ValueError("Dual DP reference budget hash differs from baseline capacity table")
    selected = _selected_queries(dual_protocol, expected)
    baseline_allocation = reference.get("dp_allocation")
    capacities = _validate_slots(baseline_allocation, "DP baseline")
    if baseline_allocation.get("dp_bank_split_used_online") is not False:
        raise ValueError("DP baseline must use capacity-only allocation")
    _validate_slots(_native_allocation(protocols["native"], directories["native"]), "Native baseline")
    dual_allocation = protocols["dp_dual"].get("dp_allocation")
    dual_capacities = _validate_slots(dual_allocation, "Dual DP")
    if dual_capacities != capacities:
        raise ValueError("Dual DP changes baseline per-head capacity_n")
    if (dual_allocation.get("dp_bank_split_used_online") is not True
            or dual_allocation.get("protect_window_included_in_recent") is not True
            or dual_allocation.get("short_cache_gate") != "legacy_global"):
        raise ValueError("Primary dual DP must use online splits, protect32 inside recent, and the baseline legacy_global warmup gate")
    recent = _matrix(dual_allocation.get("recent_budgets"), "Dual DP recent_budgets", 32)
    history = _matrix(dual_allocation.get("history_budgets"), "Dual DP history_budgets")
    if any(r + h != n for rr, hh, nn in zip(recent, history, capacities)
           for r, h, n in zip(rr, hh, nn)):
        raise ValueError("Dual recent/history splits disagree with fixed per-head capacity_n")
    rows = {arm: _read_rows(directory / "queries.jsonl") for arm, directory in directories.items()}
    indexed = {arm: _validate_rows(values, selected if arm == "dp_dual" else expected,
                                   arm, dataset, require_complete=arm != "dp_dual")
               for arm, values in rows.items()}
    matched = []
    for dual_row in rows["dp_dual"]:
        query_id = dual_row["query_id"]
        item = {arm: indexed[arm][query_id] for arm in ARMS}
        if len({row["frames"] for row in item.values()}) != 1:
            raise ValueError(f"Sampled frames differ across arms at {query_id}")
        matched.append(item)
    by_task, by_group = collections.defaultdict(list), collections.defaultdict(list)
    for row in matched:
        by_task[row["dp_dual"]["task"]].append(row)
        by_group[row["dp_dual"]["group"]].append(row)
    task_expected = collections.Counter(query["task"] for query in selected)
    group_expected = collections.Counter(query["group"] for query in selected)
    return {
        "dataset": dataset, "baseline_root": str(baseline_root), "dual_root": str(dual_root),
        "full_expected_queries": len(expected), "dual_selected_queries": len(selected),
        "completed_queries": {arm: len(values) for arm, values in rows.items()},
        "dual_complete": len(rows["dp_dual"]) == len(selected),
        "full_dataset_complete": len(rows["dp_dual"]) == len(expected),
        "full_query_list_sha256": full_hash,
        "budget": 500, "allocated_visual_slots": 56000,
        "recent_slots": sum(map(sum, recent)), "history_slots": sum(map(sum, history)),
        "short_cache_gate": "legacy_global",
        "warmup_note": "The global short-cache warmup matches both completed controls; per-head bank quotas apply once that gate opens.",
        "tasks": {task: {"selected_queries": task_expected[task], **_metrics(by_task[task])}
                  for task in sorted(task_expected)},
        "groups": {group: {"selected_queries": group_expected[group], **_metrics(by_group[group])}
                   for group in sorted(group_expected)},
        "subtotal": _metrics(matched),
        "score_note": "Matched-question weighted subtotal; do not pool different prompt/scoring protocols into an official OVO total.",
        "query_ids": [row["dp_dual"]["query_id"] for row in matched],
    }


def markdown(report):
    progress = report["completed_queries"]
    lines = [f"# OVO-Bench three-way comparison: {report['dataset']}", "",
             f"K=500; 56,000 visual KV slots. Dual recent/history slots: "
             f"{report['recent_slots']:,} / {report['history_slots']:,}.", "",
             f"Progress: native {progress['native']}/{report['full_expected_queries']}; "
             f"DP capacity-only {progress['dp']}/{report['full_expected_queries']}; "
             f"dual DP {progress['dp_dual']}/{report['dual_selected_queries']} selected "
             f"({report['full_expected_queries']} full dataset queries).", "",
             "All accuracy entries below use the same matched questions.", "",
             report["warmup_note"], "",
             "| Scope | Matched | SparseMM correct / accuracy | DP capacity-only correct / accuracy | Dual DP correct / accuracy | Dual − SparseMM | Dual − DP | Dual-only / SparseMM-only | Dual-only / DP-only |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def row(label, value):
        def accuracy(arm):
            v = value[f"{arm}_accuracy"]
            return f"{value[f'{arm}_correct']} / {100 * v:.2f}%" if v is not None else "0 / —"
        def difference(control):
            v = value[f"dp_dual_minus_{control}_percentage_points"]
            return f"{v:+.2f} pp" if v is not None else "—"
        return (f"| {label} | {value['matched_queries']} | {accuracy('native')} | "
                f"{accuracy('dp')} | {accuracy('dp_dual')} | {difference('native')} | "
                f"{difference('dp')} | {value['dual_only_vs_native']} / {value['native_only_vs_dual']} | "
                f"{value['dual_only_vs_dp']} / {value['dp_only_vs_dual']} |")
    lines.extend(row(task, value) for task, value in report["tasks"].items())
    lines.extend(row(f"{group} subtotal", value) for group, value in report["groups"].items())
    lines.extend([row("**Matched weighted subtotal**", report["subtotal"]), "", report["score_note"], ""])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--dual-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Output directory, or .json/.md file stem; writes both formats")
    args = parser.parse_args(argv)
    try:
        result = compare(args.baseline_root, args.dual_root)
    except (ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    if args.output:
        if args.output.suffix in (".json", ".md"):
            json_path, markdown_path = args.output.with_suffix(".json"), args.output.with_suffix(".md")
        else:
            json_path, markdown_path = args.output / "summary.json", args.output / "summary.md"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        markdown_path.write_text(markdown(result))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
