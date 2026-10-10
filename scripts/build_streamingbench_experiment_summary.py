#!/usr/bin/env python3
"""Build a grouped inventory of Qwen StreamingBench evaluation results."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/qwen2.5_vl_7b/streamingbench"
OUTPUT = ROOT / "reports/streamingbench_qwen_experiment_results_20260920.md"


@dataclass(frozen=True)
class Record:
    name: str
    path: Path
    samples: int
    accuracy: float
    fps: str
    budget: str
    input_mode: str
    scope: str
    status: str
    group: str


GROUP_ORDER = [
    "LU-KV",
    "HERMES / Uniform baselines",
    "V8 head-budget",
    "Specialist budget / head routing",
    "Temporal-zone / Timestamp policies",
    "Temporal utility / survival / residual",
    "Lifecycle / annealing",
    "RoleRecent / RoleDual / KV policy",
    "Defer-last / LWLF / stride",
    "HeadAlpha",
    "HorizonKV",
    "Other head-budget scores",
    "StreamingVLM",
    "Diagnostics / other",
]


def classify(name: str) -> str:
    s = name.lower()
    if "lu_" in s or "lu-" in s or "lu-kv" in s:
        return "LU-KV"
    if "horizonkv" in s:
        return "HorizonKV"
    if "headalpha" in s:
        return "HeadAlpha"
    if any(x in s for x in ("rolerecent", "roledual", "kvpolicy", "memsplit")):
        return "RoleRecent / RoleDual / KV policy"
    if any(x in s for x in ("lwlf", "deferlast", "laststride", "stf4", "-fbr")):
        return "Defer-last / LWLF / stride"
    if any(x in s for x in ("annealed", "prospective_lifecycle", "prospective-lifecycle")):
        return "Lifecycle / annealing"
    if any(x in s for x in ("survival", "residual", "relative_head", "relative_shuffled", "temporal-utility", "temporal_utility", "e0_content", "e1_global", "e2_head", "e3_full", "e4_shuffled")):
        return "Temporal utility / survival / residual"
    if any(x in s for x in ("temporal-zone", "matched_profiles", "matched_raw", "matched_selective", "timestamp-v8", "timestampv8")):
        return "Temporal-zone / Timestamp policies"
    if any(x in s for x in ("specialist", "he-route", "qhead", "answer-router")):
        return "Specialist budget / head routing"
    if "v8_blend3" in s or "v8-native" in s or "v8_budget" in s:
        return "V8 head-budget"
    if "uniform" in s:
        return "HERMES / Uniform baselines"
    if "kvheadbudget" in s or "layerbudget" in s:
        return "Other head-budget scores"
    if "streamingvlm" in s:
        return "StreamingVLM"
    if re.fullmatch(r"fps[\d.]+-kv\d+-hermes(?:-half|-fb4)?", s):
        return "HERMES / Uniform baselines"
    if "hermes" in s and not any(x in s for x in ("smoke", "trace", "readout", "diagnostic", "content-sim")):
        return "HERMES / Uniform baselines"
    return "Diagnostics / other"


def infer_scope(name: str, samples: int) -> str:
    s = name.lower()
    tags = [
        ("remaining248", "remaining-248"),
        ("videos51-250", "videos 51–250"),
        ("videos151-250", "videos 151–250"),
        ("151-250", "videos 151–250"),
        ("101-200", "videos 101–200"),
        ("101-150", "videos 101–150"),
        ("51-150", "videos 51–150"),
        ("51-100", "videos 51–100"),
        ("36-50", "videos 36–50"),
        ("21-50", "videos 21–50"),
        ("21-35", "videos 21–35"),
        ("first250", "first 250 videos"),
        ("first150", "first 150 videos"),
        ("first50", "first 50 videos"),
        ("first20", "first 20 videos"),
        ("half", "250-video half"),
        ("full", "full / tagged full"),
        ("eval50", "50-video evaluation"),
        ("pilot12", "12-video pilot"),
        ("representative12", "12-video pilot"),
        ("10v", "10-video pilot"),
        ("7v", "7-video pilot"),
        ("6v", "6-video pilot"),
        ("smoke", "smoke"),
        ("trace3", "3-question diagnostic"),
        ("readout3", "3-question diagnostic"),
    ]
    for token, label in tags:
        if token in s:
            return label
    if samples == 2495:
        return "full 500 videos"
    if samples == 1255:
        return "250-video half"
    if samples == 754:
        return "first 150 videos"
    if samples == 251:
        return "first 50 videos"
    return f"subset ({samples} questions)"


def expected_samples(scope: str) -> int | None:
    return {
        "full 500 videos": 2495,
        "250-video half": 1255,
        "first 250 videos": 1255,
        "first 150 videos": 754,
        "first 50 videos": 251,
        "videos 51–150": 503,
        "videos 101–150": 253,
        "videos 51–100": 250,
        "videos 151–250": 501,
        "remaining-248": 1240,
        "videos 51–250": 1004,
    }.get(scope)


def infer_status(name: str, samples: int, scope: str) -> str:
    expected = expected_samples(scope)
    if expected is not None and samples < expected:
        return "incomplete"
    s = name.lower()
    if "smoke" in s or samples < 20:
        return "smoke"
    if samples < 200:
        return "pilot"
    return "complete"


def parse_report(path: Path) -> Record | None:
    text = path.read_text(errors="replace")
    match = re.search(r"#Samples:\s*(\d+).*?Average qa_acc:\s*([0-9.]+)", text, re.S)
    if not match:
        return None
    name = path.parent.name
    fps_match = re.search(r"(?:^|-)fps([\d.]+)", name)
    kv_match = re.search(r"(?:^|-)kv(\d+)", name)
    ffc_match = re.search(r"ffc(\d+)", name)
    samples, accuracy = int(match.group(1)), float(match.group(2))
    scope = infer_scope(name, samples)
    return Record(
        name=name,
        path=path,
        samples=samples,
        accuracy=accuracy,
        fps=fps_match.group(1) if fps_match else "—",
        budget=kv_match.group(1) if kv_match else "—",
        input_mode=f"fixed {ffc_match.group(1)}" if ffc_match else "native / unspecified",
        scope=scope,
        status=infer_status(name, samples, scope),
        group=classify(name),
    )


def markdown_link(record: Record) -> str:
    relative = record.path.relative_to(OUTPUT.parent.parent)
    return f"[result](../{relative.as_posix()})"


def compact_config(record: Record) -> str:
    name = record.name
    prefix = re.match(r"fps[\d.]+-kv\d+-", name)
    if prefix:
        name = name[prefix.end():]
    return name.replace("|", "\\|")


def find_one(records: list[Record], *needles: str) -> Record | None:
    matches = [r for r in records if all(x in r.name for x in needles)]
    if not matches:
        return None
    status_rank = {"complete": 3, "pilot": 2, "smoke": 1, "incomplete": 0}
    return max(matches, key=lambda r: (status_rank[r.status], r.samples, r.path.stat().st_mtime))


def core_tables(records: list[Record]) -> list[str]:
    lines = [
        "## 核心可比结果",
        "",
        "以下表格只组合协议和数据子集能够明确对齐的实验。完整清单在后文。",
        "",
        "### Native 分辨率预算扫描",
        "",
        "| K | 方法 | 样本数 | 准确率 | 状态 |",
        "|---:|---|---:|---:|---|",
    ]
    sweep = [
        (250, "Uniform", ("kv250", "uniform-native-k250-first150")),
        (250, "LU-KV", ("kv250", "lu-raw-blend05-clip075-150-native-first150-k250")),
        (500, "Uniform", ("kv500", "uniform-native-k500-first150")),
        (500, "V8", ("kv500", "v8-native-k500-first150")),
        (500, "LU-KV", ("kv500", "lu-raw-blend05-clip075-150-native-first150-k500")),
        (800, "Uniform", ("kv800", "uniform-native-k800-first150")),
        (800, "LU-KV", ("kv800", "lu-raw-blend05-clip075-150-native-first150-k800")),
        (1000, "Uniform / shards", ("kv1000", "uniform-native-k1000-first50")),
        (1000, "V8 / shards", ("kv1000", "v8-native-k1000-first50")),
        (1000, "LU-KV / shards", ("kv1000", "lu-raw-blend05-clip075-150-native-first50-k1000")),
        (2000, "Uniform", ("kv2000", "uniform-native-k2000-first150")),
        (2000, "V8", ("kv2000", "v8-native-k2000-first150")),
        (2000, "LU-KV / first50", ("kv2000", "lu-raw-k2000", "first50")),
    ]
    for budget, method, needles in sweep:
        record = find_one(records, *needles)
        if record:
            lines.append(
                f"| {budget} | {method} | {record.samples} | {record.accuracy:.2f}% | "
                f"{record.status}; {markdown_link(record)} |"
            )
    rerun_status_path = ROOT / "logs/native_k800_first150/hermes_rerun.status"
    rerun_status = "未找到状态文件"
    if rerun_status_path.exists():
        status_lines = [line.strip() for line in rerun_status_path.read_text().splitlines() if line.strip()]
        if status_lines:
            rerun_status = status_lines[-1]
    lines.extend([
        "",
        f"> K=800 Uniform 的已有报告只有 5 题，是中断产物，不能与 754 题 LU-KV 结果比较。当前 `hermes_rerun` 状态：`{rerun_status}`。K=1000 的 first-150/full-250 结果分散在多个 shard 中，完整 shard 均列在 LU-KV 与 Uniform 分组。",
        "",
        "### 主要大规模评测中的最高结果",
        "",
        "同一行只按样本数筛选，输入分辨率和方法协议仍需结合配置名称判断。",
        "",
        "| 样本数 | 最佳准确率 | 方法族 | 配置 | 结果 |",
        "|---:|---:|---|---|---|",
    ])
    for n in (2495, 1255, 754, 503, 501, 253, 251, 250):
        candidates = [r for r in records if r.samples == n and r.status == "complete"]
        if not candidates:
            continue
        best = max(candidates, key=lambda r: r.accuracy)
        lines.append(
            f"| {n} | {best.accuracy:.2f}% | {best.group} | `{compact_config(best)}` | {markdown_link(best)} |"
        )
    return lines


def build(records: list[Record], skipped: list[Path]) -> str:
    counts = Counter(r.status for r in records)
    groups: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        groups[record.group].append(record)
    lines = [
        "# Qwen2.5-VL-7B StreamingBench 实验结果总表",
        "",
        f"> 快照时间：{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"> 扫描目录：`{RESULT_ROOT.relative_to(ROOT)}`  ",
        f"> 共解析 {len(records)} 份 `eval_results.txt`：完整 {counts['complete']}，pilot {counts['pilot']}，smoke {counts['smoke']}，不完整 {counts['incomplete']}。",
        "",
        "## 阅读说明",
        "",
        "- `complete` 只表示配置名称所声明的评测范围已经产生相符数量的题目，不代表不同实验之间可以直接比较。",
        "- `incomplete` 表示目录名声明了 first-50、first-150、half 等范围，但当前题数不足；这类结果不应作为正式结论。",
        "- `native / unspecified` 表示目录名没有 `ffc*` 固定画布标记；历史实验中部分未标记配置可能还需要结合启动脚本确认。",
        "- 原始报告链接保留任务级准确率。此表汇总整体准确率，避免 210 个实验 × 10 个任务形成无法阅读的超宽表格。",
        "- 样本数不同通常对应不同视频子集；排序仅在每个方法族内用于查找配置，不表示全局排行榜。",
        "",
    ]
    lines.extend(core_tables(records))
    lines.extend([
        "",
        "## 全部实验（按方法族分组）",
        "",
    ])
    for group in GROUP_ORDER:
        items = groups.get(group, [])
        if not items:
            continue
        items.sort(key=lambda r: (int(r.budget) if r.budget.isdigit() else 10**9, -r.samples, r.name))
        lines.extend([
            f"### {group}（{len(items)}）",
            "",
            "| K | FPS | 输入 | 范围 | N | Acc. | 状态 | 配置 | 原始报告 |",
            "|---:|---:|---|---|---:|---:|---|---|---|",
        ])
        for r in items:
            lines.append(
                f"| {r.budget} | {r.fps} | {r.input_mode} | {r.scope} | {r.samples} | "
                f"{r.accuracy:.2f}% | {r.status} | `{compact_config(r)}` | {markdown_link(r)} |"
            )
        lines.append("")
    if skipped:
        lines.extend(["## 无法解析的报告", ""])
        lines.extend(f"- `{p.relative_to(ROOT)}`" for p in skipped)
        lines.append("")
    lines.extend([
        "## 更新方式",
        "",
        "```bash",
        "python3 scripts/build_streamingbench_experiment_summary.py",
        "```",
        "",
        "脚本会重新扫描所有 `eval_results.txt` 并覆盖本文件。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    records, skipped = [], []
    for path in sorted(RESULT_ROOT.glob("*/eval_results.txt")):
        record = parse_report(path)
        if record is None:
            skipped.append(path)
        else:
            records.append(record)
    OUTPUT.write_text(build(records, skipped), encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)} with {len(records)} records; skipped={len(skipped)}")


if __name__ == "__main__":
    main()
