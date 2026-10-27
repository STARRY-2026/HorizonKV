"""Bounded OVO-Bench entry for the pinned recent-one/four plus capped-history arm.

This is deliberately separate from StreamingBench MCQA: it uses the official
nine-task OVO prompts, scoring and source-video grouping. Full runs remain
paused; this interface accepts only an explicit 1..8-query smoke subset.
"""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import subprocess
import sys

from . import paths


ARM = "recent1or4_history_cap4000"
ASSET = paths.PROJECT_ROOT / "assets/recent1or4/per_head_allocations.csv"


@dataclass(frozen=True)
class OvoRunConfig:
    output_dir: Path
    ids: tuple[str, ...]
    method: str = ARM
    dry_run: bool = False
    python_executable: str = field(default_factory=lambda: sys.executable)

    def selected(self):
        if self.method != ARM:
            raise ValueError(f"OVO method must be {ARM}")
        if not self.ids or len(self.ids) > 8 or any(not isinstance(value, str) or not value.strip()
                                                   or "," in value for value in self.ids):
            raise ValueError("Select 1..8 explicit OVO query IDs")
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("Duplicate OVO query IDs are not allowed")
        if not ASSET.is_file():
            raise FileNotFoundError(f"Bundled recent-1/4 calibration is missing: {ASSET}")
        from sparsemm_native.run_ovobench_sparsemm_fps1 import selected_queries
        _expected, selected = selected_queries(",".join(self.ids))
        if len(selected) != len(self.ids) or len({item["video_path"] for item in selected}) > 3:
            raise ValueError("OVO smoke validation permits at most 3 source videos")
        if any(not 0 < float(item["clip_end"]) <= 120 for item in selected):
            raise ValueError("OVO smoke validation permits video prefixes up to 120 seconds")
        return selected


def build_ovo_command(config, selected=None):
    selected = config.selected() if selected is None else selected
    groups = len({item["video_path"] for item in selected})
    return [config.python_executable, "-u", "-m", "kvbench.ovo_recent1or4_worker",
            "--output-root", str(Path(config.output_dir).expanduser().resolve()),
            "--ids", ",".join(config.ids), "--max-groups", str(groups)]


def run_ovo(config):
    if not isinstance(config, OvoRunConfig):
        raise TypeError("run_ovo requires OvoRunConfig")
    selected = config.selected()
    out = Path(config.output_dir).expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"Choose a new OVO output directory: {out}")
    command = build_ovo_command(config, selected)
    out.mkdir(parents=True)
    plan = {"method": ARM, "benchmark": "ovobench_nine_task", "query_ids": [q["query_id"] for q in selected],
            "source_videos": len({q["video_path"] for q in selected}), "command": command,
            "full_benchmarks_paused": True, "dry_run": config.dry_run}
    (out / "kvbench_ovo_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    if config.dry_run:
        return plan
    environment = dict(os.environ)
    environment.update(
        PYTHONPATH=os.pathsep.join((str(paths.PROJECT_ROOT),
                                    str(paths.PROJECT_ROOT / "sparsemm_baselines/upstream/lmms-eval"),
                                    str(paths.PROJECT_ROOT / "sparsemm_protokv/runtime_deps"))),
        HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MALLOC_ARENA_MAX="2",
    )
    with (out / "kvbench_ovo.log").open("w") as log:
        subprocess.run(command, cwd=paths.PROJECT_ROOT, env=environment,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    return audit_ovo(out)


def audit_ovo(output_dir):
    """Check complete answers, pinned source/asset hashes and recorded cache quotas."""
    from sparsemm_native import capped_history
    from sparsemm_native.run_ovobench_recent4 import digest, validate_row
    from sparsemm_native.run_ovobench_sparsemm_fps1 import selected_queries, sha
    from sparsemm_native.run_ovobench_recent1or4_cap4000 import ARM as native_arm

    out = Path(output_dir).expanduser().resolve()
    run = out / ARM if (out / ARM).is_dir() else out
    protocol = json.loads((run / "protocol.json").read_text())
    if protocol.get("arm") != ARM or native_arm != ARM or protocol.get("version") != "ovo_nine_recent1or4_history_cap4000_v1":
        raise ValueError("Output is not the registered recent-1/4 OVO arm")
    asset = Path(protocol["history_quota_source"])
    if asset.resolve() != ASSET.resolve() or sha(asset) != protocol["history_quota_source_sha256"]:
        raise ValueError("OVO recent-1/4 quota source is missing or changed")
    rows = capped_history.allocation(path=asset)
    if (protocol["per_head_allocation"] != rows
            or protocol["configured_history_visual_slots"] != capped_history.TOTAL_HISTORY
            or protocol["history_head_cap"] != capped_history.CAP
            or protocol["one_frame_heads"] != [[r["layer"], r["kv_head"]]
                                                for r in rows if r["recent_group"] == "one_frame"]):
        raise ValueError("OVO recent-1/4 allocation differs from the pinned quota source")
    for path_name, fingerprint in protocol["code_sha256"].items():
        relative = Path(path_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("OVO source path escapes its snapshot")
        snapshot = run / "source_snapshot" / relative
        if not snapshot.is_file() or sha(snapshot) != fingerprint:
            raise ValueError(f"Pinned OVO source snapshot changed: {path_name}")
    if (sha(Path(protocol["model"]) / "config.json") != protocol["model_config_sha256"]
            or sha(protocol["annotation"]) != protocol["annotation_sha256"]
            or sha(protocol["sparsemm_score_path"]) != protocol["sparsemm_score_sha256"]):
        raise ValueError("OVO model, annotation or score asset changed")
    manifest = json.loads((run / "query_manifest.json").read_text())
    expected, selected = selected_queries(",".join(protocol["query_ids"]))
    if (manifest != selected or protocol["query_count"] != len(selected)
            or protocol["full_nine_query_count"] != len(expected)
            or protocol["query_sha256"] != digest(selected)
            or protocol["full_nine_query_sha256"] != digest(expected)):
        raise ValueError("OVO query subset disagrees with its official source")
    lines = (run / "queries.jsonl").read_text().splitlines()
    answers = [json.loads(line) for line in lines]
    if len(answers) != len(selected):
        raise ValueError("OVO smoke run does not cover every selected query")
    for answer, query in zip(answers, selected):
        validate_row(answer, query)
        metrics = answer.get("cache_budget_audit", {})
        if (metrics.get("history_recent_independent_prefill") is not True
                or metrics.get("configured_history_visual_slots") != capped_history.TOTAL_HISTORY
                or metrics.get("history_head_cap") != capped_history.CAP
                or metrics.get("recent_one_frame_heads") != 56
                or metrics.get("recent_four_frame_heads") != 56
                or type(metrics.get("actual_history_visual_slots")) is not int
                or not 0 <= metrics["actual_history_visual_slots"] <= capped_history.TOTAL_HISTORY
                or type(metrics.get("actual_recent_visual_slots")) is not int
                or metrics["actual_visual_slots"] != metrics["actual_recent_visual_slots"]
                   + metrics["actual_history_visual_slots"]):
            raise ValueError(f"OVO cache-budget check failed for {query['query_id']}")
    summary = json.loads((run / "summary.json").read_text())
    if (summary.get("complete") is not True or summary.get("completed_queries") != len(answers)
            or summary.get("total_queries") != len(answers)
            or summary.get("question_accuracy") != sum(q["correct"] for q in answers) / len(answers)):
        raise ValueError("OVO result summary differs from checked answers")
    report = {"passed": True, "method": ARM, "benchmark": "ovobench_nine_task",
              "questions": len(answers), "source_videos": len({q["video_path"] for q in selected}),
              "correct": sum(q["correct"] for q in answers), "accuracy": summary["question_accuracy"],
              "configured_history_visual_slots": capped_history.TOTAL_HISTORY,
              "history_head_cap": capped_history.CAP, "full_benchmarks_paused": True,
              "limitations": ["A bounded OVO smoke subset does not establish full nine-task accuracy.",
                              "Attention rankings and tensor contents cannot be reconstructed from per-query summaries."]}
    (run / "audit_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
