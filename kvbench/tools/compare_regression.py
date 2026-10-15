"""Compare native per-question predictions and source bindings after migration."""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


METHODS = ("adakv", "pyramidkv", "lukv", "sparsemm", "uniform")


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Write a separate verification record without changing historical results")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    reference, candidate = args.reference.resolve(), args.candidate.resolve()
    records = []
    for method in METHODS:
        name = f"{method}_k1000_visual"
        old, new = reference / name, candidate / name
        assert (new / "run.status").read_text().strip() == "complete", name
        before, after = rows(old / "1_0.csv"), rows(new / "1_0.csv")
        assert len(before) == len(after) == 3
        compared = ("video_id", "question", "choices", "answer", "correct_choice", "pred_answer", "pred_choice", "qa_acc", "task")
        for a, b in zip(before, after):
            assert all(a[k] == b[k] for k in compared), (method, a, b)
        audit = read(new / "audit_report.json")
        assert audit["passed"] and audit["trace"]["compression_events"] == 9
        assert audit["trace"]["qa_restore_checks"] == 3
        checks = read(new / "runtime_checks.json")
        assert checks["status"] == "complete" and not checks["failures"]
        assert all(q["restored"] and q["returned"] and q["first_answer_native_layers"] == 28
                   and q["first_answer_pseudo_layers"] == 0 for q in checks["qa_checks"])
        old_manifest, new_manifest = read(old / "run_manifest.json"), read(new / "run_manifest.json")
        shared = set(old_manifest["code_sha256"]) & set(new_manifest["code_sha256"])
        assert shared and all(old_manifest["code_sha256"][k] == new_manifest["code_sha256"][k] for k in shared)
        assert Path(new_manifest["environment"]["PYTHONPATH"].split(":")[0]).resolve() == root
        assert Path(new_manifest["lmms_eval_file"]).resolve().is_relative_to(root)
        protocol = read(new / "protocol.json")
        assert protocol["subset_sha256"] == read(old / "protocol.json")["subset_sha256"]
        assert protocol["budget"] == 1000 and protocol["budget_scope"] == "visual"
        summary = read(new / "summary.json")
        records.append({"method": method, "predictions": [r["pred_choice"] for r in after],
                        "correct": summary["correct"], "questions": 3,
                        "identical_to_reference": True, "audit_passed": True,
                        "shared_source_hashes_equal": len(shared), "compression_events": 9,
                        "peak_gpu_allocated_gib": summary["peak_gpu_allocated_gib"]})
    collection = read(candidate / "suite_manifest.json")
    assert collection["status"] == "complete"
    interface_snapshot = candidate / "interface_source_snapshot"
    for rel, expected in collection["interface_sha256"].items():
        assert sha(interface_snapshot / rel) == expected
    migration = read(root / "MIGRATION_MANIFEST.json")
    assert all(sha(root / item["copied"]) == item.get("current_sha256", item["sha256"])
               for item in migration["copied_source_and_assets"])
    result = {"passed": True, "created_utc": datetime.now(timezone.utc).isoformat(),
              "reference": str(reference), "candidate": str(candidate), "runs": records,
              "source_and_asset_copies_verified": len(migration["copied_source_and_assets"]),
              "tested_interface_snapshot": str(interface_snapshot),
              "tested_interface_sha256": collection["interface_sha256"],
              "current_interface_sha256": {rel: sha(root / rel) for rel in collection["interface_sha256"]},
              "scope": "Five non-ProtoKV methods, same three OVO MCQA queries; no full benchmark or ProtoKV inference"}
    destination = args.output.resolve() if args.output else candidate / "migration_regression.json"
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"passed": True, "methods": len(records), "predictions_equal": 15,
                      "source_and_asset_copies_verified": result["source_and_asset_copies_verified"]}))


if __name__ == "__main__":
    main()
