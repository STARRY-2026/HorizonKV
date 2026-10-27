"""Run the 10-video LU, attention-DP, and 1000-image SparseMM calibration serially."""

import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path("/home/yujun/program/hermes/models/Qwen3-VL-8B-Instruct")
SYNTHDOG = Path("/home/yujun/program/hermes/results/myobservations/sparsemm_official_calibration_20261010/data/synthdog-en.json")


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def run(stage, command, state_path, state):
    state.update(stage=stage, status="running", updated_utc=datetime.now(timezone.utc).isoformat())
    save(state_path, state)
    print(f"FORMAL_CALIBRATION_STAGE {stage}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    state["completed_stages"].append(stage)
    save(state_path, state)


def main():
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "controller.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
    else:
        state = {"status": "running", "completed_stages": [], "model": str(MODEL),
                 "full_benchmarks_paused": True}
    lu = out / "lukv_10v"
    dp = out / "dp_attention_10v_k500"
    ocr = out / "sparsemm_synthdog_1000"
    try:
        if "lukv" not in state["completed_stages"]:
            command = [sys.executable, "-m", "tools.qwen3vl_lukv_calibrate",
                       "--model-path", str(MODEL), "--output-dir", str(lu),
                       "--num-videos", "10", "--max-time", "120",
                       "--video-max-pixels", "1200000"]
            if lu.exists():
                command.append("--resume")
            run("lukv", command, state_path, state)
        if "dp_attention" not in state["completed_stages"]:
            if dp.exists():
                manifest = dp / "manifest.json"
                if not manifest.is_file() or json.loads(manifest.read_text()).get("status") != "complete_formal":
                    raise RuntimeError("Incomplete DP output requires inspection before restart")
                state["completed_stages"].append("dp_attention")
                save(state_path, state)
            else:
                run("dp_attention", [sys.executable, "-m", "tools.qwen3vl_dp_attention_pilot",
                                     "--calibration-manifest", str(lu / "manifest.json"),
                                     "--output-dir", str(dp)], state_path, state)
        if "sparsemm_ocr" not in state["completed_stages"]:
            command = [sys.executable, "-m", "tools.qwen3vl_sparsemm_ocr_calibrate",
                       "--model-path", str(MODEL), "--annotations", str(SYNTHDOG),
                       "--output-dir", str(ocr), "--num-images", "1000",
                       "--max-pixels", "1200000", "--max-decode", "1024"]
            if ocr.exists():
                command.append("--resume")
            run("sparsemm_ocr", command, state_path, state)
        state.update(status="complete", stage=None, updated_utc=datetime.now(timezone.utc).isoformat())
        save(state_path, state)
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}",
                     updated_utc=datetime.now(timezone.utc).isoformat())
        save(state_path, state)
        raise


if __name__ == "__main__":
    main()
