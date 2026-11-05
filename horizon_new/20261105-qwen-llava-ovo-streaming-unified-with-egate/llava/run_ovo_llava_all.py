#!/usr/bin/env python3
"""Run the LLaVA-OV OVO-Bench DSCache-input baseline workflow.

This orchestrates existing method implementations in this release. It does not
embed model weights, data, baseline algorithms, or DSCache's cache policy.
By default it only checks configurations. Pass --execute to run evaluations.
"""
import argparse
import concurrent.futures
import csv
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
METHODS = ("adakv", "pyramidkv", "sparsemm", "lukv", "hermes", "fullkv")

def say(*args):
    print(time.strftime("%F %T"), *args, flush=True)

def invoke(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\\nCOMMAND " + json.dumps([str(x) for x in command], ensure_ascii=False) + "\\n")
        log.flush()
        return subprocess.run([str(x) for x in command], stdin=subprocess.DEVNULL,
                              stdout=log, stderr=subprocess.STDOUT, check=False).returncode

def command(method, gpus, dest, smoke=False, dry=False, annotation=None):
    cmd = [sys.executable, HERE/"run.py", "--benchmark", "ovobench", "--model", "llavaov",
           "--method", method, "--input-profile", "dscache", "--gpus", ",".join(gpus),
           "--output-dir", dest]
    if smoke: cmd += ["--max-videos", "1"]
    if dry: cmd += ["--dry-run"]
    if annotation: cmd += ["--annotation", annotation]
    return cmd

def smoke_method(method, gpu, root, annotation):
    say("SMOKE_START", method, "PPU", gpu)
    dest = root/"smoke"/method
    rc = invoke(command(method, [gpu], dest, smoke=True, annotation=annotation),
                root/"logs"/f"smoke_{method}.log")
    if rc:
        raise RuntimeError(f"{method} smoke exit {rc}")
    result = dest/"shard_00"/"results.csv"
    with result.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1 or rows[0].get("pred_choice") not in tuple("ABCDEFGH"):
        raise RuntimeError(f"{method} smoke did not produce one valid answer")
    text = (dest/"shard_00"/"unified.log").read_text(errors="replace")
    if "ordinary_tokens=196 latest_replay=false" not in text:
        raise RuntimeError(f"{method} smoke token counts not confirmed")
    say("SMOKE_OK", method)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--annotation", type=Path)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    gpus = args.gpus.split(",")
    if not gpus or any(not x.isdigit() for x in gpus) or len(set(gpus)) != len(gpus):
        p.error("--gpus must contain distinct numeric IDs")
    if len(set(args.methods)) != len(args.methods):
        p.error("methods must be distinct")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root/".workflow.lock").open("w") as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise SystemExit("A workflow is already using this output directory")
        if not args.execute:
            for method in args.methods:
                rc = invoke(command(method, gpus, root/"full"/method, dry=True, annotation=args.annotation),
                            root/"logs"/f"dry_{method}.log")
                if rc: raise SystemExit(f"{method} dry-run failed; see logs")
                say("DRY_RUN_OK", method)
            say("Preflight only. Add --execute to run the evaluations.")
            return 0
        # Each batch assigns one independent PPU to each smoke job.
        for start in range(0, len(args.methods), len(gpus)):
            batch = args.methods[start:start+len(gpus)]
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as pool:
                tasks = [pool.submit(smoke_method, method, gpu, root, args.annotation)
                         for method, gpu in zip(batch, gpus)]
                for task in tasks: task.result()
        say("ALL_SMOKES_PASSED")
        for method in args.methods:
            dest = root/"full"/method
            say("FULL_START", method, "PPUs", ",".join(gpus))
            rc = invoke(command(method, gpus, dest, annotation=args.annotation),
                        root/"logs"/f"full_{method}.log")
            if rc: raise SystemExit(f"{method} evaluation failed ({rc}); queue stopped")
            rc = invoke([sys.executable, HERE/"summarize.py", dest],
                        root/"logs"/f"summary_{method}.log")
            if rc: raise SystemExit(f"{method} summary failed ({rc}); queue stopped")
            say("FULL_DONE", method)
        say("ALL_METHODS_DONE")
    return 0

if __name__ == "__main__":
    sys.exit(main())
