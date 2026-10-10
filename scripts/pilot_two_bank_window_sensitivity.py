#!/usr/bin/env python3
"""Compare fixed recent-bank boundaries on identical video observations.

The data's Qwen visual time steps are four seconds apart. A window of
4/8/16/32 seconds therefore gives roughly 1/2/4/8 visual time steps.
"""

import json
from pathlib import Path

import numpy as np

import pilot_joint_two_bank as pilot


WINDOWS = (4.0, 8.0, 16.0, 32.0)
RECENTS = np.array([32, 64, 128, 256], dtype=np.int64)
OUT = Path("results/myobservations/two_bank_window_sensitivity_20260923")


def main():
    loaded = {w: pilot.load(time_window=w, recents=RECENTS) for w in WINDOWS}
    ids = set.intersection(*(set(x["observation"] for x in rows) for rows in loaded.values()))
    if not ids:
        raise ValueError("No observations shared by all windows")
    results = {}
    for w in WINDOWS:
        shared = [x for x in loaded[w] if x["observation"] in ids]
        result = pilot.run(observations=shared, time_window=w, recents=RECENTS)
        result["protocol"]["shared_observation_cohort"] = True
        result["protocol"]["visual_time_steps_approx"] = int(w / 4)
        results[str(int(w))] = result
    report = {
        "description": "Window sensitivity on one shared cohort; static oracle utility proxy only.",
        "input_observations_by_window": {str(int(w)): len(loaded[w]) for w in WINDOWS},
        "common_observations": len(ids),
        "windows": results,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    summary = {w: {"videos": r["videos"], "observations": r["observations"],
                   "retention": r["mean_video_retention"], "paired": r["paired"],
                   "soft_recent_counts": np.asarray(r["fold_assignment_counts"]["soft_joint"]).sum(axis=0).tolist()}
               for w, r in results.items()}
    print(json.dumps({"input_observations_by_window": report["input_observations_by_window"],
                      "common_observations": len(ids), "windows": summary}, indent=2))


if __name__ == "__main__":
    main()
