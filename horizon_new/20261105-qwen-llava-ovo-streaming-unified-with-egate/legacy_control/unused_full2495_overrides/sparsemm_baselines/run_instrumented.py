"""Run the unchanged StreamingBench runner with numerical and cache checks.

The runner retains ownership of pause checks, model loading, and the GPU lease.
Hooks observe tensors and reject invalid results; they never alter model output.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

CACHE_FIELDS = ("flat_k", "flat_v", "flat_pos", "head_lens")
CURSOR_FIELDS = ("_next_position", "_seen_visual_tokens", "total_processed_frames")


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class ValidationHarness:
    """Attach read-only checks to one or more loaded SparseMMQwen bridges."""
    def __init__(self, report_path):
        self.report_path = Path(report_path)
        self.report = {
            "status": "running",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "finite_norm_checks": 0, "finite_lm_head_checks": 0,
            "qa_restore_checks": 0, "qa_successes": 0, "qa_exceptions": 0,
            "peak_cache_bytes": 0, "qa_checks": [], "failures": [],
        }
        self._hooks = []
        self._questions = []
        self.write()

    def write(self):
        _atomic_json(self.report_path, self.report)

    def record_failure(self, error):
        self.report["status"] = "failed"
        description = f"{type(error).__name__}: {error}"
        if description not in self.report["failures"]:
            self.report["failures"].append(description)
        self.write()

    def _finite_hook(self, name):
        def hook(module, inputs, output):
            self.report[f"finite_{name}_checks"] += 1
            if not isinstance(output, torch.Tensor) or not torch.isfinite(output).all().item():
                error = FloatingPointError(f"Non-finite {name} output in instrumented model inference")
                self.record_failure(error)
                raise error
        return hook

    def attach(self, bridge):
        config = asdict(bridge.config)
        if "config" in self.report and self.report["config"] != config:
            raise ValueError("One validation report cannot mix baseline configurations")
        self.report.update(config=config, method=config["method"], budget=config["budget"],
                           budget_scope=config["budget_scope"])
        self._hooks.append(bridge.language_model.norm.register_forward_hook(self._finite_hook("norm")))
        self._hooks.append(bridge.model.lm_head.register_forward_hook(self._finite_hook("lm_head")))
        original = bridge.question_answering
        had_instance_method = "question_answering" in vars(bridge)
        instance_method = vars(bridge).get("question_answering")

        def checked_question(*args, **kwargs):
            cache = bridge.kv_cache
            layers = list(cache.layers)
            saved = [[(getattr(layer, field), getattr(layer, field).detach().clone())
                      for field in CACHE_FIELDS] for layer in layers]
            cursors = {field: getattr(bridge, field) for field in CURSOR_FIELDS}
            cache_bytes = int(cache.nbytes())
            self.report["peak_cache_bytes"] = max(self.report["peak_cache_bytes"], cache_bytes)
            check = {"video_id": getattr(bridge, "_active_video_id", None),
                     "question_index": getattr(bridge, "_active_question_index", None),
                     "cache_bytes": cache_bytes, "head_lengths": cache.head_lengths(),
                     "cursors": cursors, "returned": False, "restored": False}
            error = None
            try:
                result = original(*args, **kwargs)
                check["returned"] = True
                self.report["qa_successes"] += 1
                return result
            except BaseException as exc:
                error = exc
                self.report["qa_exceptions"] += 1
                self.record_failure(exc)
                raise
            finally:
                try:
                    if bridge.kv_cache is not cache or len(cache.layers) != len(layers):
                        raise AssertionError("Question answering replaced the persistent video cache")
                    for index, (layer, states) in enumerate(zip(cache.layers, saved)):
                        if layer is not layers[index]:
                            raise AssertionError(f"Question answering replaced cache layer {index}")
                        for field, (reference, value) in zip(CACHE_FIELDS, states):
                            current = getattr(layer, field)
                            if current is not reference or not torch.equal(current, value):
                                raise AssertionError(f"Question answering changed layer {index} {field}")
                    for field, value in cursors.items():
                        if getattr(bridge, field) != value:
                            raise AssertionError(f"Question answering changed persistent cursor {field}")
                    check["restored"] = True
                    self.report["qa_restore_checks"] += 1
                except BaseException as exc:
                    self.record_failure(exc)
                    if error is not None:
                        raise exc from error
                    raise
                finally:
                    self.report["qa_checks"].append(check)
                    self.write()

        bridge.question_answering = checked_question
        self._questions.append((bridge, had_instance_method, instance_method))
        self.write()
        return bridge

    def close(self):
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        for bridge, had_instance_method, instance_method in self._questions:
            if had_instance_method:
                bridge.question_answering = instance_method
            else:
                del bridge.question_answering
        self._questions.clear()

    def finish(self, success=True):
        self.close()
        if self.report["status"] != "failed":
            self.report["status"] = "complete" if success else "failed"
        self.report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        self.write()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--validation-report", type=Path, required=True)
    options, runner_args = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    from sparsemm_baselines import adapter, run_streamingbench

    harness = ValidationHarness(options.validation_report)
    harness.report["runner_arguments"] = runner_args
    harness.report["runtime_code_sha256"] = run_streamingbench.code_hashes()
    harness.write()
    original_loader = adapter.load_model

    def checked_loader(*args, **kwargs):
        return harness.attach(original_loader(*args, **kwargs))

    succeeded = False
    try:
        with patch.object(adapter, "load_model", side_effect=checked_loader):
            result = run_streamingbench.main(runner_args)
        succeeded = result in (None, 0)
        return result
    except BaseException as exc:
        harness.record_failure(exc)
        raise
    finally:
        harness.finish(succeeded)


if __name__ == "__main__":
    main()
