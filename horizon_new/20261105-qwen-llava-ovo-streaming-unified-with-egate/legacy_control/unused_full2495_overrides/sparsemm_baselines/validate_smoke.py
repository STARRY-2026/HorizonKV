"""Check a real one-question smoke result before launching a full comparison."""
import argparse
import json
from pathlib import Path


def validate(directory):
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text())
    protocol = json.loads((directory / "protocol.json").read_text())
    if summary["videos"] != 1 or summary["questions"] != 1:
        raise ValueError("Expected exactly one real-video smoke question")
    if (directory / "run.status").read_text().strip() != "complete":
        raise ValueError("The real smoke inference did not complete")
    events = [json.loads(line) for line in (directory / "cache_trace.jsonl").read_text().splitlines()]
    prefills = [e for e in events if e.get("event") == "video_prefill"]
    if not prefills:
        raise ValueError("Smoke did not record a visual forward")
    shape = None
    saw_compression = False
    for event in prefills:
        physical, layers = event.get("head_lengths"), event.get("layers")
        if not isinstance(physical, list) or not isinstance(layers, list) or not physical or len(physical) != len(layers):
            raise ValueError("Smoke has missing or inconsistent layer counts")
        if any(not isinstance(row, list) or not row or
               any(type(n) is not int or n <= 0 for n in row) for row in physical):
            raise ValueError("Smoke has invalid physical KV head lengths")
        current_shape = (len(physical), len(physical[0]))
        if any(len(row) != current_shape[1] for row in physical) or shape is not None and shape != current_shape:
            raise ValueError("Layer/head dimensions changed during smoke")
        shape = current_shape
        for lengths, metadata in zip(event["head_lengths"], event["layers"]):
            if not isinstance(metadata, dict) or type(metadata.get("compressed")) is not bool:
                raise ValueError("Smoke has invalid allocation metadata")
            if lengths != metadata["head_lengths"] or sum(lengths) != metadata["actual_total_tokens"]:
                raise ValueError("Physical cache lengths disagree with the allocation trace")
            if metadata["method"] != protocol["method"] or metadata["budget_scope"] != protocol["budget_scope"]:
                raise ValueError("Smoke trace has a different baseline configuration")
            saw_compression |= metadata["compressed"]
    if not saw_compression:
        raise ValueError("Smoke did not exercise real KV compression")
    inits = [e for e in events if e.get("event") == "init"]
    if inits and "num_layers" in inits[-1] and shape != (inits[-1]["num_layers"], inits[-1]["num_kv_heads"]):
        raise ValueError("Smoke dimensions do not match the initialized model")
    restored = [e for e in events if e.get("event") == "answer_restored"]
    if not restored or restored[-1]["head_lengths"] != prefills[-1]["head_lengths"]:
        raise ValueError("Question answering did not restore the video cache")
    return {"passed": True, "method": protocol["method"], "budget_scope": protocol["budget_scope"],
            "questions": 1, "prefills": len(prefills)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(json.dumps(validate(parser.parse_args().directory), indent=2))
