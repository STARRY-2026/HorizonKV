"""Compare Qwen3-VL text decoder logits with the bounded ragged bridge."""

import argparse
import json
from pathlib import Path

import torch

from sparsemm_baselines.core import BaselineConfig
from sparsemm_baselines.qwen3_adapter import load_model
from sparsemm_baselines.run_streamingbench import gpu_lease, write_json


def main(args):
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    with gpu_lease(out), torch.inference_mode():
        bridge = load_model(args.model_path, config=BaselineConfig(method="uniform", budget=500,
                            budget_scope="visual"))
        ids = bridge.processor.tokenizer("Hello, what is in the video?", return_tensors="pt").input_ids.to("cuda")
        pos = torch.arange(ids.shape[1], device="cuda").view(1, 1, -1).expand(3, 1, -1)
        native = bridge.language_model(input_ids=ids, position_ids=pos, use_cache=False,
                                       return_dict=True).last_hidden_state
        bridge.clear_cache()
        ported = bridge.forward_chunk(bridge.model.get_input_embeddings()(ids), pos, compress=False)
        diff = (native.float() - ported.float()).abs()
        native_logits = bridge.model.lm_head(native[:, -1])
        ported_logits = bridge.model.lm_head(ported[:, -1])
        result = {"max_hidden_abs_difference": float(diff.max()),
                  "mean_hidden_abs_difference": float(diff.mean()),
                  "native_top_token": int(native_logits.argmax()),
                  "ported_top_token": int(ported_logits.argmax()),
                  "same_top_token": bool(native_logits.argmax() == ported_logits.argmax())}
        write_json(out / "results.json", result)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    main(parser.parse_args())
