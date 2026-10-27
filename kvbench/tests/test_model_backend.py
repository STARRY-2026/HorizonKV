"""Model routing checks that do not load weights or reserve a GPU."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from kvbench import RunConfig, run
from kvbench import paths
from sparsemm_protokv.config import StreamingConfig
from sparsemm_protokv.model_loader import model_type
from sparsemm_protokv.run_streamingbench import build_command


class ModelBackendTest(unittest.TestCase):
    def test_local_config_routes_native_plugin(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for kind, expected in (("qwen2_5_vl", "qwen2_5_vl_stream"),
                                   ("llava_onevision", "llava_onevision_stream")):
                model = root / kind
                model.mkdir()
                (model / "config.json").write_text(json.dumps({"model_type": kind}))
                args = SimpleNamespace(model_path=model, dtype="bfloat16", sample_fps=0.5,
                                       encode_chunk_size=1, max_videos=1,
                                       five_video_timing=False, anno_path=root / "annotations.json")
                command, _ = build_command(args, StreamingConfig(method="adakv"), root)
                self.assertEqual(model_type(model), kind)
                self.assertEqual(command[command.index("--model") + 1], expected)

    def test_video_pixel_cap_routes_only_to_qwen_protokv(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "qwen"
            model.mkdir()
            (model / "config.json").write_text('{"model_type":"qwen2_5_vl"}')
            args = SimpleNamespace(model_path=model, dtype="bfloat16", sample_fps=0.5,
                                   encode_chunk_size=16, video_max_pixels=200704,
                                   max_videos=1, five_video_timing=False,
                                   anno_path=root / "annotations.json")
            command, _ = build_command(args, StreamingConfig(method="protokv"), root)
            model_args = command[command.index("--model_args") + 1]
            self.assertIn("video_max_pixels=200704", model_args)
            with self.assertRaisesRegex(ValueError, "only for Qwen2.5-VL ProtoKV"):
                build_command(args, StreamingConfig(method="adakv"), root)
            (model / "config.json").write_text('{"model_type":"llava_onevision"}')
            with self.assertRaisesRegex(ValueError, "only for Qwen2.5-VL ProtoKV"):
                build_command(args, StreamingConfig(method="protokv"), root)

    def test_unsupported_model_fails_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary)
            (model / "config.json").write_text('{"model_type":"unsupported"}')
            with self.assertRaisesRegex(ValueError, "Unsupported KVBench model_type"):
                model_type(model)

    def test_sparsemm_rejects_cross_model_scores_before_starting_methods(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "llava"
            model.mkdir()
            (model / "config.json").write_text('{"model_type":"llava_onevision"}')
            annotation = root / "annotations.json"
            annotation.write_text('[{"conversations":[{"end_time":1}]}]')
            output = root / "output"
            with patch.object(paths, "ensure_resources"), self.assertRaisesRegex(ValueError, "model-bound scores"):
                run(RunConfig(methods=("uniform", "sparsemm"), model_path=model,
                              anno_path=annotation, output_dir=output, dry_run=True))
            self.assertFalse(output.exists())

    def test_synthetic_sparsemm_scores_are_model_bound_and_not_official(self):
        from sparsemm_native.assets import asset_metadata
        from tools.make_sparsemm_smoke_scores import main as generate_scores

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "llava"
            model.mkdir()
            config = model / "config.json"
            config.write_text(json.dumps({"model_type": "llava_onevision", "text_config": {
                "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
            }}))
            assets = root / "assets"
            with patch("builtins.print"):
                generate_scores(["--model-path", str(model), "--output-dir", str(assets)])
            metadata = asset_metadata(assets / "head_scores.csv")
            self.assertEqual(metadata["asset_type"], "synthetic_test_scores")
            self.assertIs(metadata["official_calibration"], False)
            self.assertEqual(metadata["shape"], [2, 2])
            config.write_text(config.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "configuration fingerprint changed"):
                asset_metadata(assets / "head_scores.csv")


if __name__ == "__main__":
    unittest.main()
