"""LLaVA-OneVision entry point for the shared KVBench streaming controller."""

from lmms_eval.api.registry import register_model

from sparsemm_protokv.model_loader import model_type
from sparsemm_protokv.models.qwen2_5_vl_stream import Qwen25VLStream


@register_model("llava_onevision_stream")
class LlavaOneVisionStream(Qwen25VLStream):
    def __init__(self, pretrained, **kwargs):
        if model_type(pretrained) != "llava_onevision":
            raise ValueError("llava_onevision_stream requires a LLaVA-OneVision checkpoint")
        super().__init__(pretrained=pretrained, **kwargs)
