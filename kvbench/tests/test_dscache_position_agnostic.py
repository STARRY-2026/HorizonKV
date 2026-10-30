import pytest
import torch
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLTextConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLRotaryEmbedding,
)

from sparsemm_native.dscache_position_agnostic import (
    apply_mrope_to_key,
    remove_mrope_from_key,
)


class _Model:
    class _Language:
        pass

    def __init__(self):
        config = Qwen2_5_VLTextConfig(
            hidden_size=512, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=128000,
        )
        self.language_model = self._Language()
        self.language_model.rotary_emb = Qwen2_5_VLRotaryEmbedding(config)


@pytest.mark.parametrize('dtype,tolerance', [(torch.float32, 2e-5),
                                            (torch.bfloat16, 3e-2)])
def test_qwen_mrope_key_round_trip(dtype, tolerance):
    torch.manual_seed(17)
    model = _Model()
    raw = torch.randn(1, 2, 19, 128, dtype=dtype)
    positions = torch.stack((torch.arange(19),
                             torch.arange(19) // 4,
                             torch.arange(19) % 4))[:, None, :]
    sections = [16, 24, 24]

    rotated = apply_mrope_to_key(model, raw, positions, sections).to(dtype)
    recovered = remove_mrope_from_key(model, rotated, positions, sections)
    reassigned = apply_mrope_to_key(model, recovered, positions, sections)

    assert torch.allclose(recovered.float(), raw.float(), atol=tolerance, rtol=tolerance)
    assert torch.allclose(reassigned.float(), rotated.float(), atol=tolerance, rtol=tolerance)


def test_qwen_mrope_key_rejects_mismatched_positions():
    model = _Model()
    raw = torch.zeros(1, 2, 4, 128)
    with pytest.raises(ValueError, match='positions'):
        apply_mrope_to_key(model, raw, torch.zeros(3, 1, 3, dtype=torch.long),
                           [16, 24, 24])
