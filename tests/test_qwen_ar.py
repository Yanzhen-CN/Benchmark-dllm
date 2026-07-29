from pathlib import Path

import pytest

from dllm_bench.interfaces import GenerationRequest, PositionState
from dllm_bench.models.hf_ar import QwenARAdapter
from dllm_bench.registry import build_model_adapter


ROOT = Path(__file__).resolve().parent.parent


def test_qwen3_8b_config_builds_distinct_ar_adapter():
    adapter = build_model_adapter(ROOT / "configs" / "models" / "qwen3_8b.yaml")

    assert adapter.name == "qwen3_8b"
    assert adapter.config_name == "ar-baseline"
    assert adapter._model_name == "Qwen/Qwen3-8B"
    assert adapter._enable_thinking is False


def test_qwen3_4b_default_adapter_name_is_unchanged():
    adapter = build_model_adapter(ROOT / "configs" / "models" / "qwen3_4b.yaml")

    assert adapter.name == "qwen3_4b"


def test_qwen_trace_is_rebuilt_without_retaining_generation_scores():
    torch = pytest.importorskip("torch")

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return {
                "input_ids": torch.tensor([[1, 2]]),
                "attention_mask": torch.tensor([[1, 1]]),
            }

        def decode(self, ids, skip_special_tokens=False):
            return "".join(f"<{int(token_id)}>" for token_id in ids)

    class FakeModel:
        def __init__(self):
            self.kwargs = None

        def generate(self, **kwargs):
            self.kwargs = kwargs
            suffix = torch.tensor([[7, 8]])
            return torch.cat([kwargs["input_ids"], suffix], dim=1)

    adapter = QwenARAdapter(model_name_or_path="unused")
    adapter._tokenizer = FakeTokenizer()
    adapter._model = FakeModel()
    adapter._device = "cpu"

    result = adapter._generate_core(
        GenerationRequest(prompt="hello", max_new_tokens=2)
    )

    assert "output_scores" not in adapter._model.kwargs
    assert "return_dict_in_generate" not in adapter._model.kwargs
    assert result.num_forward_passes == 2
    assert [step.committed_positions for step in result.trace] == [[0], [1]]
    assert result.trace[0].position_states == [
        PositionState.ACCEPTED,
        PositionState.MASKED,
    ]
    assert result.trace[0].entropy_by_position is None
