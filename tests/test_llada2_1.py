from dllm_bench.interfaces import PositionState
from dllm_bench.models.llada2_1 import _build_observational_trace, _count_revision_events
from dllm_bench.registry import build_model_adapter


class _Tokenizer:
    def decode(self, token_ids, **_):
        return "".join(str(token_id) for token_id in token_ids)


def test_llada21_registry_exposes_official_quality_and_speed_modes():
    quality = build_model_adapter("configs/models/llada2_1.yaml", "qmode")
    speed = build_model_adapter("configs/models/llada2_1.yaml", "smode")
    assert (quality.threshold, quality.editing_threshold) == (0.7, 0.5)
    assert (speed.threshold, speed.editing_threshold) == (0.5, 0.0)
    assert quality.supports_trace is True
    assert speed.execution_path == "official-transformers-remote-generate"


def test_llada21_uses_shared_prompt_tokenization_for_mapping_results():
    import torch

    class MappingTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": "prompt"}]
            assert kwargs["return_dict"] is True
            return {
                "input_ids": torch.tensor([[11, 12]]),
                "attention_mask": torch.tensor([[1, 1]]),
            }

    adapter = build_model_adapter("configs/models/llada2_1.yaml", "qmode")
    adapter._tokenizer = MappingTokenizer()
    adapter.device = "cpu"

    assert adapter._prompt_ids("prompt").tolist() == [[11, 12]]


def test_observer_derives_mask_fills_and_real_edits_from_official_canvases():
    observations = [
        {"canvas": [90, 99, 99], "active_start": 1, "entropy": [0.8, 0.7], "confidence": [0.2, 0.3], "current_confidence": [0.0, 0.0], "proposed_tokens": [1, 3]},
        {"canvas": [90, 1, 99], "active_start": 1, "entropy": [0.4, 0.6], "confidence": [0.7, 0.5], "current_confidence": [0.2, 0.0], "proposed_tokens": [2, 3]},
        {"canvas": [90, 2, 3], "active_start": 1, "entropy": [0.2, 0.1], "confidence": [0.9, 0.95], "current_confidence": [0.9, 0.95], "proposed_tokens": [2, 3]},
    ]
    trace = _build_observational_trace(
        observations=observations, prompt_ids=[90], final_ids=[2, 3],
        prompt_length=1, mask_id=99, tokenizer=_Tokenizer(),
    )
    assert trace[0].committed_positions == [0]
    assert trace[1].committed_positions == [0, 1]
    assert trace[-1].committed_positions == []
    assert trace[0].position_states == [PositionState.ACCEPTED, PositionState.MASKED]
    assert trace[1].entropy_by_position == {0: 0.4, 1: 0.6}
    assert trace[1].editing_state_by_position == {
        0: "accepted_edit",
        1: "mask_fill",
    }
    assert trace[2].editing_state_by_position == {0: "stable", 1: "stable"}
    assert trace[1].current_token_confidence_by_position == {0: 0.2, 1: 0.0}
    assert trace[1].confidence_margin_by_position == {0: 0.5, 1: 0.5}
    assert _count_revision_events(trace, 99) == 1
