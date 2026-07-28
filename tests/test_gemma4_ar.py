from __future__ import annotations

from dllm_bench.interfaces import PositionState
from dllm_bench.models.gemma4_ar import Gemma4ARAdapter, _build_ar_trace
from dllm_bench.registry import build_model_adapter


class _Tokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return "".join(f"[{token_id}]" for token_id in ids)


def test_build_gemma4_ar_trace_commits_one_token_per_forward():
    trace = _build_ar_trace([10, 20, 30], _Tokenizer())

    assert len(trace) == 3
    assert trace[0].token_ids == [10, -1, -1]
    assert trace[0].position_states == [
        PositionState.ACCEPTED,
        PositionState.MASKED,
        PositionState.MASKED,
    ]
    assert trace[0].committed_positions == [0]
    assert trace[2].token_ids == [10, 20, 30]
    assert trace[2].committed_positions == [2]


def test_gemma4_model_config_builds_official_non_thinking_adapter():
    adapter = build_model_adapter("configs/models/gemma4_26b.yaml")

    assert isinstance(adapter, Gemma4ARAdapter)
    assert adapter.name == "gemma4_26b"
    assert adapter.config_name == "ar-baseline"
    assert adapter._enable_thinking is False
