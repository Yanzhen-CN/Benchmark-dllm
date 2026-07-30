from __future__ import annotations

import pytest

from dllm_bench.datasets.answer_region import (
    aggregate_answer_position_metrics,
    locate_mbpp_answer,
    locate_structeval_answer,
    position_aux,
)
from dllm_bench.datasets.base import ScoreResult


def test_mbpp_ignores_code_draft_inside_closed_thinking():
    text = (
        "<think>```python\ndef wrong(): return 0\n```</think>\n"
        "```python\ndef right(): return 1\n```"
    )
    region = locate_mbpp_answer(text)
    assert region.detected
    assert "right" in region.text
    assert "wrong" not in region.text


def test_mbpp_unclosed_thinking_is_not_treated_as_answer():
    region = locate_mbpp_answer("<think>def draft(): return 1")
    assert not region.detected
    assert region.method == "unclosed_thinking"


def test_structeval_uses_last_marker_after_thinking():
    text = (
        "<think><|BEGIN_CODE|>{\"draft\": 1}<|END_CODE|></think>"
        "<|BEGIN_CODE|>{\"final\": 2}<|END_CODE|>"
    )
    region = locate_structeval_answer(text)
    assert region.text == '{"final": 2}'
    assert region.marker_complete


def test_structeval_raw_fallback_scores_but_is_not_style_detected():
    region = locate_structeval_answer('{"raw": true}')
    assert region.text == '{"raw": true}'
    assert not region.detected
    assert region.method == "raw_fallback"


def test_formal_answer_start_aggregate_uses_token_ratio_not_char_fallback():
    detected = locate_mbpp_answer("def f(): return 1")
    first = ScoreResult(1.0, aux=position_aux(detected, "def f(): return 1"))
    second = ScoreResult(
        1.0,
        aux={
            **position_aux(detected, "def f(): return 1"),
            "answer_position_token_mapped_rate": 1.0,
            "answer_start_ratio": 0.25,
        },
    )
    summary = aggregate_answer_position_metrics([first, second])
    assert summary["answer_position_token_mapped_rate"] == pytest.approx(0.5)
    assert summary["answer_start_ratio_mean"] == pytest.approx(0.25)
    assert summary["answer_start_ratio_iqr"] == pytest.approx(0.0)

