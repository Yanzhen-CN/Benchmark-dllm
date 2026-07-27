"""Score a trace's decoded text at fixed checkpoints plus the final forward,
then combine the resulting curves into one sample's AUC/SFI.
"""

from __future__ import annotations

import pytest

from dllm_bench.datasets.structeval_t import (
    StructEvalSchema,
    checkpoint_indices,
    struct_eval_t_checkpoint_scores,
)
from dllm_bench.interfaces import PositionState, TraceStep
from dllm_bench.datasets.mbpp import mbpp_checkpoint_scores
from dllm_bench.metrics.strategy_score import strategy_score


def _step(decoded_text: str, forward_index: int = 0) -> TraceStep:
    return TraceStep(
        forward_index=forward_index,
        token_ids=[],
        position_states=[],
        committed_positions=[],
        decoded_text=decoded_text,
    )


# ---------------------------------------------------------------------------
# checkpoint_indices
# ---------------------------------------------------------------------------

def test_checkpoint_indices_every_4th_plus_final():
    assert checkpoint_indices(10, interval=4) == [0, 4, 8, 9]


def test_checkpoint_indices_final_already_on_interval_not_duplicated():
    assert checkpoint_indices(9, interval=4) == [0, 4, 8]


def test_checkpoint_indices_every_8th():
    assert checkpoint_indices(20, interval=8) == [0, 8, 16, 19]


def test_checkpoint_indices_short_trace_is_just_the_final_step():
    assert checkpoint_indices(1, interval=4) == [0]


def test_checkpoint_indices_empty_trace():
    assert checkpoint_indices(0, interval=4) == []


# ---------------------------------------------------------------------------
# struct_eval_t_checkpoint_scores
# ---------------------------------------------------------------------------

def test_struct_eval_t_checkpoint_scores_tracks_progressive_json_completion():
    schema = StructEvalSchema(format="json", required_keys=["name", "age"], critical_content=["Alice"])
    trace = [
        _step("{", 0),
        _step('{"name": "Alice"', 1),
        _step('{"name": "Alice", "age"', 2),
        _step('{"name": "Alice", "age": 30}', 3),
    ]
    structure_scores, content_scores = struct_eval_t_checkpoint_scores(trace, schema, interval=4)
    # only checkpoints 0 and 3 (final) get scored at interval=4 on a 4-step trace
    assert len(structure_scores) == 2
    assert len(content_scores) == 2
    # final checkpoint has both fields and the critical content -> complete
    assert structure_scores[-1] == 1.0
    assert content_scores[-1] == 1.0
    # first checkpoint ("{" alone) parses to nothing useful
    assert structure_scores[0] < structure_scores[-1]


def test_struct_eval_t_checkpoint_scores_empty_trace():
    schema = StructEvalSchema(format="json", required_keys=["name"])
    assert struct_eval_t_checkpoint_scores([], schema) == ([], [])


def test_struct_eval_t_checkpoint_scores_feed_directly_into_strategy_score():
    schema = StructEvalSchema(format="json", required_keys=["name", "age"])
    # structure (symbols/keys) appears immediately; values fill in late ->
    # structure-first strategy.
    trace = [
        _step('{"name": null, "age": null}', i) for i in range(3)
    ] + [_step('{"name": "Alice", "age": 30}', 3)]
    structure_scores, content_scores = struct_eval_t_checkpoint_scores(trace, schema, interval=1)
    score = strategy_score(structure_scores, content_scores)
    assert score is not None
    assert score > 0  # structure formed before content did


def test_mbpp_checkpoint_scores_feed_directly_into_sfi():
    trace = [
        _step("def solve(x):\n    if x:\n        return", 0),
        _step("def solve(x):\n    if x:\n        return x + 1", 1),
    ]
    structure, content = mbpp_checkpoint_scores(trace, interval=1)
    assert structure[-1] == pytest.approx(1.0)
    assert content[-1] == pytest.approx(1.0)
    assert structure[0] > content[0]
    assert strategy_score(structure, content) > 0
