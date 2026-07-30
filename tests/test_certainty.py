import math

import pytest

from dllm_bench.interfaces import PositionState, TraceStep
from dllm_bench.metrics.certainty import (
    accepted_ratio,
    build_certainty_curve,
    build_observed_certainty_curve,
    certainty,
    normalized_entropy,
)


def test_normalized_entropy_of_a_certain_distribution_is_zero():
    assert normalized_entropy([1.0, 0.0, 0.0, 0.0], vocab_size=4) == pytest.approx(0.0)


def test_normalized_entropy_of_uniform_distribution_is_one():
    vocab_size = 8
    probs = [1 / vocab_size] * vocab_size
    assert normalized_entropy(probs, vocab_size) == pytest.approx(1.0, abs=1e-9)


def test_normalized_entropy_vocab_size_one_is_zero():
    assert normalized_entropy([1.0], vocab_size=1) == pytest.approx(0.0)


def test_certainty_is_one_when_nothing_remains():
    assert certainty([]) == pytest.approx(1.0)


def test_certainty_is_one_minus_mean_entropy():
    entropies = [0.2, 0.4, 0.6]
    assert certainty(entropies) == pytest.approx(1 - (0.2 + 0.4 + 0.6) / 3)


def test_accepted_ratio_basic():
    assert accepted_ratio(3, 6) == pytest.approx(0.5)


def test_accepted_ratio_rejects_nonpositive_final_length():
    with pytest.raises(ValueError):
        accepted_ratio(1, 0)


def _make_step(states, entropy_by_position=None, top1_by_position=None):
    return TraceStep(
        forward_index=0,
        token_ids=[0] * len(states),
        position_states=states,
        committed_positions=[i for i, s in enumerate(states) if s == PositionState.ACCEPTED],
        decoded_text="",
        entropy_by_position=entropy_by_position,
        top1_confidence_by_position=top1_by_position,
    )


def test_build_certainty_curve_tracks_ratio_and_certainty_over_steps():
    trace = [
        _make_step(
            [PositionState.ACCEPTED, PositionState.MASKED, PositionState.MASKED],
            entropy_by_position={1: 0.8, 2: 0.6},
            top1_by_position={1: 0.3, 2: 0.4},
        ),
        _make_step(
            [PositionState.ACCEPTED, PositionState.ACCEPTED, PositionState.MASKED],
            entropy_by_position={2: 0.1},
            top1_by_position={2: 0.9},
        ),
        _make_step(
            [PositionState.ACCEPTED, PositionState.ACCEPTED, PositionState.ACCEPTED],
        ),
    ]
    curve = build_certainty_curve(trace, final_valid_length=3)
    assert len(curve) == 3

    ratio0, cert0, top1_0 = curve[0]
    assert ratio0 == pytest.approx(1 / 3)
    assert cert0 == pytest.approx(1 - (0.8 + 0.6) / 2)
    assert top1_0 == pytest.approx((0.3 + 0.4) / 2)

    ratio1, cert1, _ = curve[1]
    assert ratio1 == pytest.approx(2 / 3)
    assert cert1 == pytest.approx(1 - 0.1)

    ratio2, cert2, top1_2 = curve[2]
    assert ratio2 == pytest.approx(1.0)
    assert cert2 == pytest.approx(1.0)
    assert top1_2 == pytest.approx(1.0)


def test_build_certainty_curve_is_monotonic_in_accepted_ratio_for_well_behaved_trace():
    trace = [
        _make_step([PositionState.MASKED, PositionState.MASKED], entropy_by_position={0: 0.9, 1: 0.9}),
        _make_step([PositionState.ACCEPTED, PositionState.MASKED], entropy_by_position={1: 0.5}),
        _make_step([PositionState.ACCEPTED, PositionState.ACCEPTED]),
    ]
    curve = build_certainty_curve(trace, final_valid_length=2)
    ratios = [c[0] for c in curve]
    assert ratios == sorted(ratios)


def test_observed_certainty_does_not_turn_missing_ar_logits_into_one():
    trace = [
        _make_step([PositionState.ACCEPTED, PositionState.MASKED]),
        _make_step([PositionState.ACCEPTED, PositionState.ACCEPTED]),
    ]
    curve = build_observed_certainty_curve(trace, final_valid_length=2)
    assert curve == [(1.0, 1.0, None)]
