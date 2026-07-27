import pytest

from dllm_bench.interfaces import PositionState, TraceStep
from dllm_bench.metrics.sudoku_revision import (
    compute_revision_count,
    correction_outcomes,
    extract_cell_digit_sequences,
    revision_counts_by_stage,
)


def _step(overrides: dict[int, str]) -> TraceStep:
    position_states = [PositionState.MASKED] * 81
    token_texts: list[str | None] = [None] * 81
    for position, text in overrides.items():
        position_states[position] = PositionState.ACCEPTED
        token_texts[position] = text
    return TraceStep(
        forward_index=0,
        token_ids=[0] * 81,
        position_states=position_states,
        committed_positions=sorted(overrides),
        decoded_text="",
        token_texts=token_texts,
    )


def test_first_assignment_is_not_a_revision():
    trace = [_step({}), _step({0: "5"})]
    assert compute_revision_count(trace) == 0


def test_changing_to_a_different_value_counts_as_one_revision():
    trace = [_step({}), _step({0: "5"}), _step({0: "7"})]
    assert compute_revision_count(trace) == 1


def test_repeating_the_same_value_is_not_a_revision():
    trace = [_step({}), _step({0: "5"}), _step({0: "5"})]
    assert compute_revision_count(trace) == 0


def test_multiple_changes_on_one_cell_all_count():
    trace = [_step({}), _step({0: "5"}), _step({0: "7"}), _step({0: "3"})]
    assert compute_revision_count(trace) == 2


def test_revisions_across_independent_cells_sum():
    trace = [
        _step({}),
        _step({0: "5", 1: "2"}),
        _step({0: "7", 1: "2"}),  # only cell 0 revised
        _step({0: "7", 1: "9"}),  # only cell 1 revised
    ]
    assert compute_revision_count(trace) == 2


def test_going_back_to_masked_then_reassigning_does_not_count_that_hop():
    # mask -> 5 (not a revision) -> back to masked (not a "value" transition
    # per the formula, since current==mask) -> 7 (this IS the first
    # assignment again after having been un-set, so also not a revision by
    # the strict formula: previous must be non-mask for it to count).
    trace = [_step({}), _step({0: "5"}), _step({}), _step({0: "7"})]
    assert compute_revision_count(trace) == 0


def test_empty_trace_has_zero_revisions():
    assert compute_revision_count([]) == 0


def test_extract_cell_digit_sequences_shape_and_values():
    trace = [_step({}), _step({5: "3"})]
    sequences = extract_cell_digit_sequences(trace)
    assert len(sequences) == 81
    assert sequences[5] == [None, 3]
    assert sequences[0] == [None, None]


def test_rejects_wrong_position_count():
    bad_trace = [
        TraceStep(
            forward_index=0,
            token_ids=[1, 2, 3],
            position_states=[PositionState.ACCEPTED] * 3,
            committed_positions=[0, 1, 2],
            decoded_text="",
        )
    ]
    with pytest.raises(ValueError):
        compute_revision_count(bad_trace)


def test_unparseable_digit_text_is_treated_as_none():
    trace = [_step({0: "abc"}), _step({0: "5"})]
    # first step's "abc" parses to None (not a real digit), so the None->5
    # transition on the second step is a first assignment, not a revision.
    assert compute_revision_count(trace) == 0


def test_revision_counts_are_split_by_forward_stage():
    trace = [_step({0: "1"}), _step({0: "2"}), _step({0: "3"}), _step({0: "4"})]
    counts = revision_counts_by_stage(trace)
    assert sum(counts.values()) == 3
    assert counts == {"early": 0, "middle": 1, "late": 2}


def test_correction_outcomes_distinguish_fixed_and_still_wrong_cells():
    solution = [[1 for _ in range(9)] for _ in range(9)]
    trace = [_step({0: "2", 1: "3"}), _step({0: "1", 1: "4"})]
    corrected, still_wrong, rate = correction_outcomes(trace, solution)
    assert (corrected, still_wrong) == (1, 1)
    assert rate == pytest.approx(0.5)
