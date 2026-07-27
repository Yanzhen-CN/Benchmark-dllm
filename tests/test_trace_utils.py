from dllm_bench.interfaces import PositionState
from dllm_bench.models.trace_utils import trace_steps_from_snapshots

MASK = -1


class _Tok:
    def decode(self, ids, skip_special_tokens=False):
        return f"[{ids[0]}]"


def test_empty_snapshots_gives_empty_trace():
    assert trace_steps_from_snapshots([], MASK, _Tok()) == []


def test_first_snapshot_commits_are_whatever_isnt_still_masked():
    snapshots = [[10, MASK, 12, MASK]]
    trace = trace_steps_from_snapshots(snapshots, MASK, _Tok())
    assert len(trace) == 1
    assert trace[0].committed_positions == [0, 2]
    assert trace[0].position_states == [
        PositionState.ACCEPTED, PositionState.MASKED, PositionState.ACCEPTED, PositionState.MASKED,
    ]


def test_commits_are_diffed_against_the_previous_snapshot():
    snapshots = [
        [10, MASK, MASK, MASK],
        [10, 20, MASK, MASK],
        [10, 20, 30, 40],
    ]
    trace = trace_steps_from_snapshots(snapshots, MASK, _Tok())
    assert [step.committed_positions for step in trace] == [[0], [1], [2, 3]]


def test_revision_is_detected_generically_as_a_change_from_previous_snapshot():
    # position 0 changes value on step 1 despite already being non-mask —
    # the diff must treat this as a commit event too, since this utility
    # makes no assumption about whether the model revises.
    snapshots = [
        [10, MASK],
        [99, 20],
    ]
    trace = trace_steps_from_snapshots(snapshots, MASK, _Tok())
    assert trace[1].committed_positions == [0, 1]


def test_token_texts_use_mask_display_for_masked_positions():
    snapshots = [[10, MASK]]
    trace = trace_steps_from_snapshots(snapshots, MASK, _Tok())
    assert trace[0].token_texts == ["[10]", "▢"]


def test_final_snapshot_has_no_masked_positions_when_fully_resolved():
    snapshots = [[MASK, MASK], [1, MASK], [1, 2]]
    trace = trace_steps_from_snapshots(snapshots, MASK, _Tok())
    assert all(s == PositionState.ACCEPTED for s in trace[-1].position_states)
