"""Section 4.2.5: Sudoku trial-and-error and correction (RevisionCount).

Reuses the trace already collected for every sample (section 4.1) — no
separate instrumentation — applied only to Sudoku, since that's the one
dataset where "needs at least one trial-and-error step" (the Hard-difficulty
definition, section 1) gives this metric something to cross-check against:
if Hard's RevisionCount is clearly higher than Easy's, the trace mechanism
and the Easy/Hard split are validating each other; if not, the trace's
granularity doesn't actually line up with real trial-and-error and needs
re-checking (this is a case study, not part of the main ranking).

    RevisionCount(sample) = sum_(r,c) |{t : x_rc^(t-1) != mask
                                           AND x_rc^(t) != mask
                                           AND x_rc^(t) != x_rc^(t-1)}|

Only counts a cell changing from one non-mask value to a DIFFERENT non-mask
value — a genuine "changed my mind" event. The first mask -> value
assignment for a cell is explicitly excluded (that's an initial answer, not
a revision).
"""

from __future__ import annotations

import re

from ..interfaces import PositionState, TraceStep

_DIGITS = set("123456789")
Grid = list[list[int]]


def _parse_digit(text: str | None) -> int | None:
    if text is None:
        return None
    stripped = text.strip()
    return int(stripped) if len(stripped) == 1 and stripped in _DIGITS else None


def _token_aligned_grid(step: TraceStep) -> Grid | None:
    if len(step.position_states) != 81 or not step.token_texts:
        return None
    digits: list[int] = []
    for position in range(81):
        if step.position_states[position] == PositionState.MASKED:
            return None
        digit = _parse_digit(step.token_texts[position])
        if digit is None:
            return None
        digits.append(digit)
    return [digits[row * 9 : (row + 1) * 9] for row in range(9)]


def _decoded_grid(step: TraceStep) -> Grid | None:
    """Extract one unambiguous 81-digit candidate from a decoded canvas."""
    matches = re.findall(r"(?<![0-9])([1-9]{81})(?![0-9])", step.decoded_text)
    if not matches:
        return None
    digits = matches[-1]
    return [
        [int(digits[row * 9 + col]) for col in range(9)]
        for row in range(9)
    ]


def trace_step_grid(step: TraceStep) -> Grid | None:
    """Decode a Sudoku candidate without assuming one token per cell."""
    return _token_aligned_grid(step) or _decoded_grid(step)


def trace_parseable_step_count(trace: list[TraceStep]) -> int:
    return sum(trace_step_grid(step) is not None for step in trace)


def _uses_row_major_token_canvas(trace: list[TraceStep]) -> bool:
    return bool(trace) and len(trace[-1].position_states) == 81


def extract_cell_digit_sequences(trace: list[TraceStep]) -> list[list[int | None]]:
    """For each of 81 row-major positions, the sequence of digit values
    (``None`` while masked/undecided, or if the token text can't be parsed
    as a single digit 1-9) across trace steps. Requires an 81-position
    row-major canvas, same convention as ``report/sudoku_trace_viz.py``.
    """
    if not trace:
        return [[] for _ in range(81)]
    sequences: list[list[int | None]] = [[] for _ in range(81)]
    if _uses_row_major_token_canvas(trace):
        for step in trace:
            for position in range(81):
                if step.position_states[position] == PositionState.MASKED:
                    sequences[position].append(None)
                    continue
                digit_text = step.token_texts[position] if step.token_texts else None
                sequences[position].append(_parse_digit(digit_text))
        return sequences

    for step in trace:
        grid = _decoded_grid(step)
        for position in range(81):
            row, col = divmod(position, 9)
            sequences[position].append(grid[row][col] if grid is not None else None)
    return sequences


def _included_positions(puzzle: Grid | None) -> range | list[int]:
    if puzzle is None:
        return range(81)
    return [
        position
        for position in range(81)
        if puzzle[position // 9][position % 9] == 0
    ]


def compute_revision_count(trace: list[TraceStep], puzzle: Grid | None = None) -> int:
    """RevisionCount for one sample (design doc 4.2.5)."""
    sequences = extract_cell_digit_sequences(trace)
    count = 0
    for position in _included_positions(puzzle):
        sequence = sequences[position]
        for previous, current in zip(sequence, sequence[1:]):
            if previous is not None and current is not None and previous != current:
                count += 1
    return count


def revision_counts_by_stage(
    trace: list[TraceStep], puzzle: Grid | None = None
) -> dict[str, int]:
    """RevisionCount split over normalized forward-progress thirds."""
    counts = {"early": 0, "middle": 0, "late": 0}
    if len(trace) < 2:
        return counts
    sequences = extract_cell_digit_sequences(trace)
    denominator = max(len(trace) - 1, 1)
    for position in _included_positions(puzzle):
        sequence = sequences[position]
        for step_index, (previous, current) in enumerate(zip(sequence, sequence[1:]), start=1):
            if previous is None or current is None or previous == current:
                continue
            progress = step_index / denominator
            stage = "early" if progress < 1 / 3 else "middle" if progress < 2 / 3 else "late"
            counts[stage] += 1
    return counts


def correction_outcomes(
    trace: list[TraceStep], solution: Grid, puzzle: Grid | None = None
) -> tuple[int, int, float | None]:
    """Return ErrorThenCorrect, ErrorThenStillWrong, and their success rate."""
    sequences = extract_cell_digit_sequences(trace)
    correct = still_wrong = 0
    flat_solution = [value for row in solution for value in row]
    for position in _included_positions(puzzle):
        sequence = sequences[position]
        target = flat_solution[position]
        if not any(value is not None and value != target for value in sequence):
            continue
        final = sequence[-1] if sequence else None
        if final == target:
            correct += 1
        else:
            still_wrong += 1
    denominator = correct + still_wrong
    rate = correct / denominator if denominator else None
    return correct, still_wrong, rate
