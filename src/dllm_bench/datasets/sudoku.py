"""Sudoku: Easy/Hard Accuracy, plus Cell Accuracy, constraint-satisfaction
rate, completion rate, and conflict rate (section 1).

Difficulty (section 1's definition) is derived, not asserted: a puzzle is
**Easy** if repeated naked-single elimination alone solves it, **Hard** if
that gets stuck and would need at least one trial-and-error/backtracking
step. :func:`classify_difficulty` implements exactly that rule so difficulty
is reproducible from the puzzle grid alone.
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass

from .base import Dataset, Sample, ScoreResult

Grid = list[list[int]]

_BLANK_TOKENS = {".", "0", "_"}


def _box_cells(r: int, c: int) -> list[tuple[int, int]]:
    box_r, box_c = 3 * (r // 3), 3 * (c // 3)
    return [(i, j) for i in range(box_r, box_r + 3) for j in range(box_c, box_c + 3)]


def candidates(grid: Grid, r: int, c: int) -> set[int]:
    if grid[r][c] != 0:
        return set()
    used = set(grid[r]) | {grid[i][c] for i in range(9)}
    used |= {grid[i][j] for i, j in _box_cells(r, c)}
    return set(range(1, 10)) - used


def solve_naked_singles(grid: Grid) -> tuple[Grid, bool]:
    """Repeatedly fill any cell with exactly one candidate. Returns
    (resulting_grid, fully_solved)."""
    grid = deepcopy(grid)
    changed = True
    while changed:
        changed = False
        for r in range(9):
            for c in range(9):
                if grid[r][c] == 0:
                    cell_candidates = candidates(grid, r, c)
                    if len(cell_candidates) == 1:
                        grid[r][c] = next(iter(cell_candidates))
                        changed = True
    fully_solved = all(grid[r][c] != 0 for r in range(9) for c in range(9))
    return grid, fully_solved


def classify_difficulty(puzzle: Grid) -> str:
    _, fully_solved = solve_naked_singles(puzzle)
    return "easy" if fully_solved else "hard"


def parse_grid(text: str) -> Grid | None:
    """Best-effort extraction of a 9x9 grid from free-form model output.

    Looks for lines that plausibly encode a grid row (>= 9 digit/blank
    tokens, few stray letters) and takes the *last* 9 such lines, since a
    reasoning-style response usually states the final grid last.
    """
    candidate_rows: list[list[str]] = []
    for line in text.splitlines():
        tokens = re.findall(r"[1-9]|[.0_]", line)
        letters = re.findall(r"[A-Za-z]", line)
        if len(tokens) >= 9 and len(letters) <= 2:
            candidate_rows.append(tokens[:9])

    if len(candidate_rows) < 9:
        return None
    rows = candidate_rows[-9:]

    grid: Grid = []
    for row_tokens in rows:
        grid.append([0 if tok in _BLANK_TOKENS else int(tok) for tok in row_tokens])
    return grid


def _units() -> list[list[tuple[int, int]]]:
    rows = [[(r, c) for c in range(9)] for r in range(9)]
    cols = [[(r, c) for r in range(9)] for c in range(9)]
    boxes = [
        [(r, c) for r in range(br, br + 3) for c in range(bc, bc + 3)]
        for br in (0, 3, 6)
        for bc in (0, 3, 6)
    ]
    return rows + cols + boxes


_UNITS = _units()


def constraint_satisfaction_rate(grid: Grid) -> float:
    satisfied = 0
    for unit in _UNITS:
        values = [grid[r][c] for r, c in unit if grid[r][c] != 0]
        if len(values) == len(set(values)):
            satisfied += 1
    return satisfied / len(_UNITS)


def completion_rate(grid: Grid) -> float:
    filled = sum(1 for row in grid for v in row if v != 0)
    return filled / 81


def cell_accuracy(grid: Grid, solution: Grid) -> float:
    correct = sum(
        1 for r in range(9) for c in range(9) if grid[r][c] == solution[r][c]
    )
    return correct / 81


@dataclass
class SudokuReference:
    puzzle: Grid
    solution: Grid
    difficulty: str | None = None
    """Defaults to classify_difficulty(puzzle) if not given explicitly."""


class SudokuDataset(Dataset):
    name = "sudoku"

    def __init__(self, samples: list[Sample] | None = None) -> None:
        self._samples = samples or []

    def load_samples(self, n: int | None = None) -> list[Sample]:
        return self._samples[:n] if n is not None else list(self._samples)

    def score(self, sample: Sample, output_text: str) -> ScoreResult:
        ref: SudokuReference = sample.reference
        grid = parse_grid(output_text)

        if grid is None:
            return ScoreResult(
                primary_score=0.0,
                aux={
                    "cell_accuracy": 0.0,
                    "constraint_satisfaction_rate": 0.0,
                    "completion_rate": 0.0,
                    "conflict_rate": 1.0,
                },
                valid=False,
                complete=False,
            )

        accuracy = 1.0 if grid == ref.solution else 0.0
        satisfaction = constraint_satisfaction_rate(grid)
        return ScoreResult(
            primary_score=accuracy,
            aux={
                "cell_accuracy": cell_accuracy(grid, ref.solution),
                "constraint_satisfaction_rate": satisfaction,
                "completion_rate": completion_rate(grid),
                "conflict_rate": 1.0 - satisfaction,
            },
            valid=True,
            complete=completion_rate(grid) == 1.0,
        )

    def aggregate_records(
        self, samples: list[Sample], results: list[ScoreResult]
    ) -> dict[str, float]:
        summary = super().aggregate_records(samples, results)
        for difficulty, group in group_by_difficulty(samples, results).items():
            if group:
                summary[f"accuracy_{difficulty}"] = (
                    sum(result.primary_score for result in group) / len(group)
                )
                summary[f"n_{difficulty}"] = float(len(group))
        return summary


def group_by_difficulty(
    samples: list[Sample], results: list[ScoreResult]
) -> dict[str, list[ScoreResult]]:
    """Section 1 reports Easy/Hard accuracy separately, not blended."""
    grouped: dict[str, list[ScoreResult]] = {"easy": [], "hard": []}
    for sample, result in zip(samples, results):
        ref: SudokuReference = sample.reference
        difficulty = ref.difficulty or classify_difficulty(ref.puzzle)
        grouped[difficulty].append(result)
    return grouped
