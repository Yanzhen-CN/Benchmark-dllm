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
import random
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

    def __init__(
        self,
        samples: list[Sample] | None = None,
        easy_count: int = 50,
        hard_count: int = 50,
        seed: int = 42,
    ) -> None:
        self._samples = list(samples) if samples is not None else None
        self._easy_count = easy_count
        self._hard_count = hard_count
        self._seed = seed

    def load_samples(self, n: int | None = None) -> list[Sample]:
        samples = (
            list(self._samples)
            if self._samples is not None
            else generate_sudoku_bank(self._easy_count, self._hard_count, self._seed)
        )
        return samples[:n] if n is not None else samples

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


def _base_solution(rng: random.Random) -> Grid:
    pattern = lambda row, col: (row * 3 + row // 3 + col) % 9
    rows = [group * 3 + row for group in rng.sample(range(3), 3) for row in rng.sample(range(3), 3)]
    cols = [group * 3 + col for group in rng.sample(range(3), 3) for col in rng.sample(range(3), 3)]
    digits = rng.sample(range(1, 10), 9)
    return [[digits[pattern(row, col)] for col in cols] for row in rows]


def _count_solutions(grid: Grid, limit: int = 2) -> int:
    work = deepcopy(grid)
    count = 0

    def search() -> None:
        nonlocal count
        if count >= limit:
            return
        best: tuple[int, int, set[int]] | None = None
        for row in range(9):
            for col in range(9):
                if work[row][col] != 0:
                    continue
                options = candidates(work, row, col)
                if not options:
                    return
                if best is None or len(options) < len(best[2]):
                    best = (row, col, options)
        if best is None:
            count += 1
            return
        row, col, options = best
        for value in sorted(options):
            work[row][col] = value
            search()
            work[row][col] = 0
            if count >= limit:
                return

    search()
    return count


def _make_puzzle(solution: Grid, difficulty: str, rng: random.Random) -> Grid:
    puzzle = deepcopy(solution)
    cells = [(row, col) for row in range(9) for col in range(9)]
    rng.shuffle(cells)
    target_blanks = 40 if difficulty == "easy" else 48
    blanks = 0
    for row, col in cells:
        previous = puzzle[row][col]
        puzzle[row][col] = 0
        unique = _count_solutions(puzzle) == 1
        derived = classify_difficulty(puzzle)
        keep = unique and (
            (difficulty == "easy" and derived == "easy")
            or difficulty == "hard"
        )
        if keep:
            blanks += 1
        else:
            puzzle[row][col] = previous
        if blanks >= target_blanks and classify_difficulty(puzzle) == difficulty:
            return puzzle
    if classify_difficulty(puzzle) != difficulty:
        raise RuntimeError(f"could not generate a unique {difficulty} Sudoku puzzle")
    return puzzle


def _sudoku_prompt(puzzle: Grid) -> str:
    rows = [" ".join(str(value) if value else "." for value in row) for row in puzzle]
    return (
        "Solve this Sudoku. Return the completed 9x9 grid, one row per line, "
        "using digits 1-9 only.\n\n" + "\n".join(rows)
    )


def generate_sudoku_bank(easy_count: int, hard_count: int, seed: int = 42) -> list[Sample]:
    if easy_count < 0 or hard_count < 0 or easy_count + hard_count == 0:
        raise ValueError("Sudoku counts must be non-negative and not both zero")
    rng = random.Random(seed)
    samples: list[Sample] = []
    for difficulty, count in (("easy", easy_count), ("hard", hard_count)):
        generated = 0
        attempts = 0
        while generated < count:
            attempts += 1
            if attempts > count * 30 + 100:
                raise RuntimeError(f"failed to generate {count} unique {difficulty} puzzles")
            solution = _base_solution(rng)
            try:
                puzzle = _make_puzzle(solution, difficulty, rng)
            except RuntimeError:
                continue
            sample_id = f"sudoku-{difficulty}-{generated:04d}"
            samples.append(
                Sample(
                    sample_id=sample_id,
                    prompt=_sudoku_prompt(puzzle),
                    reference=SudokuReference(puzzle, solution, difficulty),
                    meta={
                        "source": "deterministic-generator",
                        "generator_seed": seed,
                        "difficulty_rule": "naked-singles-vs-backtracking",
                    },
                )
            )
            generated += 1
    return samples
