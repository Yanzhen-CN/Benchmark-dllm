"""Sudoku protocol adapted for general instruction checkpoints.

The paper uses Park's one-million-game dataset, rows 0..99,999 for training
and rows 100,000..100,999 for testing. This benchmark preserves the official
81-digit puzzle/solution representation. The evaluated checkpoint must return
only the completed 81-digit solution. The scorer tolerates incidental wrappers
or row formatting, extracts the final complete grid, and checks its legality;
strict direct-output compliance is retained as a separate diagnostic.
Blank-cell accuracy, exact reference match, given preservation, completion,
and constraint satisfaction are retained as diagnostics. Easy/Hard is a
reporting stratum only; it never changes the source puzzle or score protocol.
"""

from __future__ import annotations

import csv
import hashlib
import os
import random
import re
import zipfile
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from .base import Dataset, Sample, ScoreResult
from ..data_paths import ensure_data_layout

Grid = list[list[int]]

_BLANK_TOKENS = {".", "0", "_"}

SUDOKU_SOURCE_REVISION = "bryanpark-sudoku-v3"
SUDOKU_PROTOCOL_REVISION = "direct-prompt-tolerant-parser-v1"
SUDOKU_ARCHIVE_URL = "https://www.kaggle.com/api/v1/datasets/download/bryanpark/sudoku"
SUDOKU_ARCHIVE_SHA256 = "38437d3f1f47cbdd12e5cc9d86a7dafe2b23c7ebcb9c785ef881a81865651fb6"
SUDOKU_CSV_SHA256 = "5a77d5392c19c783db68961e000c17fda246f1e362655dc9675f3e7cd4f57bd6"
SUDOKU_TRAIN_ROWS = 100_000
SUDOKU_TEST_ROWS = 1_000
SUDOKU_MAX_NEW_TOKENS = 128

_FINAL_ANSWER_MARKER_RE = re.compile(r"(?im)^\s*####\s*")
_COMPACT_SOLUTION_RE = re.compile(r"(?<![0-9])([1-9]{81})(?![0-9])")


def _box_cells(r: int, c: int) -> list[tuple[int, int]]:
    box_r, box_c = 3 * (r // 3), 3 * (c // 3)
    return [(i, j) for i in range(box_r, box_r + 3) for j in range(box_c, box_c + 3)]


def candidates(grid: Grid, r: int, c: int) -> set[int]:
    if grid[r][c] != 0:
        return set()
    used = set(grid[r]) | {grid[i][c] for i in range(9)}
    used |= {grid[i][j] for i, j in _box_cells(r, c)}
    return set(range(1, 10)) - used


def naked_single_rounds(puzzle: Grid) -> int | None:
    """Synchronous naked-single rounds; ``None`` means singles get stuck."""
    grid = deepcopy(puzzle)
    rounds = 0
    while any(0 in row for row in grid):
        fills: list[tuple[int, int, int]] = []
        for row in range(9):
            for col in range(9):
                if grid[row][col] != 0:
                    continue
                options = candidates(grid, row, col)
                if len(options) == 1:
                    fills.append((row, col, next(iter(options))))
        if not fills:
            return None
        for row, col, value in fills:
            grid[row][col] = value
        rounds += 1
    return rounds


def classify_difficulty(puzzle: Grid) -> str:
    """Analysis-only split on the official test set, not an official label."""
    rounds = naked_single_rounds(puzzle)
    return "easy" if rounds is not None and rounds <= 5 else "hard"


def parse_grid(text: str) -> Grid | None:
    """Best-effort extraction of a 9x9 grid from free-form model output.

    Looks for lines that plausibly encode a grid row (>= 9 digit/blank
    tokens, few stray letters) and takes the *last* 9 such lines, since a
    reasoning-style response usually states the final grid last.
    """
    stripped = text.strip()
    if re.fullmatch(r"[0-9]{81}", stripped):
        return [
            [int(stripped[row * 9 + col]) for col in range(9)]
            for row in range(9)
        ]

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


def blank_cell_accuracy(grid: Grid, puzzle: Grid, solution: Grid) -> float:
    """Return solution accuracy on cells the model was asked to fill.

    Blanks and wrong digits both receive no credit.  A degenerate puzzle with
    no blanks falls back to exact-match scoring rather than dividing by zero.
    """
    blank_cells = [
        (r, c) for r in range(9) for c in range(9) if puzzle[r][c] == 0
    ]
    if not blank_cells:
        return 1.0 if grid == solution else 0.0
    correct = sum(grid[r][c] == solution[r][c] for r, c in blank_cells)
    return correct / len(blank_cells)


def given_preservation_rate(grid: Grid, puzzle: Grid) -> float:
    """Return the fraction of prompt-supplied cells preserved in the output."""
    given_cells = [
        (r, c) for r in range(9) for c in range(9) if puzzle[r][c] != 0
    ]
    if not given_cells:
        return 1.0
    preserved = sum(grid[r][c] == puzzle[r][c] for r, c in given_cells)
    return preserved / len(given_cells)


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
        cache_dir: str | Path | None = None,
        easy_count: int = 50,
        hard_count: int = 50,
        seed: int = 42,
    ) -> None:
        self._samples = list(samples) if samples is not None else None
        self._easy_count = int(easy_count)
        self._hard_count = int(hard_count)
        self._seed = int(seed)
        data_root = Path(cache_dir or ensure_data_layout()["datasets"])
        self._archive_path = (
            data_root / "sudoku" / SUDOKU_SOURCE_REVISION / "bryanpark-sudoku.zip"
        )

    def load_samples(self, n: int | None = None) -> list[Sample]:
        if self._samples is not None:
            samples = list(self._samples)
        else:
            official_test = _load_official_test_samples(
                _ensure_official_archive(self._archive_path)
            )
            samples = _select_formal_subset(
                official_test,
                easy_count=self._easy_count,
                hard_count=self._hard_count,
                seed=self._seed,
            )
        return samples[:n] if n is not None else samples

    def preparation_signature(self) -> dict[str, object]:
        return {
            "source_revision": SUDOKU_SOURCE_REVISION,
            "protocol_revision": SUDOKU_PROTOCOL_REVISION,
            "archive_sha256": SUDOKU_ARCHIVE_SHA256,
            "csv_sha256": SUDOKU_CSV_SHA256,
            "test_start": SUDOKU_TRAIN_ROWS,
            "test_rows": SUDOKU_TEST_ROWS,
            "formal_easy_count": self._easy_count,
            "formal_hard_count": self._hard_count,
            "formal_subset_seed": self._seed,
        }

    def score(self, sample: Sample, output_text: str) -> ScoreResult:
        ref: SudokuReference = sample.reference
        prediction = output_text.strip()
        target = _grid_to_digits(ref.solution)
        official_format_valid = bool(re.fullmatch(r"[1-9]{81}", prediction))
        official_exact = 1.0 if prediction == target else 0.0
        grid, marker_present = extract_final_grid(output_text)

        if grid is None:
            return ScoreResult(
                primary_score=0.0,
                aux={
                    "official_exact_match_accuracy": official_exact,
                    "official_format_valid": 0.0,
                    "exact_solve_rate": 0.0,
                    "blank_cell_accuracy": 0.0,
                    "cell_accuracy": 0.0,
                    "given_preservation_rate": 0.0,
                    "constraint_satisfaction_rate": 0.0,
                    "completion_rate": 0.0,
                    "conflict_rate": 1.0,
                    "constraint_valid": 0.0,
                    "reference_exact_match": 0.0,
                    "answer_marker_present": float(marker_present),
                },
                valid=False,
                complete=False,
            )

        exact = 1.0 if grid == ref.solution else 0.0
        partial_credit = blank_cell_accuracy(grid, ref.puzzle, ref.solution)
        satisfaction = constraint_satisfaction_rate(grid)
        constraint_valid = is_valid_solution(grid, ref.puzzle)
        return ScoreResult(
            primary_score=float(constraint_valid),
            aux={
                "official_exact_match_accuracy": official_exact,
                "official_format_valid": float(official_format_valid),
                "exact_solve_rate": exact,
                "blank_cell_accuracy": partial_credit,
                "cell_accuracy": cell_accuracy(grid, ref.solution),
                "given_preservation_rate": given_preservation_rate(
                    grid, ref.puzzle
                ),
                "constraint_satisfaction_rate": satisfaction,
                "completion_rate": completion_rate(grid),
                "conflict_rate": 1.0 - satisfaction,
                "constraint_valid": float(constraint_valid),
                "reference_exact_match": exact,
                "answer_marker_present": float(marker_present),
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
                summary[f"blank_cell_accuracy_{difficulty}"] = (
                    sum(result.aux["blank_cell_accuracy"] for result in group)
                    / len(group)
                )
                exact_solve_rate = (
                    sum(result.aux["exact_solve_rate"] for result in group)
                    / len(group)
                )
                summary[f"exact_solve_rate_{difficulty}"] = exact_solve_rate
                summary[f"accuracy_{difficulty}"] = (
                    sum(result.primary_score for result in group) / len(group)
                )
                summary[f"n_{difficulty}"] = float(len(group))
        return summary


def group_by_difficulty(
    samples: list[Sample], results: list[ScoreResult]
) -> dict[str, list[ScoreResult]]:
    """Group results for separate Easy/Hard partial and exact metrics."""
    grouped: dict[str, list[ScoreResult]] = {"easy": [], "hard": []}
    for sample, result in zip(samples, results):
        ref: SudokuReference = sample.reference
        difficulty = ref.difficulty or classify_difficulty(ref.puzzle)
        grouped[difficulty].append(result)
    return grouped


def _select_formal_subset(
    samples: list[Sample], *, easy_count: int, hard_count: int, seed: int
) -> list[Sample]:
    if easy_count < 0 or hard_count < 0 or easy_count + hard_count == 0:
        raise ValueError("formal Sudoku counts must be non-negative and not both zero")
    rng = random.Random(seed)
    selected: list[Sample] = []
    for difficulty, count in (("easy", easy_count), ("hard", hard_count)):
        group = sorted(
            (
                sample
                for sample in samples
                if (sample.reference.difficulty or classify_difficulty(sample.reference.puzzle))
                == difficulty
            ),
            key=lambda sample: sample.sample_id,
        )
        rng.shuffle(group)
        if len(group) < count:
            raise ValueError(
                f"official Sudoku test split has {len(group)} {difficulty} samples; "
                f"the formal subset requires {count}"
            )
        selected.extend(
            replace(
                sample,
                meta={
                    **sample.meta,
                    "formal_subset": True,
                    "formal_subset_seed": seed,
                },
            )
            for sample in group[:count]
        )
    return selected


def _grid_to_digits(grid: Grid) -> str:
    return "".join(str(value) for row in grid for value in row)


def _build_prompt(puzzle_digits: str) -> str:
    return (
        "Directly solve the following 9x9 Sudoku puzzle. The puzzle is given "
        "as an 81-digit row-major string, where 0 represents a blank cell. "
        "Return only the completed 81-digit row-major solution using digits "
        "1-9, with no spaces, separators, answer label, or explanation.\n"
        f"Puzzle: {puzzle_digits}\n"
        "Answer:"
    )


def extract_final_grid(text: str) -> tuple[Grid | None, bool]:
    """Extract the final Sudoku grid, preferring the last ``####`` marker.

    This mirrors GSM8K's marker-first/fallback-last-answer convention.  If a
    marker is present, only text after the last marker is considered so digits
    in the model's reasoning cannot be mistaken for the submitted solution.
    Without a marker, the last complete compact solution or 9-row grid is used.
    """
    markers = list(_FINAL_ANSWER_MARKER_RE.finditer(text))
    candidate_text = text[markers[-1].end() :] if markers else text
    compact = _COMPACT_SOLUTION_RE.findall(candidate_text)
    if compact:
        digits = compact[-1]
        return (
            [[int(digits[row * 9 + col]) for col in range(9)] for row in range(9)],
            bool(markers),
        )
    return parse_grid(candidate_text), bool(markers)


def is_valid_solution(grid: Grid, puzzle: Grid) -> bool:
    """Return whether ``grid`` is a complete legal solution preserving clues."""
    if len(grid) != 9 or any(len(row) != 9 for row in grid):
        return False
    required = set(range(1, 10))
    if any(set(row) != required for row in grid):
        return False
    if any({grid[row][col] for row in range(9)} != required for col in range(9)):
        return False
    for box_row in range(0, 9, 3):
        for box_col in range(0, 9, 3):
            box = {
                grid[row][col]
                for row in range(box_row, box_row + 3)
                for col in range(box_col, box_col + 3)
            }
            if box != required:
                return False
    return all(
        puzzle[row][col] == 0 or grid[row][col] == puzzle[row][col]
        for row in range(9)
        for col in range(9)
    )


def _digits_to_grid(value: str, *, row_index: int, field: str) -> Grid:
    if not re.fullmatch(r"[0-9]{81}", value):
        raise ValueError(
            f"official Sudoku row {row_index} has invalid {field}; expected 81 digits"
        )
    return [[int(value[row * 9 + col]) for col in range(9)] for row in range(9)]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_csv_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with zipfile.ZipFile(path) as archive:
        with archive.open("sudoku.csv") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _archive_is_official(path: Path) -> bool:
    if not path.is_file() or _sha256_file(path) != SUDOKU_ARCHIVE_SHA256:
        return False
    try:
        return _verified_csv_digest(path) == SUDOKU_CSV_SHA256
    except (KeyError, zipfile.BadZipFile):
        return False


def _ensure_official_archive(path: Path) -> Path:
    if _archive_is_official(path):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(SUDOKU_ARCHIVE_URL, headers={"User-Agent": "dllm-bench/0.1"})
    try:
        with urlopen(request, timeout=180) as response:  # noqa: S310 - fixed HTTPS URL
            payload = response.read()
    except (OSError, URLError) as exc:
        raise RuntimeError(f"failed to download official Sudoku data: {exc}") from exc
    digest = hashlib.sha256(payload).hexdigest()
    if digest != SUDOKU_ARCHIVE_SHA256:
        raise RuntimeError(
            "official Sudoku archive failed checksum verification: "
            f"expected {SUDOKU_ARCHIVE_SHA256}, got {digest}"
        )
    partial_path = path.with_suffix(path.suffix + ".part")
    try:
        partial_path.write_bytes(payload)
        if _verified_csv_digest(partial_path) != SUDOKU_CSV_SHA256:
            raise RuntimeError("official Sudoku CSV failed checksum verification")
        os.replace(partial_path, path)
    finally:
        if partial_path.exists():
            partial_path.unlink()
    return path


def _load_official_test_samples(
    path: Path,
    *,
    train_rows: int = SUDOKU_TRAIN_ROWS,
    test_rows: int = SUDOKU_TEST_ROWS,
) -> list[Sample]:
    samples: list[Sample] = []
    with zipfile.ZipFile(path) as archive:
        with archive.open("sudoku.csv") as raw:
            rows = csv.reader(line.decode("utf-8") for line in raw)
            header = next(rows, None)
            if header != ["quizzes", "solutions"]:
                raise ValueError(f"unexpected official Sudoku CSV header: {header!r}")
            for _ in range(train_rows):
                if next(rows, None) is None:
                    raise ValueError("official Sudoku CSV ended before the test split")
            for test_index in range(test_rows):
                row = next(rows, None)
                if row is None:
                    raise ValueError(
                        f"expected {test_rows} official Sudoku test rows, found {test_index}"
                    )
                if len(row) != 2:
                    raise ValueError(f"official Sudoku row {test_index} has {len(row)} columns")
                puzzle_digits, solution_digits = row
                puzzle = _digits_to_grid(
                    puzzle_digits, row_index=test_index, field="puzzle"
                )
                solution = _digits_to_grid(
                    solution_digits, row_index=test_index, field="solution"
                )
                difficulty = classify_difficulty(puzzle)
                rounds = naked_single_rounds(puzzle)
                samples.append(
                    Sample(
                        sample_id=f"sudoku-test-{test_index:04d}",
                        prompt=_build_prompt(puzzle_digits),
                        reference=SudokuReference(puzzle, solution, difficulty),
                        meta={
                            "source": "bryanpark/sudoku",
                            "source_revision": SUDOKU_SOURCE_REVISION,
                            "protocol_revision": SUDOKU_PROTOCOL_REVISION,
                            "source_index": train_rows + test_index,
                            "official_split": "test",
                            "source_input_format": "81_digits_zero_is_blank",
                            "prompt_protocol": "direct_81_solution_digits_only",
                            "official_output_format": "81_solution_digits",
                            "max_new_tokens": SUDOKU_MAX_NEW_TOKENS,
                            "difficulty_rule": "naked_single_rounds_le_5_vs_ge_6",
                            "naked_single_rounds": rounds,
                        },
                    )
                )
    return samples
