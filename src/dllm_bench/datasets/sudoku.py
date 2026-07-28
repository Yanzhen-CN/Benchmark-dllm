"""Official Sudoku protocol from Ye et al. (ICLR 2025).

The paper uses Park's one-million-game dataset, rows 0..99,999 for training
and rows 100,000..100,999 for testing. A puzzle is the raw 81-digit sequence
(``0`` means blank), the target is the raw 81-digit solution, and accuracy is
whole-sequence exact match. Easy/Hard is this benchmark's reporting stratum
only; it never changes the official input, target, or score.
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
import zipfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from .base import Dataset, Sample, ScoreResult
from ..data_paths import ensure_data_layout

Grid = list[list[int]]

_BLANK_TOKENS = {".", "0", "_"}

SUDOKU_SOURCE_REVISION = "bryanpark-sudoku-v3"
SUDOKU_ARCHIVE_URL = "https://www.kaggle.com/api/v1/datasets/download/bryanpark/sudoku"
SUDOKU_ARCHIVE_SHA256 = "38437d3f1f47cbdd12e5cc9d86a7dafe2b23c7ebcb9c785ef881a81865651fb6"
SUDOKU_CSV_SHA256 = "5a77d5392c19c783db68961e000c17fda246f1e362655dc9675f3e7cd4f57bd6"
SUDOKU_TRAIN_ROWS = 100_000
SUDOKU_TEST_ROWS = 1_000
SUDOKU_OFFICIAL_MAX_NEW_TOKENS = 82


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
    ) -> None:
        self._samples = list(samples) if samples is not None else None
        data_root = Path(cache_dir or ensure_data_layout()["datasets"])
        self._archive_path = (
            data_root / "sudoku" / SUDOKU_SOURCE_REVISION / "bryanpark-sudoku.zip"
        )

    def load_samples(self, n: int | None = None) -> list[Sample]:
        samples = (
            list(self._samples)
            if self._samples is not None
            else _load_official_test_samples(
                _ensure_official_archive(self._archive_path)
            )
        )
        return samples[:n] if n is not None else samples

    def preparation_signature(self) -> dict[str, object]:
        return {
            "source_revision": SUDOKU_SOURCE_REVISION,
            "archive_sha256": SUDOKU_ARCHIVE_SHA256,
            "csv_sha256": SUDOKU_CSV_SHA256,
            "test_start": SUDOKU_TRAIN_ROWS,
            "test_rows": SUDOKU_TEST_ROWS,
        }

    def score(self, sample: Sample, output_text: str) -> ScoreResult:
        ref: SudokuReference = sample.reference
        prediction = output_text.strip()
        target = _grid_to_digits(ref.solution)
        valid = bool(re.fullmatch(r"[1-9]{81}", prediction))
        return ScoreResult(
            primary_score=1.0 if prediction == target else 0.0,
            valid=valid,
            complete=valid,
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


def _grid_to_digits(grid: Grid) -> str:
    return "".join(str(value) for row in grid for value in row)


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
                        prompt=puzzle_digits,
                        reference=SudokuReference(puzzle, solution, difficulty),
                        meta={
                            "source": "bryanpark/sudoku",
                            "source_revision": SUDOKU_SOURCE_REVISION,
                            "source_index": train_rows + test_index,
                            "official_split": "test",
                            "official_input_format": "81_digits_zero_is_blank",
                            "official_output_format": "81_solution_digits",
                            "max_new_tokens": SUDOKU_OFFICIAL_MAX_NEW_TOKENS,
                            "difficulty_rule": "naked_single_rounds_le_5_vs_ge_6",
                            "naked_single_rounds": rounds,
                        },
                    )
                )
    return samples
