"""Sudoku protocol adapted for general instruction checkpoints.

The paper uses Park's one-million-game dataset, rows 0..99,999 for training
and rows 100,000..100,999 for testing. This benchmark preserves the official
81-digit puzzle/solution representation and complete-sequence accuracy.
The prompt fixes every given clue and asks checkpoints to fill only blank
cells, returning the completed grid directly without a reasoning process.
The scorer tolerates missing markers, incidental wrappers, or row formatting,
extracts the final complete grid, and compares it with the reference sequence;
direct-output/marker compliance is a separate diagnostic.
Constraint validity, blank-cell accuracy, given preservation, completion,
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

from .answer_region import (
    AnswerRegion,
    aggregate_answer_position_metrics,
    locate_digit_answer,
    position_aux,
    scored_payload_aux,
    trace_position_aux,
)
from .base import Dataset, Sample, ScoreResult
from .official_metrics import ye_sudoku_sequence_accuracy
from ..data_paths import ensure_data_layout
from ..interfaces import GenerationResult

Grid = list[list[int]]

_BLANK_TOKENS = {".", "0", "_"}

SUDOKU_SOURCE_REVISION = "bryanpark-sudoku-v3"
SUDOKU_PROTOCOL_REVISION = "fixed-clues-valid-grid-direct-v9"
SUDOKU_REASONING_PROTOCOL_REVISION = "grid-prompt-marked-answer-v3"
SUDOKU_ONE_SHOT_PROTOCOL_REVISION = "fixed-one-shot-direct-81-digit-v1"
SUDOKU_ONE_SHOT_EXAMPLE_PUZZLE = (
    "640080000700216000019000070900070004500904002800030007"
    "070000390000521006000090081"
)
SUDOKU_ONE_SHOT_EXAMPLE_ANSWER = (
    "645789123783216459219453678961872534537964812824135967"
    "172648391398521746456397281"
)
SUDOKU_ARCHIVE_URL = "https://www.kaggle.com/api/v1/datasets/download/bryanpark/sudoku"
SUDOKU_ARCHIVE_SHA256 = "38437d3f1f47cbdd12e5cc9d86a7dafe2b23c7ebcb9c785ef881a81865651fb6"
SUDOKU_CSV_SHA256 = "5a77d5392c19c783db68961e000c17fda246f1e362655dc9675f3e7cd4f57bd6"
SUDOKU_TRAIN_ROWS = 100_000
SUDOKU_TEST_ROWS = 1_000
SUDOKU_TRACE_MAX_NEW_TOKENS = 128
SUDOKU_TRACE_PROTOCOL = "compact-trace-81-digit-v1"

SUDOKU_ANSWER_BEGIN = "<|BEGIN_ANSWER|>"
SUDOKU_ANSWER_END = "<|END_ANSWER|>"
_ANSWER_BLOCK_RE = re.compile(
    r"<\|BEGIN_ANSWER\|>\s*(.*?)(?:\s*<\|END_ANSWER\|>|\Z)", re.DOTALL
)
_LEGACY_FINAL_ANSWER_MARKER_RE = re.compile(
    r"(?im)^\s*(?:####|final\s+answer\s*:)\s*"
)
_COMPACT_SOLUTION_RE = re.compile(r"(?<![0-9])([1-9]{81})(?![0-9])")


def locate_sudoku9_answer(text: str, *, enable_reasoning: bool) -> AnswerRegion:
    region = locate_digit_answer(
        text,
        expected_length=81,
        # A submitted 81-cell grid may still contain 0 placeholders.  It must
        # receive zero official exact-match credit, but remains observable for
        # Blank-cell Accuracy, clue preservation, and completion diagnostics.
        allowed_digits="0123456789",
        marker_pairs=(
            (SUDOKU_ANSWER_BEGIN, SUDOKU_ANSWER_END),
            ("<answer>", "</answer>"),
        ),
        minimum_partial_length=81,
        marker_minimum_partial_length=1,
    )
    if region.detected or region.method != "not_found":
        return region

    # Compatibility for checkpoints that label each final row instead of
    # emitting one compact string. A final-answer cue is required so an
    # initial puzzle restatement cannot become the submitted solution.
    rows: list[tuple[int, int, str]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        row_match = re.search(
            r"(?i)\brow\s*(?:[1-9])?\s*[:=-]?\s*"
            r"([1-9](?:[ \t,|/-]*[1-9]){8})\s*$",
            line.rstrip("\r\n"),
        )
        if row_match:
            rows.append(
                (
                    offset + row_match.start(1),
                    offset + row_match.end(1),
                    "".join(re.findall(r"[1-9]", row_match.group(1))),
                )
            )
        offset += len(line)
    for end_index in range(len(rows), 8, -1):
        block = rows[end_index - 9 : end_index]
        cue = text[max(0, block[0][0] - 300) : block[0][0]].lower()
        if not re.search(r"(?:final\s+grid|final\s+answer|solution\s+is)", cue):
            continue
        return AnswerRegion(
            text="".join(row[2] for row in block),
            start_char=block[0][0],
            end_char=block[-1][1],
            detected=True,
            method="final_nine_row_grid",
        )
    return region


def _reasoning_enabled(configured: bool | None = None) -> bool:
    if configured is not None:
        return bool(configured)
    return os.environ.get("DLLM_BENCH_ENABLE_REASONING", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def format_sudoku_trace_prompt(puzzle_digits: str) -> str:
    """Request a compact answer whose canvas states can map to grid cells."""
    return (
        "Solve the Sudoku puzzle below. The puzzle is an 81-digit row-major "
        "sequence, and 0 marks a blank cell.\n"
        "Return exactly the completed 81 digits in row-major order. Use only "
        "digits 1-9, with no spaces, punctuation, code fences, or explanation.\n\n"
        f"Puzzle:\n{puzzle_digits}"
    )


def format_sudoku_one_shot_prompt(puzzle_digits: str) -> str:
    return (
        "0 represents a blank cell. "
        f"Example input: {SUDOKU_ONE_SHOT_EXAMPLE_PUZZLE} "
        f"Example output: {SUDOKU_ONE_SHOT_EXAMPLE_ANSWER} "
        f"Fill in this Sudoku: {puzzle_digits} "
        "Directly return your 81-digit answer using only 1-9."
    )


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


class Sudoku9Dataset(Dataset):
    name = "sudoku9"

    def __init__(
        self,
        samples: list[Sample] | None = None,
        cache_dir: str | Path | None = None,
        easy_count: int = 50,
        hard_count: int = 50,
        seed: int = 42,
        enable_reasoning: bool | None = None,
    ) -> None:
        self._samples = list(samples) if samples is not None else None
        self._easy_count = int(easy_count)
        self._hard_count = int(hard_count)
        self._seed = int(seed)
        self._enable_reasoning = _reasoning_enabled(enable_reasoning)
        data_root = Path(cache_dir or ensure_data_layout()["datasets"])
        self._archive_path = (
            data_root / "sudoku" / SUDOKU_SOURCE_REVISION / "bryanpark-sudoku.zip"
        )

    def load_samples(self, n: int | None = None) -> list[Sample]:
        if self._samples is not None:
            samples = list(self._samples)
        else:
            official_test = _load_official_test_samples(
                _ensure_official_archive(self._archive_path),
                enable_reasoning=self._enable_reasoning,
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
            "protocol_revision": (
                SUDOKU_REASONING_PROTOCOL_REVISION
                if self._enable_reasoning
                else SUDOKU_PROTOCOL_REVISION
            ),
            "enable_reasoning": self._enable_reasoning,
            "archive_sha256": SUDOKU_ARCHIVE_SHA256,
            "csv_sha256": SUDOKU_CSV_SHA256,
            "test_start": SUDOKU_TRAIN_ROWS,
            "test_rows": SUDOKU_TEST_ROWS,
            "formal_easy_count": self._easy_count,
            "formal_hard_count": self._hard_count,
            "formal_subset_seed": self._seed,
        }

    def scoring_signature(self) -> dict[str, object]:
        return {
            "upstream_data": "bryanpark/sudoku; Ye et al. test split",
            "metric_owner": "HKUNLP/diffusion-vs-ar",
            "upstream_revision": "6743981a4ba42062c95279e590f3991de3985581",
            "metric": "complete reference-sequence accuracy",
            "official_scorer_available": True,
            "answer_extraction": "final-submission-adapter-v1",
            "enable_reasoning": self._enable_reasoning,
        }

    def score(self, sample: Sample, output_text: str) -> ScoreResult:
        ref: SudokuReference = sample.reference
        given_count = sum(
            value != 0 for row in ref.puzzle for value in row
        )
        prediction = output_text.strip()
        target = _grid_to_digits(ref.solution)
        located_region = locate_sudoku9_answer(
            output_text, enable_reasoning=self._enable_reasoning
        )
        _, marker_present, marker_complete = _extract_answer_region(
            output_text
        )
        located_marker = located_region.method.startswith("answer_marker")
        if located_marker:
            marker_present = True
            marker_complete = bool(located_region.marker_complete)
        direct_answer = (
            prediction if re.fullmatch(r"[1-9]{81}", prediction) else None
        )
        submitted_answer = located_region.text if located_region.detected else ""
        if self._enable_reasoning:
            marker_complete = bool(located_region.marker_complete)
        source_payload = (
            output_text[
                located_region.start_char : located_region.end_char
            ].strip()
            if located_region.detected
            else ""
        )
        strict_format_valid = bool(
            re.fullmatch(r"[1-9]{81}", source_payload)
        )
        strict_reference_exact = 1.0 if submitted_answer == target else 0.0
        if located_region.detected:
            grid, _ = extract_final_grid(located_region.text)
        else:
            # No final submission region means no answer. Reasoning grids,
            # copied puzzles, and rejected drafts are never scored as payload.
            grid = None

        if grid is None:
            result = ScoreResult(
                primary_score=0.0,
                aux={
                    "official_score": 0.0,
                    "strict_reference_exact_match": strict_reference_exact,
                    "strict_81_digit_format_rate": 0.0,
                    "direct_answer_instruction_following_rate": 0.0,
                    "exact_solve_rate": 0.0,
                    "blank_cell_accuracy": 0.0,
                    "cell_accuracy": 0.0,
                    "given_preservation_rate": 0.0,
                    "given_mismatch_count": float(given_count),
                    "constraint_satisfaction_rate": 0.0,
                    "completion_rate": 0.0,
                    "conflict_rate": 1.0,
                    "constraint_valid": 0.0,
                    "legal_completion": 0.0,
                    "reference_exact_match": 0.0,
                    "answer_marker_present": float(marker_present),
                    "answer_marker_complete_rate": float(marker_complete),
                },
                valid=False,
                complete=False,
            )
            result.aux.update(position_aux(located_region, output_text))
            result.aux.update(scored_payload_aux(""))
            return result

        exact = ye_sudoku_sequence_accuracy(_grid_to_digits(grid), target)
        partial_credit = blank_cell_accuracy(grid, ref.puzzle, ref.solution)
        satisfaction = constraint_satisfaction_rate(grid)
        completed = completion_rate(grid)
        clue_rate = given_preservation_rate(grid, ref.puzzle)
        clue_mismatches = sum(
            ref.puzzle[r][c] != 0 and grid[r][c] != ref.puzzle[r][c]
            for r in range(9)
            for c in range(9)
        )
        constraint_valid = completed == 1.0 and satisfaction == 1.0
        legal_completion = is_valid_solution(grid, ref.puzzle)
        result = ScoreResult(
            primary_score=exact,
            aux={
                "official_score": exact,
                "strict_reference_exact_match": strict_reference_exact,
                "strict_81_digit_format_rate": float(strict_format_valid),
                "direct_answer_instruction_following_rate": float(
                    marker_complete and strict_format_valid
                    if self._enable_reasoning
                    else direct_answer is not None
                ),
                "exact_solve_rate": exact,
                "blank_cell_accuracy": partial_credit,
                "cell_accuracy": cell_accuracy(grid, ref.solution),
                "given_preservation_rate": clue_rate,
                "given_mismatch_count": float(clue_mismatches),
                "constraint_satisfaction_rate": satisfaction,
                "completion_rate": completed,
                "conflict_rate": 1.0 - satisfaction,
                "constraint_valid": float(constraint_valid),
                "legal_completion": float(legal_completion),
                "reference_exact_match": exact,
                "answer_marker_present": float(marker_present),
                "answer_marker_complete_rate": float(marker_complete),
            },
            valid=True,
            complete=completed == 1.0,
        )
        result.aux.update(position_aux(located_region, output_text))
        result.aux.update(scored_payload_aux(_grid_to_digits(grid)))
        return result

    def score_generation(
        self, sample: Sample, generation: GenerationResult
    ) -> ScoreResult:
        result = self.score(sample, generation.output_text)
        region = locate_sudoku9_answer(
            generation.output_text, enable_reasoning=self._enable_reasoning
        )
        result.aux.update(trace_position_aux(region, generation.trace))
        result.aux.update(self.trace_aux_metrics(sample, generation.trace))
        return result

    def aggregate_records(
        self, samples: list[Sample], results: list[ScoreResult]
    ) -> dict[str, float]:
        summary = super().aggregate_records(samples, results)
        summary.update(aggregate_answer_position_metrics(results))
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
        corrected = sum(
            result.aux.get("trace_error_then_correct_count", 0.0)
            for result in results
        )
        still_wrong = sum(
            result.aux.get("trace_error_then_still_wrong_count", 0.0)
            for result in results
        )
        opportunities = corrected + still_wrong
        summary["trace_correction_opportunity_count"] = opportunities
        if opportunities:
            summary["trace_correction_success_rate"] = corrected / opportunities
        return summary

    def trace_aux_metrics(self, sample: Sample, trace: list) -> dict[str, float]:
        from ..metrics.sudoku_revision import (
            compute_revision_count,
            correction_outcomes,
            revision_counts_by_stage,
            trace_parseable_step_count,
        )

        ref: SudokuReference = sample.reference
        stages = revision_counts_by_stage(trace, puzzle=ref.puzzle)
        corrected, still_wrong, _ = correction_outcomes(
            trace, ref.solution, puzzle=ref.puzzle
        )
        parseable_steps = trace_parseable_step_count(trace)
        return {
            "trace_revision_count": float(
                compute_revision_count(trace, puzzle=ref.puzzle)
            ),
            "trace_revision_count_early": float(stages["early"]),
            "trace_revision_count_middle": float(stages["middle"]),
            "trace_revision_count_late": float(stages["late"]),
            "trace_parseable_step_count": float(parseable_steps),
            "trace_parseable_step_rate": (
                parseable_steps / len(trace) if trace else 0.0
            ),
            "trace_error_then_correct_count": float(corrected),
            "trace_error_then_still_wrong_count": float(still_wrong),
        }


class Sudoku9OneShotDataset(Sudoku9Dataset):
    """The formal Sudoku9 subset with a fixed, non-overlapping example."""

    name = "sudoku9_1shot"

    def __init__(self, *args, **kwargs) -> None:
        kwargs["enable_reasoning"] = False
        super().__init__(*args, **kwargs)

    def load_samples(self, n: int | None = None) -> list[Sample]:
        samples = super().load_samples(n=n)
        instructed: list[Sample] = []
        for sample in samples:
            puzzle_digits = _grid_to_digits(sample.reference.puzzle)
            if puzzle_digits == SUDOKU_ONE_SHOT_EXAMPLE_PUZZLE:
                raise ValueError("Sudoku9 one-shot example overlaps the formal test subset")
            instructed.append(
                replace(
                    sample,
                    prompt=format_sudoku_one_shot_prompt(puzzle_digits),
                    meta={
                        **sample.meta,
                        "protocol_revision": SUDOKU_ONE_SHOT_PROTOCOL_REVISION,
                        "prompt_protocol": "fixed_one_shot_direct_81_digits",
                        "shot_count": 1,
                    },
                )
            )
        return instructed

    def preparation_signature(self) -> dict[str, object]:
        return {
            **super().preparation_signature(),
            "protocol_revision": SUDOKU_ONE_SHOT_PROTOCOL_REVISION,
            "shot_count": 1,
            "example_puzzle": SUDOKU_ONE_SHOT_EXAMPLE_PUZZLE,
            "example_answer": SUDOKU_ONE_SHOT_EXAMPLE_ANSWER,
        }

    def scoring_signature(self) -> dict[str, object]:
        return {**super().scoring_signature(), "shot_count": 1}


class Sudoku9ThinkingDataset(Sudoku9Dataset):
    """The same 100-row Sudoku9 set with the original reasoning prompt."""

    name = "sudoku9_thinking"

    def __init__(
        self,
        samples: list[Sample] | None = None,
        cache_dir: str | Path | None = None,
        easy_count: int = 50,
        hard_count: int = 50,
        seed: int = 42,
    ) -> None:
        super().__init__(
            samples=samples,
            cache_dir=cache_dir,
            easy_count=easy_count,
            hard_count=hard_count,
            seed=seed,
            enable_reasoning=True,
        )


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


def _build_prompt(
    puzzle_digits: str, enable_reasoning: bool | None = None
) -> str:
    rows = [
        " ".join(puzzle_digits[row * 9 : (row + 1) * 9])
        for row in range(9)
    ]
    if _reasoning_enabled(enable_reasoning):
        return (
            "Solve the following 9x9 Sudoku puzzle. Each displayed row contains "
            "exactly 9 cells, and 0 represents a blank cell. Fill every blank.\n"
            "Puzzle:\n"
            + "\n".join(rows)
            + "\n\nSolve the puzzle and return the completed grid as one row-major "
            "81-digit string using only digits 1-9; do not leave any 0. You may "
            "reason before the final answer if useful. Put only the 81 digits "
            "between the answer markers, exactly as follows:\n"
            f"{SUDOKU_ANSWER_BEGIN}\n"
            "<81 digits>\n"
            f"{SUDOKU_ANSWER_END}"
        )
    return (
        f"Solve this 9x9 Sudoku puzzle: {puzzle_digits}. The puzzle is written "
        "row by row, and 0 represents an empty cell.\n"
        "Every non-zero digit is a fixed clue. Do not change, move, or omit any "
        "fixed clue. Fill only the positions containing 0.\n"
        "The completed grid must contain each digit from 1 to 9 exactly once in "
        "every row, every column, and every 3x3 subgrid.\n"
        "Directly output the COMPLETE 81-character string answer in row-major "
        "order, using only digits 1-9 and nothing else."
    )


def _extract_answer_region(text: str) -> tuple[str, bool, bool]:
    blocks = list(_ANSWER_BLOCK_RE.finditer(text))
    if blocks:
        block = blocks[-1]
        complete = SUDOKU_ANSWER_END in text[block.start() :]
        return block.group(1), True, complete
    legacy = list(_LEGACY_FINAL_ANSWER_MARKER_RE.finditer(text))
    if legacy:
        return text[legacy[-1].end() :], True, False
    return text, False, False


def extract_final_grid(text: str) -> tuple[Grid | None, bool]:
    """Extract the final grid, preferring the explicit answer-marker block.

    This mirrors GSM8K's marker-first/fallback-last-answer convention.  If a
    marker is present, only text after the last marker is considered so digits
    in the model's reasoning cannot be mistaken for the submitted solution.
    Without a marker, the last complete compact solution or 9-row grid is used.
    """
    candidate_text, marker_present, _ = _extract_answer_region(text)
    compact = _COMPACT_SOLUTION_RE.findall(candidate_text)
    if compact:
        digits = compact[-1]
        return (
            [[int(digits[row * 9 + col]) for col in range(9)] for row in range(9)],
            marker_present,
        )
    return parse_grid(candidate_text), marker_present


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
    enable_reasoning: bool | None = None,
) -> list[Sample]:
    reasoning = _reasoning_enabled(enable_reasoning)
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
                        prompt=_build_prompt(
                            puzzle_digits,
                            enable_reasoning=reasoning,
                        ),
                        reference=SudokuReference(puzzle, solution, difficulty),
                        meta={
                            "source": "bryanpark/sudoku",
                            "source_revision": SUDOKU_SOURCE_REVISION,
                            "protocol_revision": (
                                SUDOKU_REASONING_PROTOCOL_REVISION
                                if reasoning
                                else SUDOKU_PROTOCOL_REVISION
                            ),
                            "source_index": train_rows + test_index,
                            "official_split": "test",
                            "source_input_format": "81_digits_zero_is_blank",
                            "prompt_protocol": (
                                "nine_row_grid_reasoning_then_81_digits"
                                if reasoning
                                else "fixed_clues_valid_grid_direct_81_digits"
                            ),
                            "enable_reasoning": reasoning,
                            "official_output_format": "81_solution_digits",
                            "difficulty_rule": "naked_single_rounds_le_5_vs_ge_6",
                            "naked_single_rounds": rounds,
                        },
                    )
                )
    return samples


class SudokuTraceDataset(Sudoku9Dataset):
    """The same frozen rows with a compact protocol for revision analysis."""

    name = "sudoku_trace"

    def load_samples(self, n: int | None = None) -> list[Sample]:
        samples = super().load_samples(n=n)
        instructed: list[Sample] = []
        for sample in samples:
            puzzle_digits = _grid_to_digits(sample.reference.puzzle)
            instructed.append(
                replace(
                    sample,
                    prompt=format_sudoku_trace_prompt(puzzle_digits),
                    meta={
                        **sample.meta,
                        "prompt_protocol": SUDOKU_TRACE_PROTOCOL,
                        "max_new_tokens": SUDOKU_TRACE_MAX_NEW_TOKENS,
                    },
                )
            )
        return instructed

    def preparation_signature(self) -> dict[str, object]:
        return {
            **super().preparation_signature(),
            "prompt_protocol": SUDOKU_TRACE_PROTOCOL,
            "max_new_tokens": SUDOKU_TRACE_MAX_NEW_TOKENS,
        }


# Compatibility import for external callers; canonical matrix name is sudoku9.
SudokuDataset = Sudoku9Dataset
