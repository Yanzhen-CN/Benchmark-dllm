"""Sudoku's extra trace visualization on top of the unified set in
``trace_report.py``: an animated 9x9 grid walking through the solve,
specifically useful for **Hard** puzzles (naked-single elimination alone
doesn't solve them — some cells need a trial that later gets corrected).

Per-cell coloring:

- not yet decided: gray
- a *given* (prompt-supplied clue) position, echoed back matching the
  original puzzle: black text, white background
- a given position the model got wrong (echoed a different digit than the
  puzzle actually showed): yellow background — a correctness bug in the
  transcription, not the puzzle
- a *to-fill* (originally blank) position, filled with the correct digit:
  green background
- a to-fill position filled with an incorrect digit: red background

Two ways to get the per-step ``SudokuCell`` frames:

- :func:`derive_sudoku_frames` — the real path, from an actual model's
  trace, valid only when that trace has exactly 81 positions mapped
  row-major to grid cells (see ``trace_report._maybe_render_sudoku_gif``,
  which only calls this when that holds).
- :func:`simulate_sudoku_frames` — a self-contained demo/test fixture (no
  model involved) that fabricates a plausible solve, including deliberate
  wrong-then-corrected fills when ``hard=True`` — this is what exercises the
  trial-and-error visual before any real Sudoku-capable adapter exists, the
  same role :mod:`dllm_bench.models.mock` plays for the rest of the
  framework.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ..interfaces import PositionState, TraceStep
from .token_grid_viz import BG, MUTED, PANEL_BG, TEXT, _load_font

Grid = list[list[int]]


class CellState(str, Enum):
    HIDDEN = "hidden"
    GIVEN_MATCH = "given_match"
    GIVEN_MISMATCH = "given_mismatch"
    FILL_CORRECT = "fill_correct"
    FILL_WRONG = "fill_wrong"


@dataclass(frozen=True)
class SudokuCell:
    state: CellState
    digit: str = ""


Frame = list[SudokuCell]

_CELL_BG = {
    CellState.HIDDEN: (222, 222, 222),
    CellState.GIVEN_MATCH: (255, 255, 255),
    CellState.GIVEN_MISMATCH: (250, 210, 60),
    CellState.FILL_CORRECT: (196, 239, 205),
    CellState.FILL_WRONG: (247, 173, 173),
}
_CELL_TEXT = {
    CellState.GIVEN_MATCH: (20, 20, 20),
    CellState.GIVEN_MISMATCH: (110, 80, 0),
    CellState.FILL_CORRECT: (30, 120, 52),
    CellState.FILL_WRONG: (150, 30, 30),
}


def _parse_digit(text: str | None) -> int | None:
    if text is None:
        return None
    stripped = text.strip()
    if len(stripped) == 1 and stripped in "123456789":
        return int(stripped)
    return None


def _classify(
    state: PositionState, digit_text: str | None, is_given: bool, given_digit: int, solution_digit: int
) -> SudokuCell:
    if state != PositionState.ACCEPTED:
        return SudokuCell(CellState.HIDDEN)
    digit = _parse_digit(digit_text)
    shown = digit_text.strip() if digit_text else "?"
    if is_given:
        return SudokuCell(
            CellState.GIVEN_MATCH if digit == given_digit else CellState.GIVEN_MISMATCH, shown
        )
    return SudokuCell(
        CellState.FILL_CORRECT if digit == solution_digit else CellState.FILL_WRONG, shown
    )


def derive_sudoku_frames(trace: list[TraceStep], puzzle: Grid, solution: Grid) -> list[Frame]:
    """Real path: requires `trace`'s canvas to be exactly 81 positions,
    row-major (position i -> row i // 9, col i % 9), with `token_texts`
    populated so each accepted cell's actual digit is known."""
    if not trace:
        return []
    n = len(trace[-1].position_states)
    if n != 81:
        raise ValueError(f"expected an 81-position row-major canvas, got {n}")

    frames: list[Frame] = []
    for step in trace:
        frame: Frame = []
        for position in range(81):
            row, col = divmod(position, 9)
            is_given = puzzle[row][col] != 0
            digit_text = (
                step.token_texts[position]
                if step.token_texts is not None and position < len(step.token_texts)
                else None
            )
            frame.append(
                _classify(
                    step.position_states[position],
                    digit_text,
                    is_given,
                    puzzle[row][col],
                    solution[row][col],
                )
            )
        frames.append(frame)
    return frames


def simulate_sudoku_frames(
    puzzle: Grid, solution: Grid, seed: int = 42, hard: bool = False, reveal_batch: int = 3
) -> list[Frame]:
    """Demo/test fixture: fabricates a plausible solve with no model involved.

    `hard=True` deliberately fills a handful of blanks with a wrong digit
    first, holds it for a couple of frames, then erases and refills
    correctly — illustrating the trial-and-error a Hard puzzle needs, versus
    `hard=False`'s straight-line reveal (Easy puzzles need no backtracking).
    """
    rng = random.Random(seed)
    given_positions = [
        (r, c) for r in range(9) for c in range(9) if puzzle[r][c] != 0
    ]
    fill_positions = [
        (r, c) for r in range(9) for c in range(9) if puzzle[r][c] == 0
    ]
    rng.shuffle(given_positions)
    rng.shuffle(fill_positions)

    state: dict[tuple[int, int], SudokuCell] = {
        (r, c): SudokuCell(CellState.HIDDEN) for r in range(9) for c in range(9)
    }
    frames: list[Frame] = []

    def snapshot() -> Frame:
        return [state[(r, c)] for r in range(9) for c in range(9)]

    def reveal_givens_in_batches() -> Iterator[None]:
        for i in range(0, len(given_positions), reveal_batch):
            for r, c in given_positions[i : i + reveal_batch]:
                state[(r, c)] = SudokuCell(CellState.GIVEN_MATCH, str(puzzle[r][c]))
            yield

    for _ in reveal_givens_in_batches():
        frames.append(snapshot())

    trial_error_budget = max(1, len(fill_positions) // 4) if hard else 0
    for index, (r, c) in enumerate(fill_positions):
        correct_digit = solution[r][c]
        if hard and index < trial_error_budget:
            wrong_digit = ((correct_digit + rng.randint(1, 8) - 1) % 9) + 1
            state[(r, c)] = SudokuCell(CellState.FILL_WRONG, str(wrong_digit))
            frames.append(snapshot())
            state[(r, c)] = SudokuCell(CellState.HIDDEN)
            frames.append(snapshot())
        state[(r, c)] = SudokuCell(CellState.FILL_CORRECT, str(correct_digit))
        if index % reveal_batch == 0 or index == len(fill_positions) - 1:
            frames.append(snapshot())

    return frames


def render_sudoku_gif(
    frames: list[Frame],
    puzzle: Grid,
    solution: Grid,
    out_path: str | Path,
    cell_px: int = 56,
    fps: float = 3.0,
    final_hold_seconds: float = 3.0,
    title: str = "Sudoku solve",
) -> None:
    if not frames:
        return

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    grid_px = cell_px * 9
    margin = 24
    header_h = 70
    footer_h = 40

    title_font = _load_font(20)
    meta_font = _load_font(13)
    digit_font = _load_font(int(cell_px * 0.5), mono=True)

    legend_text = (
        "gray=undecided | black=given (correct) | yellow=given (mismatch) | "
        "green=filled correct | red=filled wrong"
    )
    legend_width = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), legend_text, font=meta_font)[2]
    width = max(grid_px + margin * 2, legend_width + margin * 2)
    height = header_h + grid_px + footer_h

    def render(frame_index: int) -> Image.Image:
        frame = frames[min(frame_index, len(frames) - 1)]
        image = Image.new("RGB", (width, height), BG)
        draw = ImageDraw.Draw(image)
        draw.text((margin, 14), title, fill=TEXT, font=title_font)

        solved = sum(1 for cell in frame if cell.state != CellState.HIDDEN)
        draw.text(
            (margin, 42),
            f"frame {frame_index + 1}/{len(frames)}   revealed {solved}/81",
            fill=MUTED,
            font=meta_font,
        )

        grid_x0, grid_y0 = (width - grid_px) // 2, header_h
        for position, cell in enumerate(frame):
            row, col = divmod(position, 9)
            x0, y0 = grid_x0 + col * cell_px, grid_y0 + row * cell_px
            x1, y1 = x0 + cell_px, y0 + cell_px
            draw.rectangle((x0, y0, x1, y1), fill=_CELL_BG[cell.state], outline=(200, 200, 200), width=1)
            if cell.state != CellState.HIDDEN and cell.digit:
                bbox = draw.textbbox((0, 0), cell.digit, font=digit_font)
                tx = x0 + (cell_px - (bbox[2] - bbox[0])) / 2
                ty = y0 + (cell_px - (bbox[3] - bbox[1])) / 2 - bbox[1]
                draw.text((tx, ty), cell.digit, fill=_CELL_TEXT[cell.state], font=digit_font)

        # 3x3 box borders (thick) over the thin per-cell grid lines.
        for i in range(0, 10, 3):
            draw.line((grid_x0 + i * cell_px, grid_y0, grid_x0 + i * cell_px, grid_y0 + grid_px), fill=TEXT, width=3)
            draw.line((grid_x0, grid_y0 + i * cell_px, grid_x0 + grid_px, grid_y0 + i * cell_px), fill=TEXT, width=3)

        draw.text(
            (margin, header_h + grid_px + 10),
            legend_text,
            fill=MUTED,
            font=meta_font,
        )
        return image

    duration = max(60, int(round(1000 / max(0.1, fps))))
    durations = [duration] * len(frames)
    durations[-1] = max(duration, int(round(final_hold_seconds * 1000)))

    first = render(0)

    def remaining() -> Iterator[Image.Image]:
        for index in range(1, len(frames)):
            yield render(index)

    first.save(
        out_path,
        save_all=True,
        append_images=remaining(),
        duration=durations,
        loop=0,
        optimize=False,
        disposal=2,
    )
