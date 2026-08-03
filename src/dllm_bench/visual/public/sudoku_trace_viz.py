"""Sudoku's extra trace visualization on top of the unified set in
``trace_report.py``: an animated 9x9 grid walking through the solve,
specifically useful for **Hard** puzzles (naked-single elimination alone
doesn't solve them -- some cells need a trial that later gets corrected).

Per-cell coloring:

- not yet decided: gray
- a *given* (prompt-supplied clue) position, echoed back matching the
  original puzzle: black text, white background
- a given position the model got wrong (echoed a different digit than the
  puzzle actually showed): red text on white -- a correctness bug in the
  transcription, not the puzzle
- a *to-fill* (originally blank) position, filled with the correct digit:
  green background
- a to-fill position filled with an incorrect digit: red background

Two ways to get the per-step ``SudokuCell`` frames:

- :func:`derive_sudoku_frames` -- the real path, from an actual model's
  trace. It uses direct row-major mapping for an 81-position trace, or
  extracts an exact 81-digit candidate from each decoded canvas for
  tokenized models such as DiffusionGemma.
- :func:`simulate_sudoku_frames` -- a self-contained demo/test fixture (no
  model involved) that fabricates a plausible solve, including deliberate
  wrong-then-corrected fills when ``hard=True`` -- this is what exercises the
  trial-and-error visual before any real Sudoku-capable adapter exists, the
  same role :mod:`dllm_bench.models.mock` plays for the rest of the
  framework.
"""

from __future__ import annotations

import random
import math
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ...interfaces import PositionState, TraceStep
from ...metrics.sudoku_revision import trace_step_grid
from .token_grid_viz import BG, MUTED, PANEL_BG, TEXT, _load_font

Grid = list[list[int]]
GridInput = Grid | str


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


class TokenState(str, Enum):
    NOISE = "noise"
    FRAMEWORK = "framework"
    ANSWER_CORRECT = "answer_correct"
    ANSWER_WRONG = "answer_wrong"


@dataclass(frozen=True)
class TokenCell:
    state: TokenState
    text: str = ""


TokenFrame = list[TokenCell]


@dataclass(frozen=True)
class SudokuLayoutFrame:
    prefix: TokenFrame
    board: Frame
    suffix: TokenFrame

_CELL_BG = {
    CellState.HIDDEN: (222, 222, 222),
    CellState.GIVEN_MATCH: (255, 255, 255),
    CellState.GIVEN_MISMATCH: (255, 255, 255),
    CellState.FILL_CORRECT: (196, 239, 205),
    CellState.FILL_WRONG: (247, 173, 173),
}
_CELL_TEXT = {
    CellState.GIVEN_MATCH: (20, 20, 20),
    CellState.GIVEN_MISMATCH: (170, 30, 30),
    CellState.FILL_CORRECT: (30, 120, 52),
    CellState.FILL_WRONG: (150, 30, 30),
}
_TOKEN_BG = {
    TokenState.NOISE: (205, 210, 214),
    TokenState.FRAMEWORK: (28, 31, 35),
    TokenState.ANSWER_CORRECT: (166, 226, 180),
    TokenState.ANSWER_WRONG: (241, 157, 157),
}
_TOKEN_TEXT = {
    TokenState.NOISE: (125, 132, 138),
    TokenState.FRAMEWORK: (250, 250, 250),
    TokenState.ANSWER_CORRECT: (23, 102, 48),
    TokenState.ANSWER_WRONG: (135, 24, 24),
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


def _normalize_grid(value: GridInput) -> Grid:
    if isinstance(value, str):
        size = math.isqrt(len(value))
        if size * size != len(value):
            return []
        return [
            [int(value[row * size + col]) for col in range(size)]
            for row in range(size)
        ]
    return [list(row) for row in value]


def derive_sudoku_frames(trace: list[TraceStep], puzzle: GridInput, solution: GridInput) -> list[Frame]:
    """Build row-major frames from token-aligned or decoded-canvas traces."""
    if not trace:
        return []
    puzzle = _normalize_grid(puzzle)
    solution = _normalize_grid(solution)
    if not puzzle or len(puzzle) != len(solution):
        return []
    size = len(puzzle)
    cell_count = size * size
    n = len(trace[-1].position_states)
    if n != cell_count:
        decoded_grids = [trace_step_grid(step, size=size) for step in trace]
        frames: list[Frame] = []
        for grid in decoded_grids:
            if grid is None:
                continue
            frames.append(
                [
                    _classify(
                        PositionState.ACCEPTED,
                        str(grid[row][col]),
                        puzzle[row][col] != 0,
                        puzzle[row][col],
                        solution[row][col],
                    )
                    for row in range(size)
                    for col in range(size)
                ]
            )
        return frames

    frames: list[Frame] = []
    for step in trace:
        frame: Frame = []
        for position in range(cell_count):
            row, col = divmod(position, size)
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


def _answer_token_expectations(
    trace: list[TraceStep], solution: Grid, limit: int
) -> dict[int, tuple[int, int, str]]:
    """Map the strongest final digit run onto expected Sudoku answer digits."""
    texts = trace[-1].token_texts or []
    runs: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    for position, raw_text in enumerate(texts[:limit]):
        text = raw_text.strip()
        if text and text.isdigit():
            current.append((position, text))
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    if not runs:
        return {}

    expected = "".join(str(value) for row in solution for value in row)
    eligible = [
        run
        for run in runs
        if sum(len(text) for _, text in run) >= max(4, len(expected) // 2)
    ]
    if not eligible:
        return {}
    run = max(
        eligible,
        key=lambda value: (sum(len(text) for _, text in value), value[-1][0]),
    )
    stream_length = sum(len(text) for _, text in run)
    answer_length = min(len(expected), stream_length)
    stream_start = 0
    stream_end = stream_start + answer_length

    mapping: dict[int, tuple[int, int, str]] = {}
    cursor = 0
    expected_cursor = 0
    for position, text in run:
        left = max(0, stream_start - cursor)
        right = min(len(text), stream_end - cursor)
        if left < right:
            width = right - left
            mapping[position] = (
                left,
                right,
                expected[expected_cursor : expected_cursor + width],
            )
            expected_cursor += width
        cursor += len(text)
    return mapping


def derive_sudoku_token_frames(
    trace: list[TraceStep], solution: GridInput, final_valid_length: int
) -> tuple[list[TokenFrame], list[int | None]]:
    """Build a cropped output-canvas animation with Sudoku-aware colors."""
    solution = _normalize_grid(solution)
    if not trace or not solution:
        return [], []
    cell_count = len(solution) * len(solution)
    observed_length = max(len(step.position_states) for step in trace)
    valid_length = min(observed_length, final_valid_length or observed_length)
    answer_map = _answer_token_expectations(trace, solution, valid_length)
    if answer_map:
        display_length = max(answer_map) + 1
    else:
        overhead = max(8, min(24, cell_count // 4))
        display_length = min(valid_length, cell_count + overhead)
    display_length = max(1, display_length)

    noise_frame = [TokenCell(TokenState.NOISE) for _ in range(display_length)]
    frames: list[TokenFrame] = [noise_frame]
    forward_steps: list[int | None] = [None]
    for step in trace:
        frame: TokenFrame = []
        texts = step.token_texts or []
        for position in range(display_length):
            accepted = (
                position < len(step.position_states)
                and step.position_states[position] == PositionState.ACCEPTED
            )
            if not accepted:
                frame.append(TokenCell(TokenState.NOISE))
                continue
            text = texts[position].strip() if position < len(texts) else ""
            expectation = answer_map.get(position)
            if expectation is None:
                frame.append(TokenCell(TokenState.FRAMEWORK, text))
                continue
            left, right, expected = expectation
            actual = text[left:right] if len(text) >= right else text
            state = (
                TokenState.ANSWER_CORRECT
                if actual == expected
                else TokenState.ANSWER_WRONG
            )
            frame.append(TokenCell(state, text))
        frames.append(frame)
        forward_steps.append(int(step.forward_index))
    return frames, forward_steps


def derive_sudoku_layout_frames(
    trace: list[TraceStep],
    puzzle: GridInput,
    solution: GridInput,
    final_valid_length: int,
) -> tuple[list[SudokuLayoutFrame], list[int | str | None]]:
    """Locate the final Sudoku span, then replay that fixed layout from trace."""
    puzzle = _normalize_grid(puzzle)
    solution = _normalize_grid(solution)
    if not trace or not puzzle or len(puzzle) != len(solution):
        return [], []
    size = len(puzzle)
    cell_count = size * size
    observed_length = max(len(step.position_states) for step in trace)
    valid_length = min(observed_length, final_valid_length or observed_length)
    answer_map = _answer_token_expectations(trace, solution, valid_length)
    if not answer_map:
        return [], []

    cell_sources: list[tuple[int, int]] = []
    for position in sorted(answer_map):
        left, right, _ = answer_map[position]
        cell_sources.extend((position, offset) for offset in range(left, right))
    if len(cell_sources) != cell_count:
        return [], []

    answer_start = min(answer_map)
    answer_end = max(answer_map) + 1
    prefix_positions = list(range(answer_start))
    suffix_positions = list(range(answer_end, valid_length))

    def token_cell(
        step: TraceStep | None,
        position: int,
        *,
        final_output: bool = False,
    ) -> TokenCell:
        if (
            step is None
            or position >= len(step.position_states)
            or (
                not final_output
                and step.position_states[position] != PositionState.ACCEPTED
            )
        ):
            return TokenCell(TokenState.NOISE)
        texts = step.token_texts or []
        text = texts[position].strip() if position < len(texts) else ""
        return TokenCell(TokenState.FRAMEWORK, text)

    def board_cell(
        step: TraceStep | None,
        cell_index: int,
        source: tuple[int, int],
        *,
        final_output: bool = False,
    ) -> SudokuCell:
        position, offset = source
        if (
            step is None
            or position >= len(step.position_states)
            or (
                not final_output
                and step.position_states[position] != PositionState.ACCEPTED
            )
        ):
            return SudokuCell(CellState.HIDDEN)
        texts = step.token_texts or []
        text = texts[position].strip() if position < len(texts) else ""
        if offset >= len(text) or text[offset] not in "123456789":
            return SudokuCell(CellState.FILL_WRONG, text[offset : offset + 1] or "?")
        digit = text[offset]
        row, col = divmod(cell_index, size)
        if puzzle[row][col] != 0:
            return SudokuCell(
                CellState.GIVEN_MATCH
                if digit == str(puzzle[row][col])
                else CellState.GIVEN_MISMATCH,
                digit,
            )
        expected = str(solution[row][col])
        return SudokuCell(
            CellState.FILL_CORRECT if digit == expected else CellState.FILL_WRONG,
            digit,
        )

    frames: list[SudokuLayoutFrame] = []
    forward_steps: list[int | str | None] = []
    for step in trace:
        frames.append(
            SudokuLayoutFrame(
                prefix=[token_cell(step, position) for position in prefix_positions],
                board=[
                    board_cell(step, index, source)
                    for index, source in enumerate(cell_sources)
                ],
                suffix=[token_cell(step, position) for position in suffix_positions],
            )
        )
        forward_steps.append(int(step.forward_index))

    final_step = trace[-1]
    frames.append(
        SudokuLayoutFrame(
            prefix=[
                token_cell(final_step, position, final_output=True)
                for position in prefix_positions
            ],
            board=[
                board_cell(
                    final_step,
                    index,
                    source,
                    final_output=True,
                )
                for index, source in enumerate(cell_sources)
            ],
            suffix=[
                token_cell(final_step, position, final_output=True)
                for position in suffix_positions
            ],
        )
    )
    forward_steps.append("final output")
    return frames, forward_steps


def simulate_sudoku_frames(
    puzzle: Grid, solution: Grid, seed: int = 42, hard: bool = False, reveal_batch: int = 3
) -> list[Frame]:
    """Demo/test fixture: fabricates a plausible solve with no model involved.

    `hard=True` deliberately fills a handful of blanks with a wrong digit
    first, holds it for a couple of frames, then erases and refills
    correctly -- illustrating the trial-and-error a Hard puzzle needs, versus
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


def render_sudoku_token_gif(
    token_frames: list[TokenFrame],
    puzzle: GridInput,
    solution: GridInput,
    out_path: str | Path,
    cell_px: int = 56,
    fps: float = 3.0,
    final_hold_seconds: float = 3.0,
    title: str = "Sudoku solve",
    forward_steps: list[int | None] | None = None,
) -> None:
    if not token_frames:
        return

    puzzle = _normalize_grid(puzzle)
    solution = _normalize_grid(solution)
    if not puzzle or len(puzzle) != len(solution):
        return
    size = len(puzzle)
    if size == 4 and cell_px == 56:
        cell_px = 64
    elif size == 9 and cell_px == 56:
        cell_px = 44

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    margin = 24
    token_count = len(token_frames[0])
    token_columns = size
    token_rows = math.ceil(token_count / token_columns)
    token_cell_px = cell_px
    header_h = 70
    footer_h = 40

    title_font = _load_font(20)
    meta_font = _load_font(13)

    legend_text = (
        "gray=noise | black=accepted framework | "
        "green=correct answer | red=wrong answer"
    )
    legend_width = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), legend_text, font=meta_font)[2]
    token_panel_w = token_columns * token_cell_px
    width = max(
        token_panel_w + margin * 2,
        legend_width + margin * 2,
    )
    content_h = token_rows * token_cell_px
    height = header_h + content_h + footer_h

    def render(frame_index: int) -> Image.Image:
        image = Image.new("RGB", (width, height), BG)
        draw = ImageDraw.Draw(image)
        draw.text((margin, 14), title, fill=TEXT, font=title_font)

        step = (
            forward_steps[frame_index]
            if forward_steps and frame_index < len(forward_steps)
            else None
        )
        phase = "all noise" if step is None else f"forward {step}"
        token_frame = token_frames[min(frame_index, len(token_frames) - 1)]
        progress = sum(token.state != TokenState.NOISE for token in token_frame)
        progress_text = f"accepted {progress}/{token_count} real trace positions"
        draw.text(
            (margin, 42),
            f"frame {frame_index + 1}/{len(token_frames)}   {phase}   {progress_text}",
            fill=MUTED,
            font=meta_font,
        )

        if token_frames:
            token_x0 = (width - token_panel_w) // 2
            token_y0 = header_h
            token_font = _load_font(max(10, token_cell_px // 3), mono=True)
            for position, token in enumerate(token_frame):
                row, col = divmod(position, token_columns)
                x0 = token_x0 + col * token_cell_px
                y0 = token_y0 + row * token_cell_px
                x1, y1 = x0 + token_cell_px - 2, y0 + token_cell_px - 2
                draw.rectangle(
                    (x0, y0, x1, y1),
                    fill=_TOKEN_BG[token.state],
                    outline=(185, 190, 194),
                    width=1,
                )
                shown = (
                    "<n>"
                    if token.state == TokenState.NOISE
                    else token.text.replace("\n", "\\n")[:4]
                )
                bbox = draw.textbbox((0, 0), shown, font=token_font)
                tx = x0 + (token_cell_px - 2 - (bbox[2] - bbox[0])) / 2
                ty = y0 + (token_cell_px - 2 - (bbox[3] - bbox[1])) / 2 - bbox[1]
                draw.text(
                    (tx, ty),
                    shown,
                    fill=_TOKEN_TEXT[token.state],
                    font=token_font,
                )
            draw.text(
                (margin, header_h + token_rows * token_cell_px + 10),
                legend_text,
                fill=MUTED,
                font=meta_font,
            )
            return image

    duration = max(60, int(round(1000 / max(0.1, fps))))
    durations = [duration] * len(token_frames)
    durations[0] = max(duration, 1500)
    if len(durations) > 1:
        durations[1] = max(duration, 900)
    durations[-1] = max(duration, int(round(final_hold_seconds * 1000)))

    first = render(0)

    def remaining() -> Iterator[Image.Image]:
        for index in range(1, len(token_frames)):
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


def render_sudoku_board_gif(
    frames: list[Frame],
    puzzle: GridInput,
    out_path: str | Path,
    *,
    forward_steps: list[int | None] | None = None,
    fps: float = 3.0,
    final_hold_seconds: float = 2.0,
    title: str = "Sudoku trace",
) -> None:
    """Render only reliably aligned Sudoku cells from real trace checkpoints."""
    puzzle = _normalize_grid(puzzle)
    if not frames or not puzzle:
        return
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    size = len(puzzle)
    box_size = math.isqrt(size)
    if box_size * box_size != size:
        box_size = size
    cell_px = 92 if size == 4 else 54
    grid_px = size * cell_px
    margin = 24
    header_h = 74
    footer_h = 42
    width = max(grid_px + margin * 2, 620)
    height = header_h + grid_px + footer_h

    title_font = _load_font(20)
    meta_font = _load_font(13)
    digit_font = _load_font(int(cell_px * 0.48), mono=True)
    noise_font = _load_font(max(13, int(cell_px * 0.25)), mono=True)
    legend = "black=original clue | gray <n>=unaligned/noise | green=correct | red=wrong"

    def render(frame_index: int) -> Image.Image:
        frame = frames[min(frame_index, len(frames) - 1)]
        image = Image.new("RGB", (width, height), BG)
        draw = ImageDraw.Draw(image)
        step = (
            forward_steps[frame_index]
            if forward_steps and frame_index < len(forward_steps)
            else None
        )
        phase = "initial puzzle" if step is None else f"forward {step}"
        aligned = sum(
            cell.state in {CellState.FILL_CORRECT, CellState.FILL_WRONG}
            for cell in frame
        )
        draw.text((margin, 12), title, fill=TEXT, font=title_font)
        draw.text(
            (margin, 42),
            f"frame {frame_index + 1}/{len(frames)}   {phase}   aligned blanks {aligned}",
            fill=MUTED,
            font=meta_font,
        )

        grid_x0 = (width - grid_px) // 2
        grid_y0 = header_h
        for position, cell in enumerate(frame):
            row, col = divmod(position, size)
            x0 = grid_x0 + col * cell_px
            y0 = grid_y0 + row * cell_px
            x1, y1 = x0 + cell_px, y0 + cell_px
            draw.rectangle(
                (x0, y0, x1, y1),
                fill=_CELL_BG[cell.state],
                outline=(190, 194, 198),
                width=1,
            )
            shown = "<n>" if cell.state == CellState.HIDDEN else cell.digit
            font = noise_font if cell.state == CellState.HIDDEN else digit_font
            color = MUTED if cell.state == CellState.HIDDEN else _CELL_TEXT[cell.state]
            bbox = draw.textbbox((0, 0), shown, font=font)
            tx = x0 + (cell_px - (bbox[2] - bbox[0])) / 2
            ty = y0 + (cell_px - (bbox[3] - bbox[1])) / 2 - bbox[1]
            draw.text((tx, ty), shown, fill=color, font=font)

        for index in range(0, size + 1, box_size):
            x = grid_x0 + index * cell_px
            y = grid_y0 + index * cell_px
            draw.line((x, grid_y0, x, grid_y0 + grid_px), fill=TEXT, width=3)
            draw.line((grid_x0, y, grid_x0 + grid_px, y), fill=TEXT, width=3)
        draw.text(
            (margin, header_h + grid_px + 12),
            legend,
            fill=MUTED,
            font=meta_font,
        )
        return image

    duration = max(60, int(round(1000 / max(0.1, fps))))
    durations = [duration] * len(frames)
    durations[0] = max(duration, 1500)
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

def render_sudoku_gif(
    frames: list[Frame],
    puzzle: GridInput,
    solution: GridInput,
    out_path: str | Path,
    cell_px: int = 56,
    fps: float = 3.0,
    final_hold_seconds: float = 3.0,
    title: str = "Sudoku solve",
    forward_steps: list[int | None] | None = None,
) -> None:
    """Backward-compatible board renderer for aligned Sudoku frames."""
    del solution, cell_px
    render_sudoku_board_gif(
        frames,
        puzzle,
        out_path,
        forward_steps=forward_steps,
        fps=fps,
        final_hold_seconds=final_hold_seconds,
        title=title,
    )


def render_sudoku_layout_gif(
    frames: list[SudokuLayoutFrame],
    puzzle: GridInput,
    out_path: str | Path,
    *,
    prefix_context: str = "",
    suffix_context: str = "",
    forward_steps: list[int | str | None] | None = None,
    fps: float = 3.0,
    final_hold_seconds: float = 2.0,
    title: str = "Sudoku generation trace",
) -> None:
    """Render prefix, aligned Sudoku span, and suffix as separate regions."""
    puzzle = _normalize_grid(puzzle)
    if not frames or not puzzle:
        return
    size = len(puzzle)
    box_size = math.isqrt(size)
    if box_size * box_size != size:
        box_size = size
    board_cell_px = 88 if size == 4 else 52
    board_px = size * board_cell_px
    margin = 24
    header_h = 72
    label_h = 20
    context_h = 48
    gap = 18
    footer_h = 40
    width = max(620, board_px + margin * 2)
    height = (
        header_h
        + label_h
        + context_h
        + gap
        + label_h
        + board_px
        + gap
        + label_h
        + context_h
        + footer_h
    )
    title_font = _load_font(20)
    meta_font = _load_font(13)
    board_font = _load_font(int(board_cell_px * 0.48), mono=True)
    noise_font = _load_font(max(12, int(board_cell_px * 0.24)), mono=True)
    context_font = _load_font(15, mono=True)
    legend = (
        "black=correct clue | white/red text=wrong clue | "
        "gray <n>=not accepted/re-noised | green=correct answer | red=wrong answer"
    )

    def draw_context(
        draw: ImageDraw.ImageDraw,
        text: str,
        y0: int,
        empty_label: str,
    ) -> None:
        shown = text.replace("\n", "\\n") if text else empty_label
        draw.rounded_rectangle(
            (margin, y0, width - margin, y0 + context_h),
            radius=6,
            fill=(246, 247, 248),
            outline=(185, 190, 194),
            width=1,
        )
        draw.text(
            (margin + 12, y0 + 14),
            shown,
            fill=TEXT if text else MUTED,
            font=context_font,
        )

    def render(frame_index: int) -> Image.Image:
        frame = frames[min(frame_index, len(frames) - 1)]
        image = Image.new("RGB", (width, height), BG)
        draw = ImageDraw.Draw(image)
        step = (
            forward_steps[frame_index]
            if forward_steps and frame_index < len(forward_steps)
            else None
        )
        phase = (
            step
            if isinstance(step, str)
            else f"forward {step}"
            if step is not None
            else "trace unavailable"
        )
        draw.text((margin, 10), title, fill=TEXT, font=title_font)
        draw.text(
            (margin, 40),
            f"frame {frame_index + 1}/{len(frames)}   {phase}",
            fill=MUTED,
            font=meta_font,
        )

        y = header_h
        draw.text((margin, y), "final visible text before Sudoku", fill=MUTED, font=meta_font)
        y += label_h
        draw_context(draw, prefix_context, y, "<start>")
        y += context_h + gap

        draw.text((margin, y), "Sudoku span located from final output", fill=MUTED, font=meta_font)
        y += label_h
        grid_x0 = (width - board_px) // 2
        for position, cell in enumerate(frame.board):
            row, col = divmod(position, size)
            x0 = grid_x0 + col * board_cell_px
            y0 = y + row * board_cell_px
            x1, y1 = x0 + board_cell_px, y0 + board_cell_px
            draw.rectangle(
                (x0, y0, x1, y1),
                fill=_CELL_BG[cell.state],
                outline=(190, 194, 198),
                width=1,
            )
            shown = "<n>" if cell.state == CellState.HIDDEN else cell.digit
            font = noise_font if cell.state == CellState.HIDDEN else board_font
            color = MUTED if cell.state == CellState.HIDDEN else _CELL_TEXT[cell.state]
            bbox = draw.textbbox((0, 0), shown, font=font)
            tx = x0 + (board_cell_px - (bbox[2] - bbox[0])) / 2
            ty = y0 + (board_cell_px - (bbox[3] - bbox[1])) / 2 - bbox[1]
            draw.text((tx, ty), shown, fill=color, font=font)
        for index in range(0, size + 1, box_size):
            x = grid_x0 + index * board_cell_px
            line_y = y + index * board_cell_px
            draw.line((x, y, x, y + board_px), fill=TEXT, width=3)
            draw.line((grid_x0, line_y, grid_x0 + board_px, line_y), fill=TEXT, width=3)
        y += board_px + gap

        draw.text((margin, y), "final visible text after Sudoku", fill=MUTED, font=meta_font)
        y += label_h
        draw_context(draw, suffix_context, y, "<end>")
        draw.text((margin, height - footer_h + 12), legend, fill=MUTED, font=meta_font)
        return image

    duration = max(60, int(round(1000 / max(0.1, fps))))
    durations = [duration] * len(frames)
    durations[0] = max(duration, 1500)
    durations[-1] = max(duration, int(round(final_hold_seconds * 1000)))
    first = render(0)

    def remaining() -> Iterator[Image.Image]:
        for index in range(1, len(frames)):
            yield render(index)

    first.save(
        Path(out_path),
        save_all=True,
        append_images=remaining(),
        duration=durations,
        loop=0,
        optimize=False,
        disposal=2,
    )
