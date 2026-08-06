import copy

import pytest

from dllm_bench.interfaces import PositionState, TraceStep
from dllm_bench.visual.public.sudoku_trace_viz import (
    BOARD_LEGEND_LINES,
    CellState,
    SudokuCell,
    derive_sudoku_frames,
    derive_sudoku_layout_frames,
    render_sudoku_gif,
    simulate_sudoku_frames,
)

_SOLUTION = [
    [5, 3, 4, 6, 7, 8, 9, 1, 2],
    [6, 7, 2, 1, 9, 5, 3, 4, 8],
    [1, 9, 8, 3, 4, 2, 5, 6, 7],
    [8, 5, 9, 7, 6, 1, 4, 2, 3],
    [4, 2, 6, 8, 5, 3, 7, 9, 1],
    [7, 1, 3, 9, 2, 4, 8, 5, 6],
    [9, 6, 1, 5, 3, 7, 2, 8, 4],
    [2, 8, 7, 4, 1, 9, 6, 3, 5],
    [3, 4, 5, 2, 8, 6, 1, 7, 9],
]


def _puzzle_with_blanks(positions):
    puzzle = copy.deepcopy(_SOLUTION)
    for r, c in positions:
        puzzle[r][c] = 0
    return puzzle


def test_simulate_easy_frames_never_show_wrong_fills():
    puzzle = _puzzle_with_blanks([(0, 0), (4, 4), (8, 8)])
    frames = simulate_sudoku_frames(puzzle, _SOLUTION, seed=1, hard=False)
    assert frames
    for frame in frames:
        assert all(cell.state != CellState.FILL_WRONG for cell in frame)


def test_simulate_hard_frames_include_a_wrong_then_corrected_fill():
    puzzle = _puzzle_with_blanks([(r, c) for r in range(9) for c in range(9) if (r + c) % 3 == 0])
    frames = simulate_sudoku_frames(puzzle, _SOLUTION, seed=2, hard=True)
    assert any(any(cell.state == CellState.FILL_WRONG for cell in frame) for frame in frames)
    # the puzzle must still end up fully and correctly solved
    final = frames[-1]
    for position, cell in enumerate(final):
        row, col = divmod(position, 9)
        if puzzle[row][col] == 0:
            assert cell.state == CellState.FILL_CORRECT
            assert cell.digit == str(_SOLUTION[row][col])


def test_simulate_frames_reveal_givens_before_any_fill():
    puzzle = _puzzle_with_blanks([(0, 0), (1, 1)])
    frames = simulate_sudoku_frames(puzzle, _SOLUTION, seed=3, hard=False, reveal_batch=81)
    # first frame (all givens revealed in one batch) should have zero fills yet
    first_frame = frames[0]
    assert all(cell.state != CellState.FILL_CORRECT for cell in first_frame)
    given_positions = [(r, c) for r in range(9) for c in range(9) if puzzle[r][c] != 0]
    assert all(
        first_frame[r * 9 + c].state == CellState.GIVEN_MATCH for r, c in given_positions
    )


def test_final_frame_is_fully_revealed():
    puzzle = _puzzle_with_blanks([(2, 3), (5, 6)])
    frames = simulate_sudoku_frames(puzzle, _SOLUTION, seed=4)
    assert all(cell.state != CellState.HIDDEN for cell in frames[-1])


def _make_81_position_trace(digits: list[str]) -> list[TraceStep]:
    assert len(digits) == 81
    return [
        TraceStep(
            forward_index=0,
            token_ids=list(range(81)),
            position_states=[PositionState.ACCEPTED] * 81,
            committed_positions=list(range(81)),
            decoded_text="",
            token_texts=digits,
        )
    ]


def test_derive_sudoku_frames_classifies_given_match():
    puzzle = _puzzle_with_blanks([(0, 0)])
    digits = [str(_SOLUTION[r][c]) for r in range(9) for c in range(9)]
    trace = _make_81_position_trace(digits)
    frames = derive_sudoku_frames(trace, puzzle, _SOLUTION)
    assert frames[-1][0] == SudokuCell(CellState.FILL_CORRECT, str(_SOLUTION[0][0]))
    assert frames[-1][1] == SudokuCell(CellState.GIVEN_MATCH, str(_SOLUTION[0][1]))


def test_derive_sudoku_frames_classifies_mismatches():
    puzzle = _puzzle_with_blanks([(0, 0)])
    digits = [str(_SOLUTION[r][c]) for r in range(9) for c in range(9)]
    # corrupt position (0,1), a given cell, and position (0,0), a fill cell
    digits[0] = "9" if _SOLUTION[0][0] != 9 else "8"
    digits[1] = "9" if _SOLUTION[0][1] != 9 else "8"
    trace = _make_81_position_trace(digits)
    frames = derive_sudoku_frames(trace, puzzle, _SOLUTION)
    assert frames[-1][0].state == CellState.FILL_WRONG
    assert frames[-1][1].state == CellState.GIVEN_MISMATCH


def test_derive_sudoku_frames_skips_unparseable_non_cell_canvas():
    puzzle = _puzzle_with_blanks([(0, 0)])
    trace = [
        TraceStep(
            forward_index=0,
            token_ids=[1, 2, 3],
            position_states=[PositionState.ACCEPTED] * 3,
            committed_positions=[0, 1, 2],
            decoded_text="",
        )
    ]
    assert derive_sudoku_frames(trace, puzzle, _SOLUTION) == []


def test_derive_sudoku_frames_decodes_diffusion_canvas_candidates():
    puzzle = _puzzle_with_blanks([(0, 0)])
    correct = "".join(str(value) for row in _SOLUTION for value in row)
    wrong_digit = "9" if correct[0] != "9" else "8"
    wrong = wrong_digit + correct[1:]
    trace = [
        TraceStep(
            forward_index=index,
            token_ids=list(range(256)),
            position_states=[PositionState.VISIBLE] * 256,
            committed_positions=[],
            decoded_text=digits,
        )
        for index, digits in enumerate((wrong, correct))
    ]

    frames = derive_sudoku_frames(trace, puzzle, _SOLUTION)

    assert frames[0][0].state == CellState.FILL_WRONG
    assert frames[1][0].state == CellState.FILL_CORRECT
    assert frames[1][1].state == CellState.GIVEN_MATCH


def test_derive_sudoku_frames_masked_position_is_hidden():
    puzzle = _puzzle_with_blanks([(0, 0)])
    digits = [str(_SOLUTION[r][c]) for r in range(9) for c in range(9)]
    trace = _make_81_position_trace(digits)
    trace[0].position_states[0] = PositionState.MASKED
    frames = derive_sudoku_frames(trace, puzzle, _SOLUTION)
    assert frames[-1][0].state == CellState.HIDDEN


def test_layout_keeps_last_forward_state_and_appends_final_output():
    puzzle = _puzzle_with_blanks([(0, 0)])
    digits = [str(_SOLUTION[row][col]) for row in range(9) for col in range(9)]
    trace = [
        TraceStep(
            forward_index=0,
            token_ids=list(range(81)),
            position_states=[PositionState.ACCEPTED] * 81,
            committed_positions=list(range(81)),
            decoded_text="".join(digits),
            token_texts=digits,
        ),
        TraceStep(
            forward_index=1,
            token_ids=list(range(81)),
            position_states=[PositionState.VISIBLE]
            + [PositionState.ACCEPTED] * 80,
            committed_positions=list(range(1, 81)),
            decoded_text="".join(digits),
            token_texts=digits,
        ),
    ]

    frames, steps = derive_sudoku_layout_frames(
        trace,
        puzzle,
        _SOLUTION,
        final_valid_length=81,
    )

    assert steps == [0, 1, "final output"]
    assert frames[-2].board[0].state == CellState.HIDDEN
    assert frames[-1].board[0].state == CellState.FILL_CORRECT


def test_render_sudoku_gif_writes_file_and_legend_fits_canvas(tmp_path):
    puzzle = _puzzle_with_blanks([(0, 0), (4, 4)])
    frames = simulate_sudoku_frames(puzzle, _SOLUTION, seed=5)
    out_path = tmp_path / "sudoku.gif"
    render_sudoku_gif(frames, puzzle, _SOLUTION, out_path)
    assert out_path.exists()

    from PIL import Image, ImageDraw

    with Image.open(out_path) as img:
        assert getattr(img, "n_frames", 1) == len(frames)
        # Each of the two legend rows must fit while the board occupies most
        # of the canvas width.
        from dllm_bench.visual.public.sudoku_trace_viz import _load_font

        meta_font = _load_font(13)
        draw = ImageDraw.Draw(img)
        assert all(
            draw.textbbox((0, 0), line, font=meta_font)[2] <= img.size[0] - 48
            for line in BOARD_LEGEND_LINES
        )
        assert (9 * 58) / img.size[0] > 0.85


def test_render_sudoku_gif_is_a_no_op_on_empty_frames(tmp_path):
    out_path = tmp_path / "empty.gif"
    render_sudoku_gif([], _SOLUTION, _SOLUTION, out_path)
    assert not out_path.exists()
