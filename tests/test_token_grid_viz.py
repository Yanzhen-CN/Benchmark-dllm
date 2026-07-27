from pathlib import Path

import pytest

from dllm_bench.interfaces import GenerationRequest, PositionState, TraceStep
from dllm_bench.models.mock import MockDiffusionAdapter
from dllm_bench.report.token_grid_viz import (
    FIRST_ACCEPT_FILL,
    MASK_FILL,
    PANEL_BG,
    VISIBLE_FILL,
    cell_fill_color,
    compute_running_accept_counts,
    gradient_color_for_count,
    plot_token_position_forward_heatmap,
    render_frame,
    render_token_grid_final_png,
    render_token_grid_gif,
    token_grid_geometry,
)
from dllm_bench.report.trace_distribution_viz import plot_commit_speed, plot_position_vs_first_commit


def test_gradient_color_never_touched_is_mask_fill():
    assert gradient_color_for_count(0, max_count=5) == MASK_FILL


def test_gradient_color_accepted_once_is_neutral_even_when_that_is_the_max():
    # this is the bug this function exists to avoid: max_count == 1 must not
    # collapse every accepted-once position to the gradient's darkest stop.
    assert gradient_color_for_count(1, max_count=1) == PANEL_BG


def test_gradient_color_accepted_once_is_neutral_when_others_were_revised():
    assert gradient_color_for_count(1, max_count=10) == PANEL_BG


def test_gradient_color_revised_twice_differs_from_neutral_and_from_max():
    low = gradient_color_for_count(2, max_count=10)
    high = gradient_color_for_count(10, max_count=10)
    assert low != PANEL_BG
    assert low != high


def test_token_grid_geometry_thresholds():
    assert token_grid_geometry(200)[0] == 16
    assert token_grid_geometry(400)[0] == 24
    assert token_grid_geometry(1000)[0] == 32


def _make_trace(n_positions: int, steps: int) -> list[TraceStep]:
    adapter = MockDiffusionAdapter(steps=steps)
    prompt = " ".join(f"tok{i}" for i in range(n_positions))
    request = GenerationRequest(prompt=prompt, max_new_tokens=n_positions, seed=1)
    return adapter.generate(request).trace


def test_compute_running_accept_counts_is_monotonic_non_decreasing():
    trace = _make_trace(n_positions=6, steps=4)
    running = compute_running_accept_counts(trace)
    for position in range(6):
        counts = [step_counts.get(position, 0) for step_counts in running]
        assert counts == sorted(counts)


def test_compute_running_accept_counts_reflects_revision():
    # fabricate a trace where position 0 is committed twice (a revision).
    trace = [
        TraceStep(
            forward_index=0,
            token_ids=[1, -1],
            position_states=[PositionState.ACCEPTED, PositionState.MASKED],
            committed_positions=[0],
            decoded_text="a [MASK]",
        ),
        TraceStep(
            forward_index=1,
            token_ids=[2, 3],
            position_states=[PositionState.ACCEPTED, PositionState.ACCEPTED],
            committed_positions=[0, 1],  # position 0 revised
            decoded_text="b c",
        ),
    ]
    running = compute_running_accept_counts(trace)
    assert running[-1][0] == 2
    assert running[-1][1] == 1


def test_render_token_grid_final_png_writes_file(tmp_path):
    trace = _make_trace(n_positions=5, steps=4)
    out_path = tmp_path / "final.png"
    render_token_grid_final_png(trace, out_path, title="test")
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_render_token_grid_gif_writes_animated_file(tmp_path):
    trace = _make_trace(n_positions=5, steps=4)
    out_path = tmp_path / "trace.gif"
    render_token_grid_gif(trace, out_path, title="test", fps=4)
    assert out_path.exists()

    from PIL import Image

    with Image.open(out_path) as img:
        assert getattr(img, "n_frames", 1) == len(trace)


def test_render_empty_trace_is_a_no_op(tmp_path):
    out_path = tmp_path / "empty.gif"
    render_token_grid_gif([], out_path)
    assert not out_path.exists()


def test_render_frame_marks_just_committed_position(tmp_path):
    trace = _make_trace(n_positions=5, steps=4)
    running = compute_running_accept_counts(trace)
    image = render_frame(trace, 0, running, max_accept_count=1, title="t")
    assert image.size[0] > 0 and image.size[1] > 0


def test_plot_position_vs_first_commit_writes_file(tmp_path):
    trace = _make_trace(n_positions=6, steps=4)
    out_path = tmp_path / "pos.png"
    plot_position_vs_first_commit(trace, out_path, title="t")
    assert out_path.exists()


def test_plot_commit_speed_writes_file(tmp_path):
    trace = _make_trace(n_positions=6, steps=4)
    out_path = tmp_path / "speed.png"
    plot_commit_speed(trace, out_path, title="t")
    assert out_path.exists()


def test_plot_functions_are_no_ops_on_empty_trace(tmp_path):
    plot_position_vs_first_commit([], tmp_path / "a.png")
    plot_commit_speed([], tmp_path / "b.png")
    assert not (tmp_path / "a.png").exists()
    assert not (tmp_path / "b.png").exists()


# ---------------------------------------------------------------------------
# cell_fill_color / static heatmap (design doc 4.1's "Token Position x
# Forward 热图")
# ---------------------------------------------------------------------------

def test_cell_fill_color_masked_is_mask_fill():
    assert cell_fill_color(PositionState.MASKED, count=0, max_count=1, just_committed=False) == MASK_FILL


def test_cell_fill_color_visible_is_visible_fill():
    assert cell_fill_color(PositionState.VISIBLE, count=0, max_count=5, just_committed=False) == VISIBLE_FILL


def test_cell_fill_color_just_committed_is_first_accept_fill():
    color = cell_fill_color(PositionState.ACCEPTED, count=1, max_count=1, just_committed=True)
    assert color == FIRST_ACCEPT_FILL


def test_cell_fill_color_stable_not_just_committed_is_neutral():
    color = cell_fill_color(PositionState.ACCEPTED, count=1, max_count=1, just_committed=False)
    assert color == PANEL_BG


def test_cell_fill_color_revised_uses_the_gradient():
    color = cell_fill_color(PositionState.ACCEPTED, count=3, max_count=5, just_committed=False)
    assert color == gradient_color_for_count(3, 5)


def test_render_frame_actually_paints_cells_with_cell_fill_color(tmp_path):
    """Pixel-level check (not just re-calling the same function): the
    animated grid's rendered cell color must match what `cell_fill_color`
    predicts, since `render_frame` is supposed to use it directly — this is
    what keeps the GIF and the static heatmap from silently drifting apart.
    """
    trace = _make_trace(n_positions=6, steps=4)
    running = compute_running_accept_counts(trace)
    max_count = max((max(c.values(), default=0) for c in running), default=1) or 1
    step_index = 1
    step = trace[step_index]
    committed = set(step.committed_positions)

    image = render_frame(trace, step_index, running, max_count, title="t")
    cols, _rows, cell_w, cell_h = token_grid_geometry(len(step.position_states))
    grid_x, grid_y = 20, 72

    for position in range(len(step.position_states)):
        row, col = divmod(position, cols)
        # bottom-right corner, well inside even the thicker 3px
        # just-committed outline and clear of both the position-index label
        # (top-left) and the centered token glyph.
        cx = grid_x + col * cell_w + (cell_w - 6)
        cy = grid_y + row * cell_h + (cell_h - 6)
        expected = cell_fill_color(
            step.position_states[position], running[step_index].get(position, 0), max_count, position in committed
        )
        assert image.getpixel((cx, cy)) == expected, f"position {position}"


def test_plot_token_position_forward_heatmap_writes_file(tmp_path):
    trace = _make_trace(n_positions=6, steps=4)
    out_path = tmp_path / "heatmap.png"
    plot_token_position_forward_heatmap(trace, out_path, title="t")
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_token_position_forward_heatmap_is_a_no_op_on_empty_trace(tmp_path):
    out_path = tmp_path / "heatmap.png"
    plot_token_position_forward_heatmap([], out_path)
    assert not out_path.exists()
