"""Unified per-sample token-grid visualization, used identically by every
dataset (section 4.6's "每个数据集要统一画图方式" requirement) so a reviewer
sees the same visual language whether they're looking at a GSM8K trace or a
Sudoku trace.

Visual language and color palette are carried over from
``Gemma/DGtest/visual.py`` (*How DiffusionGemma Actually Commits Tokens*'
own trace visualizer) for continuity with that prior art:

- gray:  masked / not yet touched
- brown text on white: visible-but-uncommitted ("noisy") token
- light-green fill + green text: just accepted this frame (first time)
- black text on white: stable (accepted once, unchanged since)
- green -> teal -> blue -> purple -> near-black gradient: revised / re-accepted
  multiple times, colored by cumulative accept count (log scale)
- red outline: position committed *this* frame

Unlike the original DiffusionGemma visualizer (which reads multi-canvas CSVs off disk and compares
several sampler configs side by side in one figure), this operates on a
single in-memory trace for one sample/run — the orchestration across
model/config/dataset combinations happens one level up, in
``report/trace_report.py``.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ..interfaces import PositionState, TraceStep

# --- palette (matches Gemma/DGtest/visual.py) ------------------------------

BG = (247, 247, 247)
PANEL_BG = (255, 255, 255)
BORDER = (185, 185, 185)
TEXT = (25, 25, 25)
MUTED = (95, 95, 95)
MASK_FILL = (238, 238, 238)
CELL_BORDER = (210, 210, 210)
STRIP_BORDER = (120, 120, 120)
CURRENT_MARK = (220, 66, 47)

NOISE_TEXT = (145, 110, 70)
FIRST_ACCEPT_FILL = (196, 239, 205)
FIRST_ACCEPT_TEXT = (30, 120, 52)
FINAL_TEXT = (25, 25, 25)
# The animated grid conveys VISIBLE (uncommitted-but-showing-a-token, e.g.
# DiffusionGemma's renoised positions) through NOISE_TEXT *text* color on a
# white cell — but the static heatmap (plot_token_position_forward_heatmap)
# has no text, only a single fill color per cell, so it needs its own
# background tint for this state instead.
VISIBLE_FILL = (232, 218, 196)

_GRADIENT_STOPS = [
    (145, 225, 110),  # green (first re-accept)
    (72, 203, 170),   # teal
    (58, 132, 224),   # blue
    (104, 72, 196),   # purple
    (26, 16, 48),     # near-black purple (heavily revised)
]


def _lerp(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


def gradient_color_for_count(count: int, max_count: int) -> tuple[int, int, int]:
    """Log-scaled *revision* gradient: `count` is a position's cumulative
    accept count, and this only means anything once a position has been
    accepted more than once (i.e. revised/re-masked-and-re-accepted).

    `count <= 0` (never touched) -> neutral mask fill. `count == 1` (accepted
    exactly once, never revised) -> neutral panel background — deliberately
    NOT the gradient's first stop, so a trace with zero revisions (e.g. the
    mock adapter, or any sampler that never re-masks) renders as neutral
    throughout rather than collapsing to the gradient's single endpoint
    (which is what a naive `count / max_count` normalization would do when
    `max_count == 1`).
    """
    if count <= 0:
        return MASK_FILL
    if count == 1 or max_count <= 1:
        return PANEL_BG
    t = math.log(count) / math.log(max_count)
    t = t**0.86
    stops = _GRADIENT_STOPS
    segment = t * (len(stops) - 1)
    index = min(len(stops) - 2, int(segment))
    return _lerp(stops[index], stops[index + 1], segment - index)


def matplotlib_gradient_cmap():
    """The same green->teal->blue->purple->near-black revision gradient as
    :func:`gradient_color_for_count`, as a matplotlib colormap — used by
    ``report/trace_distribution_viz.py`` so the static charts and the GIF/PNG
    read as one visual system."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "dllm_bench_revision", [tuple(c / 255.0 for c in stop) for stop in _GRADIENT_STOPS], N=256
    )


def count_ticks(max_count: int) -> list[int]:
    if max_count <= 0:
        return [1]
    ticks = [1]
    value = 2
    while value < max_count:
        ticks.append(value)
        value *= 2
    if ticks[-1] != max_count:
        ticks.append(max_count)
    return sorted(set(ticks))


def _load_font(size: int, mono: bool = False) -> ImageFont.ImageFont:
    candidates = (
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
            "C:/Windows/Fonts/consola.ttf",
        ]
        if mono
        else [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _compact(token: str, limit: int = 6) -> str:
    token = str(token).replace("\n", "↵").replace("\r", "").replace("\t", "⇥").replace(" ", "·")
    if token == "":
        return "∅"
    return token if len(token) <= limit else token[: limit - 1] + "…"


def token_grid_geometry(n_positions: int) -> tuple[int, int, int, int]:
    if n_positions <= 256:
        cols = 16
    elif n_positions <= 512:
        cols = 24
    else:
        cols = 32
    rows = max(1, math.ceil(n_positions / cols))
    return cols, rows, 54, 31


def compute_running_accept_counts(trace: list[TraceStep]) -> list[dict[int, int]]:
    """Per-step cumulative accept count per position (position i's count
    increments every time it appears in that step's `committed_positions`,
    whether that's its first acceptance or a later revision)."""
    counts: dict[int, int] = {}
    running: list[dict[int, int]] = []
    for step in trace:
        for position in step.committed_positions:
            counts[position] = counts.get(position, 0) + 1
        running.append(dict(counts))
    return running


def cell_fill_color(
    state: PositionState, count: int, max_count: int, just_committed: bool
) -> tuple[int, int, int]:
    """The single representative background color for one (position, step)
    cell — shared by the animated grid's cell fill (`render_frame`) and the
    static heatmap (`plot_token_position_forward_heatmap`), so the two never
    drift into inconsistent color meanings."""
    if state == PositionState.MASKED:
        return MASK_FILL
    if state == PositionState.VISIBLE:
        return VISIBLE_FILL
    if count <= 1 and just_committed:
        return FIRST_ACCEPT_FILL
    if count <= 1:
        return PANEL_BG
    return gradient_color_for_count(count, max_count)


def _cell_token_text(step: TraceStep, position: int) -> str:
    if step.token_texts is not None and position < len(step.token_texts):
        return step.token_texts[position]
    return str(step.token_ids[position])


def render_frame(
    trace: list[TraceStep],
    step_index: int,
    accept_counts_by_step: list[dict[int, int]],
    max_accept_count: int,
    title: str,
    n_positions: int | None = None,
) -> Image.Image:
    step = trace[step_index]
    n = n_positions or max(len(item.position_states) for item in trace)
    cols, rows, cell_w, cell_h = token_grid_geometry(n)

    prev_committed = set(trace[step_index - 1].committed_positions) if step_index > 0 else set()
    committed = set(step.committed_positions)
    counts = accept_counts_by_step[step_index]

    width = 28 + cols * cell_w
    height = 150 + rows * cell_h
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)

    title_font = _load_font(20)
    meta_font = _load_font(14)
    token_font = _load_font(12, mono=True)
    index_font = _load_font(9, mono=True)

    draw.rounded_rectangle((6, 6, width - 6, height - 6), radius=10, fill=PANEL_BG, outline=BORDER, width=2)
    draw.text((20, 16), title, fill=TEXT, font=title_font)
    entropy_values = list(step.entropy_by_position.values()) if step.entropy_by_position else []
    mean_entropy = sum(entropy_values) / len(entropy_values) if entropy_values else None
    entropy_text = "" if mean_entropy is None else f"{mean_entropy:.4f}"
    draw.text(
        (20, 44),
        f"forward={step.forward_index}   committed_this_step={len(committed)}   mean_entropy={entropy_text}",
        fill=MUTED,
        font=meta_font,
    )

    grid_x, grid_y = 20, 72
    for position in range(n):
        row, col = divmod(position, cols)
        cx0, cy0 = grid_x + col * cell_w, grid_y + row * cell_h
        cx1, cy1 = cx0 + cell_w - 2, cy0 + cell_h - 2

        # Older AR traces grew by one position per forward instead of
        # recording their full final canvas. Treat not-yet-emitted positions
        # as masked so those persisted traces remain visualizable.
        state = (
            step.position_states[position]
            if position < len(step.position_states)
            else PositionState.MASKED
        )
        count = counts.get(position, 0)
        fill = cell_fill_color(state, count, max_accept_count, position in committed)

        if state == PositionState.MASKED:
            text_fill = MUTED
        elif state == PositionState.VISIBLE:
            text_fill = NOISE_TEXT
        elif count <= 1 and position in committed:
            text_fill = FIRST_ACCEPT_TEXT
        elif count <= 1:
            text_fill = FINAL_TEXT
        else:
            text_fill = TEXT

        outline = CURRENT_MARK if position in committed else CELL_BORDER
        outline_width = 3 if position in committed else 1
        draw.rectangle((cx0, cy0, cx1, cy1), fill=fill, outline=outline, width=outline_width)

        if position in prev_committed and position not in committed and count <= 1:
            draw.rectangle((cx0, cy0, cx1, cy1), fill=PANEL_BG, outline=CELL_BORDER, width=1)

        draw.text((cx0 + 2, cy0 + 1), str(position), fill=MUTED, font=index_font)
        if state != PositionState.MASKED:
            shown = _compact(_cell_token_text(step, position))
            bbox = draw.textbbox((0, 0), shown, font=token_font)
            tx = cx0 + max(2, (cell_w - (bbox[2] - bbox[0])) // 2)
            ty = cy0 + max(8, (cell_h - (bbox[3] - bbox[1])) // 2 + 4)
            draw.text((tx, ty), shown, fill=text_fill, font=token_font)

    # bottom strip: gradient by cumulative accept count, red mark = committed this step
    strip_x0, strip_x1 = grid_x, width - 20
    strip_y0 = grid_y + rows * cell_h + 14
    strip_y1 = strip_y0 + 18
    draw.text((strip_x0, strip_y0 - 16), "position history (cumulative accept count)", fill=MUTED, font=meta_font)
    strip_width = max(1, strip_x1 - strip_x0)
    for position in range(n):
        px0 = strip_x0 + round(position * strip_width / n)
        px1 = strip_x0 + round((position + 1) * strip_width / n)
        # gradient_color_for_count already maps count<=1 to a neutral color,
        # so this is correct whether or not the trace has any revisions at all.
        color = gradient_color_for_count(counts.get(position, 0), max_accept_count)
        draw.rectangle((px0, strip_y0, max(px0, px1 - 1), strip_y1), fill=color)
        if position in committed:
            draw.rectangle((px0, strip_y0, max(px0, px1 - 1), strip_y0 + 5), fill=CURRENT_MARK)
    draw.rectangle((strip_x0, strip_y0, strip_x1, strip_y1), outline=STRIP_BORDER, width=1)
    draw.text(
        (strip_x0, strip_y1 + 4),
        "gray=masked | brown=visible | green=first accept | black=stable | teal/blue/purple=revised | red=committed now",
        fill=MUTED,
        font=meta_font,
    )

    return image


def render_token_grid_gif(
    trace: list[TraceStep],
    out_path: str | Path,
    title: str = "",
    fps: float = 2.0,
    final_hold_seconds: float = 2.0,
) -> None:
    if not trace:
        return
    accept_counts_by_step = compute_running_accept_counts(trace)
    max_accept_count = max((max(c.values(), default=0) for c in accept_counts_by_step), default=1) or 1
    n_positions = max(len(step.position_states) for step in trace)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def frame(index: int) -> Image.Image:
        return render_frame(
            trace, index, accept_counts_by_step, max_accept_count, title,
            n_positions=n_positions,
        )

    duration = max(40, int(round(1000 / max(0.1, fps))))
    durations = [duration] * len(trace)
    durations[-1] = max(duration, int(round(final_hold_seconds * 1000)))

    first = frame(0)

    def remaining() -> Iterator[Image.Image]:
        for index in range(1, len(trace)):
            yield frame(index)

    first.save(
        out_path,
        save_all=True,
        append_images=remaining(),
        duration=durations,
        loop=0,
        optimize=False,
        disposal=2,
    )


def render_token_grid_final_png(trace: list[TraceStep], out_path: str | Path, title: str = "") -> None:
    if not trace:
        return
    accept_counts_by_step = compute_running_accept_counts(trace)
    max_accept_count = max((max(c.values(), default=0) for c in accept_counts_by_step), default=1) or 1
    n_positions = max(len(step.position_states) for step in trace)
    image = render_frame(
        trace, len(trace) - 1, accept_counts_by_step, max_accept_count, title,
        n_positions=n_positions,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def plot_token_position_forward_heatmap(
    trace: list[TraceStep], out_path: str | Path, title: str = ""
) -> None:
    """Section 4.1's "Token Position x Forward 热图" — the literal static
    heatmap (every forward step visible at once: position on x, forward step
    on y), using the exact same per-cell colors as the animated grid
    (`cell_fill_color`) so it reads as the same visual system, just in a
    single flattened image instead of an animation.
    """
    if not trace:
        return

    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    accept_counts_by_step = compute_running_accept_counts(trace)
    max_accept_count = max((max(c.values(), default=0) for c in accept_counts_by_step), default=1) or 1
    n_positions = max(len(step.position_states) for step in trace)
    n_steps = len(trace)

    rgb = np.zeros((n_steps, n_positions, 3), dtype=np.uint8)
    for t, step in enumerate(trace):
        committed = set(step.committed_positions)
        counts = accept_counts_by_step[t]
        for position in range(n_positions):
            state = (
                step.position_states[position]
                if position < len(step.position_states)
                else PositionState.MASKED
            )
            color = cell_fill_color(
                state, counts.get(position, 0), max_accept_count, position in committed
            )
            rgb[t, position] = color

    fig, ax = plt.subplots(figsize=(min(14, 3 + n_positions * 0.03), min(10, 2 + n_steps * 0.18)))
    ax.imshow(rgb, aspect="auto", origin="upper", interpolation="nearest")
    ax.set_xlabel("Token position")
    ax.set_ylabel("Forward step")
    ax.set_title(title or "Token Position x Forward")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
