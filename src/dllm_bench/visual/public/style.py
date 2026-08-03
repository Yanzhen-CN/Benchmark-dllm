"""One visual language for all benchmark-owned figures."""

from __future__ import annotations

from typing import Any, Iterable

import matplotlib

BACKGROUND = "#f6f3ec"
PANEL = "#fffdf8"
GRID = "#d8d1c4"
TEXT = "#17201f"
MUTED = "#66706d"
ACCENT = "#0f766e"

VARIANT_COLORS = (
    "#0f766e",
    "#c2410c",
    "#2563a6",
    "#7c5c20",
    "#9f2f52",
    "#4f6f3d",
    "#5b4b8a",
    "#3f6b73",
    "#a15c38",
    "#576574",
)

STATE_COLORS = {
    "masked": "#ddd8ce",
    "visible": "#e0a44c",
    "accepted": "#18786d",
}

PUBLIC_RCPARAMS = {
    "figure.facecolor": BACKGROUND,
    "axes.facecolor": PANEL,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": TEXT,
    "axes.titlecolor": TEXT,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.alpha": 0.48,
    "grid.linewidth": 0.7,
    "text.color": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "legend.frameon": False,
    "font.family": "DejaVu Sans",
    "savefig.facecolor": BACKGROUND,
    "savefig.bbox": "tight",
}


def install_public_style() -> None:
    """Install the benchmark style before any public renderer is imported."""
    matplotlib.rcParams.update(PUBLIC_RCPARAMS)


def variant_color(index: int) -> str:
    return VARIANT_COLORS[index % len(VARIANT_COLORS)]


def _unique_legend_items(
    handles: Iterable[Any], labels: Iterable[str]
) -> tuple[list[Any], list[str]]:
    unique: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        if label and label not in unique:
            unique[label] = handle
    return list(unique.values()), list(unique)


def place_legend(
    ax: Any,
    *,
    outside_threshold: int = 4,
    fontsize: int = 8,
    **kwargs: Any,
) -> Any | None:
    """Keep short legends inside and move crowded comparison labels right."""
    handles, labels = _unique_legend_items(*ax.get_legend_handles_labels())
    if not handles:
        return None
    if len(labels) >= outside_threshold:
        kwargs.setdefault("loc", "upper left")
        kwargs.setdefault("bbox_to_anchor", (1.02, 1.0))
        kwargs.setdefault("borderaxespad", 0.0)
        kwargs.setdefault("ncol", 1)
    else:
        kwargs.setdefault("loc", "best")
    return ax.legend(handles, labels, fontsize=fontsize, **kwargs)


def place_figure_legend(
    fig: Any,
    handles: Iterable[Any],
    labels: Iterable[str],
    *,
    outside_threshold: int = 4,
    fontsize: int = 8,
    **kwargs: Any,
) -> Any | None:
    """Place crowded figure-wide comparison labels in a right-hand column."""
    clean_handles, clean_labels = _unique_legend_items(handles, labels)
    if not clean_handles:
        return None
    if len(clean_labels) >= outside_threshold:
        kwargs.setdefault("loc", "center left")
        kwargs.setdefault("bbox_to_anchor", (1.01, 0.5))
        kwargs.setdefault("ncol", 1)
    else:
        kwargs.setdefault("loc", "upper center")
        kwargs.setdefault("bbox_to_anchor", (0.5, 0.98))
        kwargs.setdefault("ncol", len(clean_labels))
    return fig.legend(
        clean_handles,
        clean_labels,
        fontsize=fontsize,
        frameon=False,
        **kwargs,
    )
