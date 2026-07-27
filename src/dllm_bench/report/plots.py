"""Section 3.4's chart list: Quality-TPS/SPS/EPS/CPS, Score per Unit
Energy, Score per Compute, Best vs Fast, and the two scenario rankings.

Every function takes the same row-dict shape :mod:`tables` produces and
writes a PNG to ``out_path`` — no chart holds state, so callers can generate
any subset of the section-1 chart list independently.
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _label(row: dict[str, Any]) -> str:
    return f"{row['Model']}/{row['Config']}"


def plot_quality_vs_resource(
    rows: list[dict[str, Any]], resource_key: str, out_path: str, title: str | None = None
) -> None:
    """Quality-TPS / Quality-SPS / Quality-EPS / Quality-CPS scatter."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for row in rows:
        x, y = row.get(resource_key), row.get("q")
        if x is None or y is None:
            continue
        ax.scatter(x, y, s=60)
        ax.annotate(_label(row), (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(resource_key)
    ax.set_ylabel("q")
    ax.set_title(title or f"Quality vs {resource_key}")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_score_per_unit(
    rows: list[dict[str, Any]], score_key: str, out_path: str, title: str | None = None
) -> None:
    """Score per Unit Energy / Score per Compute bar chart."""
    labeled = [(_label(r), r.get(score_key)) for r in rows if r.get(score_key) is not None]
    if not labeled:
        return
    labels, values = zip(*labeled)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(range(len(labels)), values)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(score_key)
    ax.set_title(title or score_key)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_best_vs_fast(rows: list[dict[str, Any]], metric_key: str, out_path: str) -> None:
    """Grouped bar comparing `metric_key` (usually q or TPS) between
    each model's Best and Fast config, for models that have both."""
    by_model: dict[str, dict[str, float]] = {}
    for row in rows:
        config = row["Config"].lower()
        if config not in ("best", "fast"):
            continue
        value = row.get(metric_key)
        if value is None:
            continue
        by_model.setdefault(row["Model"], {})[config] = value

    models = [m for m, cfgs in by_model.items() if "best" in cfgs and "fast" in cfgs]
    if not models:
        return

    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = range(len(models))
    width = 0.35
    best_values = [by_model[m]["best"] for m in models]
    fast_values = [by_model[m]["fast"] for m in models]
    ax.bar([i - width / 2 for i in x], best_values, width=width, label="best")
    ax.bar([i + width / 2 for i in x], fast_values, width=width, label="fast")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(metric_key)
    ax.set_title(f"Best vs Fast: {metric_key}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_scenario_ranking(
    converted_rows: list[dict[str, Any]], scenario_key: str, out_path: str
) -> None:
    """`scenario_key` is "Speed-priority" or "Energy-priority" — one bar per
    model/config, sorted descending (section 3.3's two deployment-preference
    rankings, kept separate rather than combined)."""
    labeled = [
        (_label(r), r.get(scenario_key)) for r in converted_rows if r.get(scenario_key) is not None
    ]
    if not labeled:
        return
    labeled.sort(key=lambda pair: pair[1], reverse=True)
    labels, values = zip(*labeled)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(range(len(labels)), values)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(scenario_key)
    ax.set_title(f"{scenario_key} ranking")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
