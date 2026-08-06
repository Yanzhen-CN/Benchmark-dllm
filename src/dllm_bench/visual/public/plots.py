"""Measured-only plots from design-document section 3.4."""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .style import place_legend


def _label(row: dict[str, Any]) -> str:
    n = row.get("N")
    suffix = f" (N={n})" if n is not None else ""
    return f"{row['Model']}/{row['Config']}{suffix}"


def plot_quality_vs_resource(
    rows: list[dict[str, Any]], resource_key: str, out_path: str, title: str | None = None
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    plotted = False
    for row in rows:
        x, y = row.get(resource_key), row.get("q")
        if x is None or y is None:
            continue
        plotted = True
        ax.scatter(x, y, s=60, label=_label(row))
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel(resource_key)
    ax.set_ylabel("q (measured primary score)")
    ax.set_title(title or f"Quality vs {resource_key}")
    place_legend(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_score_per_unit(
    rows: list[dict[str, Any]], score_key: str, out_path: str, title: str | None = None
) -> None:
    labeled = [(_label(row), row.get(score_key)) for row in rows if row.get(score_key) is not None]
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


def plot_p1_vs_p2(rows: list[dict[str, Any]], metric_key: str, out_path: str) -> None:
    by_model: dict[str, dict[str, float]] = {}
    for row in rows:
        config = str(row["Config"]).lower()
        if config not in ("p1", "p2"):
            continue
        value = row.get(metric_key)
        if value is None:
            continue
        by_model.setdefault(row["Model"], {})[config] = value
    models = [model for model, configs in by_model.items() if {"p1", "p2"} <= configs.keys()]
    if not models:
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = list(range(len(models)))
    width = 0.35
    ax.bar([value - width / 2 for value in x], [by_model[m]["p1"] for m in models], width=width, label="P1")
    ax.bar([value + width / 2 for value in x], [by_model[m]["p2"] for m in models], width=width, label="P2")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(metric_key)
    ax.set_title(f"Planned parallelism P1 vs P2: {metric_key}")
    place_legend(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_answer_region_diagnostics(
    rows: list[dict[str, Any]], out_path: str, title: str | None = None
) -> None:
    labeled = [
        (_label(row), row.get("Answer Start Ratio"), row.get("Answer Detect Rate"))
        for row in rows
        if row.get("Answer Start Ratio") is not None or row.get("Answer Detect Rate") is not None
    ]
    if not labeled:
        return
    labels = [item[0] for item in labeled]
    starts = [item[1] if item[1] is not None else float("nan") for item in labeled]
    detects = [item[2] if item[2] is not None else float("nan") for item in labeled]
    x = list(range(len(labels)))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.8), 4.5))
    ax.bar([value - width / 2 for value in x], starts, width=width, label="Answer Start Ratio")
    ax.bar([value + width / 2 for value in x], detects, width=width, label="Answer Region Detected Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Ratio")
    ax.set_title(title or "Answer region diagnostics")
    place_legend(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_accepts_per_forward_vs_tps(rows: list[dict[str, Any]], out_path: str) -> None:
    usable = [
        row
        for row in rows
        if row.get("Accepted tokens/forward") is not None and row.get("Accepted TPS") is not None
    ]
    if not usable:
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for row in usable:
        ax.scatter(row["Accepted tokens/forward"], row["Accepted TPS"], s=60, label=_label(row))
    ax.set_xlabel("Accepted tokens / accepting forward")
    ax.set_ylabel("Accepted-token TPS")
    ax.set_title("Acceptance parallelism vs accepted-token throughput")
    place_legend(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_task4_curve_overlay(
    rows: list[dict[str, Any]],
    curve_key: str,
    out_path: str,
    *,
    xlabel: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = False
    for row in rows:
        curve = (row.get("Trace Summary") or {}).get("curves", {}).get(curve_key, [])
        if not curve:
            continue
        plotted = True
        x = [point["bin_center"] for point in curve]
        y = [point["stats"]["mean"] for point in curve]
        low = [point["stats"]["ci_low"] for point in curve]
        high = [point["stats"]["ci_high"] for point in curve]
        label = _label(row)
        if curve_key in {"certainty", "top1"}:
            observation = (row.get("Trace Summary") or {}).get(
                "certainty_observation", {}
            )
            prefix = "entropy" if curve_key == "certainty" else "top1"
            label += (
                f" [{observation.get(f'{prefix}_scope', 'unknown')}, "
                f"position coverage={observation.get(f'{prefix}_position_coverage', 0):.2f}]"
            )
        ax.plot(x, y, marker="o", markersize=3, label=label)
        ax.fill_between(x, low, high, alpha=0.10)
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    place_legend(ax)
    fig.savefig(out_path)
    plt.close(fig)


def plot_task4_tau_windows(rows: list[dict[str, Any]], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = False
    for row in rows:
        values = (row.get("Trace Summary") or {}).get("commit_order_tau", {})
        if not values:
            continue
        windows = sorted(int(value) for value in values)
        means = [values[str(window)]["mean"] for window in windows]
        lows = [values[str(window)]["ci_low"] for window in windows]
        highs = [values[str(window)]["ci_high"] for window in windows]
        ax.errorbar(
            windows,
            means,
            yerr=[
                [max(0.0, mean - low) for mean, low in zip(means, lows)],
                [max(0.0, high - mean) for mean, high in zip(means, highs)],
            ],
            marker="o",
            capsize=3,
            label=_label(row),
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Window Size (tokens)")
    ax.set_ylabel("Per-sample Mean Kendall tau-b")
    ax.set_title("Commit order by local window")
    place_legend(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_task4_block_local_tau(rows: list[dict[str, Any]], out_path: str) -> None:
    """Compare within-block acceptance order without global block scheduling."""
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    plotted = False
    for row in rows:
        metric = (row.get("Trace Summary") or {}).get(
            "block_local_commit_order_tau", {}
        )
        by_block = metric.get("by_block", {})
        if not by_block:
            continue
        blocks = sorted(int(value) for value in by_block)
        means = [by_block[str(block)]["mean"] for block in blocks]
        lows = [by_block[str(block)]["ci_low"] for block in blocks]
        highs = [by_block[str(block)]["ci_high"] for block in blocks]
        line = ax.plot(blocks, means, marker="o", markersize=3, label=_label(row))[0]
        ax.fill_between(
            blocks,
            lows,
            highs,
            color=line.get_color(),
            alpha=0.12,
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Block index")
    ax.set_ylabel("Kendall tau-b within block")
    ax.set_title("Within-block acceptance order")
    place_legend(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_task4_finalization_share(rows: list[dict[str, Any]], out_path: str) -> None:
    usable = [
        row
        for row in rows
        if (row.get("Trace Summary") or {}).get("finalization_share")
    ]
    if not usable:
        return
    labels = [_label(row) for row in usable]
    x = list(range(len(labels)))
    bottom = [0.0] * len(labels)
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.9), 4.8))
    for stage in ("early", "middle", "late"):
        values = [
            row["Trace Summary"]["finalization_share"][stage]["mean"]
            for row in usable
        ]
        ax.bar(x, values, bottom=bottom, label=stage)
        bottom = [left + value for left, value in zip(bottom, values)]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Final valid token share")
    ax.set_title("Early / middle / late finalization")
    place_legend(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_task4_parallelism_signature(
    rows: list[dict[str, Any]], out_path: str
) -> None:
    """Compact Task 4 headline: burst size, concentration, and tail stability."""
    usable = [
        row
        for row in rows
        if (row.get("Trace Summary") or {}).get("parallelism_signature")
        and (row.get("Trace Summary") or {}).get("final_stable_progress")
    ]
    if not usable:
        return
    labels = [_label(row) for row in usable]
    x = list(range(len(labels)))
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(max(8, len(labels) * 0.9), 10.5),
        constrained_layout=True,
    )
    burst = [
        row["Trace Summary"]["parallelism_signature"]["peak_to_mean_tpf"]["mean"]
        for row in usable
    ]
    concentration = [
        row["Trace Summary"]["parallelism_signature"]
        ["busiest_10pct_finalization_share"]["mean"]
        for row in usable
    ]
    tail = [
        row["Trace Summary"]["final_stable_progress"]["p90"]["mean"]
        for row in usable
    ]
    axes[0].bar(x, burst, color="#E15759")
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Peak / mean final-stable gain")
    axes[0].set_title("Final-stable burst ratio (1 = flat schedule)")
    axes[1].bar(x, concentration, color="#F28E2B")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Final token share")
    axes[1].set_title("Tokens finalized in the busiest 10% of forwards")
    axes[2].bar(x, tail, color="#4C78A8")
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Normalized progress")
    axes[2].set_title("P90 final-stable progress (lower = earlier convergence)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    fig.suptitle("Task 4 parallelism and convergence signature")
    fig.savefig(out_path)
    plt.close(fig)


def plot_task4_draft_volatility(rows: list[dict[str, Any]], out_path: str) -> None:
    usable = [
        row
        for row in rows
        if (row.get("Trace Summary") or {}).get("draft_volatility")
    ]
    if not usable:
        return
    labels = [_label(row) for row in usable]
    x = list(range(len(labels)))
    revised = [
        row["Trace Summary"]["draft_volatility"]["revised_position_share"]["mean"]
        for row in usable
    ]
    changes = [
        row["Trace Summary"]["draft_volatility"]
        ["mean_revisions_per_position"]["mean"]
        for row in usable
    ]
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(max(8, len(labels) * 0.9), 7.5),
        constrained_layout=True,
    )
    axes[0].bar(x, revised, color="#B07AA1")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Final-position share")
    axes[0].set_title("Positions whose visible draft changed at least once")
    axes[1].bar(x, changes, color="#76B7B2")
    axes[1].set_ylabel("Changes / final position")
    axes[1].set_title("Mean visible-draft changes per final position")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    fig.suptitle("Task 4 draft volatility (not task correctness)")
    fig.savefig(out_path)
    plt.close(fig)


def plot_task4_forward_yield(rows: list[dict[str, Any]], out_path: str) -> None:
    """Native accepted-token TPF and DFlash target-verification yield."""
    values: list[tuple[dict[str, Any], float, str]] = []
    for row in rows:
        mean_tpf = row.get("Mean TPF")
        speculative = (row.get("Aux") or {}).get(
            "speculative_mean_acceptance_length"
        )
        if mean_tpf is not None:
            values.append((row, float(mean_tpf), "accepted events / model forward"))
        elif speculative is not None:
            values.append((row, float(speculative), "accepted / target verification"))
    if not values:
        return
    labels = [f"{_label(row)}\n{basis}" for row, _, basis in values]
    fig, ax = plt.subplots(figsize=(max(8, len(values) * 0.95), 5.2))
    ax.bar(
        range(len(values)),
        [value for _, value, _ in values],
        color=["#4C78A8" if basis.startswith("accepted events") else "#F28E2B" for _, _, basis in values],
    )
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Tokens advanced per primary/target forward")
    ax.set_title("Forward yield (re-accepted positions count again; DFlash uses a separate basis)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_task4_update_geometry(rows: list[dict[str, Any]], out_path: str) -> None:
    usable = [
        row for row in rows if (row.get("Trace Summary") or {}).get("update_geometry")
    ]
    if not usable:
        return
    labels = [_label(row) for row in usable]
    x = list(range(len(labels)))
    run_length = [
        row["Trace Summary"]["update_geometry"]["mean_finalization_run_length"]["mean"]
        for row in usable
    ]
    density = [
        row["Trace Summary"]["update_geometry"]
        ["mean_finalization_span_density"]["mean"]
        for row in usable
    ]
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(max(8, len(labels) * 0.9), 7.5),
        constrained_layout=True,
    )
    axes[0].bar(x, run_length, color="#4C78A8")
    axes[0].set_ylabel("Contiguous final tokens")
    axes[0].set_title("Mean contiguous finalization-run length")
    axes[1].bar(x, density, color="#59A14F")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Positions / enclosing span")
    axes[1].set_title("Finalization span density (1 = one compact region)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    fig.suptitle("Task 4 block-update geometry")
    fig.savefig(out_path)
    plt.close(fig)


def plot_task4_visible_draft_correction(
    rows: list[dict[str, Any]], out_path: str
) -> None:
    traced = [
        row
        for row in rows
        if (row.get("Trace Summary") or {}).get("visible_draft_correction")
    ]
    if not traced:
        return
    labels = [_label(row) for row in traced]
    x = list(range(len(labels)))
    observable = [
        row["Trace Summary"]["visible_draft_correction"]["observable_sample_rate"]
        for row in traced
    ]
    eligible = [
        (index, row["Trace Summary"]["visible_draft_correction"])
        for index, row in enumerate(traced)
        if row["Trace Summary"]["visible_draft_correction"].get(
            "observation_status"
        )
        == "observable"
    ]
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(max(8, len(labels) * 0.9), 14),
        constrained_layout=True,
    )
    axes[0].bar(x, observable, color="#9C9C9C")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Sample ratio")
    axes[0].set_title("Provisional visible-draft observability (0 means N/A, not no corrections)")
    if eligible:
        indices = [index for index, _ in eligible]
        width_pair = 0.34
        axes[1].bar(
            [index - width_pair / 2 for index in indices],
            [value["first_visible_final_match_rate"]["mean"] for _, value in eligible],
            width_pair,
            color="#59A14F",
            label="first-visible final match",
        )
        axes[1].bar(
            [index + width_pair / 2 for index in indices],
            [value["wrong_draft_exposure_auc"]["mean"] for _, value in eligible],
            width_pair,
            color="#E15759",
            label="wrong-draft exposure AUC",
        )
        axes[1].set_ylim(0, 1)
        axes[1].set_ylabel("Rate / normalized area")
        axes[1].set_title("First-visible final match and wrong-draft exposure")
        place_legend(axes[1])
        width = 0.24
        for offset, (key, label, color) in enumerate(
            (
                ("helpful_revision_share", "toward final", "#59A14F"),
                ("lateral_revision_share", "wrong -> other wrong", "#F28E2B"),
                ("harmful_revision_share", "away from final", "#E15759"),
            )
        ):
            axes[2].bar(
                [index + (offset - 1) * width for index in indices],
                [value[key]["mean"] for _, value in eligible],
                width,
                label=label,
                color=color,
            )
        axes[2].set_ylim(0, 1)
        axes[2].set_ylabel("Revision-event share")
        axes[2].set_title("Direction of visible-draft revisions")
        place_legend(axes[2])
        bottom = [0.0] * len(indices)
        for key, label, color in (
            ("revision_early_share", "early", "#4C78A8"),
            ("revision_middle_share", "middle", "#F28E2B"),
            ("revision_late_share", "late", "#E15759"),
        ):
            vals = [value[key]["mean"] for _, value in eligible]
            axes[3].bar(indices, vals, bottom=bottom, label=label, color=color)
            bottom = [left + value for left, value in zip(bottom, vals)]
        axes[3].set_ylim(0, 1)
        axes[3].set_ylabel("Revision-event share")
        axes[3].set_title("When visible-draft revisions happen")
        place_legend(axes[3])
    else:
        for ax in axes[1:]:
            ax.text(
                0.5,
                0.5,
                "N/A: selected traces expose commitments only",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    fig.suptitle("Task 4 visible-draft correction (coverage-gated)")
    fig.savefig(out_path)
    plt.close(fig)


def plot_task4_confidence_dynamics(rows: list[dict[str, Any]], out_path: str) -> None:
    usable = []
    for row in rows:
        trace = row.get("Trace Summary") or {}
        dynamics = trace.get("confidence_dynamics") or {}
        if dynamics.get("backslide_step_rate"):
            usable.append((row, trace, dynamics))
    if not usable:
        return
    labels = []
    for row, trace, _ in usable:
        observation = trace.get("certainty_observation", {})
        labels.append(
            f"{_label(row)}\n{observation.get('entropy_scope', 'unknown')} "
            f"cov={observation.get('entropy_position_coverage', 0):.2f}"
        )
    x = list(range(len(labels)))
    backslide = [value["backslide_step_rate"]["mean"] for _, _, value in usable]
    magnitude = [
        value["mean_backslide_magnitude_per_transition"]["mean"]
        for _, _, value in usable
    ]
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(max(8, len(labels) * 1.0), 7.5),
        constrained_layout=True,
    )
    axes[0].bar(x, backslide, color="#E15759")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Observed transition share")
    axes[0].set_title("Certainty-backslide step rate")
    axes[1].bar(x, magnitude, color="#F28E2B")
    axes[1].set_ylabel("Mean certainty decrease")
    axes[1].set_title("Backslide magnitude per observed transition")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    fig.suptitle("Task 4 confidence correction (compare matching coverage scopes)")
    fig.savefig(out_path)
    plt.close(fig)


def plot_speculative_acceptance(rows: list[dict[str, Any]], out_path: str) -> None:
    usable = [
        row
        for row in rows
        if (row.get("Aux") or {}).get("speculative_draft_acceptance_rate")
        is not None
    ]
    if not usable:
        return
    labels = [_label(row) for row in usable]
    x = list(range(len(labels)))
    acceptance = [
        row["Aux"]["speculative_draft_acceptance_rate"] for row in usable
    ]
    lengths = [
        row["Aux"]["speculative_mean_acceptance_length"] for row in usable
    ]
    fig, axes = plt.subplots(1, 2, figsize=(max(9, len(labels) * 1.2), 4.8))
    axes[0].bar(x, acceptance, color="#59A14F")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Accepted draft tokens / drafted tokens")
    axes[0].set_title("DFlash draft acceptance rate")
    axes[1].bar(x, lengths, color="#F28E2B")
    axes[1].set_ylabel("Tokens / target verification")
    axes[1].set_title("Mean accepted span (+ target token)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_task4_style_coverage(rows: list[dict[str, Any]], out_path: str) -> None:
    usable = [row for row in rows if (row.get("Trace Summary") or {}).get("style")]
    if not usable:
        return
    labels = [_label(row) for row in usable]
    x = list(range(len(labels)))
    width = 0.25
    keys = (
        ("answer_region_detected_rate", "answer detected"),
        ("style_trace_mappable_rate", "trace mappable"),
        ("style_eligible_ratio", "eligible"),
    )
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.9), 4.8))
    for offset, (key, label) in enumerate(keys):
        values = [row["Trace Summary"]["style"].get(key, 0.0) for row in usable]
        ax.bar(
            [value + (offset - 1) * width for value in x],
            values,
            width=width,
            label=label,
        )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Coverage ratio")
    ax.set_title("Answer-local structure analysis coverage")
    place_legend(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_task4_structure_first(rows: list[dict[str, Any]], out_path: str) -> None:
    usable = []
    for row in rows:
        style = (row.get("Trace Summary") or {}).get("style") or {}
        stats = style.get("answer_local_structure_first_score")
        if style.get("style_eligible_ratio", 0.0) >= 0.5 and stats:
            usable.append((row, stats))
    if not usable:
        return
    labels = [_label(row) for row, _ in usable]
    means = [stats["mean"] for _, stats in usable]
    lows = [stats["ci_low"] for _, stats in usable]
    highs = [stats["ci_high"] for _, stats in usable]
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.9), 4.8))
    ax.errorbar(
        x,
        means,
        yerr=[
            [max(0.0, mean - low) for mean, low in zip(means, lows)],
            [max(0.0, high - mean) for mean, high in zip(means, highs)],
        ],
        fmt="o",
        capsize=4,
    )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Answer-local Structure-First Score")
    ax.set_title("Structure-first preference (coverage-gated)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_sudoku_revision_diagnostics(rows: list[dict[str, Any]], out_path: str) -> None:
    """Plot Easy/Hard revision and correction only behind mapping coverage."""
    available = [
        row
        for row in rows
        if (row.get("Trace Summary") or {}).get("sudoku_revision", {}).get(
            "by_difficulty"
        )
    ]
    if not available:
        return
    entries: list[tuple[dict[str, Any], str, dict[str, Any]]] = []
    for row in available:
        by_difficulty = row["Trace Summary"]["sudoku_revision"]["by_difficulty"]
        for difficulty in ("easy", "hard"):
            if difficulty in by_difficulty:
                entries.append((row, difficulty, by_difficulty[difficulty]))

    labels = [f"{_label(row)}\n{difficulty}" for row, difficulty, _ in entries]
    x = list(range(len(entries)))
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(max(8, len(entries) * 0.85), 12),
        constrained_layout=True,
    )

    mapping = [float(group["mapping_eligible_ratio"]) for _, _, group in entries]
    bars = axes[0].bar(x, mapping, color="#4C78A8")
    for bar, (_, _, group) in zip(bars, entries):
        step_rate = group["trace_parseable_step_rate"]["mean"]
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            min(0.98, bar.get_height() + 0.025),
            f"steps={step_rate:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
        )
    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("Eligible sample ratio")
    axes[0].set_title("Sudoku trace-to-cell mapping coverage")

    interpretable = [
        (index, group)
        for index, (_, _, group) in enumerate(entries)
        if group.get("interpretation_status") == "interpretable"
        and group.get("revision_count_by_stage")
    ]
    if interpretable:
        width = 0.24
        for offset, stage in enumerate(("early", "middle", "late")):
            axes[1].bar(
                [index + (offset - 1) * width for index, _ in interpretable],
                [group["revision_count_by_stage"][stage]["mean"] for _, group in interpretable],
                width,
                label=stage,
            )
        place_legend(axes[1])
        axes[1].set_ylabel("Mean visible-token revisions / sample")
    else:
        axes[1].text(
            0.5,
            0.5,
            "N/A: no Easy/Hard stratum reaches mapping coverage >= 0.5",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
    axes[1].set_title("Revision timing by forward-progress third")

    correction = [
        (index, group)
        for index, group in interpretable
        if group.get("correction_success_rate") is not None
    ]
    if correction:
        correction_bars = axes[2].bar(
            [index for index, _ in correction],
            [group["correction_success_rate"]["mean"] for _, group in correction],
            color="#59A14F",
            yerr=[
                [
                    max(
                        0.0,
                        group["correction_success_rate"]["mean"]
                        - group["correction_success_rate"]["ci_low"],
                    )
                    for _, group in correction
                ],
                [
                    max(
                        0.0,
                        group["correction_success_rate"]["ci_high"]
                        - group["correction_success_rate"]["mean"],
                    )
                    for _, group in correction
                ],
            ],
            capsize=3,
        )
        for bar, (_, group) in zip(correction_bars, correction):
            opportunities = (
                group["error_then_correct_count"]
                + group["error_then_still_wrong_count"]
            )
            axes[2].text(
                bar.get_x() + bar.get_width() / 2,
                min(0.98, bar.get_height() + 0.025),
                f"n={opportunities}",
                ha="center",
                fontsize=8,
            )
        axes[2].set_ylim(0, 1.08)
        axes[2].set_ylabel("Correction success rate")
    else:
        axes[2].text(
            0.5,
            0.5,
            "N/A: insufficient mapping or no observed wrong-visible opportunity",
            ha="center",
            va="center",
            transform=axes[2].transAxes,
        )
    axes[2].set_title("Wrong-visible token corrected by final state")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    fig.suptitle("Sudoku9 revision diagnostics (coverage-gated Easy/Hard case study)")
    fig.savefig(out_path)
    plt.close(fig)
