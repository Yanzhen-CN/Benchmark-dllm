"""Dataset-level Task 4 trace aggregation and visualizations."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..datasets.base import Sample
from ..interfaces import GenerationResult, PositionState
from ..metrics.certainty import build_observed_certainty_curve
from ..metrics.commit_order import aggregate_commit_order, commit_order_tau_windows
from ..metrics.stats_utils import BinnedPoint, aggregate_curve_by_bins, summarize
from ..metrics.strategy_score import normalized_progress_series, strategy_score
from ..metrics.trace_parallelism import (
    compute_final_stable_steps,
    effective_tokens_per_forward,
    finalization_share,
    mean_peak_effective_tokens_per_forward,
    normalized_forward_progress,
)


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _plot_curve(
    points: list[BinnedPoint], path: Path, *, xlabel: str, ylabel: str
) -> None:
    if not points:
        return
    x = [point.bin_center for point in points]
    y = [point.stats.mean for point in points]
    low = [point.stats.ci_low for point in points]
    high = [point.stats.ci_high for point in points]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(x, y, marker="o")
    ax.fill_between(x, low, high, alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_two_curves(
    first: list[BinnedPoint],
    second: list[BinnedPoint],
    path: Path,
    *,
    first_label: str,
    second_label: str,
    xlabel: str,
    ylabel: str,
) -> None:
    if not first or not second:
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for points, label in ((first, first_label), (second, second_label)):
        x = [point.bin_center for point in points]
        y = [point.stats.mean for point in points]
        low = [point.stats.ci_low for point in points]
        high = [point.stats.ci_high for point in points]
        ax.plot(x, y, marker="o", label=label)
        ax.fill_between(x, low, high, alpha=0.16)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_tau(summary: dict[str, Any], path: Path) -> None:
    tau = summary.get("commit_order_tau", {})
    if not tau:
        return
    windows = sorted((int(window) for window in tau))
    medians = [tau[str(window)]["median"] for window in windows]
    lows = [tau[str(window)]["ci_low"] for window in windows]
    highs = [tau[str(window)]["ci_high"] for window in windows]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.errorbar(
        windows,
        medians,
        yerr=[
            [max(0.0, median - low) for median, low in zip(medians, lows)],
            [max(0.0, high - median) for median, high in zip(medians, highs)],
        ],
        marker="o",
        capsize=4,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Window Size (tokens)")
    ax.set_ylabel("Kendall tau-b")
    ax.set_title("Commit order by local window")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_finalization_share(summary: dict[str, Any], path: Path) -> None:
    shares = summary.get("finalization_share", {})
    labels = [label for label in ("early", "middle", "late") if label in shares]
    if not labels:
        return
    medians = [shares[label]["median"] for label in labels]
    lows = [shares[label]["ci_low"] for label in labels]
    highs = [shares[label]["ci_high"] for label in labels]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(
        labels,
        medians,
        yerr=[
            [max(0.0, median - low) for median, low in zip(medians, lows)],
            [max(0.0, high - median) for median, high in zip(medians, highs)],
        ],
        capsize=4,
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Final valid token share")
    ax.set_title("When tokens become finally stable")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_draft_volatility(summary: dict[str, Any], path: Path) -> None:
    burden = summary.get("draft_volatility", {})
    keys = [
        key
        for key in ("revised_position_share", "mean_revisions_per_position")
        if key in burden
    ]
    if not keys:
        return
    labels = ["Revised position share", "Mean revisions / position"][: len(keys)]
    means = [burden[key]["mean"] for key in keys]
    lows = [burden[key]["ci_low"] for key in keys]
    highs = [burden[key]["ci_high"] for key in keys]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(
        labels,
        means,
        yerr=[
            [max(0.0, mean - low) for mean, low in zip(means, lows)],
            [max(0.0, high - mean) for mean, high in zip(means, highs)],
        ],
        capsize=4,
    )
    ax.set_ylabel("Dataset-level mean (bootstrap 95% CI)")
    ax.set_title("Draft-token volatility before final stabilization")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_parallelism_signature(summary: dict[str, Any], path: Path) -> None:
    signature = summary.get("parallelism_signature", {})
    if not signature:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    burst = signature["peak_to_mean_tpf"]["mean"]
    axes[0].bar(["Peak / Mean TPF"], [burst], color="#E15759")
    axes[0].axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel("Ratio")
    axes[0].set_title("Finalization burstiness")

    labels = ["Active forwards", "Busiest 10%\nfinalization share"]
    values = [
        signature["active_forward_ratio"]["mean"],
        signature["busiest_10pct_finalization_share"]["mean"],
    ]
    axes[1].bar(labels, values, color=["#4C78A8", "#F28E2B"])
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Share")
    axes[1].set_title("How concentrated is final token production?")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_finalization_quantiles(summary: dict[str, Any], path: Path) -> None:
    quantiles = summary.get("final_stable_progress", {})
    if not quantiles:
        return
    labels = ["P50", "P90", "P99"]
    values = [quantiles[label.lower()]["mean"] for label in labels]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.bar(labels, values, color=["#59A14F", "#F28E2B", "#E15759"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Normalized Forward Progress")
    ax.set_title("When final tokens become stable")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_finalization_map(
    records: list[tuple[Sample, GenerationResult]], path: Path, bins: int = 32
) -> None:
    """Equal-sample density of final position versus final-stable forward."""
    sample_histograms = []
    for _, result in records:
        if not result.trace:
            continue
        length = result.final_valid_length or len(result.trace[-1].token_ids)
        if length <= 0:
            continue
        sequences = [step.token_ids[:length] for step in result.trace]
        stable = compute_final_stable_steps(sequences)
        if not stable:
            continue
        x = np.asarray(
            [index / (len(stable) - 1) if len(stable) > 1 else 0.0 for index in range(len(stable))]
        )
        y = np.asarray(
            [normalized_forward_progress(step, len(sequences)) for step in stable]
        )
        histogram, _, _ = np.histogram2d(
            x, y, bins=bins, range=((0, 1), (0, 1))
        )
        sample_histograms.append(histogram / histogram.sum())
    if not sample_histograms:
        return
    density = np.mean(sample_histograms, axis=0).T
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(
        density,
        origin="lower",
        extent=(0, 1, 0, 1),
        aspect="auto",
        cmap="magma",
    )
    ax.set_xlabel("Normalized Final Token Position")
    ax.set_ylabel("Normalized Final-Stable Forward")
    ax.set_title("Dataset-level token finalization map")
    fig.colorbar(image, ax=ax, label="Equal-sample token mass")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _envelope(values: list[float]) -> list[float]:
    normalized = normalized_progress_series(values)
    result: list[float] = []
    highest = 0.0
    for value in normalized:
        highest = max(highest, value)
        result.append(highest)
    return result


def _style_observation(
    dataset_name: str, sample: Sample, result: GenerationResult
) -> dict[str, Any] | None:
    if dataset_name == "mbpp":
        from ..datasets.answer_region import (
            answer_local_checkpoint_texts,
            locate_mbpp_answer,
        )
        from ..datasets.mbpp import mbpp_text_checkpoint_scores
        from ..datasets.structeval_t import checkpoint_indices

        region = locate_mbpp_answer(result.output_text)
        scorer = lambda texts: mbpp_text_checkpoint_scores(texts)
    elif dataset_name == "structeval_t":
        from ..datasets.answer_region import (
            answer_local_checkpoint_texts,
            locate_structeval_answer,
        )
        from ..datasets.structeval_t import (
            checkpoint_indices,
            struct_eval_t_text_checkpoint_scores,
        )

        region = locate_structeval_answer(result.output_text)
        scorer = lambda texts: struct_eval_t_text_checkpoint_scores(texts, sample.reference)
    else:
        return None
    observation: dict[str, Any] = {"detected": bool(region.detected), "mapped": False}
    if not result.trace or not region.detected:
        return observation
    texts, mapped = answer_local_checkpoint_texts(
        result.trace, region, checkpoint_indices(len(result.trace), 4)
    )
    observation["mapped"] = bool(mapped)
    if not mapped:
        return observation
    structure, content = scorer(texts)
    if not structure or not content:
        return observation
    score = strategy_score(structure, content)
    eligible = score is not None and structure[-1] >= 0.5 and content[-1] >= 0.5
    observation.update(
        {
            "eligible": eligible,
            "score": score if eligible else None,
            "structure": _envelope(structure),
            "content": _envelope(content),
        }
    )
    return observation


def _revision_metrics(result: GenerationResult, length: int) -> tuple[float, float]:
    revisions: list[int] = []
    for position in range(length):
        previous = None
        changes = 0
        for step in result.trace:
            if position >= len(step.token_ids) or position >= len(step.position_states):
                continue
            if step.position_states[position] == PositionState.MASKED:
                continue
            token = step.token_ids[position]
            if previous is not None and token != previous:
                changes += 1
            previous = token
        revisions.append(changes)
    if not revisions:
        return 0.0, 0.0
    return (
        sum(value > 0 for value in revisions) / len(revisions),
        sum(revisions) / len(revisions),
    )


def _sudoku9_revision_summary(
    records: list[tuple[Sample, GenerationResult]], *, seed: int
) -> dict[str, Any]:
    """Build the coverage-gated Easy/Hard Task 4.2.4 diagnostics.

    Only samples with at least half of their trace checkpoints mappable to an
    unambiguous row-major Sudoku trajectory contribute revision/correction
    values.  A difficulty stratum is interpretable only when at least half of
    its traced samples meet that condition.  Counts are otherwise left N/A,
    rather than emitting visually plausible but meaningless zero bars.
    """

    from ..datasets.sudoku9 import SudokuReference, classify_difficulty
    from ..metrics.sudoku_revision import (
        correction_outcomes,
        revision_counts_by_stage,
        trace_parseable_step_count,
    )

    grouped: dict[str, list[dict[str, Any]]] = {"easy": [], "hard": []}
    for sample, result in records:
        if not result.trace or not isinstance(sample.reference, SudokuReference):
            continue
        reference = sample.reference
        difficulty = reference.difficulty or classify_difficulty(reference.puzzle)
        if difficulty not in grouped:
            continue
        parseable_rate = trace_parseable_step_count(result.trace) / len(result.trace)
        observation: dict[str, Any] = {
            "parseable_step_rate": parseable_rate,
            "eligible": parseable_rate >= 0.5,
        }
        if observation["eligible"]:
            observation["revision_count_by_stage"] = revision_counts_by_stage(
                result.trace, puzzle=reference.puzzle
            )
            corrected, still_wrong, success_rate = correction_outcomes(
                result.trace, reference.solution, puzzle=reference.puzzle
            )
            observation.update(
                {
                    "error_then_correct": corrected,
                    "error_then_still_wrong": still_wrong,
                    "correction_success_rate": success_rate,
                }
            )
        grouped[difficulty].append(observation)

    output: dict[str, Any] = {
        "mapping_threshold": 0.5,
        "note": (
            "Revision/correction statistics require >=0.5 mappable trace-step "
            "coverage per sample and >=0.5 eligible samples per difficulty stratum."
        ),
        "by_difficulty": {},
    }
    for difficulty, observations in grouped.items():
        if not observations:
            continue
        parseable_rates = [item["parseable_step_rate"] for item in observations]
        eligible = [item for item in observations if item["eligible"]]
        eligible_ratio = len(eligible) / len(observations)
        interpretable = eligible_ratio >= 0.5
        group_summary: dict[str, Any] = {
            "n_trace_samples": len(observations),
            "trace_parseable_step_rate": _jsonable(
                summarize(parseable_rates, seed=seed)
            ),
            "mapping_eligible_ratio": eligible_ratio,
            "interpretation_status": (
                "interpretable" if interpretable else "insufficient_mapping"
            ),
            "revision_count_by_stage": None,
            "error_then_correct_count": None,
            "error_then_still_wrong_count": None,
            "correction_success_rate": None,
            "pooled_correction_success_rate": None,
        }
        if interpretable:
            group_summary["revision_count_by_stage"] = {
                stage: _jsonable(
                    summarize(
                        [
                            float(item["revision_count_by_stage"][stage])
                            for item in eligible
                        ],
                        seed=seed,
                    )
                )
                for stage in ("early", "middle", "late")
            }
            corrected = sum(item["error_then_correct"] for item in eligible)
            still_wrong = sum(
                item["error_then_still_wrong"] for item in eligible
            )
            opportunities = corrected + still_wrong
            per_sample_correction_rates = [
                float(item["correction_success_rate"])
                for item in eligible
                if item["correction_success_rate"] is not None
            ]
            group_summary.update(
                {
                    "error_then_correct_count": corrected,
                    "error_then_still_wrong_count": still_wrong,
                    "correction_success_rate": (
                        _jsonable(
                            summarize(per_sample_correction_rates, seed=seed)
                        )
                        if per_sample_correction_rates
                        else None
                    ),
                    "pooled_correction_success_rate": (
                        corrected / opportunities if opportunities else None
                    ),
                }
            )
        output["by_difficulty"][difficulty] = group_summary
    return output


def build_dataset_trace_summary(
    dataset_name: str,
    records: list[tuple[Sample, GenerationResult]],
    *,
    seed: int = 42,
    model_name: str | None = None,
    config_name: str | None = None,
) -> tuple[dict[str, Any], dict[str, list[BinnedPoint]]]:
    usable = [(sample, result) for sample, result in records if result.trace]
    summary: dict[str, Any] = {
        "dataset": dataset_name,
        "model": model_name,
        "config": config_name,
        "selected_samples": len(records),
        "trace_samples": len(usable),
        "trace_coverage_rate": len(usable) / len(records) if records else 0.0,
    }
    curves: dict[str, list[BinnedPoint]] = {}
    if not usable:
        return summary, curves

    tpf_samples: list[list[tuple[float, float]]] = []
    certainty_samples: list[list[tuple[float, float]]] = []
    top1_samples: list[list[tuple[float, float]]] = []
    style_structure_samples: list[list[tuple[float, float]]] = []
    style_content_samples: list[list[tuple[float, float]]] = []
    means: list[float] = []
    peaks: list[float] = []
    revised_shares: list[float] = []
    revision_means: list[float] = []
    peak_to_mean_values: list[float] = []
    active_forward_ratios: list[float] = []
    busiest_shares: list[float] = []
    stable_progress_quantiles: dict[str, list[float]] = {
        "p50": [],
        "p90": [],
        "p99": [],
    }
    shares: dict[str, list[float]] = {"early": [], "middle": [], "late": []}
    taus = []
    observed_entropy_steps = observed_top1_steps = total_steps = 0
    observed_entropy_positions = observed_top1_positions = expected_remaining_positions = 0
    style_detected = style_mapped = style_eligible = 0
    style_scores: list[float] = []
    total_time = 0.0
    total_tokens = 0

    for sample, result in usable:
        length = result.final_valid_length or len(result.trace[-1].token_ids)
        if length <= 0:
            continue
        sequences = [step.token_ids[:length] for step in result.trace]
        stable = compute_final_stable_steps(sequences)
        counts = effective_tokens_per_forward(stable, len(sequences))
        tpf_samples.append(
            [
                (normalized_forward_progress(index, len(sequences)), float(counts[index]))
                for index in range(len(sequences))
            ]
        )
        mean_tpf, peak_tpf = mean_peak_effective_tokens_per_forward(
            stable, len(sequences)
        )
        means.append(mean_tpf)
        peaks.append(float(peak_tpf))
        peak_to_mean_values.append(peak_tpf / mean_tpf if mean_tpf > 0 else 0.0)
        active_forward_ratios.append(
            sum(value > 0 for value in counts.values()) / len(counts)
            if counts
            else 0.0
        )
        busiest_count = max(1, int(np.ceil(0.1 * len(counts))))
        busiest_shares.append(
            sum(sorted(counts.values(), reverse=True)[:busiest_count]) / len(stable)
        )
        stable_progress = [
            normalized_forward_progress(step, len(sequences)) for step in stable
        ]
        for label, quantile in (("p50", 0.5), ("p90", 0.9), ("p99", 0.99)):
            stable_progress_quantiles[label].append(
                float(np.quantile(stable_progress, quantile))
            )
        for stage, value in finalization_share(stable, len(sequences)).items():
            shares[stage].append(value)
        taus.append(commit_order_tau_windows(list(range(len(stable))), stable))

        certainty = build_observed_certainty_curve(result.trace, length)
        if len(certainty) >= 2:
            certainty_samples.append([(point[0], point[1]) for point in certainty])
        top1 = [(point[0], point[2]) for point in certainty if point[2] is not None]
        if len(top1) >= 2:
            top1_samples.append([(x, float(value)) for x, value in top1])
        total_steps += len(result.trace)
        observed_entropy_steps += sum(bool(step.entropy_by_position) for step in result.trace)
        observed_top1_steps += sum(
            bool(step.top1_confidence_by_position) for step in result.trace
        )
        for step in result.trace:
            remaining = {
                index
                for index, state in enumerate(step.position_states[:length])
                if state != PositionState.ACCEPTED
            }
            expected_remaining_positions += len(remaining)
            if step.entropy_by_position:
                observed_entropy_positions += len(
                    remaining.intersection(step.entropy_by_position)
                )
            if step.top1_confidence_by_position:
                observed_top1_positions += len(
                    remaining.intersection(step.top1_confidence_by_position)
                )

        revised_share, revision_mean = _revision_metrics(result, length)
        revised_shares.append(revised_share)
        revision_means.append(revision_mean)

        style = _style_observation(dataset_name, sample, result)
        if style is not None:
            style_detected += int(style.get("detected", False))
            style_mapped += int(style.get("mapped", False))
            style_eligible += int(style.get("eligible", False))
            if style.get("eligible"):
                structure = style["structure"]
                content = style["content"]
                style_structure_samples.append(
                    [
                        (index / (len(structure) - 1) if len(structure) > 1 else 0.0, value)
                        for index, value in enumerate(structure)
                    ]
                )
                style_content_samples.append(
                    [
                        (index / (len(content) - 1) if len(content) > 1 else 0.0, value)
                        for index, value in enumerate(content)
                    ]
                )
                style_scores.append(float(style["score"]))

        if result.timing and result.timing.wall_clock_seconds > 0:
            total_time += result.timing.wall_clock_seconds
            total_tokens += result.final_valid_length

    if not means:
        return summary, curves
    curves["tpf"] = aggregate_curve_by_bins(tpf_samples, seed=seed)
    if certainty_samples:
        curves["certainty"] = aggregate_curve_by_bins(certainty_samples, seed=seed)
    if top1_samples:
        curves["top1"] = aggregate_curve_by_bins(top1_samples, seed=seed)
    if style_structure_samples and style_content_samples:
        curves["style_structure"] = aggregate_curve_by_bins(
            style_structure_samples, seed=seed
        )
        curves["style_content"] = aggregate_curve_by_bins(
            style_content_samples, seed=seed
        )

    summary["mean_tpf"] = _jsonable(summarize(means, seed=seed))
    summary["peak_tpf"] = _jsonable(summarize(peaks, seed=seed))
    summary["parallelism_signature"] = {
        "peak_to_mean_tpf": _jsonable(summarize(peak_to_mean_values, seed=seed)),
        "active_forward_ratio": _jsonable(
            summarize(active_forward_ratios, seed=seed)
        ),
        "busiest_10pct_finalization_share": _jsonable(
            summarize(busiest_shares, seed=seed)
        ),
    }
    summary["final_stable_progress"] = {
        label: _jsonable(summarize(values, seed=seed))
        for label, values in stable_progress_quantiles.items()
    }
    summary["tps"] = total_tokens / total_time if total_time > 0 else None
    summary["tpf_tps"] = {
        "mean_tpf": summary["mean_tpf"]["mean"],
        "tps": summary["tps"],
    }
    summary["finalization_share"] = {
        stage: _jsonable(summarize(values, seed=seed)) for stage, values in shares.items()
    }
    summary["commit_order_tau"] = _jsonable(aggregate_commit_order(taus, seed=seed))
    summary["certainty_observation"] = {
        "entropy_step_rate": observed_entropy_steps / total_steps if total_steps else 0.0,
        "top1_step_rate": observed_top1_steps / total_steps if total_steps else 0.0,
        "entropy_position_coverage": (
            observed_entropy_positions / expected_remaining_positions
            if expected_remaining_positions
            else 0.0
        ),
        "top1_position_coverage": (
            observed_top1_positions / expected_remaining_positions
            if expected_remaining_positions
            else 0.0
        ),
        "curve_sample_rate": len(certainty_samples) / len(usable),
        "top1_curve_sample_rate": len(top1_samples) / len(usable),
    }
    for prefix in ("entropy", "top1"):
        coverage = summary["certainty_observation"][f"{prefix}_position_coverage"]
        summary["certainty_observation"][f"{prefix}_scope"] = (
            "full_remaining"
            if coverage >= 0.95
            else "partial_or_active_subset"
            if coverage > 0
            else "unavailable"
        )
    # Compatibility alias used by earlier reports: certainty is entropy-based.
    summary["certainty_observation"]["scope"] = summary[
        "certainty_observation"
    ]["entropy_scope"]
    summary["draft_volatility"] = {
        "revised_position_share": _jsonable(summarize(revised_shares, seed=seed)),
        "mean_revisions_per_position": _jsonable(summarize(revision_means, seed=seed)),
    }
    if dataset_name in {"mbpp", "structeval_t"}:
        summary["style"] = {
            "answer_region_detected_rate": style_detected / len(usable),
            "style_trace_mappable_rate": style_mapped / len(usable),
            "style_eligible_ratio": style_eligible / len(usable),
            "answer_local_structure_first_score": (
                _jsonable(summarize(style_scores, seed=seed)) if style_scores else None
            ),
        }
    if dataset_name in {"sudoku9", "sudoku9_thinking"}:
        summary["sudoku_revision"] = _sudoku9_revision_summary(usable, seed=seed)
    return summary, curves


def render_dataset_trace_report(
    dataset_name: str,
    records: list[tuple[Sample, GenerationResult]],
    out_dir: str | Path,
    *,
    seed: int = 42,
    model_name: str | None = None,
    config_name: str | None = None,
) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary, curves = build_dataset_trace_summary(
        dataset_name,
        records,
        seed=seed,
        model_name=model_name,
        config_name=config_name,
    )
    summary_path = out / "dataset_trace_summary.json"
    summary["curves"] = _jsonable(curves)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    written = {"summary": str(summary_path)}
    curve_specs = {
        "tpf": ("Normalized Forward Progress", "Tokens per Forward"),
        "certainty": ("Accepted Ratio", "Remaining-token Certainty"),
        "top1": ("Accepted Ratio", "Remaining-token Mean Top-1 Confidence"),
    }
    for name, (xlabel, ylabel) in curve_specs.items():
        path = out / f"dataset_{name}.png"
        _plot_curve(curves.get(name, []), path, xlabel=xlabel, ylabel=ylabel)
        if path.exists():
            written[name] = str(path)

    style_path = out / "dataset_structure_content_progress.png"
    _plot_two_curves(
        curves.get("style_structure", []),
        curves.get("style_content", []),
        style_path,
        first_label="Structure Progress",
        second_label="Content Progress",
        xlabel="Answer-local Normalized Progress",
        ylabel="Normalized cumulative formation",
    )
    if style_path.exists():
        written["style"] = str(style_path)

    finalization_map_path = out / "dataset_finalization_map.png"
    _plot_finalization_map(records, finalization_map_path)
    if finalization_map_path.exists():
        written["finalization_map"] = str(finalization_map_path)

    for key, plotter, filename in (
        ("commit_order_tau", _plot_tau, "dataset_commit_order_tau.png"),
        ("finalization_share", _plot_finalization_share, "dataset_finalization_share.png"),
        (
            "parallelism_signature",
            _plot_parallelism_signature,
            "dataset_parallelism_signature.png",
        ),
        (
            "final_stable_progress",
            _plot_finalization_quantiles,
            "dataset_final_stable_progress.png",
        ),
        (
            "draft_volatility",
            _plot_draft_volatility,
            "dataset_draft_volatility.png",
        ),
    ):
        path = out / filename
        if summary.get(key):
            plotter(summary, path)
        if path.exists():
            written[key] = str(path)

    tpf_tps_path = out / "dataset_tpf_tps.txt"
    tpf_tps = summary.get("tpf_tps")
    if tpf_tps:
        tpf_tps_path.write_text(
            "Model | Mean TPF (token/forward) | Tps (token/s)\n"
            f"{model_name or '-'} / {config_name or '-'} | "
            f"{tpf_tps['mean_tpf']:.6g} | "
            f"{tpf_tps['tps']:.6g}\n",
            encoding="utf-8",
        )
        written["tpf_tps"] = str(tpf_tps_path)
    return written
