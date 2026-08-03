"""Dataset-level Task 4 trace aggregation and visualizations."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

from ...datasets.base import Sample
from ...interfaces import GenerationResult, PositionState
from ...metrics.certainty import build_observed_certainty_curve
from .trace_metrics import (
    build_auxiliary_performance_summary,
    visible_revision_profile,
    write_auxiliary_performance_csv,
)
from ...metrics.commit_order import aggregate_commit_order, commit_order_tau_windows
from ...metrics.stats_utils import BinnedPoint, aggregate_curve_by_bins, summarize
from ...metrics.strategy_score import normalized_progress_series, strategy_score
from ...metrics.trace_parallelism import (
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
        from ...datasets.answer_region import (
            answer_local_checkpoint_texts,
            locate_mbpp_answer,
        )
        from ...datasets.mbpp import mbpp_text_checkpoint_scores
        from ...datasets.structeval_t import checkpoint_indices

        region = locate_mbpp_answer(result.output_text)
        scorer = lambda texts: mbpp_text_checkpoint_scores(texts)
    elif dataset_name == "structeval_t":
        from ...datasets.answer_region import (
            answer_local_checkpoint_texts,
            locate_structeval_answer,
        )
        from ...datasets.structeval_t import (
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


def _revision_metrics(
    result: GenerationResult, length: int
) -> tuple[float | None, float | None]:
    profile = visible_revision_profile(result, length)
    if profile is None:
        return None, None
    return (
        profile["revised_position_share"],
        profile["revisions_per_final_position"],
    )


def _draft_correction_metrics(
    result: GenerationResult, length: int
) -> dict[str, float] | None:
    """Correction dynamics for traces that expose provisional visible tokens.

    Commitment-only traces (MASKED -> ACCEPTED with no VISIBLE draft state)
    cannot reveal whether an internal proposal was corrected. They return N/A
    rather than a misleading perfect first-pass or zero-revision score.
    """

    trace = result.trace
    if not trace or not any(
        PositionState.VISIBLE in step.position_states[:length] for step in trace
    ):
        return None
    final_tokens = trace[-1].token_ids[:length]
    first_matches: list[float] = []
    correction_lags: list[float] = []
    wrong_rates: list[float] = []
    helpful = harmful = lateral = total_changes = relapsed = 0
    changes_by_stage = {"early": 0, "middle": 0, "late": 0}
    denominator = max(len(trace) - 1, 1)

    for step in trace:
        visible = [
            position
            for position in range(
                min(length, len(step.token_ids), len(step.position_states))
            )
            if step.position_states[position] != PositionState.MASKED
        ]
        if visible:
            wrong_rates.append(
                sum(step.token_ids[position] != final_tokens[position] for position in visible)
                / len(visible)
            )

    for position, final_token in enumerate(final_tokens):
        sequence: list[tuple[int, int]] = []
        for step_index, step in enumerate(trace):
            if (
                position < len(step.token_ids)
                and position < len(step.position_states)
                and step.position_states[position] != PositionState.MASKED
            ):
                sequence.append((step_index, step.token_ids[position]))
        if not sequence:
            continue
        first_matches.append(float(sequence[0][1] == final_token))
        stable_step = len(trace) - 1
        for step_index in range(len(trace) - 1, -1, -1):
            step = trace[step_index]
            if position < len(step.token_ids) and step.token_ids[position] == final_token:
                stable_step = step_index
            else:
                break
        correction_lags.append((stable_step - sequence[0][0]) / denominator)
        left_final = False
        for (previous_step, previous), (current_step, current) in zip(
            sequence, sequence[1:]
        ):
            if previous == current:
                continue
            total_changes += 1
            progress = current_step / denominator
            stage = (
                "early"
                if progress < 1 / 3
                else "middle"
                if progress < 2 / 3
                else "late"
            )
            changes_by_stage[stage] += 1
            if previous != final_token and current == final_token:
                helpful += 1
            elif previous == final_token and current != final_token:
                harmful += 1
                left_final = True
            else:
                lateral += 1
        relapsed += int(left_final)

    positions = max(len(first_matches), 1)
    changes = max(total_changes, 1)
    wrong_draft_exposure_auc = (
        wrong_rates[0]
        if len(wrong_rates) == 1
        else sum(
            (previous + current) / 2
            for previous, current in zip(wrong_rates, wrong_rates[1:])
        )
        / (len(wrong_rates) - 1)
        if wrong_rates
        else 0.0
    )
    return {
        "first_visible_final_match_rate": sum(first_matches) / positions,
        "wrong_draft_exposure_auc": wrong_draft_exposure_auc,
        "mean_correction_lag": (
            sum(correction_lags) / len(correction_lags)
            if correction_lags
            else 0.0
        ),
        "changes_per_final_position": total_changes / positions,
        "helpful_revision_share": helpful / changes,
        "harmful_revision_share": harmful / changes,
        "lateral_revision_share": lateral / changes,
        "relapse_position_share": relapsed / positions,
        **{
            f"revision_{stage}_share": count / changes
            for stage, count in changes_by_stage.items()
        },
    }


def _update_geometry(stable_steps: list[int], num_steps: int) -> dict[str, float]:
    run_lengths: list[int] = []
    span_densities: list[float] = []
    run_counts: list[int] = []
    for step in range(num_steps):
        positions = [
            position
            for position, stable_step in enumerate(stable_steps)
            if stable_step == step
        ]
        if not positions:
            continue
        runs: list[int] = []
        current = 1
        for previous, position in zip(positions, positions[1:]):
            if position == previous + 1:
                current += 1
            else:
                runs.append(current)
                current = 1
        runs.append(current)
        run_lengths.extend(runs)
        run_counts.append(len(runs))
        span_densities.append(len(positions) / (positions[-1] - positions[0] + 1))
    return {
        "mean_finalization_run_length": (
            sum(run_lengths) / len(run_lengths) if run_lengths else 0.0
        ),
        "mean_finalization_run_count": (
            sum(run_counts) / len(run_counts) if run_counts else 0.0
        ),
        "mean_finalization_span_density": (
            sum(span_densities) / len(span_densities) if span_densities else 0.0
        ),
    }


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

    from ...datasets.sudoku9 import SudokuReference, classify_difficulty
    from ...metrics.sudoku_revision import (
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

    auxiliary_performance, _ = build_auxiliary_performance_summary(
        dataset_name=dataset_name,
        records=records,
        model_name=model_name,
        config_name=config_name,
    )
    summary["auxiliary_performance"] = auxiliary_performance

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
    geometry_values: dict[str, list[float]] = {
        "mean_finalization_run_length": [],
        "mean_finalization_run_count": [],
        "mean_finalization_span_density": [],
    }
    draft_metric_values: dict[str, list[float]] = {}
    draft_observable_samples = 0
    confidence_metric_values: dict[str, list[float]] = {
        "backslide_step_rate": [],
        "mean_backslide_magnitude_per_transition": [],
        "mean_total_variation_per_transition": [],
        "net_certainty_gain": [],
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
        for key, value in _update_geometry(stable, len(sequences)).items():
            geometry_values[key].append(value)

        certainty = build_observed_certainty_curve(result.trace, length)
        if len(certainty) >= 2:
            certainty_samples.append([(point[0], point[1]) for point in certainty])
        # The curve appends a synthetic fully-certain endpoint for presentation.
        # Backslide metrics must use only actually observed entropy checkpoints.
        certainty_values = [
            1.0 - sum(step.entropy_by_position.values()) / len(step.entropy_by_position)
            for step in result.trace
            if step.entropy_by_position
        ]
        if len(certainty_values) >= 2:
            changes = [
                current - previous
                for previous, current in zip(certainty_values, certainty_values[1:])
            ]
            negative = [max(0.0, -change) for change in changes]
            confidence_metric_values["backslide_step_rate"].append(
                sum(value > 0 for value in negative) / len(changes)
            )
            confidence_metric_values[
                "mean_backslide_magnitude_per_transition"
            ].append(sum(negative) / len(changes))
            confidence_metric_values["mean_total_variation_per_transition"].append(
                sum(abs(value) for value in changes) / len(changes)
            )
            confidence_metric_values["net_certainty_gain"].append(
                certainty_values[-1] - certainty_values[0]
            )
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
        if revised_share is not None and revision_mean is not None:
            revised_shares.append(revised_share)
            revision_means.append(revision_mean)
        draft_metrics = _draft_correction_metrics(result, length)
        if draft_metrics is not None:
            draft_observable_samples += 1
            for key, value in draft_metrics.items():
                draft_metric_values.setdefault(key, []).append(value)

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
    summary["update_geometry"] = {
        key: _jsonable(summarize(values, seed=seed))
        for key, values in geometry_values.items()
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
    summary["confidence_dynamics"] = {
        "observable_sample_rate": (
            len(confidence_metric_values["backslide_step_rate"]) / len(usable)
        ),
        **{
            key: _jsonable(summarize(values, seed=seed)) if values else None
            for key, values in confidence_metric_values.items()
        },
        "comparison_note": (
            "Direct comparison requires matching entropy scope/coverage; active-subset "
            "curves remain model-local diagnostics."
        ),
    }
    summary["draft_volatility"] = {
        "semantics": (
            "changed-token re-acceptance only; proposal refresh and re-noising "
            "are excluded"
        ),
        "revised_position_share": (
            _jsonable(summarize(revised_shares, seed=seed))
            if revised_shares
            else None
        ),
        "mean_revisions_per_position": (
            _jsonable(summarize(revision_means, seed=seed))
            if revision_means
            else None
        ),
    }
    summary["visible_draft_correction"] = {
        "metric_status": "deprecated_draft_churn_diagnostic_not_revision",
        "observable_sample_rate": draft_observable_samples / len(usable),
        "observation_status": (
            "observable" if draft_observable_samples else "commitment_only_trace"
        ),
        **{
            key: _jsonable(summarize(values, seed=seed)) if values else None
            for key, values in draft_metric_values.items()
        },
        "comparison_note": (
            "Adjacent visible-canvas changes include proposal refresh and re-noising. "
            "Do not report this block as revision or correction; use draft_volatility "
            "for changed-token re-acceptance."
        ),
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
    for filename in (
        "dataset_tpf.png",
        "dataset_certainty.png",
        "dataset_top1.png",
        "dataset_structure_content_progress.png",
        "dataset_finalization_map.png",
        "dataset_commit_order_tau.png",
        "dataset_finalization_share.png",
        "dataset_parallelism_signature.png",
        "dataset_final_stable_progress.png",
        "dataset_draft_volatility.png",
        "dataset_update_geometry.png",
        "dataset_visible_draft_correction.png",
        "dataset_confidence_dynamics.png",
    ):
        (out / filename).unlink(missing_ok=True)
    written = {"summary": str(summary_path)}

    _, auxiliary_rows = build_auxiliary_performance_summary(
        dataset_name=dataset_name,
        records=records,
        model_name=model_name,
        config_name=config_name,
    )
    auxiliary_path = out / "dataset_auxiliary_performance.csv"
    if write_auxiliary_performance_csv(auxiliary_rows, auxiliary_path):
        written["auxiliary_performance"] = str(auxiliary_path)

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
