"""Deterministic, dataset-aware sampling for formal evaluation files."""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Any, Iterable

from ..datasets.base import Sample
from ..datasets.sudoku import classify_difficulty


def _shuffled(samples: Iterable[Sample], rng: random.Random) -> list[Sample]:
    ordered = sorted(samples, key=lambda sample: sample.sample_id)
    rng.shuffle(ordered)
    return ordered


def _take(
    samples: Iterable[Sample],
    count: int,
    rng: random.Random,
    description: str,
) -> list[Sample]:
    available = _shuffled(samples, rng)
    if len(available) < count:
        raise ValueError(
            f"{description} requires {count} samples, but only {len(available)} are available"
        )
    return available[:count]


def _with_meta(sample: Sample, **values: Any) -> Sample:
    return replace(sample, meta={**sample.meta, **values})


def _generic_count(dataset_config: dict[str, Any], n_samples: int | None) -> int:
    count = n_samples if n_samples is not None else dataset_config.get("sample_size")
    if count is None:
        raise ValueError(
            f"dataset {dataset_config.get('dataset')!r} does not declare sample_size"
        )
    count = int(count)
    if count <= 0:
        raise ValueError("sample count must be greater than zero")
    return count


def _balanced_counts(total: int, group_count: int) -> list[int]:
    quotient, remainder = divmod(total, group_count)
    return [quotient + (1 if index < remainder else 0) for index in range(group_count)]


def _sudoku_difficulty(sample: Sample) -> str:
    reference = sample.reference
    difficulty = getattr(reference, "difficulty", None)
    if difficulty is None:
        puzzle = getattr(reference, "puzzle", None)
        if puzzle is None:
            raise ValueError(f"Sudoku sample {sample.sample_id!r} has no puzzle")
        difficulty = classify_difficulty(puzzle)
    return str(difficulty).lower()


def _select_sudoku(
    samples: list[Sample],
    config: dict[str, Any],
    n_samples: int | None,
    rng: random.Random,
) -> list[Sample]:
    configured = config.get("difficulty_counts")
    if not isinstance(configured, dict) or not configured:
        return _take(samples, _generic_count(config, n_samples), rng, "Sudoku")

    difficulties = list(configured)
    if n_samples is None:
        counts = [int(configured[difficulty]) for difficulty in difficulties]
    else:
        counts = _balanced_counts(_generic_count(config, n_samples), len(difficulties))

    selected: list[Sample] = []
    for difficulty, count in zip(difficulties, counts):
        group = [sample for sample in samples if _sudoku_difficulty(sample) == difficulty]
        selected.extend(
            _take(group, count, rng, f"Sudoku difficulty {difficulty!r}")
        )
    return selected


def _select_hellobench(
    samples: list[Sample],
    config: dict[str, Any],
    n_samples: int | None,
    rng: random.Random,
) -> list[Sample]:
    profiles = config.get("output_profiles")
    if not isinstance(profiles, list) or not profiles:
        return _take(samples, _generic_count(config, n_samples), rng, "HelloBench")

    if n_samples is None:
        per_length = int(config["samples_per_length"])
        counts = [per_length] * len(profiles)
    else:
        counts = _balanced_counts(_generic_count(config, n_samples), len(profiles))

    selected: list[Sample] = []
    for profile, count in zip(profiles, counts):
        target_words = int(profile["target_words"])
        max_new_tokens = int(profile["max_new_tokens"])
        group = [
            sample
            for sample in samples
            if int(getattr(sample.reference, "target_length_words", -1)) == target_words
        ]
        chosen = _take(group, count, rng, f"HelloBench {target_words}-word profile")
        selected.extend(
            _with_meta(
                sample,
                target_output_words=target_words,
                max_new_tokens=max_new_tokens,
            )
            for sample in chosen
        )
    return selected


def _positive_int(value: Any, name: str) -> int:
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a positive integer, got {value!r}; set its environment variable if needed"
        ) from exc
    if resolved <= 0:
        raise ValueError(f"{name} must be a positive integer, got {resolved}")
    return resolved


def _ruler_windows(
    dataset_config: dict[str, Any], model_config: dict[str, Any]
) -> list[int]:
    common = _positive_int(
        dataset_config.get("common_context_window_tokens"),
        "common_context_window_tokens",
    )
    windows = [common]
    if dataset_config.get("include_model_max_context_window", False):
        model_max = _positive_int(
            model_config.get("max_context_tokens"), "model max_context_tokens"
        )
        if model_max < common:
            raise ValueError(
                f"model max_context_tokens={model_max} is smaller than the common RULER window {common}"
            )
        if model_max != common:
            windows.append(model_max)
    return windows


def _ruler_sample_window(
    sample: Sample, windows: list[int], max_output_tokens: int
) -> int | None:
    reference = sample.reference
    raw = sample.meta.get("context_window_tokens")
    if raw is None:
        raw = sample.meta.get("input_tokens")
    if raw is None:
        raw = getattr(reference, "context_length", None)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    if value in windows:
        return value
    prompt_window = value + max_output_tokens
    return prompt_window if prompt_window in windows else None


def _annotate_ruler(
    sample: Sample, window: int, max_output_tokens: int
) -> Sample:
    return _with_meta(
        sample,
        context_window_tokens=window,
        input_tokens=window - max_output_tokens,
        target_input_tokens=window - max_output_tokens,
        max_new_tokens=max_output_tokens,
    )


def _eligible_ruler_samples(
    samples: list[Sample], windows: list[int], max_output_tokens: int
) -> list[tuple[Sample, int]]:
    eligible: list[tuple[Sample, int]] = []
    for sample in samples:
        window = _ruler_sample_window(sample, windows, max_output_tokens)
        if window is not None:
            eligible.append((sample, window))
    return eligible


def _select_ruler(
    samples: list[Sample],
    config: dict[str, Any],
    model_config: dict[str, Any],
    n_samples: int | None,
    rng: random.Random,
) -> list[Sample]:
    windows = _ruler_windows(config, model_config)
    max_output_tokens = _positive_int(
        config.get("max_output_tokens"), "RULER max_output_tokens"
    )
    if any(window <= max_output_tokens for window in windows):
        raise ValueError("every RULER context window must exceed max_output_tokens")

    eligible = _eligible_ruler_samples(samples, windows, max_output_tokens)
    if n_samples is not None:
        count = _generic_count(config, n_samples)
        chosen = _take(
            [sample for sample, _ in eligible], count, rng, "RULER override"
        )
        window_by_id = {sample.sample_id: window for sample, window in eligible}
        return [
            _annotate_ruler(sample, window_by_id[sample.sample_id], max_output_tokens)
            for sample in chosen
        ]

    task_types = list(config.get("task_types", []))
    positions = list(config.get("positions", []))
    if not task_types or not positions:
        raise ValueError("RULER config must declare task_types and positions")
    per_group = _positive_int(
        config.get("samples_per_context_window_position"),
        "samples_per_context_window_position",
    )
    task_counts = _balanced_counts(per_group, len(task_types))

    selected: list[Sample] = []
    for window in windows:
        for position_index, position in enumerate(positions):
            # Rotate which task receives the remainder so 10 samples per
            # position become 10 samples per task across all three positions.
            rotated_tasks = (
                task_types[position_index:] + task_types[:position_index]
            )
            for task_type, count in zip(rotated_tasks, task_counts):
                group = [
                    sample
                    for sample, sample_window in eligible
                    if sample_window == window
                    and getattr(sample.reference, "task_type", None) == task_type
                    and getattr(sample.reference, "position", None) == position
                ]
                chosen = _take(
                    group,
                    count,
                    rng,
                    f"RULER window={window} position={position!r} task={task_type!r}",
                )
                selected.extend(
                    _annotate_ruler(sample, window, max_output_tokens)
                    for sample in chosen
                )
    return selected


def select_configured_samples(
    samples: list[Sample],
    dataset_config: dict[str, Any],
    model_config: dict[str, Any],
    *,
    n_samples: int | None = None,
    seed: int = 42,
) -> list[Sample]:
    """Select the configured formal subset and attach per-sample run limits."""
    if not samples:
        raise ValueError("cannot select samples from an empty input")
    if n_samples is not None and n_samples <= 0:
        raise ValueError("n_samples must be greater than zero")

    rng = random.Random(seed)
    dataset_name = dataset_config.get("dataset")
    if dataset_name in {"sudoku", "sudoku_trace"}:
        return _select_sudoku(samples, dataset_config, n_samples, rng)
    if dataset_name == "hellobench":
        return _select_hellobench(samples, dataset_config, n_samples, rng)
    if dataset_name == "ruler":
        return _select_ruler(samples, dataset_config, model_config, n_samples, rng)
    return _take(
        samples,
        _generic_count(dataset_config, n_samples),
        rng,
        str(dataset_name or "dataset"),
    )
