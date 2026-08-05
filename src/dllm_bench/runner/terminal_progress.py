"""Compact terminal rendering for long generation runs."""

from __future__ import annotations


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    rounded = max(0, int(round(seconds)))
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def progress_bar(index: int, total: int, *, width: int = 20) -> str:
    ratio = min(1.0, max(0.0, index / total)) if total > 0 else 0.0
    filled = min(width, max(0, int(round(width * ratio))))
    return f"[{'#' * filled}{'-' * (width - filled)}]"


def generation_progress_text(
    *,
    variant: str,
    index: int,
    total: int,
    sample_id: str,
    run_elapsed: float,
    estimated_total: float | None,
    sample_elapsed: float,
    expected_sample: float | None,
    status: str | None = None,
) -> str:
    sample_estimate = "--" if expected_sample is None else f"{expected_sample:.1f}"
    failure = f" | {status}" if status and status != "success" else ""
    return (
        f"[{variant}] {progress_bar(index, total)} {index}/{total} "
        f"{format_duration(run_elapsed)}/~{format_duration(estimated_total)} | "
        f"{sample_id} {sample_elapsed:.1f}/{sample_estimate}s{failure}"
    )
