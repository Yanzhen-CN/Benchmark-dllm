"""Stage 2: scoring only. Reads generations back from ``model_output/`` (no
model adapter, no GPU, no torch needed — see ``persistence.generation_result_from_dict``),
writes one `ScoreResult` JSON per sample plus an aggregate ``summary.json``
under ``output/score_output/<model>_<config>/<dataset>/``, and skips
re-scoring samples that already have a score file — the same per-sample
resume behavior as ``generate_stage``, useful since some scorers (MBPP's code
execution) aren't free.

This is deliberately decoupled from :class:`~dllm_bench.interfaces.ModelAdapter`:
scoring a run that was generated on a different machine only ever needs
``model_output/*.json`` plus the dataset's scoring logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..datasets.base import Dataset, Sample, ScoreResult
from ..interfaces import RunStatus
from .orchestrator import RunSummary, SampleRecord, summarize_records
from .persistence import (
    load_generation_result,
    load_meta,
    load_score_result,
    save_run_summary,
    save_score_result,
)


@dataclass
class ScoreStageResult:
    summary: RunSummary
    scored: int
    skipped: int
    missing_sample_ids: list[str] = field(default_factory=list)


class InvalidTestError(RuntimeError):
    """Raised when generation marked the complete model×dataset test invalid."""


class IncompleteTestError(RuntimeError):
    """Raised when not every selected generation is available for aggregation."""


def ensure_test_valid(model_output_dir: str | Path) -> dict:
    """Return generation metadata, rejecting OOM-invalidated tests."""
    model_output_dir = Path(model_output_dir)
    meta_path = model_output_dir / "_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"no _meta.json under {model_output_dir} — run generation first "
            f"(dllm-bench generate ...) before scoring"
        )
    meta = load_meta(meta_path)
    oom_info_path = model_output_dir / "oom_info.json"
    invalid_detail = meta if meta.get("test_valid") is False else None
    invalid_path = oom_info_path
    if invalid_detail is not None or oom_info_path.exists():
        detail = (
            invalid_detail
            if invalid_detail is not None
            else load_meta(oom_info_path)
        )
        stage = detail.get("failure_stage") or detail.get("early_stop", {}).get(
            "failure_stage", "generation"
        )
        ordinal = detail.get("sample_ordinal") or detail.get("early_stop", {}).get(
            "sample_ordinal"
        )
        sample_id = detail.get("sample_id") or detail.get("early_stop", {}).get(
            "sample_id"
        )
        location = f"sample {ordinal} ({sample_id})" if ordinal else stage
        raise InvalidTestError(
            f"test is invalid because of OOM at {location}; see {invalid_path}"
        )
    if meta.get("test_complete") is False:
        completed = int(meta.get("completed_samples", 0))
        selected = meta.get("selected_samples", "unknown")
        raise IncompleteTestError(
            f"generation is incomplete under {model_output_dir}: "
            f"completed {completed} of {selected} selected samples"
        )
    return meta


def run_scoring(
    dataset: Dataset,
    samples: list[Sample],
    model_output_dir: str | Path,
    score_output_dir: str | Path,
    resume: bool = True,
) -> ScoreStageResult:
    if not samples:
        raise ValueError("samples must be non-empty")

    model_output_dir = Path(model_output_dir)
    score_output_dir = Path(score_output_dir)
    meta = ensure_test_valid(model_output_dir)
    score_output_dir.mkdir(parents=True, exist_ok=True)

    records: list[SampleRecord] = []
    missing: list[str] = []
    scored = skipped = 0

    for sample in samples:
        generation_path = model_output_dir / f"{sample.sample_id}.json"
        if not generation_path.exists():
            missing.append(sample.sample_id)
            continue

        score_path = score_output_dir / f"{sample.sample_id}.json"
        generation = load_generation_result(generation_path)

        if resume and score_path.exists():
            score = load_score_result(score_path)
            skipped += 1
        else:
            if generation.status == RunStatus.SUCCESS:
                score = dataset.score(sample, generation.output_text)
                score.aux.update(dataset.trace_aux_metrics(sample, generation.trace))
            else:
                score = ScoreResult(primary_score=0.0, valid=False, complete=False)
            save_score_result(score, score_path)
            scored += 1

        records.append(SampleRecord(sample=sample, generation=generation, score=score))

    if not records:
        raise RuntimeError(
            f"no generated samples found under {model_output_dir} for the "
            f"requested sample set — run generation first"
        )

    if missing:
        # Per-sample scores are resumable, but a partial formal aggregate must
        # never survive as a reportable benchmark row.
        (score_output_dir / "summary.json").unlink(missing_ok=True)
        raise IncompleteTestError(
            f"{len(missing)} of {len(samples)} selected generation(s) are missing "
            f"under {model_output_dir}: {missing}"
        )

    summary = summarize_records(
        meta["model_name"], meta["config_name"], dataset, records,
        run_metadata=meta.get("run_metadata", {}),
    )
    save_run_summary(summary, score_output_dir / "summary.json")

    return ScoreStageResult(
        summary=summary, scored=scored, skipped=skipped, missing_sample_ids=missing
    )
