"""Stage 2: scoring only. Reads generations back from ``model_output/`` (no
model adapter, no GPU, no torch needed — see ``persistence.generation_result_from_dict``),
writes one `ScoreResult` JSON per sample plus an aggregate ``summary.json``
under ``output/score_output/<model>/<config>/<dataset>/``, and skips
re-scoring samples that already have a score file — the same per-sample
resume behavior as ``generate_stage``, useful since some scorers (MBPP's code
execution) aren't free.

This is deliberately decoupled from :class:`~dllm_bench.interfaces.ModelAdapter`:
scoring a run that was generated on a different machine only ever needs
``model_output/*.json`` plus the dataset's scoring logic.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..datasets.base import Dataset, Sample, ScoreResult
from ..interfaces import RunStatus
from .orchestrator import RunSummary, SampleRecord, summarize_records
from .persistence import (
    load_generation_result,
    load_meta,
    load_score_metadata,
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
    preview: bool = False


class InvalidTestError(RuntimeError):
    """Raised when generation marked the complete model×dataset test invalid."""


class IncompleteTestError(RuntimeError):
    """Raised when not every selected generation is available for aggregation."""


def _hash_payload(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, default=repr, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _scorer_revision(dataset: Dataset) -> str:
    paths = {Path(__file__), Path(inspect.getsourcefile(type(dataset)) or __file__)}
    dataset_root = Path(inspect.getsourcefile(Dataset) or __file__).parent
    package_root = dataset_root.parent
    for candidate in (
        dataset_root / "answer_region.py",
        dataset_root / "official_metrics.py",
        package_root / "metrics" / "strategy_score.py",
        Path(__file__).with_name("orchestrator.py"),
    ):
        if candidate.exists():
            paths.add(candidate)
    digest = hashlib.sha256()
    for path in sorted(paths, key=str):
        digest.update(str(path.name).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _dataset_revision(dataset: Dataset, samples: list[Sample]) -> str:
    signature = getattr(dataset, "preparation_signature", None)
    if callable(signature):
        return _hash_payload(signature())
    return _hash_payload(
        [
            {
                key: sample.meta.get(key)
                for key in (
                    "source",
                    "source_revision",
                    "protocol_revision",
                    "prompt_protocol",
                )
            }
            for sample in samples
        ]
    )


def _generation_hash(generation) -> str:
    return _hash_payload(
        {
            "status": generation.status.value,
            "output_text": generation.output_text,
            "request_prompt": generation.request.prompt,
            "request_max_new_tokens": generation.request.max_new_tokens,
            "request_config": generation.request.config,
            "request_seed": generation.request.seed,
        }
    )


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
    primary_metric: str | None = None,
) -> ScoreStageResult:
    if not samples:
        raise ValueError("samples must be non-empty")

    model_output_dir = Path(model_output_dir)
    score_output_dir = Path(score_output_dir)
    preview = os.environ.get("DLLM_SCORE_PREVIEW") == "1"
    meta = ensure_test_valid(model_output_dir)
    expected_ids = [sample.sample_id for sample in samples]
    generated_ids = list(meta.get("selected_sample_ids", []))
    if generated_ids != expected_ids:
        if not preview:
            (score_output_dir / "summary.json").unlink(missing_ok=True)
        raise IncompleteTestError(
            "ordered selected_sample_ids mismatch: generation has "
            f"{generated_ids}, scoring requested {expected_ids}"
        )
    if not preview:
        score_output_dir.mkdir(parents=True, exist_ok=True)

    metric_name = primary_metric or f"{dataset.name}_score"
    dataset_revision = _dataset_revision(dataset, samples)
    prompt_protocol_revision = _hash_payload(
        [
            {
                "sample_id": sample.sample_id,
                "prompt": sample.prompt,
                "protocol_revision": sample.meta.get("protocol_revision"),
                "prompt_protocol": sample.meta.get("prompt_protocol"),
            }
            for sample in samples
        ]
    )
    scorer_revision = _scorer_revision(dataset)
    scoring_protocol = dataset.scoring_signature()
    sample_set_hash = _hash_payload(expected_ids)

    records: list[SampleRecord] = []
    missing: list[str] = []
    scored = skipped = 0
    score_fingerprints: list[str] = []

    for sample in samples:
        generation_path = model_output_dir / f"{sample.sample_id}.json"
        if not generation_path.exists():
            missing.append(sample.sample_id)
            continue

        score_path = score_output_dir / f"{sample.sample_id}.json"
        generation = load_generation_result(generation_path)

        if generation.status not in {RunStatus.SUCCESS, RunStatus.TRUNCATED}:
            if not preview:
                (score_output_dir / "summary.json").unlink(missing_ok=True)
            raise InvalidTestError(
                f"{sample.sample_id} has infrastructure status "
                f"{generation.status.value}; the complete dataset row is invalid"
            )

        generation_hash = _generation_hash(generation)
        score_metadata = {
            "sample_id": sample.sample_id,
            "generation_status": generation.status.value,
            "dataset_revision": dataset_revision,
            "prompt_protocol_revision": prompt_protocol_revision,
            "scorer_revision": scorer_revision,
            "sample_set_hash": sample_set_hash,
            "generation_text_hash": generation_hash,
            "primary_metric": metric_name,
            "scoring_protocol": scoring_protocol,
        }
        score_metadata["fingerprint"] = _hash_payload(score_metadata)
        score_fingerprints.append(score_metadata["fingerprint"])

        if (
            not preview
            and resume
            and score_path.exists()
            and load_score_metadata(score_path).get("fingerprint")
            == score_metadata["fingerprint"]
        ):
            score = load_score_result(score_path)
            skipped += 1
        else:
            score = dataset.score_generation(sample, generation)
            score.aux["truncated_rate"] = float(
                generation.status == RunStatus.TRUNCATED
            )
            if generation.status == RunStatus.TRUNCATED:
                score.complete = False
            if not preview:
                save_score_result(score, score_path, metadata=score_metadata)
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
        if not preview:
            (score_output_dir / "summary.json").unlink(missing_ok=True)
        raise IncompleteTestError(
            f"{len(missing)} of {len(samples)} selected generation(s) are missing "
            f"under {model_output_dir}: {missing}"
        )

    generation_protocol_revision = _hash_payload(
        [
            {
                "sample_id": record.sample.sample_id,
                "prompt": record.generation.request.prompt,
                "max_new_tokens": record.generation.request.max_new_tokens,
                "config": record.generation.request.config,
                "seed": record.generation.request.seed,
            }
            for record in records
        ]
    )

    summary = summarize_records(
        meta["model_name"], meta["config_name"], dataset, records,
        run_metadata=meta.get("run_metadata", {}),
    )
    summary.scoring_metadata = {
        "fingerprint": _hash_payload(score_fingerprints),
        "dataset_revision": dataset_revision,
        "prompt_protocol_revision": prompt_protocol_revision,
        "generation_protocol_revision": generation_protocol_revision,
        "scorer_revision": scorer_revision,
        "sample_set_hash": sample_set_hash,
        "primary_metric": metric_name,
        "scoring_protocol": scoring_protocol,
        "expected_sample_count": len(expected_ids),
        "actual_scored_sample_count": len(records),
        "aggregation_method": "micro_mean_over_exact_selected_sample_set",
    }
    if not preview:
        save_run_summary(summary, score_output_dir / "summary.json")

    return ScoreStageResult(
        summary=summary,
        scored=scored,
        skipped=skipped,
        missing_sample_ids=missing,
        preview=preview,
    )
