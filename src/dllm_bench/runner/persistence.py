"""JSON (de)serialization for the three stage artifacts:

- :class:`~dllm_bench.interfaces.GenerationResult` (full fidelity, including
  ``trace`` — this is the ``model_output/`` per-sample file, and the only
  thing the score/visualize stages read to avoid re-running generation).
- :class:`~dllm_bench.datasets.base.ScoreResult` (the ``score_output/``
  per-sample file).
- :class:`~dllm_bench.runner.orchestrator.RunSummary` (the ``score_output/
  summary.json`` aggregate — deliberately trace-and-per-sample-record-free,
  see :func:`run_summary_to_dict`).

Round-tripping ``GenerationResult`` is what lets generation run on one
machine (e.g. a GPU box) and scoring/visualization run on another — only
``model_output/*.json`` needs to cross that gap.
"""

from __future__ import annotations

import dataclasses
import json
from enum import Enum
from pathlib import Path
from typing import Any

from ..datasets.base import ScoreResult
from ..interfaces import (
    GenerationRequest,
    GenerationResult,
    ForwardProfile,
    PositionState,
    RunStatus,
    TimingResult,
    TraceStep,
)
from .orchestrator import RunSummary


_LEGACY_EOS_TOKEN_TEXTS = frozenset(
    {
        "<[EOS]>",
        "</s>",
        "<|eos|>",
        "<|eot_id|>",
        "<|endoftext|>",
        "<|im_end|>",
    }
)


def _legacy_trace_eos_position(trace: list[TraceStep]) -> int | None:
    """Recover EOS from old artifacts whose ``output_text`` hid it."""
    if not trace or not trace[-1].token_texts:
        return None
    for position, token_text in enumerate(trace[-1].token_texts or []):
        if token_text.strip() in _LEGACY_EOS_TOKEN_TEXTS:
            return position
    return None


def _truncate_trace_step(step: TraceStep, valid_length: int) -> TraceStep:
    limit = min(valid_length, len(step.token_ids))
    token_texts = (
        step.token_texts[:limit] if step.token_texts is not None else None
    )
    return TraceStep(
        forward_index=step.forward_index,
        token_ids=step.token_ids[:limit],
        position_states=step.position_states[:limit],
        committed_positions=[
            position
            for position in step.committed_positions
            if position < valid_length
        ],
        decoded_text=(
            "".join(token_texts)
            if token_texts is not None
            else step.decoded_text
        ),
        entropy_by_position=(
            {
                position: value
                for position, value in step.entropy_by_position.items()
                if int(position) < valid_length
            }
            if step.entropy_by_position is not None
            else None
        ),
        top1_confidence_by_position=(
            {
                position: value
                for position, value in step.top1_confidence_by_position.items()
                if int(position) < valid_length
            }
            if step.top1_confidence_by_position is not None
            else None
        ),
        token_texts=token_texts,
    )


def _recover_legacy_eos_boundary(
    generation: GenerationResult,
) -> GenerationResult:
    position = _legacy_trace_eos_position(generation.trace)
    if position is None:
        return generation
    final_token_texts = generation.trace[-1].token_texts or []
    generation.output_text = "".join(final_token_texts[:position])
    generation.final_valid_length = position
    generation.trace = [
        _truncate_trace_step(step, position) for step in generation.trace
    ]
    generation.extra = dict(generation.extra)
    generation.extra.setdefault("stop_reason", "eos")
    generation.extra.setdefault("stop_position", position)
    generation.extra.setdefault("eos_boundary_recovered_from_trace", True)
    return generation


def _to_jsonable(obj: Any, include_trace: bool) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        result = {}
        for f in dataclasses.fields(obj):
            if f.name == "trace" and not include_trace:
                result[f.name] = []
                continue
            if f.name == "records":
                continue  # per-sample records are large; summary.json stays aggregate-only
            result[f.name] = _to_jsonable(getattr(obj, f.name), include_trace)
        return result
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: _to_jsonable(v, include_trace) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v, include_trace) for v in obj]
    return obj


def run_summary_to_dict(summary: RunSummary, include_trace: bool = False) -> dict[str, Any]:
    return _to_jsonable(summary, include_trace)


def save_run_summary(summary: RunSummary, path: str | Path, include_trace: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(run_summary_to_dict(summary, include_trace=include_trace), f, indent=2, ensure_ascii=False)


def load_run_summary_dict(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def generation_result_to_dict(generation: GenerationResult) -> dict[str, Any]:
    """Full-fidelity dict, including ``trace`` — unlike ``run_summary_to_dict``,
    this is meant to be read back into an equivalent ``GenerationResult``."""
    return {
        "output_text": generation.output_text,
        "status": generation.status.value,
        "num_forward_passes": generation.num_forward_passes,
        "final_valid_length": generation.final_valid_length,
        "timing": (
            {"wall_clock_seconds": generation.timing.wall_clock_seconds, "source": generation.timing.source}
            if generation.timing
            else None
        ),
        "energy_joules": generation.energy_joules,
        "compute_tflops": generation.compute_tflops,
        "peak_vram_gb": generation.peak_vram_gb,
        "error_message": generation.error_message,
        "extra": generation.extra,
        "request": {
            "prompt": generation.request.prompt,
            "max_new_tokens": generation.request.max_new_tokens,
            "config": generation.request.config,
            "sample_id": generation.request.sample_id,
            "seed": generation.request.seed,
        },
        "trace": [
            {
                "forward_index": step.forward_index,
                "token_ids": step.token_ids,
                "position_states": [state.value for state in step.position_states],
                "committed_positions": step.committed_positions,
                "decoded_text": step.decoded_text,
                "entropy_by_position": step.entropy_by_position,
                "top1_confidence_by_position": step.top1_confidence_by_position,
                "token_texts": step.token_texts,
            }
            for step in generation.trace
        ],
        "forward_profiles": [dataclasses.asdict(profile) for profile in generation.forward_profiles],
    }


def generation_result_from_dict(data: dict[str, Any]) -> GenerationResult:
    request_data = data["request"]
    request = GenerationRequest(
        prompt=request_data["prompt"],
        max_new_tokens=request_data["max_new_tokens"],
        config=request_data.get("config", {}),
        sample_id=request_data.get("sample_id"),
        seed=request_data.get("seed", 42),
    )
    trace = [
        TraceStep(
            forward_index=step["forward_index"],
            token_ids=step["token_ids"],
            position_states=[PositionState(v) for v in step["position_states"]],
            committed_positions=step["committed_positions"],
            decoded_text=step["decoded_text"],
            entropy_by_position=step.get("entropy_by_position"),
            top1_confidence_by_position=step.get("top1_confidence_by_position"),
            token_texts=step.get("token_texts"),
        )
        for step in data.get("trace", [])
    ]
    timing = TimingResult(**data["timing"]) if data.get("timing") else None
    forward_profiles = [
        ForwardProfile(**profile) for profile in data.get("forward_profiles", [])
    ]
    generation = GenerationResult(
        request=request,
        output_text=data["output_text"],
        status=RunStatus(data["status"]),
        trace=trace,
        forward_profiles=forward_profiles,
        num_forward_passes=data["num_forward_passes"],
        final_valid_length=data["final_valid_length"],
        timing=timing,
        energy_joules=data.get("energy_joules"),
        compute_tflops=data.get("compute_tflops"),
        peak_vram_gb=data.get("peak_vram_gb"),
        error_message=data.get("error_message"),
        extra=data.get("extra", {}),
    )
    return _recover_legacy_eos_boundary(generation)


def save_generation_result(generation: GenerationResult, path: str | Path) -> None:
    _write_json(generation_result_to_dict(generation), path)


def load_generation_result(path: str | Path) -> GenerationResult:
    return generation_result_from_dict(_read_json(path))


def score_result_to_dict(score: ScoreResult) -> dict[str, Any]:
    return {
        "primary_score": score.primary_score,
        "aux": score.aux,
        "valid": score.valid,
        "complete": score.complete,
    }


def score_result_from_dict(data: dict[str, Any]) -> ScoreResult:
    return ScoreResult(
        primary_score=data["primary_score"],
        aux=data.get("aux", {}),
        valid=data.get("valid", True),
        complete=data.get("complete", True),
    )


def save_score_result(
    score: ScoreResult, path: str | Path, metadata: dict[str, Any] | None = None
) -> None:
    payload = score_result_to_dict(score)
    if metadata is not None:
        payload["_score_metadata"] = metadata
    _write_json(payload, path)


def load_score_result(path: str | Path) -> ScoreResult:
    return score_result_from_dict(_read_json(path))


def load_score_metadata(path: str | Path) -> dict[str, Any]:
    return dict(_read_json(path).get("_score_metadata", {}))


def save_meta(meta: dict[str, Any], path: str | Path) -> None:
    _write_json(meta, path)


def load_meta(path: str | Path) -> dict[str, Any]:
    return _read_json(path)
