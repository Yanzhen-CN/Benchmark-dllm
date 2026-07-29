"""Stage 1: generation only. Writes one JSON per sample under
``output/model_output/<model>_<config>/<dataset>/`` (see ``output_layout.py``)
and skips samples that already have a file there — the resume behavior that
lets a half-finished dataset run continue without redoing already-generated
samples, and lets generation happen entirely separately from scoring (e.g.
on a GPU box, with ``model_output/`` copied elsewhere afterward).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from ..datasets.base import Sample
from ..interfaces import GenerationRequest, GenerationResult, ModelAdapter, RunStatus
from .persistence import load_generation_result, save_generation_result, save_meta
from .sampling import DEFAULT_SEED, collect_run_metadata


@dataclass
class GenerateStageSummary:
    out_dir: str
    generated: int
    skipped: int
    total: int
    stopped_early: bool = False
    stop_reason: str | None = None
    first_oom_sample_id: str | None = None
    first_oom_ordinal: int | None = None


OOM_CONTEXT_META_KEYS = (
    "target_output_words",
    "max_new_tokens",
    "context_window_tokens",
    "target_input_tokens",
    "position",
    "task_type",
    "measurement_role",
    "declared_max_context_tokens",
    "model_context_fraction",
)


GenerationProgress = Callable[
    [str, int, int, Sample, GenerationResult | None], None
]

MEASUREMENT_PROTOCOL = "gpu-synced-v4-trace-excluded-compute-deferred"
OOM_INFO_FILENAME = "oom_info.json"


def _measurement_protocol(adapter: ModelAdapter) -> str:
    return str(getattr(adapter, "measurement_protocol", MEASUREMENT_PROTOCOL))


class OOMInvalidTestError(RuntimeError):
    """The current model×dataset test is invalid because CUDA OOM occurred."""


def oom_invalid_test_error(out_dir: str | Path, detail: dict) -> OOMInvalidTestError:
    out_dir = Path(out_dir)
    ordinal = detail.get("sample_ordinal")
    sample_id = detail.get("sample_id")
    stage = detail.get("failure_stage", "generation")
    error_message = detail.get("error_message") or "CUDA out of memory"
    location = (
        f"sample {ordinal} ({sample_id})"
        if ordinal is not None and sample_id
        else stage
    )
    return OOMInvalidTestError(
        f"OOM invalidated the complete model×dataset test at {location}: "
        f"{error_message}. Details: {out_dir / OOM_INFO_FILENAME}"
    )


def is_cuda_oom_error(exc: BaseException) -> bool:
    """Recognize CUDA OOM without requiring torch in the local scoring venv."""
    return (
        exc.__class__.__name__ == "OutOfMemoryError"
        or "cuda out of memory" in str(exc).lower()
        or "cuda error: out of memory" in str(exc).lower()
    )


def persist_setup_oom_invalidation(
    *,
    adapter: ModelAdapter,
    dataset_name: str,
    out_dir: str | Path,
    selected_samples: int,
    failure_stage: str,
    error: BaseException,
    seed: int,
    measure_compute: bool,
    require_all_metrics: bool,
    capture_trace: bool,
) -> dict:
    """Record an OOM during model loading or warmup before sample generation."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_metadata = collect_run_metadata(
        adapter,
        {
            "seed": seed,
            "measurement_protocol": _measurement_protocol(adapter),
            "measure_compute": measure_compute,
            "require_all_metrics": require_all_metrics,
            "trace_scope": "all_samples" if capture_trace else "none",
            "energy_backend": "nvml-total-energy",
        },
    )
    early_stop = {
        "reason": "oom",
        "failure_stage": failure_stage,
        "sample_ordinal": None,
        "sample_id": None,
        "attempted_samples": 0,
        "selected_samples": selected_samples,
        "remaining_samples_not_attempted": selected_samples,
        "sample_context": {},
    }
    meta = {
        "model_name": adapter.name,
        "config_name": adapter.config_name,
        "dataset_name": dataset_name,
        "test_valid": False,
        "test_complete": False,
        "invalid_reason": "oom",
        "oom_info_file": OOM_INFO_FILENAME,
        "early_stop": early_stop,
        "run_metadata": run_metadata,
    }
    save_meta(meta, out_dir / "_meta.json")
    oom_info = {
        "schema_version": 1,
        "test_valid": False,
        "invalid_reason": "oom",
        "scope": "model_x_variant_x_dataset",
        "model_name": adapter.name,
        "config_name": adapter.config_name,
        "dataset_name": dataset_name,
        "sample_ordinal": None,
        "sample_id": None,
        "sample_context": {},
        "attempted_samples": 0,
        "selected_samples": selected_samples,
        "remaining_samples_not_attempted": selected_samples,
        "failure_stage": failure_stage,
        "error_type": "cuda_out_of_memory",
        "error_class": error.__class__.__name__,
        "error_message": str(error),
        "gpu": {
            "cuda_visible_devices": run_metadata.get("cuda_visible_devices"),
            "cuda_current_device": run_metadata.get("cuda_current_device"),
            "devices": run_metadata.get("cuda_device_details", []),
        },
        "run_metadata": run_metadata,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_meta(oom_info, out_dir / OOM_INFO_FILENAME)
    return oom_info


def _oom_sample_context(sample: Sample) -> dict:
    return {
        key: sample.meta[key]
        for key in OOM_CONTEXT_META_KEYS
        if key in sample.meta
    }


def _persist_oom_invalidation(
    *,
    meta_path: Path,
    sample_path: Path,
    sample: Sample,
    sample_ordinal: int,
    selected_samples: int,
    generation: GenerationResult,
) -> None:
    """Persist the first OOM as a test-level invalidation, not a scoreable sample."""
    sample_context = _oom_sample_context(sample)
    generation.extra.update(
        {
            "long_task_oom_stop": True,
            "test_valid": False,
            "invalid_reason": "oom",
            "oom_sample_ordinal": sample_ordinal,
            "oom_sample_id": sample.sample_id,
            "attempted_samples": sample_ordinal,
            "selected_samples": selected_samples,
            "oom_sample_context": sample_context,
        }
    )
    save_generation_result(generation, sample_path)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    early_stop = {
        "reason": "oom",
        "sample_ordinal": sample_ordinal,
        "sample_id": sample.sample_id,
        "attempted_samples": sample_ordinal,
        "selected_samples": selected_samples,
        "remaining_samples_not_attempted": selected_samples - sample_ordinal,
        "sample_context": sample_context,
    }
    meta.update(
        {
            "test_valid": False,
            "test_complete": False,
            "invalid_reason": "oom",
            "oom_info_file": OOM_INFO_FILENAME,
            "early_stop": early_stop,
        }
    )
    save_meta(meta, meta_path)

    run_metadata = meta.get("run_metadata", {})
    oom_info = {
        "schema_version": 1,
        "test_valid": False,
        "invalid_reason": "oom",
        "scope": "model_x_variant_x_dataset",
        "model_name": meta.get("model_name"),
        "config_name": meta.get("config_name"),
        "dataset_name": meta.get("dataset_name"),
        "sample_ordinal": sample_ordinal,
        "sample_id": sample.sample_id,
        "sample_context": sample_context,
        "attempted_samples": sample_ordinal,
        "selected_samples": selected_samples,
        "remaining_samples_not_attempted": selected_samples - sample_ordinal,
        "failure_stage": "generation",
        "error_type": "cuda_out_of_memory",
        "error_message": generation.error_message,
        "gpu": {
            "cuda_visible_devices": run_metadata.get("cuda_visible_devices"),
            "cuda_current_device": run_metadata.get("cuda_current_device"),
            "devices": run_metadata.get("cuda_device_details", []),
        },
        "run_metadata": run_metadata,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_meta(oom_info, meta_path.parent / OOM_INFO_FILENAME)


def _validate_required_metrics(
    adapter: ModelAdapter,
    generation: GenerationResult,
    *,
    require_compute: bool,
    require_trace: bool,
) -> None:
    if generation.status.value != "success":
        return
    missing: list[str] = []
    if generation.timing is None or generation.timing.wall_clock_seconds <= 0:
        missing.append("timing")
    if not adapter.natively_measures_resources:
        if generation.energy_joules is None:
            missing.append("energy_joules")
        if generation.peak_vram_gb is None:
            missing.append("peak_vram_gb")
        if require_compute and generation.compute_tflops is None:
            missing.append("compute_tflops")
    if require_trace and adapter.supports_trace and not generation.trace:
        missing.append("trace")
    if missing:
        raise RuntimeError(
            f"sample {generation.request.sample_id} is missing required formal metrics: "
            f"{', '.join(missing)}"
        )


def run_generation(
    adapter: ModelAdapter,
    dataset_name: str,
    samples: list[Sample],
    max_new_tokens: int,
    out_dir: str | Path,
    extra_config: dict | None = None,
    measure_compute: bool = False,
    require_all_metrics: bool = False,
    capture_trace: bool = True,
    seed: int = DEFAULT_SEED,
    resume: bool = True,
    force_max_new_tokens: bool = False,
    progress: GenerationProgress | None = None,
) -> GenerateStageSummary:
    if not samples:
        raise ValueError("samples must be non-empty")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir / "_meta.json"
    if resume and meta_path.exists():
        existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        existing_run = existing_meta.get("run_metadata", {})
        expected_protocol = _measurement_protocol(adapter)
        if existing_run.get("measurement_protocol") != expected_protocol:
            raise RuntimeError(
                f"existing outputs under {out_dir} use an incompatible measurement "
                "protocol; rerun with --no-resume (overwrites this dataset) or "
                "choose a fresh --output-root"
            )
        early_stop = existing_meta.get("early_stop")
        if early_stop and early_stop.get("reason") == "oom":
            oom_info_path = out_dir / OOM_INFO_FILENAME
            if not oom_info_path.exists():
                ordinal = int(early_stop["sample_ordinal"])
                oom_sample = next(
                    (
                        sample
                        for sample in samples
                        if sample.sample_id == early_stop.get("sample_id")
                    ),
                    samples[ordinal - 1],
                )
                oom_sample_path = out_dir / f"{oom_sample.sample_id}.json"
                if oom_sample_path.exists():
                    _persist_oom_invalidation(
                        meta_path=meta_path,
                        sample_path=oom_sample_path,
                        sample=oom_sample,
                        sample_ordinal=ordinal,
                        selected_samples=len(samples),
                        generation=load_generation_result(oom_sample_path),
                    )
            detail = (
                json.loads(oom_info_path.read_text(encoding="utf-8"))
                if oom_info_path.exists()
                else {
                    **early_stop,
                    "error_message": "CUDA out of memory",
                }
            )
            raise oom_invalid_test_error(out_dir, detail)
    if not (resume and meta_path.exists()):
        save_meta(
            {
                "model_name": adapter.name,
                "config_name": adapter.config_name,
                "dataset_name": dataset_name,
                "test_valid": True,
                "test_complete": False,
                "selected_samples": len(samples),
                "selected_sample_ids": [sample.sample_id for sample in samples],
                "run_metadata": collect_run_metadata(
                    adapter,
                    {
                        "measurement_protocol": _measurement_protocol(adapter),
                        "measure_compute": measure_compute,
                        "require_all_metrics": require_all_metrics,
                        "trace_scope": "all_samples" if capture_trace else "none",
                        "energy_backend": "nvml-total-energy",
                        "max_new_tokens_override": (
                            max_new_tokens if force_max_new_tokens else None
                        ),
                    },
                ),
            },
            meta_path,
        )

    generated = skipped = 0
    stopped_early = False
    first_oom_sample_id: str | None = None
    first_oom_ordinal: int | None = None
    compute_queue: list[
        tuple[int, Sample, Path, GenerationResult]
    ] = []
    for index, sample in enumerate(samples, start=1):
        sample_path = out_dir / f"{sample.sample_id}.json"
        sample_max_new_tokens = (
            int(max_new_tokens)
            if force_max_new_tokens
            else int(sample.meta.get("max_new_tokens", max_new_tokens))
        )
        if sample_max_new_tokens <= 0:
            raise ValueError(
                f"sample {sample.sample_id} has invalid max_new_tokens={sample_max_new_tokens}"
        )
        if resume and sample_path.exists():
            existing = load_generation_result(sample_path)
            skipped += 1
            if existing.status is RunStatus.OOM:
                stopped_early = True
                first_oom_sample_id = sample.sample_id
                first_oom_ordinal = index
                _persist_oom_invalidation(
                    meta_path=meta_path,
                    sample_path=sample_path,
                    sample=sample,
                    sample_ordinal=index,
                    selected_samples=len(samples),
                    generation=existing,
                )
                detail = json.loads(
                    (out_dir / OOM_INFO_FILENAME).read_text(encoding="utf-8")
                )
                raise oom_invalid_test_error(out_dir, detail)
            if existing.request.max_new_tokens != sample_max_new_tokens:
                raise RuntimeError(
                    f"existing sample {sample.sample_id} under {out_dir} used "
                    f"max_new_tokens={existing.request.max_new_tokens}, but the current "
                    f"matrix requires {sample_max_new_tokens}; rerun with --no-resume "
                    "(overwrites this dataset) or choose a fresh --output-root"
                )
            if (
                measure_compute
                and existing.status.value == "success"
                and existing.compute_tflops is None
            ):
                compute_queue.append((index, sample, sample_path, existing))
            continue

        request_config = dict(extra_config or {})
        request_config["capture_trace"] = capture_trace
        if "target_input_tokens" in sample.meta:
            request_config["target_input_tokens"] = int(
                sample.meta["target_input_tokens"]
            )
        request = GenerationRequest(
            prompt=sample.prompt,
            max_new_tokens=sample_max_new_tokens,
            config=request_config,
            sample_id=sample.sample_id,
            seed=seed,
        )
        if progress is not None:
            progress("start", index, len(samples), sample, None)
        generation = adapter.generate(request)

        if generation.status is RunStatus.OOM:
            stopped_early = True
            first_oom_sample_id = sample.sample_id
            first_oom_ordinal = index

        if require_all_metrics:
            _validate_required_metrics(
                adapter,
                generation,
                require_compute=False,
                require_trace=capture_trace,
            )

        if stopped_early:
            _persist_oom_invalidation(
                meta_path=meta_path,
                sample_path=sample_path,
                sample=sample,
                sample_ordinal=index,
                selected_samples=len(samples),
                generation=generation,
            )
        else:
            save_generation_result(generation, sample_path)
        if measure_compute and generation.status.value == "success":
            compute_queue.append((index, sample, sample_path, generation))
        generated += 1
        if progress is not None:
            progress("finish", index, len(samples), sample, generation)
        if stopped_early:
            detail = json.loads(
                (out_dir / OOM_INFO_FILENAME).read_text(encoding="utf-8")
            )
            raise oom_invalid_test_error(out_dir, detail)

    # Compute profiling is intentionally a second phase. Interleaving a full
    # replay between timed samples changes GPU thermal/cache state and can bias
    # subsequent latency and energy. Persist formal generation first so a
    # profiler failure is resumable without generating the sample again.
    for index, sample, sample_path, generation in compute_queue:
        if progress is not None:
            progress("compute", index, len(samples), sample, generation)
        profile_compute = getattr(adapter, "profile_compute", None)
        if callable(profile_compute):
            compute_handle = profile_compute(generation.request)
            generation.compute_tflops = (
                compute_handle.tflops if compute_handle.available else None
            )
        if require_all_metrics:
            _validate_required_metrics(
                adapter,
                generation,
                require_compute=True,
                require_trace=capture_trace,
            )
        save_generation_result(generation, sample_path)

    completed_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    completed_meta.update(
        {
            "test_complete": True,
            "selected_samples": len(samples),
            "completed_samples": len(samples),
            "selected_sample_ids": [sample.sample_id for sample in samples],
        }
    )
    save_meta(completed_meta, meta_path)

    return GenerateStageSummary(
        out_dir=str(out_dir),
        generated=generated,
        skipped=skipped,
        total=len(samples),
        stopped_early=stopped_early,
        stop_reason="oom" if stopped_early else None,
        first_oom_sample_id=first_oom_sample_id,
        first_oom_ordinal=first_oom_ordinal,
    )
