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
from pathlib import Path
from typing import Callable

from ..datasets.base import Sample
from ..interfaces import GenerationRequest, GenerationResult, ModelAdapter
from .persistence import save_generation_result, save_meta
from .sampling import DEFAULT_SEED, collect_run_metadata


@dataclass
class GenerateStageSummary:
    out_dir: str
    generated: int
    skipped: int
    total: int


GenerationProgress = Callable[
    [str, int, int, Sample, GenerationResult | None], None
]

MEASUREMENT_PROTOCOL = "gpu-synced-v3-trace-excluded"


def run_generation(
    adapter: ModelAdapter,
    dataset_name: str,
    samples: list[Sample],
    max_new_tokens: int,
    out_dir: str | Path,
    extra_config: dict | None = None,
    measure_compute: bool = False,
    require_all_metrics: bool = False,
    seed: int = DEFAULT_SEED,
    resume: bool = True,
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
        if (
            existing_run.get("measurement_protocol") != MEASUREMENT_PROTOCOL
            or bool(existing_run.get("measure_compute")) != measure_compute
            or bool(existing_run.get("require_all_metrics")) != require_all_metrics
        ):
            raise RuntimeError(
                f"existing outputs under {out_dir} use an incompatible measurement "
                "protocol or compute setting; rerun with --no-resume (overwrites this "
                "dataset) or choose a fresh --output-root"
            )
    if not (resume and meta_path.exists()):
        save_meta(
            {
                "model_name": adapter.name,
                "config_name": adapter.config_name,
                "dataset_name": dataset_name,
                "run_metadata": collect_run_metadata(
                    adapter,
                    {
                        "measurement_protocol": MEASUREMENT_PROTOCOL,
                        "measure_compute": measure_compute,
                        "require_all_metrics": require_all_metrics,
                        "energy_backend": "nvml-total-energy",
                    },
                ),
            },
            meta_path,
        )

    generated = skipped = 0
    for index, sample in enumerate(samples, start=1):
        sample_path = out_dir / f"{sample.sample_id}.json"
        if resume and sample_path.exists():
            skipped += 1
            continue

        sample_max_new_tokens = int(sample.meta.get("max_new_tokens", max_new_tokens))
        if sample_max_new_tokens <= 0:
            raise ValueError(
                f"sample {sample.sample_id} has invalid max_new_tokens={sample_max_new_tokens}"
            )
        request = GenerationRequest(
            prompt=sample.prompt,
            max_new_tokens=sample_max_new_tokens,
            config=dict(extra_config or {}),
            sample_id=sample.sample_id,
            seed=seed,
        )
        if progress is not None:
            progress("start", index, len(samples), sample, None)
        generation = adapter.generate(request)

        if measure_compute and hasattr(adapter, "profile_compute"):
            if progress is not None:
                progress("compute", index, len(samples), sample, generation)
            compute_handle = adapter.profile_compute(request)
            generation.compute_tflops = compute_handle.tflops if compute_handle.available else None

        if require_all_metrics and generation.status.value == "success":
            missing: list[str] = []
            if generation.timing is None or generation.timing.wall_clock_seconds <= 0:
                missing.append("timing")
            if not adapter.natively_measures_resources:
                if generation.energy_joules is None:
                    missing.append("energy_joules")
                if generation.peak_vram_gb is None:
                    missing.append("peak_vram_gb")
                if measure_compute and generation.compute_tflops is None:
                    missing.append("compute_tflops")
            if adapter.supports_trace and not generation.trace:
                missing.append("trace")
            if missing:
                raise RuntimeError(
                    f"sample {sample.sample_id} is missing required formal metrics: "
                    f"{', '.join(missing)}"
                )

        save_generation_result(generation, sample_path)
        generated += 1
        if progress is not None:
            progress("finish", index, len(samples), sample, generation)

    return GenerateStageSummary(
        out_dir=str(out_dir), generated=generated, skipped=skipped, total=len(samples)
    )
