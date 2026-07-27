"""Stage 1: generation only. Writes one JSON per sample under
``output/model_output/<model>_<config>/<dataset>/`` (see ``output_layout.py``)
and skips samples that already have a file there — the resume behavior that
lets a half-finished dataset run continue without redoing already-generated
samples, and lets generation happen entirely separately from scoring (e.g.
on a GPU box, with ``model_output/`` copied elsewhere afterward).
"""

from __future__ import annotations

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


def run_generation(
    adapter: ModelAdapter,
    dataset_name: str,
    samples: list[Sample],
    max_new_tokens: int,
    out_dir: str | Path,
    extra_config: dict | None = None,
    measure_compute: bool = False,
    seed: int = DEFAULT_SEED,
    resume: bool = True,
    progress: GenerationProgress | None = None,
) -> GenerateStageSummary:
    if not samples:
        raise ValueError("samples must be non-empty")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir / "_meta.json"
    if not (resume and meta_path.exists()):
        save_meta(
            {
                "model_name": adapter.name,
                "config_name": adapter.config_name,
                "dataset_name": dataset_name,
                "run_metadata": collect_run_metadata(adapter),
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
            compute_handle = adapter.profile_compute(request)
            generation.compute_tflops = compute_handle.tflops if compute_handle.available else None

        save_generation_result(generation, sample_path)
        generated += 1
        if progress is not None:
            progress("finish", index, len(samples), sample, generation)

    return GenerateStageSummary(
        out_dir=str(out_dir), generated=generated, skipped=skipped, total=len(samples)
    )
