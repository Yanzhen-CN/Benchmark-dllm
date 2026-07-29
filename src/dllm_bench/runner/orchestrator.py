"""Runs one (model, config) x (dataset) experiment: generate -> score ->
measure resources -> aggregate into the section 3.4 raw-results-table shape.

This is the one piece of code every model/dataset combination flows through —
per Appendix D's "只写一套通用的执行 + 测量流程", swapping models or datasets
never touches this file, only the registry config that builds them.
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field

from ..datasets.base import Dataset, Sample, ScoreResult
from ..interfaces import GenerationRequest, GenerationResult, ModelAdapter, RunStatus
from ..metrics.quality_resource import score_per_compute as _score_per_compute
from ..metrics.quality_resource import score_per_unit_energy as _score_per_unit_energy
from .sampling import DEFAULT_SEED, collect_run_metadata


@dataclass
class SampleRecord:
    sample: Sample
    generation: GenerationResult
    score: ScoreResult


@dataclass
class RunSummary:
    model_name: str
    config_name: str
    dataset_name: str
    q: float
    tps: float | None
    sps: float | None
    eps: float | None
    cps: float | None
    time_per_sample: float | None
    energy_per_sample: float | None
    compute_per_sample: float | None
    peak_vram_gb: float | None
    score_per_energy: float | None
    score_per_compute: float | None
    status_counts: dict[str, int]
    n_samples: int
    timing_source: str
    aux: dict[str, float] = field(default_factory=dict)
    run_metadata: dict[str, object] = field(default_factory=dict)
    records: list[SampleRecord] = field(default_factory=list, repr=False)


def run_experiment(
    adapter: ModelAdapter,
    dataset: Dataset,
    samples: list[Sample],
    max_new_tokens: int,
    extra_config: dict | None = None,
    measure_compute: bool = False,
    seed: int = DEFAULT_SEED,
) -> RunSummary:
    if not samples:
        raise ValueError("samples must be non-empty")

    records: list[SampleRecord] = []
    for sample in samples:
        sample_max_new_tokens = int(sample.meta.get("max_new_tokens", max_new_tokens))
        if sample_max_new_tokens <= 0:
            raise ValueError(
                f"sample {sample.sample_id} has invalid max_new_tokens={sample_max_new_tokens}"
            )
        request = GenerationRequest(
            prompt=sample.prompt,
            max_new_tokens=sample_max_new_tokens,
            config={
                **dict(extra_config or {}),
                **(
                    {"target_input_tokens": int(sample.meta["target_input_tokens"])}
                    if "target_input_tokens" in sample.meta
                    else {}
                ),
            },
            sample_id=sample.sample_id,
            seed=seed,
        )
        generation = adapter.generate(request)

        if measure_compute and hasattr(adapter, "profile_compute"):
            compute_handle = adapter.profile_compute(request)
            generation.compute_tflops = compute_handle.tflops if compute_handle.available else None

        if generation.status == RunStatus.SUCCESS:
            score = dataset.score(sample, generation.output_text)
            score.aux.update(dataset.trace_aux_metrics(sample, generation.trace))
        else:
            score = ScoreResult(primary_score=0.0, valid=False, complete=False)

        records.append(SampleRecord(sample=sample, generation=generation, score=score))

    return summarize_records(
        adapter.name, adapter.config_name, dataset, records,
        run_metadata=collect_run_metadata(adapter),
    )


def summarize_records(
    model_name: str,
    config_name: str,
    dataset: Dataset,
    records: list[SampleRecord],
    run_metadata: dict | None = None,
) -> RunSummary:
    """Aggregates already-generated-and-scored records into a `RunSummary`.

    Deliberately takes `model_name`/`config_name` strings rather than a live
    `ModelAdapter` — this is what lets `score_stage.run_scoring` aggregate
    results without ever instantiating a model (no GPU/torch needed to score
    generations that already happened, possibly on a different machine).
    `run_metadata` should come from whatever machine actually ran generation
    (persisted in `model_output/.../_meta.json`), not be recomputed here.
    """
    score_results = [r.score for r in records]
    agg = dataset.aggregate_records([r.sample for r in records], score_results)
    generation_agg = dataset.aggregate_generation_records(
        [r.sample for r in records], [r.generation for r in records]
    )
    agg.update(generation_agg)
    q = agg[f"{dataset.name}_score"]
    aux = {k: v for k, v in agg.items() if k != f"{dataset.name}_score"}

    successful = [r for r in records if r.generation.status == RunStatus.SUCCESS]
    timed = [
        r for r in records
        if r.generation.timing and r.generation.timing.wall_clock_seconds > 0
    ]
    complete_timing = len(timed) == len(records)
    times = [r.generation.timing.wall_clock_seconds for r in timed]
    energies = [
        r.generation.energy_joules
        for r in records
        if r.generation.energy_joules is not None
    ]
    computes = [
        r.generation.compute_tflops
        for r in records
        if r.generation.compute_tflops is not None
    ]
    vrams = [
        r.generation.peak_vram_gb
        for r in records
        if r.generation.peak_vram_gb is not None
    ]

    time_per_sample = statistics.fmean(times) if complete_timing else None
    energy_per_sample = (
        statistics.fmean(energies) if len(energies) == len(records) else None
    )
    compute_per_sample = (
        statistics.fmean(computes) if len(computes) == len(records) else None
    )
    peak_vram_gb = max(vrams) if vrams else None

    total_time = sum(times)
    total_tokens = sum(r.generation.final_valid_length for r in timed)
    # Appendix B requires ratios of window totals, never a mean of per-sample
    # rates.  Energy/compute rates are only available if every timed sample in
    # the measurement window has the corresponding counter.
    tps = total_tokens / total_time if complete_timing and total_time > 0 else None
    sps = len(records) / total_time if complete_timing and total_time > 0 else None
    eps = (
        sum(r.generation.energy_joules for r in timed) / total_time
        if complete_timing
        and total_time > 0
        and all(r.generation.energy_joules is not None for r in timed)
        else None
    )
    cps = (
        sum(r.generation.compute_tflops for r in timed) / total_time
        if complete_timing
        and total_time > 0
        and all(r.generation.compute_tflops is not None for r in timed)
        else None
    )

    score_per_energy = _score_per_unit_energy(q, eps) if eps else None
    score_per_compute = _score_per_compute(q, cps) if cps else None

    status_counts = dict(Counter(r.generation.status.value for r in records))
    timing_sources = {
        r.generation.timing.source for r in records if r.generation.timing
    }
    if len(timing_sources) == 1:
        timing_source = next(iter(timing_sources))
    elif not timing_sources:
        timing_source = "unavailable"
    else:
        timing_source = "mixed"

    return RunSummary(
        model_name=model_name,
        config_name=config_name,
        dataset_name=dataset.name,
        q=q,
        tps=tps,
        sps=sps,
        eps=eps,
        cps=cps,
        time_per_sample=time_per_sample,
        energy_per_sample=energy_per_sample,
        compute_per_sample=compute_per_sample,
        peak_vram_gb=peak_vram_gb,
        score_per_energy=score_per_energy,
        score_per_compute=score_per_compute,
        status_counts=status_counts,
        n_samples=len(records),
        timing_source=timing_source,
        aux=aux,
        run_metadata=run_metadata or {},
        records=records,
    )
