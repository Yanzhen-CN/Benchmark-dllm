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


_NON_ACCEPTING_FORWARD_PHASES = frozenset(
    {"prefill", "prefill_or_cache_build", "finalization"}
)


def _accepted_token_count(generation: GenerationResult) -> int | None:
    productive = [
        profile
        for profile in generation.forward_profiles
        if profile.phase not in _NON_ACCEPTING_FORWARD_PHASES
    ]
    if productive and all(profile.accepted_tokens is not None for profile in productive):
        return sum(int(profile.accepted_tokens or 0) for profile in productive)

    accepted_draft = generation.extra.get("accepted_draft_tokens")
    verification_passes = generation.extra.get("target_verification_passes")
    if isinstance(accepted_draft, (int, float)) and isinstance(
        verification_passes, (int, float)
    ):
        return int(accepted_draft) + int(verification_passes)

    if generation.trace:
        # ``committed_positions`` is the accepted-event set for that forward,
        # not a cumulative set.  Summing per-step events deliberately counts a
        # DG position again after re-noise and re-acceptance.
        if any(step.committed_positions for step in generation.trace):
            return sum(
                len({int(position) for position in step.committed_positions})
                for step in generation.trace
            )
    return None


@dataclass
class RunSummary:
    model_name: str
    config_name: str
    dataset_name: str
    q: float
    total_time_seconds: float | None
    total_energy_joules: float | None
    total_accepted_tokens: int | None
    accepted_tokens_per_sample: float | None
    timed_sample_count: int
    energy_sample_count: int
    tps: float | None
    tpf: float | None
    accepted_tps: float | None
    accepted_tokens_per_forward: float | None
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
    scoring_metadata: dict[str, object] = field(default_factory=dict)
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

        if generation.status in {RunStatus.SUCCESS, RunStatus.TRUNCATED}:
            score = dataset.score_generation(sample, generation)
            score.aux["truncated_rate"] = float(
                generation.status == RunStatus.TRUNCATED
            )
            if generation.status == RunStatus.TRUNCATED:
                score.complete = False
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

    speculative = [
        record.generation.extra
        for record in records
        if "target_verification_passes" in record.generation.extra
    ]
    if speculative:
        drafts = sum(int(item.get("target_verification_passes", 0)) for item in speculative)
        drafted_tokens = sum(int(item.get("drafted_tokens", 0)) for item in speculative)
        accepted_tokens = sum(int(item.get("accepted_draft_tokens", 0)) for item in speculative)
        aux.update(
            {
                "speculative_target_verification_passes": float(drafts),
                "speculative_drafted_tokens": float(drafted_tokens),
                "speculative_accepted_draft_tokens": float(accepted_tokens),
                "speculative_draft_acceptance_rate": (
                    accepted_tokens / drafted_tokens if drafted_tokens else 0.0
                ),
                "speculative_mean_acceptance_length": (
                    1.0 + accepted_tokens / drafts if drafts else 1.0
                ),
            }
        )
        ttfts = [
            float(item["time_to_first_token_seconds"])
            for item in speculative
            if item.get("time_to_first_token_seconds") is not None
        ]
        tpots = [
            float(item["time_per_output_token_seconds"])
            for item in speculative
            if item.get("time_per_output_token_seconds") is not None
        ]
        if ttfts:
            aux["mean_time_to_first_token_seconds"] = statistics.fmean(ttfts)
        if tpots:
            aux["mean_time_per_output_token_seconds"] = statistics.fmean(tpots)

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
    total_energy = sum(energies) if len(energies) == len(records) else None
    # Appendix B requires ratios of window totals, never a mean of per-sample
    # rates.  Energy/compute rates are only available if every timed sample in
    # the measurement window has the corresponding counter.
    accepted_counts = [_accepted_token_count(r.generation) for r in timed]
    complete_acceptance = (
        complete_timing
        and bool(accepted_counts)
        and all(value is not None for value in accepted_counts)
    )
    total_accepted_tokens = (
        sum(int(value or 0) for value in accepted_counts)
        if complete_acceptance
        else None
    )
    accepted_tps = (
        total_accepted_tokens / total_time
        if total_accepted_tokens is not None and total_time > 0
        else None
    )
    accepted_tokens_per_sample = (
        total_accepted_tokens / len(records)
        if total_accepted_tokens is not None and records
        else None
    )
    generation_forward_counts = [
        int(record.generation.num_forward_passes) for record in timed
    ]
    complete_generation_forwards = (
        complete_timing
        and bool(generation_forward_counts)
        and all(value > 0 for value in generation_forward_counts)
    )
    total_generation_forwards = (
        sum(generation_forward_counts) if complete_generation_forwards else None
    )
    accepted_tokens_per_forward = (
        total_accepted_tokens / total_generation_forwards
        if total_accepted_tokens is not None
        and total_generation_forwards is not None
        and total_generation_forwards > 0
        else None
    )
    tps = accepted_tps
    tpf = accepted_tokens_per_forward
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

    score_per_energy = (
        _score_per_unit_energy(q, energy_per_sample)
        if energy_per_sample
        else None
    )
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
        total_time_seconds=total_time if complete_timing else None,
        total_energy_joules=total_energy,
        total_accepted_tokens=total_accepted_tokens,
        accepted_tokens_per_sample=accepted_tokens_per_sample,
        timed_sample_count=len(timed),
        energy_sample_count=len(energies),
        tps=tps,
        tpf=tpf,
        accepted_tps=accepted_tps,
        accepted_tokens_per_forward=accepted_tokens_per_forward,
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
