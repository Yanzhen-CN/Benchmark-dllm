"""``dllm-bench generate`` / ``score`` / ``visualize`` / ``report``.

Three separate stages instead of one ``run`` command — this is what lets you
run each model independently (skip W1 entirely, run iLLaDA without touching
DreamReasoner), lets a half-finished dataset resume without redoing already-done
samples, and lets generation happen on one machine (a GPU box) while
scoring/visualization happen on another (see README "三阶段 pipeline").

The atomic unit of testing is the **model**, not model+variant: P1/P2 (or
standard/jump/gidd) get tested *together*, in one process, so the expensive
part (loading weights onto the GPU) happens once and every variant just
changes the generation-time config (see `models/model_cache.py`). So by
default, every command below sweeps **every variant** declared in
`--model-config`'s `configs:` block:

    dllm-bench generate --model-config configs/models/illada.yaml \\
                         --dataset-config configs/datasets/gsm8k.yaml \\
                         --demo --n-samples 5 --max-new-tokens 32
    # runs both `p1` and `p2`, loading the checkpoint only once

Pass `--variant p1` for just one, or `--variants p1,p2` to name a
subset explicitly (the default already covers this specific example, but
matters when a file declares more variants than you want this run).

    dllm-bench score     --model-config configs/models/mock.yaml \\
                         --dataset-config configs/datasets/gsm8k.yaml --demo --n-samples 5

    dllm-bench visualize --model-config configs/models/mock.yaml \\
                         --dataset-config configs/datasets/gsm8k.yaml --demo --n-samples 5

    dllm-bench report --output-root output --dataset-config configs/datasets/gsm8k.yaml

``--demo`` selects the tiny built-in samples in ``runner/demo_samples.py``;
``--samples-file`` reads local JSON/JSONL, and ``--no-demo`` without a file
uses the dataset's official loader where implemented (currently GSM8K).
Scoring and visualization reconstruct the same deterministic sample list,
so pass matching source/count/seed values to every stage.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import click

from .hf_cache import configure_default_cache_dir
from .interfaces import GenerationRequest, ModelAdapter
from .registry import (
    build_dataset,
    build_model_adapter,
    dataset_run_defaults,
    list_model_variants,
    load_yaml,
    model_name,
)
from .visual.public.tables import raw_results_row, render_raw_results_table
from .visual.public.raw_report import write_raw_report
from .visual.public.pairwise import (
    PairwiseCompatibilityError,
    compute_pairwise_row,
    summary_label,
    write_pairwise_outputs,
)
from .visual import (
    render_dataset_visualization,
    render_model_comparison_visualization,
    render_sample_visualization,
)
from .runner.demo_samples import build_demo_samples
from .runner.generate_stage import (
    OOMInvalidTestError,
    is_cuda_oom_error,
    oom_invalid_test_error,
    persist_setup_oom_invalidation,
    run_generation,
)
from .runner.output_layout import (
    model_output_dir,
    model_profiling_dir,
    resolve_model_output_dir,
    resolve_model_profiling_dir,
    resolve_score_output_dir,
    score_output_dir,
    visualization_output_dir,
    model_comparison_visualization_output_dir,
)
from .runner.persistence import (
    load_generation_result,
    load_meta,
    load_run_summary_dict,
    load_score_result,
)
from .runner.score_stage import (
    IncompleteTestError,
    InvalidTestError,
    ensure_test_valid,
    run_scoring,
)
from .runner.matrix import load_matrix_jobs
from .runner.evaluation_sampling import select_configured_samples
from .runner.data_preparation import (
    DataPreparationError,
    load_prepared_samples,
    prepare_dataset,
)



@click.group()
def main() -> None:
    """dLLM Benchmark CLI."""
    # Deliberately not a module-level call: this only mutates HF_HOME when a
    # command actually runs, not merely when `dllm_bench.cli` gets imported
    # (e.g. by tests reusing `main` via Click's CliRunner).
    configure_default_cache_dir()


def _common_options(f):
    f = click.option("--model-config", required=True, type=click.Path(exists=True), help="Path to configs/models/*.yaml")(f)
    f = click.option("--variant", default=None, help="Run just this one named config (e.g. p1)")(f)
    f = click.option("--variants", default=None, help="Comma-separated named configs to run together (default: every variant in --model-config)")(f)
    f = click.option("--dataset-config", required=True, type=click.Path(exists=True), help="Path to configs/datasets/*.yaml")(f)
    f = click.option("--demo/--no-demo", default=True, show_default=True, help="Use demos; --no-demo loads official data where supported")(f)
    f = click.option("--samples-file", default=None, type=click.Path(exists=True), help="Local JSON/JSONL samples; overrides --demo")(f)
    f = click.option("--n-samples", default=None, type=int, help="Defaults to the dataset config's `sample_size`")(f)
    f = click.option(
        "--hellobench-length", "hellobench_lengths", multiple=True,
        type=click.Choice(["2k", "4k", "2000", "4000"]),
        help="HelloBench output profile; repeat to include both",
    )(f)
    f = click.option("--seed", default=None, type=int, help="Defaults to the dataset config's `seed` (42)")(f)
    f = click.option(
        "--output-root",
        default="output/model_output",
        show_default=True,
        type=click.Path(),
        help="Exact generation-output root; model run directories are created directly below it",
    )(f)
    return f


def _resolve_variants(model_config: str, variant: str | None, variants: str | None) -> list[str]:
    if variant and variants:
        raise click.UsageError("pass either --variant or --variants, not both")
    if variant:
        return [variant]
    if variants:
        return [v.strip() for v in variants.split(",") if v.strip()]
    return list_model_variants(model_config)  # atomic unit = model: sweep everything it declares


def _resolve_samples(
    dataset_config: str,
    model_config: str,
    dataset,
    demo: bool,
    samples_file: str | None,
    n_samples: int | None,
    seed: int | None,
    hellobench_lengths: tuple[str, ...] = (),
) -> tuple[list, int]:
    defaults = dataset_run_defaults(dataset_config)
    resolved_n = n_samples if n_samples is not None else (defaults["sample_size"] or 5)
    resolved_seed = seed if seed is not None else defaults["seed"]

    if resolved_n <= 0:
        raise click.UsageError("--n-samples must be greater than zero")
    if hellobench_lengths and dataset.name != "hellobench":
        raise click.UsageError(
            "--hellobench-length can only be used with the hellobench dataset"
        )
    if hellobench_lengths and demo:
        raise click.UsageError(
            "--hellobench-length requires real HelloBench data; use --real-data/--no-demo"
        )
    if samples_file or not demo:
        try:
            prepared = prepare_dataset(
                dataset_config,
                samples_file=samples_file,
                dataset=dataset,
            )
            available = load_prepared_samples(prepared)
            sampling_config = load_yaml(dataset_config)
            if hellobench_lengths:
                target_words = {
                    2000 if value in {"2k", "2000"} else 4000
                    for value in hellobench_lengths
                }
                sampling_config["output_profiles"] = [
                    profile
                    for profile in sampling_config.get("output_profiles", [])
                    if int(profile["target_words"]) in target_words
                ]
            samples = select_configured_samples(
                available,
                sampling_config,
                load_yaml(model_config),
                n_samples=n_samples,
                seed=resolved_seed,
            )
        except (DataPreparationError, ValueError) as exc:
            raise click.UsageError(str(exc)) from exc
        if prepared.prepared_now:
            click.echo(
                f"Prepared {prepared.sample_count} {dataset.name} samples -> "
                f"{prepared.samples_path}"
            )
        return samples, resolved_seed
    if demo:
        samples = build_demo_samples(dataset.name, n=resolved_n)
        return samples, resolved_seed

    raise AssertionError("unreachable sample-source state")


@main.command("prepare-data")
@click.option("--dataset-config", required=True, type=click.Path(exists=True))
@click.option("--samples-file", default=None, type=click.Path(exists=True))
@click.option("--force", is_flag=True, help="Rebuild the matching prepared artifact")
def prepare_data_command(dataset_config: str, samples_file: str | None, force: bool) -> None:
    """Prepare one real dataset now; normal runs do this automatically if absent."""
    try:
        prepared = prepare_dataset(
            dataset_config, samples_file=samples_file, force=force
        )
    except DataPreparationError as exc:
        raise click.ClickException(str(exc)) from exc
    action = "prepared" if prepared.prepared_now else "cached"
    click.echo(
        f"[{prepared.dataset_name}] {action}: {prepared.sample_count} samples -> "
        f"{prepared.samples_path}"
    )


@main.command()
@_common_options
@click.option("--max-new-tokens", required=True, type=int)
@click.option("--measure-compute/--no-measure-compute", default=False, show_default=True)
@click.option("--profiling-output/--standard-output", default=False, show_default=True)
@click.option("--capture-trace/--no-capture-trace", "capture_trace_override", default=None)
@click.option("--require-all-metrics/--allow-missing-metrics", default=False, show_default=True)
@click.option("--resume/--no-resume", default=True, show_default=True, help="Skip samples that already have a model_output file")
def generate(
    model_config: str,
    variant: str | None,
    variants: str | None,
    dataset_config: str,
    demo: bool,
    samples_file: str | None,
    n_samples: int | None,
    seed: int | None,
    hellobench_lengths: tuple[str, ...],
    output_root: str,
    max_new_tokens: int,
    measure_compute: bool,
    profiling_output: bool,
    require_all_metrics: bool,
    resume: bool,
    capture_trace_override: bool | None = None,
    force_max_new_tokens: bool = False,
    adapter_cache: dict[tuple[str, str], ModelAdapter] | None = None,
) -> None:
    variant_list = _resolve_variants(model_config, variant, variants)
    dataset_settings = load_yaml(dataset_config)
    trace_scope = str(dataset_settings.get("trace_scope", "all_samples"))
    if trace_scope not in {"all_samples", "none"}:
        raise click.UsageError(
            f"unsupported trace_scope={trace_scope!r} in {dataset_config}; "
            "use 'all_samples' or 'none'"
        )
    capture_trace = (
        trace_scope == "all_samples"
        if capture_trace_override is None
        else capture_trace_override
    )
    if profiling_output and not measure_compute:
        raise click.UsageError(
            "profiling_output requires --measure-compute; no files were generated"
        )
    dataset = build_dataset(dataset_config)
    samples, resolved_seed = _resolve_samples(dataset_config, model_config, dataset, demo, samples_file, n_samples, seed, hellobench_lengths)

    click.echo(f"Sweeping variants {variant_list} of {model_config} on {dataset.name} ({len(samples)} samples)")
    invalid_variant_errors: list[str] = []
    for v in variant_list:
        adapter_key = (str(Path(model_config).resolve()), v)
        adapter = adapter_cache.get(adapter_key) if adapter_cache is not None else None
        if adapter is None:
            adapter = build_model_adapter(model_config, variant=v)
            if adapter_cache is not None:
                adapter_cache[adapter_key] = adapter
        out_dir = (
            model_profiling_dir(
                output_root, adapter.name, adapter.config_name, dataset.name
            )
            if profiling_output
            else model_output_dir(
                output_root, adapter.name, adapter.config_name, dataset.name
            )
        )
        meta_path = out_dir / "_meta.json"
        if (
            resume
            and meta_path.exists()
        ):
            early_stop = load_meta(meta_path).get("early_stop")
            if early_stop and early_stop.get("reason") == "oom":
                oom_info_path = out_dir / "oom_info.json"
                detail = (
                    load_meta(oom_info_path)
                    if oom_info_path.exists()
                    else {
                        **early_stop,
                        "error_message": "CUDA out of memory",
                    }
                )
                invalid_variant_errors.append(str(oom_invalid_test_error(out_dir, detail)))
                click.echo(f"[{v}] INVALID OOM DATASET: {invalid_variant_errors[-1]}", err=True)
                continue
        needs_generation = not resume or any(
            not (out_dir / f"{sample.sample_id}.json").exists()
            for sample in samples
        )
        warm = getattr(adapter, "warm", None)
        if needs_generation and callable(warm):
            click.echo(f"[{v}] loading model into runtime device (outside sample timing) ...")
            try:
                warm()
            except Exception as exc:
                if not is_cuda_oom_error(exc):
                    raise
                detail = persist_setup_oom_invalidation(
                    adapter=adapter,
                    dataset_name=dataset.name,
                    out_dir=out_dir,
                    selected_samples=len(samples),
                    failure_stage="model_load",
                    error=exc,
                    seed=resolved_seed,
                    measure_compute=measure_compute,
                    require_all_metrics=require_all_metrics,
                    capture_trace=capture_trace,
                )
                invalid_variant_errors.append(str(oom_invalid_test_error(out_dir, detail)))
                click.echo(f"[{v}] INVALID OOM DATASET: {invalid_variant_errors[-1]}", err=True)
                continue
            click.echo(f"[{v}] model ready")
        elif not needs_generation:
            click.echo(f"[{v}] all sample outputs already exist; model load skipped")

        warmup_generation = getattr(adapter, "warmup_generation", None)
        if needs_generation and callable(warmup_generation):
            warmup_sample = next(
                sample for sample in samples
                if not resume or not (out_dir / f"{sample.sample_id}.json").exists()
            )
            adapter_warmup_tokens = int(
                getattr(adapter, "warmup_new_tokens", 8)
            )
            if adapter_warmup_tokens <= 0:
                raise ValueError(
                    f"{adapter.name} declares invalid "
                    f"warmup_new_tokens={adapter_warmup_tokens}"
                )
            warmup_tokens = min(
                adapter_warmup_tokens,
                (
                    max_new_tokens
                    if force_max_new_tokens
                    else int(
                        warmup_sample.meta.get("max_new_tokens", max_new_tokens)
                    )
                ),
            )
            click.echo(f"[{v}] running untimed {warmup_tokens}-token warmup ...")
            try:
                warmup_generation(
                    GenerationRequest(
                        # Warmup initializes kernels/caches only. Reusing the first
                        # 8K RULER prompt here can OOM before the formal sample is
                        # timed and persisted, which loses the actual failure
                        # boundary and duplicates the long-context workload.
                        prompt="Warm up.",
                        max_new_tokens=warmup_tokens,
                        config={},
                        sample_id="__warmup__",
                        seed=resolved_seed,
                    )
                )
            except Exception as exc:
                if not is_cuda_oom_error(exc):
                    raise
                detail = persist_setup_oom_invalidation(
                    adapter=adapter,
                    dataset_name=dataset.name,
                    out_dir=out_dir,
                    selected_samples=len(samples),
                    failure_stage="warmup",
                    error=exc,
                    seed=resolved_seed,
                    measure_compute=measure_compute,
                    require_all_metrics=require_all_metrics,
                    capture_trace=capture_trace,
                )
                invalid_variant_errors.append(str(oom_invalid_test_error(out_dir, detail)))
                click.echo(f"[{v}] INVALID OOM DATASET: {invalid_variant_errors[-1]}", err=True)
                continue
            click.echo(f"[{v}] warmup complete")

        from threading import Event, Lock, Thread

        compute_started: dict[str, float] = {}
        generation_stops: dict[str, Event] = {}
        generation_threads: dict[str, Thread] = {}
        completed_generation_seconds: list[float] = []
        generation_run_started: list[float] = []
        progress_lock = Lock()
        dynamic_generation_progress = (
            not measure_compute and click.get_text_stream("stdout").isatty()
        )

        def format_duration(seconds: float) -> str:
            seconds = max(0, int(round(seconds)))
            hours, remainder = divmod(seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            if hours:
                return f"{hours:d}:{minutes:02d}:{seconds:02d}"
            return f"{minutes:02d}:{seconds:02d}"

        def generation_progress_text(
            prefix: str,
            state: str,
            elapsed: float,
            index: int,
            total: int,
            *,
            finished: bool = False,
        ) -> str:
            if completed_generation_seconds:
                average = sum(completed_generation_seconds) / len(
                    completed_generation_seconds
                )
            else:
                average = elapsed
            timing = f"{elapsed:.1f}s / avg {average:.1f}s"
            if not completed_generation_seconds:
                return f"{prefix}: {state} ({timing}) | ETA calculating"
            remaining_current = 0.0 if finished else max(0.0, average - elapsed)
            eta = remaining_current + average * max(0, total - index)
            run_elapsed = (
                perf_counter() - generation_run_started[0]
                if generation_run_started
                else elapsed
            )
            return (
                f"{prefix}: {state} ({timing}) | ETA {format_duration(eta)}"
                f" | total ~{format_duration(run_elapsed + eta)}"
            )

        def render_generation_progress(text: str, *, final: bool = False) -> None:
            with progress_lock:
                click.echo(f"\r\033[2K{text}", nl=final)

        def stop_generation_refresh(sample_id: str) -> None:
            stop = generation_stops.pop(sample_id, None)
            if stop is not None:
                stop.set()
            thread = generation_threads.pop(sample_id, None)
            if thread is not None:
                thread.join(timeout=2.0)

        def log_progress(event, index, total, sample, generation):
            prefix = f"[{v}] [{index}/{total}] {sample.sample_id}"
            if event == "start":
                if measure_compute:
                    click.echo(f"{prefix}: [timing] generating ...")
                elif not dynamic_generation_progress:
                    click.echo(f"{prefix}: generating ...")
                else:
                    started = perf_counter()
                    if not generation_run_started:
                        generation_run_started.append(started)
                    stop = Event()
                    generation_stops[sample.sample_id] = stop

                    def refresh() -> None:
                        while True:
                            render_generation_progress(
                                generation_progress_text(
                                    prefix,
                                    "generating ...",
                                    perf_counter() - started,
                                    index,
                                    total,
                                )
                            )
                            if stop.wait(1.0):
                                return

                    thread = Thread(target=refresh, daemon=True)
                    generation_threads[sample.sample_id] = thread
                    thread.start()
                return
            if event == "timing_finish":
                elapsed = (
                    generation.timing.wall_clock_seconds
                    if generation is not None and generation.timing is not None
                    else 0.0
                )
                click.echo(f"{prefix}: [timing] step collection complete ({elapsed:.2f}s)")
                return
            if event == "compute":
                compute_started[sample.sample_id] = perf_counter()
                expected = (
                    len(generation.forward_profiles)
                    if generation is not None
                    else 0
                )
                click.echo(
                    f"{prefix}: [compute] replay started ({expected} forwards)"
                )
                return
            if event == "compute_progress":
                detail = (
                    generation.extra.get("_compute_replay_progress", {})
                    if generation is not None
                    else {}
                )
                completed = int(detail.get("completed_steps", 0))
                expected = int(detail.get("expected_steps", 0))
                started = compute_started.get(sample.sample_id)
                elapsed = perf_counter() - started if started is not None else 0.0
                if expected:
                    percent = min(100.0, 100.0 * completed / expected)
                    detail_text = (
                        f"{completed}/{expected} ({percent:.0f}%, "
                        f"{elapsed:.1f}s elapsed)"
                    )
                else:
                    detail_text = f"{completed} steps ({elapsed:.1f}s elapsed)"
                click.echo(f"{prefix}: [compute] replay {detail_text}")
                return
            if event == "compute_finish":
                started = compute_started.pop(sample.sample_id, None)
                elapsed = perf_counter() - started if started is not None else 0.0
                validation = (
                    generation.extra.get("compute_replay_validation", {}).get(
                        "status", "unavailable"
                    )
                    if generation is not None
                    else "unavailable"
                )
                click.echo(
                    f"{prefix}: [profiling] complete, replay {validation} "
                    f"({elapsed:.1f}s)"
                )
                return
            elapsed = (
                generation.timing.wall_clock_seconds
                if generation is not None and generation.timing is not None
                else 0.0
            )
            status = generation.status.value if generation is not None else "unknown"
            if dynamic_generation_progress:
                stop_generation_refresh(sample.sample_id)
                completed_generation_seconds.append(elapsed)
                render_generation_progress(
                    generation_progress_text(
                        prefix,
                        status,
                        elapsed,
                        index,
                        total,
                        finished=True,
                    ),
                    final=index == total,
                )
            else:
                click.echo(f"{prefix}: {status} ({elapsed:.2f}s)")

        try:
            summary = run_generation(
                adapter, dataset.name, samples, max_new_tokens,
                out_dir=out_dir, measure_compute=measure_compute,
                require_all_metrics=require_all_metrics, seed=resolved_seed,
                capture_trace=capture_trace, resume=resume,
                force_max_new_tokens=force_max_new_tokens,
                progress=log_progress,
            )
        except OOMInvalidTestError as exc:
            invalid_variant_errors.append(str(exc))
            click.echo(f"[{v}] INVALID OOM DATASET: {exc}", err=True)
            continue
        click.echo(f"[{v}] generated={summary.generated} skipped={summary.skipped} total={summary.total} -> {out_dir}")
        if summary.stopped_early:
            click.echo(
                f"[{v}] stopped after first long-task OOM at sample "
                f"{summary.first_oom_ordinal}/{summary.total} "
                f"({summary.first_oom_sample_id}); later samples were not attempted"
            )
            invalid_variant_errors.append(
                f"{adapter.name} x {adapter.config_name} x {dataset.name} OOM"
            )

    if invalid_variant_errors:
        raise OOMInvalidTestError("; ".join(invalid_variant_errors))


@main.command()
@_common_options
@click.option("--resume/--no-resume", default=True, show_default=True, help="Skip samples that already have a score_output file")
def score(
    model_config: str,
    variant: str | None,
    variants: str | None,
    dataset_config: str,
    demo: bool,
    samples_file: str | None,
    n_samples: int | None,
    seed: int | None,
    hellobench_lengths: tuple[str, ...],
    output_root: str,
    resume: bool,
) -> None:
    variant_list = _resolve_variants(model_config, variant, variants)
    dataset = build_dataset(dataset_config)
    dataset_settings = load_yaml(dataset_config)
    primary_metric = dataset_settings.get("primary_metric")
    preview_aux_metrics = dataset_settings.get("aux_metrics", [])
    samples, _ = _resolve_samples(dataset_config, model_config, dataset, demo, samples_file, n_samples, seed, hellobench_lengths)
    configured_model = model_name(model_config)

    invalid_rows: list[str] = []
    incomplete_rows: list[str] = []
    for v in variant_list:
        model_out = resolve_model_output_dir(output_root, configured_model, v, dataset.name)
        score_out = score_output_dir(output_root, configured_model, v, dataset.name)
        try:
            result = run_scoring(
                dataset,
                samples,
                model_out,
                score_out,
                resume=resume,
                primary_metric=primary_metric,
            )
        except InvalidTestError as exc:
            invalid_rows.append(str(exc))
            click.echo(f"[{v}] INVALID OOM DATASET: {exc}")
            continue
        except IncompleteTestError as exc:
            incomplete_rows.append(str(exc))
            click.echo(f"[{v}] INCOMPLETE DATASET: {exc}")
            continue
        except FileNotFoundError as exc:
            # Matrix scoring is intentionally allowed before every generation
            # row exists. A missing model_output directory is one incomplete
            # dataset, not a reason to abort scoring later completed rows.
            incomplete_rows.append(str(exc))
            click.echo(f"[{v}] MISSING DATASET OUTPUT: {exc}")
            continue

        if result.preview:
            summary = result.summary
            metric_rows = [
                (primary_metric or f"{dataset.name}_score", summary.q),
                ("valid_rate", summary.aux.get("valid_rate")),
                ("complete_rate", summary.aux.get("complete_rate")),
            ]
            metric_rows.extend(
                (name, summary.aux[name])
                for name in preview_aux_metrics
                if name in summary.aux
                and name not in {"valid_rate", "complete_rate"}
            )
            label_width = max(len(str(name)) for name, _ in metric_rows)
            click.echo("")
            click.echo("=" * 72)
            click.echo(
                f"PREVIEW  model={configured_model}  variant={v}  "
                f"dataset={dataset.name}  samples={summary.n_samples}"
            )
            click.echo("-" * 72)
            for name, value in metric_rows:
                rendered = "N/A" if value is None else f"{float(value):.6f}"
                click.echo(f"{str(name):<{label_width}}  {rendered}")
            click.echo("-" * 72)
            click.echo("No score files were created or updated.")
        else:
            click.echo(f"[{v}] q={result.summary.q:.4f}  scored={result.scored}  skipped={result.skipped}  -> {score_out / 'summary.json'}")
        if result.missing_sample_ids:
            click.echo(f"[{v}] WARNING: {len(result.missing_sample_ids)} sample(s) not yet generated: {result.missing_sample_ids}")

    if invalid_rows:
        raise InvalidTestError("; ".join(invalid_rows))
    if incomplete_rows:
        raise IncompleteTestError("; ".join(incomplete_rows))


@main.command()
@_common_options
@click.option("--n-representative", default=None, type=int, help="Only visualize the first N samples (default: all)")
@click.option("--sample-ids", default=None, help="Comma-separated sample ids to visualize (overrides --n-representative)")
@click.option(
    "--figures",
    default=None,
    help="Comma-separated model comparison figures: all,trace,state,convergence,yield,forward",
)
@click.option("--profiling-output/--standard-output", default=False, show_default=True)
def visualize(
    model_config: str,
    variant: str | None,
    variants: str | None,
    dataset_config: str,
    demo: bool,
    samples_file: str | None,
    n_samples: int | None,
    seed: int | None,
    hellobench_lengths: tuple[str, ...],
    output_root: str,
    n_representative: int | None,
    sample_ids: str | None,
    figures: str | None,
    profiling_output: bool,
) -> None:
    variant_list = _resolve_variants(model_config, variant, variants)
    dataset = build_dataset(dataset_config)
    samples, resolved_seed = _resolve_samples(dataset_config, model_config, dataset, demo, samples_file, n_samples, seed, hellobench_lengths)
    configured_model = model_name(model_config)

    all_samples = samples
    if sample_ids:
        wanted = {s.strip() for s in sample_ids.split(",") if s.strip()}
        representative_ids = {
            sample.sample_id for sample in all_samples if sample.sample_id in wanted
        }
    elif n_representative is not None:
        representative_ids = {
            sample.sample_id for sample in all_samples[:n_representative]
        }
    else:
        representative_ids = {sample.sample_id for sample in all_samples}

    model_settings = load_yaml(model_config)
    invalid_rows: list[str] = []
    incomplete_rows: list[str] = []
    comparison_records = {}
    comparison_block_lengths: set[int] = set()
    for v in variant_list:
        variant_config = model_settings["configs"][v]
        block_length = (
            variant_config.get("step_config", {}).get("block_length")
            or model_settings.get("trace_block_length")
        )
        if block_length:
            comparison_block_lengths.add(int(block_length))
        if profiling_output:
            model_out = resolve_model_profiling_dir(
                output_root, configured_model, v, dataset.name
            )
            score_out = model_out / "_score_output"
            viz_out = visualization_output_dir(
                output_root, configured_model, v, dataset.name
            )
        else:
            model_out = resolve_model_output_dir(
                output_root, configured_model, v, dataset.name
            )
            score_out = resolve_score_output_dir(
                output_root, configured_model, v, dataset.name
            )
            viz_out = visualization_output_dir(
                output_root, configured_model, v, dataset.name
            )

        if not (model_out / "_meta.json").exists():
            raise click.UsageError(f"no _meta.json under {model_out} — run `dllm-bench generate` first")
        try:
            ensure_test_valid(model_out)
        except InvalidTestError as exc:
            invalid_rows.append(str(exc))
            click.echo(f"[{v}] INVALID OOM DATASET: {exc}")
            continue
        except IncompleteTestError as exc:
            incomplete_rows.append(str(exc))
            click.echo(f"[{v}] INCOMPLETE DATASET: {exc}")
            continue

        rendered = 0
        trace_records = []
        for sample in all_samples:
            gen_path = model_out / f"{sample.sample_id}.json"
            if not gen_path.exists():
                click.echo(f"[{v}] skipping {sample.sample_id}: not generated yet")
                continue
            generation = load_generation_result(gen_path)
            trace_records.append((sample, generation))

            if sample.sample_id not in representative_ids:
                continue

            score_path = score_out / f"{sample.sample_id}.json"
            score_result = load_score_result(score_path) if score_path.exists() else None

            render_sample_visualization(
                model_name=configured_model,
                sample_id=sample.sample_id,
                trace=generation.trace,
                final_valid_length=generation.final_valid_length,
                out_dir=str(viz_out),
                final_output_text=generation.output_text,
                final_score=score_result.primary_score if score_result else None,
                dataset_name=dataset.name,
                sample=sample,
                block_length=block_length,
            )
            rendered += 1

        if trace_records:
            comparison_records[v] = trace_records
            render_dataset_visualization(
                model_name=configured_model,
                dataset_name=dataset.name,
                records=trace_records,
                out_dir=viz_out,
                seed=resolved_seed,
                config_name=v,
                block_length=block_length,
            )

        click.echo(f"[{v}] rendered {rendered} sample(s) -> {viz_out}")

    if comparison_records:
        comparison_out = model_comparison_visualization_output_dir(
            output_root,
            configured_model,
            dataset.name,
        )
        render_model_comparison_visualization(
            model_name=configured_model,
            dataset_name=dataset.name,
            records_by_variant=comparison_records,
            out_dir=comparison_out,
            seed=resolved_seed,
            block_length=(
                next(iter(comparison_block_lengths))
                if len(comparison_block_lengths) == 1
                else None
            ),
            figures=(
                {value.strip() for value in figures.split(",") if value.strip()}
                if figures
                else None
            ),
        )

    if invalid_rows:
        raise InvalidTestError("; ".join(invalid_rows))
    if incomplete_rows:
        raise IncompleteTestError("; ".join(incomplete_rows))


def _valid_summary_paths(paths: list[str]) -> list[str]:
    valid_paths = []
    for raw_path in dict.fromkeys(paths):
        summary_path = Path(raw_path)
        dataset_dir = summary_path.parent
        run_dir = dataset_dir.parent
        score_root = run_dir.parent
        model_out = score_root.parent / "model_output" / run_dir.name / dataset_dir.name
        if score_root.name == "score_output" and (model_out / "_meta.json").exists():
            try:
                ensure_test_valid(model_out)
            except InvalidTestError as exc:
                click.echo(
                    f"WARNING: excluding OOM-invalid test: {summary_path} ({exc})", err=True
                )
                continue
            except IncompleteTestError as exc:
                click.echo(
                    f"WARNING: excluding incomplete test: {summary_path} ({exc})", err=True
                )
                continue
        valid_paths.append(str(summary_path))
    return valid_paths


@main.command()
@click.option("--run", "run_paths", multiple=True, type=click.Path(exists=True), help="One or more summary.json files from `dllm-bench score`")
@click.option("--output-root", default=None, type=click.Path(), help="Auto-discover summary.json files under output_root/score_output/*/<dataset>/")
@click.option("--model", "model_names", multiple=True, help="Only include these model names")
@click.option("--dataset", "dataset_names", multiple=True, help="Only include these datasets")
def report(
    run_paths: tuple[str, ...],
    output_root: str | None,
    model_names: tuple[str, ...],
    dataset_names: tuple[str, ...],
) -> None:
    """Build the measured-only report. Resource conversion is a separate command."""
    paths = list(run_paths)
    if output_root:
        score_root = Path(output_root) / "score_output"
        paths.extend(str(p) for p in sorted(score_root.glob("*/*/summary.json")))

    if not paths:
        raise click.UsageError("no summary.json files found (pass --run, or --output-root [--dataset])")

    paths = _valid_summary_paths(paths)
    if not paths:
        raise click.UsageError("no valid summary.json files remain after exclusions")

    summaries = [load_run_summary_dict(p) for p in paths]
    if model_names:
        summaries = [s for s in summaries if s.get("model_name") in model_names]
    if dataset_names:
        summaries = [
            s
            for s in summaries
            if s.get("dataset_name") in dataset_names
            or ("sudoku" in dataset_names and str(s.get("dataset_name", "")).startswith("sudoku"))
        ]
    if not summaries:
        raise click.UsageError("no summaries match the selected model/dataset filters")
    rows = [raw_results_row(s) for s in summaries]
    click.echo(render_raw_results_table(rows))

    if output_root:
        report_root = Path(output_root) / "report"
        written = write_raw_report(summaries, report_root)
        click.echo(f"\nMeasured-only report ({len(written)} files) -> {report_root}")


@main.command("pairwise-report")
@click.option("--output-root", default="output", show_default=True, type=click.Path())
@click.option("--model", "model_names", multiple=True, required=True, help="Comparison model A; repeat for more independent A-vs-base analyses")
@click.option("--model-config", "model_configs", multiple=True, help="Optional A variant/config filter")
@click.option("--base-model", required=True, help="Explicit baseline model B")
@click.option("--base-config", default=None, help="Baseline variant/config; required if B has multiple configs")
@click.option("--dataset", "dataset_names", multiple=True, help="Dataset filter")
@click.option("--beta", default=100.0, show_default=True, type=click.FloatRange(0, 100))
@click.option("--gamma", default=50.0, show_default=True, type=click.FloatRange(0, 100), help="Energy weight percent; speed weight is 100-gamma")
def pairwise_report(
    output_root: str,
    model_names: tuple[str, ...],
    model_configs: tuple[str, ...],
    base_model: str,
    base_config: str | None,
    dataset_names: tuple[str, ...],
    beta: float,
    gamma: float,
) -> None:
    """Write isolated A-relative-to-B sensitivity artifacts; never a leaderboard."""
    score_root = Path(output_root) / "score_output"
    paths = _valid_summary_paths(
        [str(path) for path in sorted(score_root.glob("*/*/summary.json"))]
    )
    summaries = [load_run_summary_dict(path) for path in paths]
    if dataset_names:
        summaries = [
            s
            for s in summaries
            if s.get("dataset_name") in dataset_names
            or ("sudoku" in dataset_names and str(s.get("dataset_name", "")).startswith("sudoku"))
        ]
    bases = [s for s in summaries if s.get("model_name") == base_model]
    if base_config:
        bases = [s for s in bases if s.get("config_name") == base_config]
    base_configs = sorted({s.get("config_name") for s in bases})
    if not bases:
        raise click.UsageError("no baseline summaries match --base-model/--base-config")
    if not base_config and len(base_configs) > 1:
        raise click.UsageError(
            f"baseline {base_model} has multiple configs {base_configs}; pass --base-config"
        )
    selected_base_config = base_config or base_configs[0]
    base_by_dataset = {s["dataset_name"]: s for s in bases}
    comparisons = [s for s in summaries if s.get("model_name") in model_names]
    if model_configs:
        comparisons = [s for s in comparisons if s.get("config_name") in model_configs]
    comparisons = [
        s
        for s in comparisons
        if (s.get("model_name"), s.get("config_name"))
        != (base_model, selected_base_config)
        and s.get("dataset_name") not in {"hellobench", "ruler_context_probe"}
    ]
    if not comparisons:
        raise click.UsageError("no comparison summaries match the requested filters")

    written_count = 0
    incompatibilities: list[str] = []
    for model_summary in comparisons:
        dataset_name = model_summary["dataset_name"]
        base_summary = base_by_dataset.get(dataset_name)
        if base_summary is None:
            incompatibilities.append(
                f"{summary_label(model_summary)} / {dataset_name}: baseline row missing"
            )
            continue
        try:
            row, metadata = compute_pairwise_row(
                model_summary, base_summary, beta=beta, gamma=gamma
            )
        except PairwiseCompatibilityError as exc:
            incompatibilities.append(
                f"{summary_label(model_summary)} / {dataset_name}: {exc}"
            )
            continue
        pair_slug = (
            f"{model_summary['model_name']}_{model_summary['config_name']}__relative_to__"
            f"{base_summary['model_name']}_{base_summary['config_name']}"
        )
        out_dir = (
            Path(output_root)
            / "conversion_output"
            / pair_slug
            / dataset_name
            / f"beta-{beta:g}_gamma-{gamma:g}"
        )
        written_count += len(write_pairwise_outputs(row, metadata, out_dir))
        click.echo(f"{metadata['direction']} / {dataset_name} -> {out_dir}")
    for message in incompatibilities:
        click.echo(f"WARNING: skipped incompatible pair: {message}", err=True)
    if not written_count:
        raise click.UsageError("no compatible pairwise rows were produced")
    click.echo(f"Pairwise sensitivity artifacts: {written_count} file(s)")


@main.command("matrix")
@click.option("--experiment-config", required=True, type=click.Path(exists=True))
@click.option("--model", "model_names", multiple=True, required=True, help="Exactly one model name; run_bench.py dispatches multiple isolated environments")
@click.option("--variants", "variant_names", default=None, help="Comma-separated variant subset for the selected model")
@click.option(
    "--hellobench-length", "hellobench_lengths", multiple=True,
    type=click.Choice(["2k", "4k", "2000", "4000"]),
)
@click.option(
    "--dataset",
    "dataset_names",
    multiple=True,
    help=(
        "Dataset name to include; repeat for multiple. 'sudoku' selects every "
        "sudoku* variant in the matrix"
    ),
)
@click.option("--stage", type=click.Choice(["generate", "score", "visualize", "all"]), default="all", show_default=True)
@click.option("--demo/--no-demo", default=False, show_default=True, help="Use demo data for every matrix row")
@click.option("--n-samples", default=None, type=int)
@click.option(
    "--max-new-tokens",
    "max_new_tokens_values",
    multiple=True,
    type=click.IntRange(min=1),
    help=(
        "Temporary override; repeat for a same-process length sweep. Multiple "
        "values write to <output-root>/len<value>/"
    ),
)
@click.option(
    "--output-root",
    default=None,
    type=click.Path(),
    help=(
        "Exact generation-output root. Model run directories are created "
        "directly below this path; defaults to the experiment YAML value."
    ),
)
@click.option("--measure-compute/--no-measure-compute", default=False, show_default=True)
@click.option("--require-all-metrics/--allow-missing-metrics", default=False, show_default=True)
@click.option("--resume/--no-resume", default=True, show_default=True)
@click.option(
    "--n-representative",
    default=0,
    show_default=True,
    type=int,
    help="Automatically render N sample traces; use --sample-ids for curated examples.",
)
@click.option(
    "--sample-ids",
    default=None,
    help="Comma-separated curated IDs for per-sample visuals; aggregate Task 4 still uses all traces",
)
@click.option(
    "--figures",
    default=None,
    help="Comma-separated model comparison figures passed to the selected model visualizer",
)
@click.pass_context
def matrix_command(
    ctx: click.Context,
    experiment_config: str,
    model_names: tuple[str, ...],
    variant_names: str | None,
    hellobench_lengths: tuple[str, ...],
    dataset_names: tuple[str, ...],
    stage: str,
    demo: bool,
    n_samples: int | None,
    max_new_tokens_values: tuple[int, ...],
    output_root: str | None,
    measure_compute: bool,
    require_all_metrics: bool,
    resume: bool,
    n_representative: int,
    sample_ids: str | None,
    figures: str | None,
) -> None:
    """Run every model-variant x dataset row declared in an experiment YAML."""
    experiment_settings = load_yaml(experiment_config)
    if len(model_names) != 1:
        raise click.UsageError(
            "matrix runs inside one model environment and accepts exactly one --model; "
            "use run_bench.py for multiple models"
        )
    try:
        jobs, seed = load_matrix_jobs(
            experiment_config,
            model_names=model_names,
            dataset_names=dataset_names,
        )
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    click.echo(f"Matrix contains {len(jobs)} model x dataset jobs")
    profiling_jobs = [job for job in jobs if job.profiling_output]
    if profiling_jobs and len(profiling_jobs) != len(jobs):
        raise click.UsageError(
            "one matrix cannot mix profiling_output and standard-output jobs"
        )
    profiling_matrix = bool(profiling_jobs)
    expected_output_stage = "model_profiling" if profiling_matrix else "model_output"
    configured_output_stage = str(
        experiment_settings.get("output_stage", expected_output_stage)
    )
    if configured_output_stage != expected_output_stage:
        raise click.UsageError(
            f"experiment output_stage={configured_output_stage!r} conflicts with "
            f"profiling_output={profiling_matrix}; expected {expected_output_stage!r}"
        )
    if output_root is None:
        configured_output_dir = Path(
            str(
                experiment_settings.get(
                    "output_root", Path("output") / expected_output_stage
                )
            )
        )
        if configured_output_dir.name != expected_output_stage:
            raise click.UsageError(
                f"experiment output_root={str(configured_output_dir)!r} must end "
                f"with output_stage={expected_output_stage!r}"
            )
        output_root = str(configured_output_dir)
    else:
        output_root = str(output_root)
    if profiling_matrix and stage in {"generate", "all"} and not measure_compute:
        raise click.UsageError(
            "this profiling matrix requires --measure-compute; nothing was run"
        )
    if profiling_matrix and stage == "score":
        raise click.UsageError(
            "profiling JSON is not scored; use --stage generate or visualize"
        )
    valid_jobs = 0
    invalid_jobs = 0
    incomplete_jobs = 0
    adapter_cache: dict[tuple[str, str], ModelAdapter] = {}
    length_overrides = tuple(dict.fromkeys(max_new_tokens_values))
    length_cases: tuple[int | None, ...] = length_overrides or (None,)
    split_output_by_length = len(length_overrides) > 1
    run_cases = [
        (length_override, job)
        for length_override in length_cases
        for job in jobs
    ]
    valid_output_roots: set[str] = set()
    for index, (length_override, job) in enumerate(run_cases, start=1):
        selected_variants = job.variants
        if variant_names:
            requested_variants = tuple(
                value.strip() for value in variant_names.split(",") if value.strip()
            )
            # The experiment matrix declares the default operating points, not
            # the full allow-list. Explicit research points such as P4/P8 live
            # in the model YAML and are selectable with ``-v`` without making
            # every ordinary matrix run execute them automatically.
            available_variants = tuple(list_model_variants(str(job.model_config)))
            unknown_variants = set(requested_variants).difference(available_variants)
            if unknown_variants:
                raise click.UsageError(
                    f"unknown variant(s) for {job.model_name}: "
                    f"{', '.join(sorted(unknown_variants))}; available: "
                    f"{', '.join(available_variants)}"
                )
            selected_variants = requested_variants
        variants = ",".join(selected_variants)
        samples_file = str(job.samples_file) if job.samples_file else None
        case_output_root = (
            str(Path(output_root) / f"len{length_override}")
            if split_output_by_length
            else output_root
        )
        common = dict(
            model_config=str(job.model_config), variant=None, variants=variants,
            dataset_config=str(job.dataset_config), demo=demo,
            samples_file=samples_file,
            n_samples=n_samples if n_samples is not None else job.n_samples,
            seed=seed,
            output_root=case_output_root,
            hellobench_lengths=(
                hellobench_lengths or job.hellobench_lengths
                if job.dataset_config.stem == "hellobench"
                else ()
            ),
        )
        length_label = (
            f" [max_new_tokens={length_override}]"
            if length_override is not None
            else ""
        )
        click.echo(
            f"[{index}/{len(run_cases)}] {job.model_config.name} x "
            f"{job.dataset_config.name}{length_label}"
        )
        try:
            if stage in {"generate", "all"}:
                ctx.invoke(
                    generate,
                    **common,
                    max_new_tokens=(
                        length_override
                        if length_override is not None
                        else job.max_new_tokens
                    ),
                    force_max_new_tokens=length_override is not None,
                    measure_compute=measure_compute,
                    profiling_output=job.profiling_output,
                    require_all_metrics=require_all_metrics, resume=resume,
                    capture_trace_override=job.capture_trace,
                    adapter_cache=adapter_cache,
                )
            if stage in {"score", "all"} and not job.profiling_output:
                ctx.invoke(score, **common, resume=resume)
            if stage in {"visualize", "all"}:
                ctx.invoke(
                    visualize,
                    **common,
                    n_representative=n_representative,
                    sample_ids=sample_ids,
                    figures=figures,
                    profiling_output=job.profiling_output,
                )
        except (IncompleteTestError, FileNotFoundError) as exc:
            incomplete_jobs += 1
            click.echo(
                f"ERROR: {job.model_name} x {job.dataset_config.stem} is "
                f"incomplete; skipping this test and continuing. {exc}",
                err=True,
            )
            continue
        except (OOMInvalidTestError, InvalidTestError) as exc:
            invalid_jobs += 1
            click.echo(
                f"ERROR: {job.model_name} x {job.dataset_config.stem} is invalid; "
                f"skipping this test and continuing. {exc}",
                err=True,
            )
            continue
        valid_jobs += 1
        # Reporting consumes the common parent that contains score_output;
        # case_output_root itself is the exact generation-output directory.
        valid_output_roots.add(str(Path(case_output_root).parent))
    if stage == "all" and not profiling_matrix:
        if valid_jobs:
            for report_root in sorted(valid_output_roots):
                ctx.invoke(
                    report,
                    run_paths=(),
                    output_root=report_root,
                    dataset_name=None,
                )
        else:
            click.echo("WARNING: no valid tests completed; aggregate report skipped", err=True)
    if invalid_jobs or incomplete_jobs:
        click.echo(
            "Matrix completed with "
            f"{invalid_jobs} invalid test(s), {incomplete_jobs} incomplete "
            f"test(s) excluded, and {valid_jobs} valid test(s)."
        )


if __name__ == "__main__":
    main()
