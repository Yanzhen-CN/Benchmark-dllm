"""``dllm-bench generate`` / ``score`` / ``visualize`` / ``report``.

Three separate stages instead of one ``run`` command — this is what lets you
run each model independently (skip W1 entirely, run iLLaDA without touching
DreamReasoner), lets a half-finished dataset resume without redoing already-done
samples, and lets generation happen on one machine (a GPU box) while
scoring/visualization happen on another (see README "三阶段 pipeline").

The atomic unit of testing is the **model**, not model+variant: Best/Fast (or
standard/jump/gidd) get tested *together*, in one process, so the expensive
part (loading weights onto the GPU) happens once and every variant just
changes the generation-time config (see `models/model_cache.py`). So by
default, every command below sweeps **every variant** declared in
`--model-config`'s `configs:` block:

    dllm-bench generate --model-config configs/models/illada.yaml \\
                         --dataset-config configs/datasets/gsm8k.yaml \\
                         --demo --n-samples 5 --max-new-tokens 32
    # runs both `best` and `fast`, loading the checkpoint only once

Pass `--variant best` for just one, or `--variants best,fast` to name a
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

import click

from .hf_cache import configure_default_cache_dir
from .interfaces import GenerationRequest
from .registry import (
    build_dataset,
    build_model_adapter,
    dataset_run_defaults,
    list_model_variants,
    load_yaml,
    model_name,
)
from .report.tables import (
    compute_converted_row,
    is_resource_baseline,
    raw_results_row,
    render_converted_results_table,
    render_raw_results_table,
    select_resource_baselines,
)
from .report.dataset_trace_report import render_dataset_trace_report
from .report.plots import (
    plot_best_vs_fast,
    plot_quality_vs_resource,
    plot_scenario_ranking,
    plot_score_per_unit,
)
from .report.trace_report import render_sample_report
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
    resolve_model_output_dir,
    resolve_score_output_dir,
    score_output_dir,
    visualization_output_dir,
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
    f = click.option("--variant", default=None, help="Run just this one named config (e.g. best)")(f)
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
    f = click.option("--output-root", default="output", show_default=True, type=click.Path(), help="Root of model_output/score_output/visualization_output")(f)
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
    require_all_metrics: bool,
    resume: bool,
    force_max_new_tokens: bool = False,
) -> None:
    variant_list = _resolve_variants(model_config, variant, variants)
    dataset_settings = load_yaml(dataset_config)
    trace_scope = str(dataset_settings.get("trace_scope", "all_samples"))
    if trace_scope not in {"all_samples", "none"}:
        raise click.UsageError(
            f"unsupported trace_scope={trace_scope!r} in {dataset_config}; "
            "use 'all_samples' or 'none'"
        )
    capture_trace = trace_scope == "all_samples"
    dataset = build_dataset(dataset_config)
    samples, resolved_seed = _resolve_samples(dataset_config, model_config, dataset, demo, samples_file, n_samples, seed, hellobench_lengths)

    click.echo(f"Sweeping variants {variant_list} of {model_config} on {dataset.name} ({len(samples)} samples)")
    invalid_variant_errors: list[str] = []
    for v in variant_list:
        adapter = build_model_adapter(model_config, variant=v)
        out_dir = model_output_dir(output_root, adapter.name, adapter.config_name, dataset.name)
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
        pending_count = (
            len(samples)
            if not resume
            else sum(
                not (out_dir / f"{sample.sample_id}.json").exists()
                for sample in samples
            )
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

        def log_progress(event, index, total, sample, generation):
            prefix = f"[{v}] [{index}/{total}] {sample.sample_id}"
            if event == "start":
                click.echo(f"{prefix}: generating ...")
                return
            if event == "compute":
                click.echo(f"{prefix}: profiling compute replay ...")
                return
            elapsed = (
                generation.timing.wall_clock_seconds
                if generation is not None and generation.timing is not None
                else 0.0
            )
            status = generation.status.value if generation is not None else "unknown"
            click.echo(f"{prefix}: {status} ({elapsed:.2f}s)")

        output_stream = click.get_text_stream("stdout")
        try:
            if pending_count and output_stream.isatty():
                with click.progressbar(
                    length=pending_count,
                    label=f"[{v}] {dataset.name}",
                    show_pos=True,
                    show_percent=True,
                    item_show_func=lambda item: str(item or ""),
                    file=output_stream,
                ) as sample_bar:
                    def bar_progress(event, index, total, sample, generation):
                        if event == "start":
                            sample_bar.update(0, current_item=f"{sample.sample_id} generating")
                            sample_bar.render_progress()
                            return
                        if event == "compute":
                            sample_bar.update(
                                0, current_item=f"{sample.sample_id} compute replay"
                            )
                            sample_bar.render_progress()
                            return
                        elapsed = (
                            generation.timing.wall_clock_seconds
                            if generation is not None and generation.timing is not None
                            else 0.0
                        )
                        status = generation.status.value if generation is not None else "unknown"
                        sample_bar.update(
                            1,
                            current_item=f"{sample.sample_id} {status} {elapsed:.2f}s",
                        )

                    summary = run_generation(
                        adapter, dataset.name, samples, max_new_tokens,
                        out_dir=out_dir, measure_compute=measure_compute,
                        require_all_metrics=require_all_metrics, seed=resolved_seed,
                        capture_trace=capture_trace, resume=resume,
                        force_max_new_tokens=force_max_new_tokens,
                        progress=bar_progress,
                    )
            else:
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
    samples, _ = _resolve_samples(dataset_config, model_config, dataset, demo, samples_file, n_samples, seed, hellobench_lengths)
    configured_model = model_name(model_config)

    invalid_rows: list[str] = []
    incomplete_rows: list[str] = []
    for v in variant_list:
        model_out = resolve_model_output_dir(output_root, configured_model, v, dataset.name)
        score_out = score_output_dir(output_root, configured_model, v, dataset.name)
        try:
            result = run_scoring(dataset, samples, model_out, score_out, resume=resume)
        except InvalidTestError as exc:
            invalid_rows.append(str(exc))
            click.echo(f"[{v}] INVALID OOM DATASET: {exc}")
            continue
        except IncompleteTestError as exc:
            incomplete_rows.append(str(exc))
            click.echo(f"[{v}] INCOMPLETE DATASET: {exc}")
            continue

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

    invalid_rows: list[str] = []
    incomplete_rows: list[str] = []
    for v in variant_list:
        model_out = resolve_model_output_dir(output_root, configured_model, v, dataset.name)
        score_out = resolve_score_output_dir(output_root, configured_model, v, dataset.name)
        viz_out = visualization_output_dir(output_root, configured_model, v, dataset.name)

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

            render_sample_report(
                sample_id=sample.sample_id,
                trace=generation.trace,
                final_valid_length=generation.final_valid_length,
                out_dir=str(viz_out),
                final_output_text=generation.output_text,
                final_score=score_result.primary_score if score_result else None,
                dataset_name=dataset.name,
                sample=sample,
            )
            rendered += 1

        if trace_records:
            render_dataset_trace_report(dataset.name, trace_records, viz_out, seed=resolved_seed)

        click.echo(f"[{v}] rendered {rendered} sample(s) -> {viz_out}")

    if invalid_rows:
        raise InvalidTestError("; ".join(invalid_rows))
    if incomplete_rows:
        raise IncompleteTestError("; ".join(incomplete_rows))


@main.command()
@click.option("--run", "run_paths", multiple=True, type=click.Path(exists=True), help="One or more summary.json files from `dllm-bench score`")
@click.option("--output-root", default=None, type=click.Path(), help="Auto-discover summary.json files under output_root/score_output/*/<dataset>/")
@click.option("--dataset", "dataset_name", default=None, help="Dataset name to filter to when using --output-root")
def report(run_paths: tuple[str, ...], output_root: str | None, dataset_name: str | None) -> None:
    paths = list(run_paths)
    if output_root:
        score_root = Path(output_root) / "score_output"
        pattern = f"*/{dataset_name}/summary.json" if dataset_name else "*/*/summary.json"
        paths.extend(str(p) for p in sorted(score_root.glob(pattern)))

    if not paths:
        raise click.UsageError("no summary.json files found (pass --run, or --output-root [--dataset])")

    valid_paths = []
    for raw_path in paths:
        summary_path = Path(raw_path)
        dataset_dir = summary_path.parent
        run_dir = dataset_dir.parent
        score_root = run_dir.parent
        model_out = (
            score_root.parent
            / "model_output"
            / run_dir.name
            / dataset_dir.name
        )
        if score_root.name == "score_output" and (model_out / "_meta.json").exists():
            try:
                ensure_test_valid(model_out)
            except InvalidTestError as exc:
                click.echo(
                    f"WARNING: excluding OOM-invalid test from report: "
                    f"{summary_path} ({exc})",
                    err=True,
                )
                continue
            except IncompleteTestError as exc:
                click.echo(
                    f"WARNING: excluding incomplete test from report: "
                    f"{summary_path} ({exc})",
                    err=True,
                )
                continue
        valid_paths.append(raw_path)
    paths = valid_paths
    if not paths:
        raise click.UsageError("no valid summary.json files remain after exclusions")

    summaries = [load_run_summary_dict(p) for p in paths]
    rows = [raw_results_row(s) for s in summaries]
    click.echo(render_raw_results_table(rows))

    baselines = select_resource_baselines(summaries)
    non_ranked_diagnostics = {"hellobench", "ruler_context_probe"}
    converted = [
        compute_converted_row(summary, baselines[summary["dataset_name"]])
        for summary in summaries
        if not is_resource_baseline(summary)
        and summary["dataset_name"] in baselines
        and summary["dataset_name"] not in non_ranked_diagnostics
    ]
    if converted:
        click.echo("\nAR-relative converted results:\n")
        click.echo(render_converted_results_table(converted))

    if output_root:
        report_root = Path(output_root) / "report"
        dataset_names = sorted({row["Dataset"] for row in rows})
        for name in dataset_names:
            dataset_rows = [row for row in rows if row["Dataset"] == name]
            dataset_converted = [row for row in converted if row["Dataset"] == name]
            out_dir = report_root / name
            out_dir.mkdir(parents=True, exist_ok=True)
            if name in non_ranked_diagnostics:
                continue
            for key, filename in (
                ("TPS", "quality_tps.png"),
                ("SPS", "quality_sps.png"),
                ("EPS", "quality_eps.png"),
                ("CPS", "quality_cps.png"),
            ):
                plot_quality_vs_resource(dataset_rows, key, str(out_dir / filename))
            plot_score_per_unit(dataset_rows, "Score/J", str(out_dir / "score_per_energy.png"))
            plot_score_per_unit(dataset_rows, "Score/TFLOP", str(out_dir / "score_per_compute.png"))
            plot_best_vs_fast(dataset_rows, "q", str(out_dir / "best_vs_fast_quality.png"))
            plot_best_vs_fast(dataset_rows, "TPS", str(out_dir / "best_vs_fast_tps.png"))
            plot_scenario_ranking(dataset_converted, "Speed-priority", str(out_dir / "speed_priority.png"))
            plot_scenario_ranking(dataset_converted, "Energy-priority", str(out_dir / "energy_priority.png"))
        click.echo(f"\nCharts -> {report_root}")


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
    default=None,
    type=click.IntRange(min=1),
    help="Temporary override; supersedes matrix and per-sample output limits",
)
@click.option("--output-root", default="output", show_default=True, type=click.Path())
@click.option("--measure-compute/--no-measure-compute", default=False, show_default=True)
@click.option("--require-all-metrics/--allow-missing-metrics", default=False, show_default=True)
@click.option("--resume/--no-resume", default=True, show_default=True)
@click.option("--n-representative", default=3, show_default=True, type=int)
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
    max_new_tokens: int | None,
    output_root: str,
    measure_compute: bool,
    require_all_metrics: bool,
    resume: bool,
    n_representative: int,
) -> None:
    """Run every model-variant x dataset row declared in an experiment YAML."""
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
    valid_jobs = 0
    invalid_jobs = 0
    for index, job in enumerate(jobs, start=1):
        selected_variants = job.variants
        if variant_names:
            requested_variants = tuple(
                value.strip() for value in variant_names.split(",") if value.strip()
            )
            unknown_variants = set(requested_variants).difference(job.variants)
            if unknown_variants:
                raise click.UsageError(
                    f"unknown variant(s) for {job.model_name}: "
                    f"{', '.join(sorted(unknown_variants))}; available: "
                    f"{', '.join(job.variants)}"
                )
            selected_variants = requested_variants
        variants = ",".join(selected_variants)
        samples_file = str(job.samples_file) if job.samples_file else None
        common = dict(
            model_config=str(job.model_config), variant=None, variants=variants,
            dataset_config=str(job.dataset_config), demo=demo,
            samples_file=samples_file,
            n_samples=n_samples if n_samples is not None else job.n_samples,
            seed=seed,
            output_root=output_root,
            hellobench_lengths=(
                hellobench_lengths or job.hellobench_lengths
                if job.dataset_config.stem == "hellobench"
                else ()
            ),
        )
        click.echo(f"[{index}/{len(jobs)}] {job.model_config.name} x {job.dataset_config.name}")
        try:
            if stage in {"generate", "all"}:
                ctx.invoke(
                    generate,
                    **common,
                    max_new_tokens=(
                        max_new_tokens
                        if max_new_tokens is not None
                        else job.max_new_tokens
                    ),
                    force_max_new_tokens=max_new_tokens is not None,
                    measure_compute=measure_compute,
                    require_all_metrics=require_all_metrics, resume=resume,
                )
            if stage in {"score", "all"}:
                ctx.invoke(score, **common, resume=resume)
            if stage in {"visualize", "all"}:
                ctx.invoke(
                    visualize, **common, n_representative=n_representative, sample_ids=None,
                )
        except (OOMInvalidTestError, InvalidTestError, IncompleteTestError) as exc:
            invalid_jobs += 1
            click.echo(
                f"ERROR: {job.model_name} x {job.dataset_config.stem} is invalid; "
                f"skipping this test and continuing. {exc}",
                err=True,
            )
            continue
        valid_jobs += 1
    if stage == "all":
        if valid_jobs:
            ctx.invoke(report, run_paths=(), output_root=output_root, dataset_name=None)
        else:
            click.echo("WARNING: no valid tests completed; aggregate report skipped", err=True)
    if invalid_jobs:
        click.echo(
            f"Matrix completed with {invalid_jobs} OOM-invalid test(s) excluded "
            f"and {valid_jobs} valid test(s)."
        )


if __name__ == "__main__":
    main()
