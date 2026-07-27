"""``dllm-bench generate`` / ``score`` / ``visualize`` / ``report``.

Three separate stages instead of one ``run`` command — this is what lets you
run each model independently (skip W1 entirely, run iLLaDA without touching
Dream), lets a half-finished dataset resume without redoing already-done
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
``--no-demo`` uses a dataset's real loader where implemented (currently the
official GSM8K test split). `score`/`visualize` deterministically reconstruct
the same sample list from ``--n-samples``/``--seed``, so pass matching values
to every stage of one run.
"""

from __future__ import annotations

import random
from pathlib import Path

import click

from .hf_cache import configure_default_cache_dir
from .registry import build_dataset, build_model_adapter, dataset_run_defaults, list_model_variants
from .report.tables import raw_results_row, render_raw_results_table
from .report.trace_report import render_sample_report
from .runner.demo_samples import build_demo_samples
from .runner.generate_stage import run_generation
from .runner.output_layout import model_output_dir, score_output_dir, visualization_output_dir
from .runner.persistence import load_generation_result, load_run_summary_dict, load_score_result
from .runner.score_stage import run_scoring


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
    f = click.option("--demo/--no-demo", default=True, show_default=True, help="Use built-in demo samples; --no-demo loads the real dataset where supported")(f)
    f = click.option("--n-samples", default=None, type=int, help="Defaults to the dataset config's `sample_size`")(f)
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


def _resolve_samples(dataset_config: str, dataset, demo: bool, n_samples: int | None, seed: int | None) -> tuple[list, int]:
    defaults = dataset_run_defaults(dataset_config)
    resolved_n = n_samples if n_samples is not None else (defaults["sample_size"] or 5)
    resolved_seed = seed if seed is not None else defaults["seed"]

    if resolved_n <= 0:
        raise click.UsageError("--n-samples must be greater than zero")
    if demo:
        samples = build_demo_samples(dataset.name, n=resolved_n)
        return samples, resolved_seed

    try:
        available = dataset.load_samples()
    except (OSError, RuntimeError, ValueError) as exc:
        raise click.ClickException(f"failed to load real {dataset.name} samples: {exc}") from exc
    if not available:
        raise click.UsageError(
            f"real sample loading is not implemented for dataset {dataset.name!r}; use --demo"
        )
    if resolved_n > len(available):
        raise click.UsageError(
            f"requested {resolved_n} real {dataset.name} samples, but only {len(available)} are available"
        )

    samples = list(available)
    random.Random(resolved_seed).shuffle(samples)
    samples = samples[:resolved_n]
    return samples, resolved_seed


@main.command()
@_common_options
@click.option("--max-new-tokens", required=True, type=int)
@click.option("--measure-compute/--no-measure-compute", default=False, show_default=True)
@click.option("--resume/--no-resume", default=True, show_default=True, help="Skip samples that already have a model_output file")
def generate(
    model_config: str,
    variant: str | None,
    variants: str | None,
    dataset_config: str,
    demo: bool,
    n_samples: int | None,
    seed: int | None,
    output_root: str,
    max_new_tokens: int,
    measure_compute: bool,
    resume: bool,
) -> None:
    variant_list = _resolve_variants(model_config, variant, variants)
    dataset = build_dataset(dataset_config)
    samples, resolved_seed = _resolve_samples(dataset_config, dataset, demo, n_samples, seed)

    click.echo(f"Sweeping variants {variant_list} of {model_config} on {dataset.name} ({len(samples)} samples)")
    for v in variant_list:
        adapter = build_model_adapter(model_config, variant=v)
        out_dir = model_output_dir(output_root, adapter.name, adapter.config_name, dataset.name)
        summary = run_generation(
            adapter, dataset.name, samples, max_new_tokens,
            out_dir=out_dir, measure_compute=measure_compute, seed=resolved_seed, resume=resume,
        )
        click.echo(f"[{v}] generated={summary.generated} skipped={summary.skipped} total={summary.total} -> {out_dir}")


@main.command()
@_common_options
@click.option("--resume/--no-resume", default=True, show_default=True, help="Skip samples that already have a score_output file")
def score(
    model_config: str,
    variant: str | None,
    variants: str | None,
    dataset_config: str,
    demo: bool,
    n_samples: int | None,
    seed: int | None,
    output_root: str,
    resume: bool,
) -> None:
    variant_list = _resolve_variants(model_config, variant, variants)
    dataset = build_dataset(dataset_config)
    samples, _ = _resolve_samples(dataset_config, dataset, demo, n_samples, seed)

    for v in variant_list:
        adapter = build_model_adapter(model_config, variant=v)
        model_out = model_output_dir(output_root, adapter.name, adapter.config_name, dataset.name)
        score_out = score_output_dir(output_root, adapter.name, adapter.config_name, dataset.name)
        result = run_scoring(dataset, samples, model_out, score_out, resume=resume)

        click.echo(f"[{v}] q={result.summary.q:.4f}  scored={result.scored}  skipped={result.skipped}  -> {score_out / 'summary.json'}")
        if result.missing_sample_ids:
            click.echo(f"[{v}] WARNING: {len(result.missing_sample_ids)} sample(s) not yet generated: {result.missing_sample_ids}")


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
    n_samples: int | None,
    seed: int | None,
    output_root: str,
    n_representative: int | None,
    sample_ids: str | None,
) -> None:
    variant_list = _resolve_variants(model_config, variant, variants)
    dataset = build_dataset(dataset_config)
    samples, _ = _resolve_samples(dataset_config, dataset, demo, n_samples, seed)

    if sample_ids:
        wanted = {s.strip() for s in sample_ids.split(",") if s.strip()}
        samples = [s for s in samples if s.sample_id in wanted]
    elif n_representative is not None:
        samples = samples[:n_representative]

    for v in variant_list:
        adapter = build_model_adapter(model_config, variant=v)
        model_out = model_output_dir(output_root, adapter.name, adapter.config_name, dataset.name)
        score_out = score_output_dir(output_root, adapter.name, adapter.config_name, dataset.name)
        viz_out = visualization_output_dir(output_root, adapter.name, adapter.config_name, dataset.name)

        if not (model_out / "_meta.json").exists():
            raise click.UsageError(f"no _meta.json under {model_out} — run `dllm-bench generate` first")

        rendered = 0
        for sample in samples:
            gen_path = model_out / f"{sample.sample_id}.json"
            if not gen_path.exists():
                click.echo(f"[{v}] skipping {sample.sample_id}: not generated yet")
                continue
            generation = load_generation_result(gen_path)

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

        click.echo(f"[{v}] rendered {rendered} sample(s) -> {viz_out}")


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

    summaries = [load_run_summary_dict(p) for p in paths]
    rows = [raw_results_row(s) for s in summaries]
    click.echo(render_raw_results_table(rows))


if __name__ == "__main__":
    main()
