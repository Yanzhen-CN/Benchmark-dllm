# dLLM Benchmark

Benchmark harness for diffusion LLMs (iLLaDA, Dream, W1, DiffusionGemma) vs
an AR baseline (Qwen3-4B), implementing the design in
`dLLM_benchmark_设计文档.md`: task quality, long-context robustness,
resource cost, and generation-process analysis (trace, parallelism,
commit-order, certainty).

## Status: framework-first

This is a **framework-first** build: everything that can be verified without
a GPU, real model checkpoints, or network access — the unified model
interface, all Part 3/4 metric math, all 7 dataset scorers, resource
measurement plumbing, the 3-stage pipeline, and report/visualization
generation — is implemented and unit-tested (211 tests, see
[Testing](#testing)). A pure-Python mock model backend (`models/mock.py`)
exercises the entire pipeline end-to-end without any of those dependencies.

What is **not** wired up yet, and why:

| Piece | Status | Why |
| --- | --- | --- |
| iLLaDA / Dream sampler loop | `NotImplementedError` in `models/hf_diffusion.py` | Design doc Appendix D.1/D.2 flags the checkpoint name and HF-integration status as needing verification before this can be written for real. |
| W1 | Configuration only (`configs/models/w1.yaml`) | Project decision: W1 will be integrated against a custom/internal API later — no adapter code changes planned until then, just keep the config shape ready. |
| DG `accept_canvas` hook | Best-effort placeholder in `models/dg.py` | Exact hook argument names aren't pinned down here; adjust `_on_accept` once run against the real checkpoint. |
| HelloEval score | Heuristic fallback in `datasets/hellobench.py` | The real metric is an LLM-judge rubric. Pass `judge_fn` to `HelloBenchDataset` to use a real judge. |
| RULER / StructEval-T / IFEval task banks | Synthetic/representative samples | No official task-bank files wired in; `ruler.build_niah_sample` and `runner/demo_samples.py` are placeholders. |
| Real dataset loading (GSM8K, MBPP, ...) | Not implemented | `Dataset.load_samples` takes whatever `Sample` list you hand it; nothing downloads real data yet. `--demo` samples are hand-authored smoke-test fixtures, regenerated deterministically by every stage (no Sample persistence yet — see below). |
| Batch experiment runner | Not implemented | Each CLI command runs one model-config/variant x dataset-config pair. `configs/experiments/full_matrix.yaml` documents the intended full matrix as a checklist for a future batch script. |

None of these block the framework from being extended — each is isolated
behind the same `ModelAdapter`/`Dataset` interface, with a TODO at the exact
point that needs a real checkpoint/API/judge to finish.

## Layout

```
setup_env.py / run_tests.py   # root install / test-runner scripts (see below)
prepare_model.py               # pre-warm a model's HF checkpoint cache (see below)
configs/
  models/       # one YAML per model, one or more named `configs:` variants (Appendix D)
  datasets/     # one YAML per dataset (section 1/6): dataset class + sample size + seed
  experiments/  # model x dataset matrix checklist (not yet auto-run)
src/dllm_bench/
  interfaces.py     # GenerationRequest/Result, TraceStep, ModelAdapter protocol
  registry.py       # YAML -> instantiated ModelAdapter/Dataset
  hf_cache.py       # project-relative HF cache directory (see below)
  models/           # base.py (resource-measurement wrapper), model_cache.py (shared
                     # loaded-weights cache), hf_ar.py (Qwen3-4B),
                     # hf_diffusion.py (iLLaDA/Dream base), dg.py, w1_api.py, mock.py
  datasets/         # base.py + gsm8k/mbpp/structeval_t/ifeval/sudoku/ruler/hellobench
  resource/         # timing.py/energy.py/compute.py/vram.py (Appendix B protocol)
  metrics/          # quality_resource.py/long_context.py (Part 3),
                     # trace_parallelism.py/strategy_score.py/commit_order.py/
                     # certainty.py (Part 4), stats_utils.py (shared aggregation)
  runner/           # output_layout.py (output/ dir conventions), generate_stage.py,
                     # score_stage.py, orchestrator.py (all-in-one convenience path),
                     # demo_samples.py, persistence.py, sampling.py
  report/           # tables.py/plots.py (3.4), trace_report.py (unified per-sample
                     # entry point), token_grid_viz.py + trace_distribution_viz.py
                     # (trace visuals), sudoku_trace_viz.py (Sudoku's extra GIF)
  cli.py            # `dllm-bench generate/score/visualize/report`
tests/              # one file per module area, 211 tests total
```

## Install

```bash
python setup_env.py                 # core + dev extra (pytest)
python setup_env.py --extras hf,gpu # + torch/transformers + pynvml
```

This just runs `pip install -e .[...]` against whatever Python you invoke it
with — no virtualenv is created for you; activate one yourself first if you
want isolation. A manual `pip install -e ".[dev,hf,gpu]"` works identically.

## Testing

```bash
python run_tests.py             # pytest -q
python run_tests.py -k gsm8k    # extra args pass straight through to pytest
```

All metrics (Part 3/4 formulas), dataset scorers, the resource-wrapping base
adapter, and the visualization modules are tested with hand-constructed
inputs — no GPU or model weights required. `tests/test_registry.py` builds
every shipped `configs/models/*.yaml` variant and `configs/datasets/*.yaml`.
`tests/test_cli.py` and `tests/test_stages.py` run the full
generate → score → visualize → report pipeline (including resume behavior)
through the mock adapter.

## The three-stage pipeline

Generation, scoring, and visualization are three separate CLI commands
instead of one combined "run" — this is what lets you:

- run each model independently (skip W1 entirely; run iLLaDA without
  touching Dream's output);
- resume a half-finished dataset without redoing already-generated or
  already-scored samples (each stage checks per-sample files first);
- generate on one machine (e.g. a GPU box) and score/visualize on another —
  only `model_output/` needs to make that trip.

The atomic unit of testing is the **model**, not model+variant: Best/Fast
(or standard/jump/gidd) run *together*, in one process, by default — loading
weights onto the GPU only happens once, and every variant just changes the
generation-time config (`models/model_cache.py`). So leaving out
`--variant`/`--variants` sweeps every variant declared in the file:

```bash
dllm-bench generate --model-config configs/models/mock.yaml \
                     --dataset-config configs/datasets/gsm8k.yaml \
                     --demo --n-samples 5 --max-new-tokens 32
# runs every variant mock.yaml declares (`default` and `fast`), one process,
# one model load

dllm-bench score    --model-config configs/models/mock.yaml \
                     --dataset-config configs/datasets/gsm8k.yaml --demo --n-samples 5

dllm-bench visualize --model-config configs/models/mock.yaml \
                     --dataset-config configs/datasets/gsm8k.yaml --demo --n-samples 5

dllm-bench report --output-root output --dataset gsm8k
```

Pass `--variant best` to run just one, or `--variants best,fast` to name an
explicit subset. `score`/`visualize` regenerate the same `--demo`/
`--n-samples`/`--seed` sample list rather than reading it back off disk —
pass matching flags to every stage of one run (real dataset loading will
need to persist `Sample`s properly; today's `--demo` samples are cheap and
deterministic enough that regenerating them is fine).

Output lands under `output/` (override with `--output-root`), split by
stage, then by `<model>_<config>`, then by dataset — so `iLLaDA-best` and
`iLLaDA-fast` never collide, and you can `rsync`/copy just `model_output/`
off a GPU box:

```
output/
  model_output/<model>_<config>/<dataset>/
    _meta.json          # model/config/dataset name + run metadata (section 6)
    <sample_id>.json    # full GenerationResult, including trace
  score_output/<model>_<config>/<dataset>/
    <sample_id>.json    # ScoreResult for that sample
    summary.json        # RunSummary — section 3.4's raw-results-table row
  visualization_output/<model>_<config>/<dataset>/
    <sample_id>_*.png / .gif
```

To run the real AR baseline once `torch`/`transformers` are installed and
`Qwen/Qwen3-4B` is reachable, just point at its config instead:

```bash
dllm-bench generate --model-config configs/models/qwen3_4b.yaml \
                     --dataset-config configs/datasets/gsm8k.yaml \
                     --demo --n-samples 5 --max-new-tokens 64
```

## Model checkpoints and caching

Every model here loads via `from_pretrained(repo_id)` — `configs/models/
*.yaml`'s `model_name_or_path` is an HF Hub repo id (e.g. `Qwen/Qwen3-4B`),
never a local checkpoint path. `from_pretrained` downloads it on first use,
same as any other HF model.

By default, that download would land under `~/.cache/huggingface`. On a
cloud GPU box the large/persistent storage is usually a network volume
mounted at the project directory, while the home directory sits on small
local/ephemeral disk — so `hf_cache.py`'s `configure_default_cache_dir()`
points `HF_HOME` at `./.hf_cache` (relative to wherever you launch from)
instead, unless you've already set `HF_HOME`/`HF_HUB_CACHE`/
`TRANSFORMERS_CACHE` yourself (any of those always wins). `cli.py` and
`prepare_model.py` both call this before any model gets touched.

To download/load a model ahead of time instead of lazily mid-benchmark:

```bash
python prepare_model.py --model-config configs/models/illada.yaml
# warms every variant declared in the file — for illada.yaml that's
# `best`+`fast`, but since they share one checkpoint (models/model_cache.py)
# this only downloads/loads it once, same as `dllm-bench generate` does.
```

## Visualization

Every dataset renders through the same unified set (`report/trace_report.py`
calling `token_grid_viz.py` + `trace_distribution_viz.py` + the Part 4
metric curves) — the visual language (colors, layout) is carried over from
`Gemma/DGtest/visual.py` (*How DiffusionGemma Actually Commits Tokens*'
own trace visualizer) for continuity with that prior art:

- **token-grid GIF/PNG** — one cell per token position; gray = masked,
  brown text = visible-but-uncommitted, light green = just accepted,
  black = stable, green→teal→blue→purple→near-black gradient = revised
  multiple times (log-scaled by revision count), red outline = committed
  this frame.
- **position vs first-commit scatter** and **commit-speed chart** — static
  matplotlib companions, same color gradient.
- **Effective Tokens per Forward**, **Structure/Content formation**,
  **Accepted-Ratio × Certainty** — the design doc's own Part 4 formulas
  (`metrics/trace_parallelism.py`/`strategy_score.py`/`certainty.py`).

**Sudoku** gets one more artifact on top: an animated 9x9 grid walking
through the solve (`report/sudoku_trace_viz.py`), most useful for **Hard**
puzzles that need at least one trial-and-error step. Per-cell coloring:

- gray — not yet decided
- black text, white background — a given (prompt-supplied) cell, echoed
  correctly
- yellow — a given cell the model mis-transcribed
- green — a to-fill cell, filled correctly
- red — a to-fill cell, filled incorrectly (visible mid-solve for puzzles
  that need to revise a wrong trial before landing on the right digit)

This requires the trace's canvas to be exactly 81 positions, row-major
(`derive_sudoku_frames`) — no Sudoku-capable model is wired in yet, so
`simulate_sudoku_frames` is a self-contained demo/test fixture (the same
role `models/mock.py` plays for the rest of the framework) used to exercise
and demo the renderer today.

## Configuration: what lives where

Two separate config trees, both under `configs/`, both loaded by
`registry.py` — never edit test *code* to change a run, edit the YAML:

- **`configs/models/*.yaml`** — one file per model, with one or more named
  configs nested under `configs:` (e.g. `illada.yaml`'s `best`/`fast`,
  `w1.yaml`'s `standard`/`jump`/`gidd`). Each variant has `adapter` (dotted
  class path), `init_kwargs` (passed straight to the constructor),
  `step_config` (diffusion-only: `gen_length`/`steps`/`block_length`/
  `steps_per_block`, see Appendix D). `registry.build_model_adapter(path,
  variant=...)` builds one; the CLI defaults to building *every* variant a
  file declares (see "The three-stage pipeline" above) since they're meant
  to be tested together.
- **`configs/datasets/*.yaml`** — one file per dataset. `dataset_class`
  picks the `Dataset` subclass; `dataset_kwargs` is passed straight to its
  constructor (e.g. MBPP's `timeout_s`); `sample_size`/`seed` are read by the
  CLI (`registry.dataset_run_defaults`) as the default `--n-samples`/
  `--seed` when you don't pass them explicitly. `primary_metric`/
  `aux_metrics` are documentation of what `Dataset.aggregate()` reports, not
  inputs — nothing reads them back.

Anything not exposed as a YAML field (a dataset's actual scoring logic, a
model adapter's denoising loop) is Python code in `datasets/*.py` /
`models/*.py`, on purpose — the config layer is for *run parameters*, not
for redefining behavior.

`pytest`'s own configuration (test discovery path) lives in `pyproject.toml`
under `[tool.pytest.ini_options]`; the tests themselves in `tests/*.py` use
hand-constructed fixtures, not the `configs/` YAML files, so changing a
dataset's YAML never silently changes what the unit tests assert.

## Adding a new model

1. Implement `ModelAdapter` (either subclass `BaseModelAdapter` and implement
   `_generate_core`, for anything running resource measurement locally, or
   implement the protocol directly for an API-backed model like `w1_api.py`).
   If it loads HF weights, route `_ensure_loaded` through
   `models/model_cache.get_or_load(model_name_or_path, device, loader)` (see
   `hf_ar.py`/`hf_diffusion.py`/`dg.py`) so multiple variants pointing at the
   same checkpoint share one in-memory copy instead of reloading.
2. Add `configs/models/<name>.yaml` with a `configs:` block naming each
   variant you want, pointing `adapter:` at the dotted class path and
   `init_kwargs:`/`step_config:` at its constructor args. If it's HF-backed,
   `init_kwargs.model_name_or_path` is a repo id, not a local path (see
   "Model checkpoints and caching" above).
3. No other file needs to change — `registry.build_model_adapter` and the
   generate/score/visualize stages are model-agnostic.

## Adding a new dataset

1. Subclass `Dataset` in `datasets/<name>.py`: `load_samples` and `score`.
   `ScoreResult.primary_score` must be normalized to `[0, 1]`.
2. Add `configs/datasets/<name>.yaml` pointing `dataset_class:` at it, plus
   `sample_size`/`seed` and a `dataset_kwargs:` block for any constructor
   args (see `mbpp.yaml`'s `timeout_s` for an example).
3. Optionally add a builder to `runner/demo_samples.py` so `--demo` can
   exercise it.
4. Add it to `tests/test_datasets_scoring.py` (a handful of hand-authored
   correct/incorrect/edge-case samples, no config file needed for this) and
   it'll automatically be covered by `tests/test_registry.py`'s
   "every shipped config builds" parametrized test once step 2 is done.

## Seed and reproducibility

Every run defaults to `seed = 42` (section 6). `runner/sampling.py`'s
`collect_run_metadata` records the seed, model/config name, Python/torch
version, CUDA availability, and (if available) the current git commit into
`model_output/.../_meta.json` at generation time; `score`/`visualize` read
that back rather than recomputing it, so it reflects the machine that
actually generated the run even when scoring happens elsewhere.
