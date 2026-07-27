# dLLM Benchmark

Benchmark harness for diffusion LLMs (iLLaDA, DreamReasoner, W1, DiffusionGemma) vs
an AR baseline (Qwen3-4B), implementing the design in
`dLLM_benchmark_设计文档.md`: task quality, long-context robustness,
resource cost, and generation-process analysis (trace, parallelism,
commit-order, certainty).

## Status

The framework (unified model interface, all Part 3/4 metric math, all six formal
dataset scorers, resource measurement plumbing, the 3-stage pipeline, and
report/visualization generation) is fully implemented and unit-tested (255
tests, see [Testing](#testing)). A pure-Python mock model backend
(`models/mock.py`) exercises the entire pipeline end-to-end without a GPU.

On top of that, every **local** model's real sampling loop is implemented —
not just interface stubs:

| Model | Status | Verified against |
| --- | --- | --- |
| iLLaDA | **Real, ported sampler** (`models/illada.py`) | `iLLaDAtest`'s reference `generate.py` — checkpoint id, `mask_id=5` override, block-wise low-confidence unmasking, gumbel-noise formula. Algorithm-level tests in `tests/test_illada_sampling.py` (fake-logits, no GPU) check the actual selection order, not just wiring. |
| DiffusionGemma | **Real** (`models/diffusiongemma.py`) | Verified against the upstream `DiffusionGemmaForBlockDiffusion`/`EntropyBoundSampler` implementation. Trace capture wraps `accept_canvas` through `_prepare_sampler`. Tests live in `tests/test_diffusiongemma_sampling.py`. |
| DreamReasoner | **Real, ported sampler** (`models/dreamreasoner.py`) | GitHub's `DreamLM/DreamReasoner` repo (the design doc's own link) ships only a README + assets, no python — so verified instead by fetching the real `generation_utils.py`/`modeling_dream.py`/`config.json` from the `Dream-org/DreamReasoner-8B` HF repo directly. Confirmed from that source (not assumed): `block_diffusion_generate` returns only `sequences`/`nfe`, no per-step history at all (unlike regular Dream-7B's `output_history`), so this adapter ports the real `_denoise_current_block`/`_select_transfer_index` loop itself (like iLLaDA) rather than calling the model's own convenience method; default `block_length`=`config.block_size`=32; default per-block `denoising_steps`=`block_length` unless `remasking_strategy='low_confidence_static'`; the library's own default remasking strategy (`low_confidence_dynamic`) is already confidence-based, so — unlike regular Dream, where the library default was overridden — no override is applied; `mask_token_id`=`config.mask_token_id`=151669, shipped directly; the real default path uses a prefix KV cache (ported faithfully, since it directly affects real Time/Energy/Compute cost, not just trace fidelity); never revises (same structural reason as iLLaDA). This is a genuinely different, independently trained model from regular Dream-7B (`Dream-org/Dream-v0-Instruct-7B`) — not a config variant of it — and the design doc's own model roster (section 5) no longer includes regular Dream at all, so that adapter/config/tests were removed rather than kept alongside this one. Tests in `tests/test_dreamreasoner_sampling.py`. |

What's still open, and why:

| Piece | Status | Why |
| --- | --- | --- |
| W1 | API adapter + configuration (`models/w1_api.py`, `configs/models/w1.yaml`) | The transport/timing path is implemented; the private endpoint and its trace payload still need to be validated against the real service before Part 4 is enabled. |
| HelloEval score | Heuristic fallback in `datasets/hellobench.py` | The real metric is an LLM-judge rubric. Pass `judge_fn` to `HelloBenchDataset` to use a real judge. |
| Real dataset loading | GSM8K + local files | `--no-demo` downloads pinned/checksummed official GSM8K; every dataset also accepts local JSON/JSONL through `--samples-file`. Remaining official task-bank downloaders are external preparation. |
| Batch experiment runner | Implemented with isolated environments | `run_bench.py` reads the matrix and delegates each model to its own script/venv. |
| Running the real models end-to-end | Not done in this environment | No GPU / multi-GB checkpoint downloads here — the sampling loops themselves are ported/verified against reference code and algorithm-tested with fake logits (see table above), but nobody has run them against the actual weights yet. First real run should sanity-check output quality before trusting the pipeline's numbers. |

None of these block the framework from being extended — each is isolated
behind the same `ModelAdapter`/`Dataset` interface, with a TODO at the exact
point that needs a real checkpoint/API/judge/GPU to finish.

## Layout

```
run_bench.py                   # main entry: full matrix by default, -m filters models
setup_venv.py / run_tests.py   # venv dispatcher / test runner (see below)
prepare_model.py               # pre-warm a model's HF checkpoint cache (see below)
prepare_data.py                # prepare/cache every real dataset in the matrix
venv_scripts/                  # one Python venv/install/run script per model
configs/
  models/       # one YAML per model, one or more named `configs:` variants (Appendix D)
  datasets/     # one YAML per dataset (section 1/6): dataset class + sample size + seed
  experiments/  # executable model x dataset matrices
src/dllm_bench/
  interfaces.py     # GenerationRequest/Result, TraceStep, ModelAdapter protocol
  registry.py       # YAML -> instantiated ModelAdapter/Dataset
  hf_cache.py       # project-relative HF cache directory (see below)
  models/           # base.py (resource-measurement wrapper), model_cache.py (shared
                     # loaded-weights cache), hf_ar.py (Qwen3-4B),
                     # hf_diffusion.py (iLLaDA/DreamReasoner shared base + DiffusionStepConfig),
                     # illada.py, dreamreasoner.py (each model's real sampler — see Status),
                     # diffusiongemma.py, w1_api.py, mock.py
  datasets/         # base.py + gsm8k/mbpp/structeval_t/sudoku/ruler/hellobench
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
  cli.py            # `dllm-bench generate/score/visualize/report/matrix`
tests/              # one file per module area; see Testing for the current suite
```

## Main entry point

`run_bench.py` is the normal way to launch the benchmark. With no `-m`, it
runs every model and variant declared in `configs/experiments/full_matrix.yaml`:

```bash
python run_bench.py                    # all matrix models
python run_bench.py -m illada          # iLLaDA Best + Fast
python run_bench.py -m dreamreasoner   # DreamReasoner Best + Fast
python run_bench.py -m qwen3_4b        # AR baseline only
python run_bench.py -m diffusiongemma  # large-model reference
python run_bench.py -m illada -m qwen3_4b
```

Useful controls:

```bash
python run_bench.py --list-models
python run_bench.py --dry-run -m illada
python run_bench.py -m illada --stage generate --n-samples 20
python run_bench.py --real-data  # use samples_file entries from the matrix
```

Runs are resumable by default. The built-in demo dataset is the default. A
real-data run checks the normalized cache first and automatically invokes the
same preparation logic as `prepare_data.py` only when its artifact is absent.

## Environment setup

```bash
python setup_venv.py                   # every model declared in the matrix
python setup_venv.py -m illada         # only .venvs/illada
python setup_venv.py -m dreamreasoner  # only .venvs/dreamreasoner
python setup_venv.py -m diffusiongemma # only .venvs/diffusiongemma
```

`setup_venv.py` is only a dispatcher. It calls
`venv_scripts/<model>.py setup`, and each Python script creates its own venv
with model-specific torch/transformers pins. No model packages are installed
into the Python running `setup_venv.py`. Model environments are grouped under
the single `.venvs/` directory; the root `.venv/` remains the development and
test environment.

### Model scripts

Every model has one public script with the same four actions:

| Action | Purpose |
| --- | --- |
| `setup` | Create or update the model-specific virtualenv |
| `check` | Validate dependencies and construct every configured adapter |
| `prepare` | Download and load the checkpoint without generating samples |
| `run` | Run generation, scoring, and the result table |

Available entry points and environments:

| Model | Script | Environment |
| --- | --- | --- |
| Qwen3-4B AR | `venv_scripts/qwen3_4b.py` | `.venvs/qwen3_4b` |
| iLLaDA | `venv_scripts/illada.py` | `.venvs/illada` |
| DreamReasoner | `venv_scripts/dreamreasoner.py` | `.venvs/dreamreasoner` |
| DiffusionGemma | `venv_scripts/diffusiongemma.py` | `.venvs/diffusiongemma` |
| W1 | `venv_scripts/w1.py` | `.venvs/w1` |
| Mock | `venv_scripts/mock.py` | `.venvs/mock` |

Example lifecycle:

```bash
python venv_scripts/qwen3_4b.py setup
python venv_scripts/qwen3_4b.py check
python venv_scripts/qwen3_4b.py prepare
python venv_scripts/qwen3_4b.py run
```

`run` creates the environment automatically when it is missing, then launches
the model's matrix rows explicitly with that venv's Python executable. It does
not depend on shell activation. Override settings with environment variables:

```bash
DATA_SOURCE=demo N_SAMPLES=1 STAGE=all \
  OUTPUT_ROOT=output/checks/qwen3_4b python venv_scripts/qwen3_4b.py run
```

CUDA 12.4 is the default package index. Override it during setup when supported:

```bash
python venv_scripts/qwen3_4b.py setup --cuda-index cu126
python venv_scripts/dreamreasoner.py setup --cuda-index cu121
```

For formal data declared through `samples_file` in the experiment matrix:

```bash
DATA_SOURCE=real OUTPUT_ROOT=output/formal \
  python venv_scripts/qwen3_4b.py run
```

The default diagnostic suite uses 100 samples for each regular-capability
dataset. Sudoku is stratified 50 easy / 50 hard. RULER selects 10 samples for
each `context-window × position` cell at both the common 8192-token point and
the selected model's own maximum window; NIAH, multi-hop tracing, and
aggregation are balanced inside those cells. The 64-token
answer allowance is included in that window, so the corresponding prompt
targets are 8128 and `model_max - 64` tokens. If both windows are 8192 (as for
iLLaDA), the same common point is run once: 30 samples rather than a duplicated
60. The formal strengthened profile can raise each cell from 10 to 20.

HelloBench independently measures long output from short prompts: 10 samples
target 2K words with `max_new_tokens=3072`, and 10 target 4K words with
`max_new_tokens=6144`. These are attached per sample by the runner, so the
matrix-wide fallback cannot accidentally reduce both groups to 256 tokens.

Model weights, official/generated datasets, and package wheels are all kept
under the repository-root `.data/` directory (`huggingface/`, `datasets/`,
and `pip-cache/`). This makes the cache follow the project onto a mounted
cloud volume. Set `DLLM_DATA_ROOT=/mounted/path/.data` to override it. W1
additionally requires `W1_API_BASE_URL` and, when applicable, `W1_API_KEY` at
run time; data preparation does not need either. Reference-only W1 uses the
common/base 8192-token RULER point.

## Data preparation

Prepare every real dataset declared in the matrix before allocating a GPU:

```bash
python prepare_data.py
python prepare_data.py --force  # rebuild matching prepared artifacts
```

Or prepare one dataset/source explicitly:

```bash
dllm-bench prepare-data \
  --dataset-config configs/datasets/gsm8k.yaml

dllm-bench prepare-data \
  --dataset-config configs/datasets/mbpp.yaml \
  --samples-file /mounted/raw/mbpp-sanitized.jsonl
```

Prepared samples land at
`.data/datasets/prepared/<dataset>/<fingerprint>/samples.jsonl`, accompanied
by a manifest. The fingerprint covers the dataset YAML, loader implementation,
and raw source contents, so rerunning is idempotent while source/config changes
create a distinct cache artifact.

`python run_bench.py --real-data` uses exactly the same function: cache hit
means immediate reuse; cache miss means prepare first, then start model work.
Data download, validation, normalization, and loading therefore remain outside
the per-sample timing window. A dataset without an official loader must provide
`samples_file`; the error is raised during preparation, before any model is
loaded or timed.

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

The three real sampler implementations are also algorithm-tested against
fake logits/models (`tests/test_illada_sampling.py`, `test_diffusiongemma_sampling.py`,
`test_dreamreasoner_sampling.py`) — these check the actual
selection/trace-construction logic (e.g. iLLaDA's top-k-by-confidence commit
order, DiffusionGemma's `_prepare_sampler` patch-and-restore under exceptions,
DreamReasoner's KV-cache store_kv call pattern and force-accept-on-last-step),
not just that the classes import and wire together. They still can't replace
running against real weights on a GPU (see the Status table).

## The three-stage pipeline

Generation, scoring, and visualization are three separate CLI commands
instead of one combined "run" — this is what lets you:

- run each model independently (skip W1 entirely; run iLLaDA without
  touching DreamReasoner's output);
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

For real local data, replace `--demo` with a JSON/JSONL file. Records use
`sample_id`, `prompt`, `reference`, and optional `meta`; common GSM8K
(`question`/`answer`) and MBPP (`text`/`test_list`) exports are also accepted:

```bash
dllm-bench generate --model-config configs/models/illada.yaml \
  --dataset-config configs/datasets/gsm8k.yaml --no-demo \
  --samples-file data/gsm8k.jsonl --max-new-tokens 256
```

Run the complete declared matrix with isolated model environments:

```bash
python prepare_data.py          # recommended: finish data work first
python run_bench.py --real-data # also auto-prepares any missing artifact
```

Pass `--variant best` to run just one, or `--variants best,fast` to name an
explicit subset. `score`/`visualize` deterministically reconstruct the same
sample list from `--demo`/`--no-demo`, `--n-samples`, and `--seed`; pass matching
values to every stage. The official GSM8K loader uses stable source indices as
sample IDs and a pinned source revision.

Formal RULER records must provide `task_type`, `position`, `required_answers`,
and `context_length` in `reference`. `context_length` may be either the target
prompt-token count (for example 8128) or the named context-window point (8192);
`meta.context_window_tokens` or `meta.input_tokens` can state it explicitly.
Formal HelloBench records provide `reference.target_length_words` as either
2000 or 4000. Dataset-aware sampling is deterministic under `--seed`.

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
                     --no-demo --seed 42 --n-samples 5 --max-new-tokens 512
```

iLLaDA and DiffusionGemma work the same way (`configs/models/illada.yaml`,
`configs/models/diffusiongemma.yaml`). Their isolated scripts install the
appropriate transformer versions. Nobody has run either against real
weights yet in this environment (no GPU here) — see the Status table above.

## Model checkpoints and caching

Every model here loads via `from_pretrained(repo_id)` — `configs/models/
*.yaml`'s `model_name_or_path` is an HF Hub repo id (e.g. `Qwen/Qwen3-4B`),
never a local checkpoint path. `from_pretrained` downloads it on first use,
same as any other HF model.

By default, that download would land under `~/.cache/huggingface`. On a
cloud GPU box the large/persistent storage is usually a network volume
mounted at the project directory, while the home directory sits on small
local/ephemeral disk — so `hf_cache.py` points `HF_HOME` at the repository's
`.data/huggingface` directory regardless of the launch working directory.
An explicit `HF_HOME`/`HF_HUB_CACHE`/`TRANSFORMERS_CACHE` still wins.
`cli.py` and `prepare_model.py` apply this before any model is touched.

To download/load a model ahead of time instead of lazily mid-benchmark:

```bash
# all models in the matrix, each through its own isolated environment
python prepare_model.py

# selected matrix models
python prepare_model.py -m illada -m qwen3_4b

# direct single-config mode inside the current compatible environment
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
   `hf_ar.py`/`hf_diffusion.py`/`diffusiongemma.py`) so variants pointing at the
   same checkpoint share one in-memory copy instead of reloading. If the
   model's own convenience generate method doesn't expose a usable per-step
   trace (confirm this from its real source first, the way
   `dreamreasoner.py`/`illada.py`'s module docstrings do — don't assume),
   reimplement its real denoising loop yourself, calling the model's forward
   pass directly each step and building `TraceStep`s from whatever
   confidence/selected-positions signal *that* algorithm actually produces
   (see `illada.py`/`dreamreasoner.py`); if it exposes something richer
   through its own generate method (DiffusionGemma's `accepted_token_mask`),
   hook that instead (see `diffusiongemma.py`).
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
