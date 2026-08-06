# dLLM Benchmark

A reproducible benchmark harness for comparing diffusion language models, autoregressive baselines, and deployment-optimized generation paths under one task, resource, and trace contract.

The project separates GPU generation from local scoring and visualization. Model outputs are immutable JSON artifacts, so scoring rules and figures can be updated without rerunning expensive inference.

## Documentation

- The desktop document `dLLM_benchmark_设计文档.md` is the research protocol: questions, fairness rules, metric definitions, comparison boundaries, and report design. It is intentionally kept outside the repository.
- This README is the repository's only maintained documentation: installation, commands, configuration, implementation contracts, and extension workflow.

Experimental results and temporary progress do not belong in the repository documentation. Keep generated artifacts under `output/` and write result analysis in a separate report.

## What the system measures

| Axis | Primary outputs |
| --- | --- |
| Quality | Dataset primary score plus a small, task-specific set of failure diagnostics |
| Parallelism | Accepted-token TPS, accepted tokens per forward, actual model forwards, acceptance trace |
| Cost | Seconds/sample, joules/sample, average power, compute/sample, peak VRAM |

Results are compared only inside compatible dataset, sample-set, prompt/output-budget, hardware, and measurement-protocol groups. Missing or unobservable metrics are `N/A`, never zero-filled.

## Pipeline

```text
configs + prepared data
        |
        v
run_model.py  ----------> output/model_output/       (GPU server)
                                |
                                +--> run_score.py     (local CPU)
                                |
                                +--> run_visualization.py
```

The normal workflow has three user-facing execution stages:

1. `run_model.py` generates model output and trace on the GPU server.
2. `run_score.py` evaluates transferred JSON locally without loading model weights.
3. `run_visualization.py` builds trace artifacts and measured-only reports locally.

## Quick start

### 1. Prepare a server

The top-level scripts may be launched with any available system Python, which
acts only as a dispatcher and never receives model dependencies. Each model
command executes with `.venvs/<model>/bin/python`, while data preparation uses
`.venvs/root/bin/python`. If an interactive command finds its environment
missing or incomplete, it offers to run setup and changes the environment only
after a `y` confirmation; non-interactive commands exit instead. The explicit
setup commands below remain the normal setup path.

```bash
git clone <repository-url> dllm
cd dllm

# Prepare environments, data, and model snapshots explicitly.
python setup_venv.py --cuda-index cu124 --check
python prepare_data.py --experiment-config configs/experiments/full_matrix.yaml
python prepare_model.py --matrix configs/experiments/full_matrix.yaml
```

Selected models or datasets use the same selectors:

```bash
python setup_venv.py -m qwen3_8b illada dreamreasoner --cuda-index cu124 --check
python prepare_data.py --experiment-config configs/experiments/full_matrix.yaml \
  -d gsm8k mbpp structeval_t
python prepare_model.py --matrix configs/experiments/full_matrix.yaml \
  -m qwen3_8b illada dreamreasoner
```

Large caches default to the project volume:

```text
data/huggingface/
data/datasets/prepared/
data/tmp/
data/torch-extensions/
```

`DLLM_DATA_ROOT` is the only cache/data location control and applies to every
model environment. Hugging Face Hub, Xet, datasets, build temporary files and
Torch extensions are all bound below that tree; inherited `HF_HOME` and legacy
cache variables cannot redirect individual models elsewhere. Pip/uv download
caches are disabled because their wheels duplicate packages already installed
inside each venv. Model dependencies remain isolated under `.venvs/<model>/`,
with the local non-model tools under `.venvs/root`. `DLLM_VENV_ROOT` is the only
venv location override.

To deliberately discard and rebuild selected environments:

```bash
python setup_venv.py -m diffusiongemma --recreate --check
```

### 2. Generate on the GPU server

```bash
# All models and formal variants in the matrix.
python run_model.py --real-data

# Selected models.
python run_model.py -m qwen3_8b dreamreasoner --real-data

# Selected datasets; `sudoku` selects every sudoku* matrix entry.
python run_model.py -m qwen3_8b -d gsm8k mbpp structeval_t
python run_model.py -m qwen3_8b -d sudoku

# Selected model variants.
python run_model.py -m dreamreasoner -v p1 p2
```

Generation is resumable by default. Use `--no-resume` only when intentionally replacing the selected generation row.

### 3. Transfer the raw generation layer

Copy only the immutable generation artifacts to the local machine:

```text
output/model_output/
```

Do not copy model environments or checkpoints for local scoring.

### 4. Score and visualize locally

```bash
python run_score.py --real-data
python run_visualization.py --real-data
```

The selectors are parallel across stages:

```bash
python run_score.py -m illada dreamreasoner -d gsm8k mbpp -v p1 p2
python run_visualization.py -m illada dreamreasoner -d gsm8k mbpp -v p1 p2
```

Re-scoring never regenerates model output:

```bash
python run_score.py -m qwen3_8b -d gsm8k --no-resume
```

## Common selectors

| Option | Meaning |
| --- | --- |
| `-m/--model` | One or more model names from the experiment matrix |
| `-d/--dataset` | One or more datasets; `sudoku` expands to every `sudoku*` row |
| `-v/--variant` | One or more model variants such as `p1 p2 p4 p8` |
| `-max/--max-new-tokens` | Temporary output budget override; repeat for a length sweep |
| `--n-samples` | Diagnostic sample-count override |
| `--output-root` | Isolated artifact tree for a diagnostic run |
| `--resume/--no-resume` | Reuse or replace compatible per-sample artifacts |
| `--dry-run` | Print commands without running them |

List configured models before running:

```bash
python run_model.py --list-models
python setup_venv.py --list-models
```

## Length and parallelism probes

Multiple output budgets are executed inside one model process, so the loaded model can be reused:

```bash
python run_model.py -m illada_vargen \
  -d gsm8k mbpp structeval_t \
  -v p1 p2 p4 p8 \
  -max 1024 2048 \
  --n-samples 1 --no-resume \
  --output-root output/length_probe

python run_score.py -m illada_vargen \
  -d gsm8k mbpp structeval_t \
  -v p1 p2 p4 p8 \
  -max 1024 2048 \
  --output-root output/length_probe
```

When multiple lengths are selected, each length is isolated under `len<tokens>/` inside the output root. Length probes are diagnostic and must not overwrite formal rows.

## Profiling 3x3

`model_profiling` is a second `model_output` tree, not a separate artifact
format. Both trees use the identical
`<model>/<config>/<dataset>/` structure and contain the same `_meta.json`,
per-sample JSON, completion state, and OOM records. Profiling sample JSON only
adds the profiling measurements collected by this protocol.

The deep profiling matrix runs one fixed sample from MBPP, GSM8K, and
StructEval-T for DiffusionGemma official, iLLaDA P2, iLLaDA-VARGEN P2, and
DreamReasoner P2: 4 model execution paths x 3 tasks x 1 sample. One sample is
intentional because each row performs a deterministic Torch operator/module
replay in addition to the timed generation. Token identity trace capture is
disabled; it is unrelated to systems profiling.

Every adapter uses the same in-repository protocol and persists raw integer
FLOPs plus TFLOPs. Shared stage names cover input preparation, prefill/cache
build when present, denoise steps, token selection, canvas/cache updates,
and output decoding. Torch Profiler writes `torch_trace.json` and
`torch_summary.json` under each sample's `_profiling/` directory. The summary
groups CUDA/CPU time, calls, and FLOPs by attention, linear, MLP/MoE,
normalization, embedding, sampling, and KV-cache categories, while module
ranges retain per-layer names. A stage that a model does not execute is absent,
not recorded as zero.

```bash
python run_model.py --matrix configs/experiments/profiling_matrix.yaml \
  -m diffusiongemma illada illada_vargen dreamreasoner \
  -d mbpp gsm8k structeval_t --measure-compute --no-resume \
  --output-root output

python run_visualization.py --matrix configs/experiments/profiling_matrix.yaml \
  -m diffusiongemma illada illada_vargen dreamreasoner \
  -d mbpp gsm8k structeval_t --output-root output
```

Each dataset visualization directory contains `dataset_step_profiling.png`
and `dataset_stage_profiling.png`.
The plot is derived directly from per-sample profiling JSON. Per-step and
per-stage compute comes from the same deterministic replay; the sample also
retains total `compute_flops` and `compute_tflops`. Optional Nsight runs may
consume the emitted `dllm::stage::*` and `dllm::step::*` NVTX ranges for
kernel-level diagnosis, but Nsight is not a formal cross-model metric because
its output depends on the installed driver and profiler build.

The profiling matrix declares `profiling_output: true`. Raw `_meta.json` and
per-sample JSON files are written under
`output/model_profiling/<model>/<config>/<dataset>/`, exactly parallel to
`output/model_output/<model>/<config>/<dataset>/`. PNG reports remain under the
matching `output/visualization_output/<model>/<config>/<dataset>/`. Profiling
fields distinguish collection modes inside JSON; they do not change the output
layout. Profiling runs are diagnostic and are not passed through
`run_score.py`.

## Curated and dataset-level visualization

Dataset-level Task 4 summaries always use every available trace in the selected row. `--sample-ids` and `--n-representative` control only the additional single-sample evidence.

```bash
python run_visualization.py \
  -m diffusiongemma \
  -d gsm8k mbpp structeval_t \
  -v official SC2 SC05 SC0 EB2 EB05 Lg2 Lg05 \
  --n-representative 0

python run_visualization.py \
  -m diffusiongemma \
  -d mbpp structeval_t \
  --sample-ids mbpp-sanitized-0131,structeval-t-180530
```

Model-specific comparison modules may also expose a direct CLI:

```bash
python -m dllm_bench.visual.models.diffusiongemma \
  -d structeval_t \
  -v official SC2 SC05 SC0 EB2 EB05 Lg2 Lg05 \
  --figure trace state convergence yield forward
```

## Optional pairwise conversion

Measured results are the default report. The optional pairwise ideal-retry sensitivity analysis remains an advanced CLI command:

```bash
python -m dllm_bench.cli pairwise-report \
  -m illada dreamreasoner \
  --base-model qwen3_8b \
  --base-config ar-baseline \
  --beta 50 --gamma 30
```

Each `A relative to B` pair receives a separate directory. The command rejects mismatched samples, prompts, budgets, dataset revisions, timing sources, and measurement protocols. It does not create a cross-pair leaderboard.

## Environment model

The project uses one environment per model because Torch, Transformers, CUDA, vLLM, and checkpoint-specific code may be incompatible.

```text
.venvs/root/               # preparation, scoring, visualization, checking
.venvs/qwen3_8b/
.venvs/illada/
.venvs/illada_vargen/
.venvs/dreamreasoner/
.venvs/diffusiongemma/
.venvs/gemma/
.venvs/gemma_dflash/
...
```

Create or validate environments explicitly:

```bash
python setup_venv.py
python setup_venv.py -m illada dreamreasoner --cuda-index cu124
python setup_venv.py -m diffusiongemma --check
python venv_scripts/root.py check
```

Every model script supports four lifecycle actions:

| Action | Purpose |
| --- | --- |
| `setup` | Create/update the isolated environment |
| `check` | Validate installed packages and construct configured adapters |
| `prepare` | Download checkpoint/tokenizer files only |
| `run` | Internal generation dispatch used by the top-level scripts |

`prepare_model.py` does not construct an adapter, load weights, move a model to GPU, or run warmup.

## Data preparation

Prepare all datasets selected by a matrix:

```bash
python prepare_data.py
python prepare_data.py -d gsm8k mbpp structeval_t
python prepare_data.py -d sudoku
python prepare_data.py -d sudoku --force
```

Preparation downloads or generates the source data, validates its revision/checksum where applicable, and writes a deterministic prepared bank. A real-data run checks this bank first and invokes the same preparation logic only when an artifact is missing.

For reproducible server runs, prepare data explicitly before generation so network and preprocessing work never occurs during the benchmark session.

## Output contract

```text
output/
  model_output/<model>/<config>/<dataset>/
    _meta.json
    <sample_id>.json
    oom_info.json              # only when the complete row is invalid
  score_output/<model>/<config>/<dataset>/
    <sample_id>.json
    summary.json
  model_profiling/<model>/<config>/<dataset>/
    _meta.json
    <sample_id>.json
  visualization_output/
    <model>/<config>/<dataset>/...
    <model>/model_comparison/<dataset>/...
    profiling_comparison/...
  report/...
  conversion_output/...
```

`model_output` is the raw source of truth. A generation row is reportable only when `_meta.json` marks it complete and valid and every selected sample exists.

Remove local test caches and temporary files from the repository root:

```bash
python tests/clean_test.py --dry-run
python tests/clean_test.py
```

The cleaner removes only test, coverage, and Python cache artifacts. It does not touch `output`, prepared datasets, model files, or virtual environments.

Score reuse is guarded by fingerprints over the generation text, dataset revision, prompt protocol, scorer revision, ordered sample-set hash, and primary metric. Changing a scorer triggers re-scoring without requiring a new GPU run.

## OOM, interruption, and resume

- The first OOM invalidates the current model x variant x dataset row.
- The runner writes `oom_info.json`, stops later samples in that row, and continues with later datasets.
- Scoring and visualization refuse to aggregate an invalid row.
- An interrupted incomplete row may retain per-sample JSON for diagnosis, but it cannot produce a formal `summary.json`.
- Resume reuses only sample artifacts compatible with the current matrix. Score and visualization reject incomplete or incompatible rows.

## Core implementation contracts

These contracts keep server generation, local scoring, and later extensions compatible.

- `_meta.json.selected_sample_ids` is ordered and defines the exact dataset row. A row is reportable only when metadata marks it complete and valid and every selected sample artifact exists.
- `SUCCESS` runs the dataset scorer normally. `TRUNCATED` scores the available output, preserves any already-present valid answer, and records `complete=false`. OOM, model-load failure, infrastructure failure, or a missing selected sample invalidates the complete model × variant × dataset row.
- Score reuse requires matching dataset revision, prompt/protocol revision, scorer revision, ordered sample-set hash, generation-text hash, and primary metric. A scoring change therefore causes re-scoring, not model regeneration.
- Model download, construction, device transfer, warmup, trace copying, serialization, scoring, visualization, and optional compute replay are outside the measured generation window.
- TPS is `sum(accepted_tokens_per_step) / total timed generation seconds`. Missing accepted-token observations remain `N/A`; final text length is not used as a throughput substitute. Average power is also calculated from compatible window totals. Seconds/Sample and Energy/Sample divide total timed seconds or joules by completed samples; they are not means of per-sample ratios.
- Trace fields are capability-gated. Missing entropy, provisional-token, or verification information is reported as `N/A`, never inferred or replaced with zero.
- `run_visualization.py` is the only user-facing visualization entry point. Shared renderers live under `dllm_bench.visual.public`; optional model-specific hooks live in `dllm_bench.visual.<model_name>`.

## Configuration

### Model YAML

`configs/models/<name>.yaml` owns the adapter, checkpoint, and named variants:

```yaml
model: example_model
configs:
  default:
    adapter: dllm_bench.models.example.ExampleAdapter
    init_kwargs:
      model_name_or_path: org/checkpoint
      torch_dtype: bfloat16
    step_config:
      gen_length: 512
      block_length: 32
      steps_per_block: 32
```

`step_config` is optional and is used only by adapters that accept `DiffusionStepConfig`.

### Dataset YAML

`configs/datasets/<name>.yaml` owns the dataset class and default sample policy:

```yaml
dataset: example_dataset
dataset_class: dllm_bench.datasets.example.ExampleDataset
primary_metric: accuracy
aux_metrics: [complete_rate, answer_region_detected_rate]
sample_size: 100
seed: 42
dataset_kwargs: {}
```

`primary_metric` and `aux_metrics` document the result contract. The actual scorer remains Python code.

### Experiment matrix

`configs/experiments/full_matrix.yaml` selects formal models, variants, datasets, output ceilings, and per-model overrides. Change the matrix to change a run; do not edit runner code for experiment selection.

LLaDA2.1-mini is intentionally outside the main cross-model matrix. Standard
1-shot Sudoku4/9 runs use `configs/experiments/llada2_1_sudoku.yaml`; editable
controlled-repair diagnostics use `llada2_1_repair.yaml`. The adapter calls the
remote checkpoint's `generate()` unchanged, while observational wrappers record
the real canvas edits, confidence, and entropy used by the public trace plots.

## Adding a new model

The pipeline is model-agnostic after a model is registered. A new backend normally requires the following bounded changes.

### 1. Implement the adapter

Add `src/dllm_bench/models/<name>.py` and implement `ModelAdapter` directly or subclass `BaseModelAdapter`.

The adapter must return a standard `GenerationResult` containing:

- the original `GenerationRequest`
- output text and `RunStatus`
- measured timing/resource fields when locally observable
- step count and final valid length
- optional `TraceStep[]`
- backend-specific audit fields under `extra`

Do not include model loading, device transfer, warmup, trace serialization, scoring, or compute replay inside the measured generation window.

If several variants share one checkpoint, use `models/model_cache.py` so the model is loaded once per process.

### 2. Add the model configuration

Create `configs/models/<name>.yaml`. Put checkpoint and inference parameters in `init_kwargs`; put each formal sampler setting under a named `configs:` variant.

### 3. Register the isolated environment

Add a `ModelProfile` entry to `venv_scripts/_model_script.py` with the model config, optional dependency extras, Torch/Transformers pins, supported CUDA indexes, and required distributions.

Add the five-line dispatcher `venv_scripts/<name>.py`:

```python
#!/usr/bin/env python3
from _model_script import main

if __name__ == "__main__":
    raise SystemExit(main("<name>"))
```

This explicit environment registration is intentional: dependency changes for one model must not mutate another model's environment.

### 4. Declare visualization capability

Add `src/dllm_bench/visual/<name>.py`:

```python
from .base import public_model_visual

MODEL_VISUAL = public_model_visual("<name>")
main = MODEL_VISUAL.main
```

This declaration reuses every compatible implementation under `visual/public/`.
Only add a private renderer when the backend exposes model-specific information
such as its own entropy definition, confidence signal, or sampler diagnostics.
The public dispatcher discovers the module automatically; there is no central
model list to edit.

Never fabricate an unavailable trace field. Record its capability as `N/A`.

### 5. Add the model to an experiment matrix

```yaml
models:
  - config: configs/models/example_model.yaml
    variants: [default]
```

Keep diagnostic variants out of the formal matrix unless they are intended to be part of every normal run.

### 6. Test the integration

At minimum add tests for:

- adapter construction and configuration
- deterministic sampling decisions on fake logits or a tiny fixture
- GenerationResult and trace serialization
- OOM/error propagation
- environment/profile registration
- one mock or one-sample end-to-end generation/score/visualization path

Then run:

```bash
python setup_venv.py -m <name> --check
python prepare_model.py -m <name>
python run_model.py -m <name> -d gsm8k --n-samples 1 --output-root output/smoke
python run_score.py -m <name> -d gsm8k --n-samples 1 --output-root output/smoke
python run_visualization.py -m <name> -d gsm8k --n-samples 1 --output-root output/smoke
```

## Adding a new dataset

### 1. Implement the dataset

Subclass `Dataset` in `src/dllm_bench/datasets/<name>.py` and implement sample loading plus scoring. `ScoreResult.primary_score` must be normalized to `[0,1]`.

Use a task-specific final-answer extractor when reasoning, drafts, or formatting can surround the answer. Keep extraction diagnostics separate from the primary task score.

### 2. Add preparation support

Network-backed or synthetic datasets must produce a deterministic prepared artifact with source revision, checksum/signature, sample IDs, and seed.

### 3. Add dataset YAML and matrix entry

Create `configs/datasets/<name>.yaml`, then add it to the relevant experiment matrix with its task-specific output ceiling.

### 4. Test scoring boundaries

Add fixtures for:

- a correct answer
- a wrong answer
- malformed/empty output
- truncated but scoreable output
- answer marker and rejected-draft behavior
- aggregation and missing auxiliary fields

The registry tests should also construct every shipped YAML.

## Project layout

```text
setup_venv.py / prepare_data.py / prepare_model.py
run_bench.py / run_model.py / run_score.py / run_visualization.py
configs/
  models/
  datasets/
  experiments/
venv_scripts/
src/dllm_bench/
  interfaces.py
  registry.py
  models/
  datasets/
  resource/
  metrics/
  runner/
  visual/
    public/
    <model>.py
tests/
  core/
  data/
  models/
  metrics/
  runtime/
  visual/
  clean_test.py
```

## Testing

Tests are part of the repository and are grouped by responsibility:

- `core`: matrix execution, stages, persistence, output layout, and CLI contracts
- `data`: preparation, sampling, dataset I/O, and scoring
- `models`: model adapters, prompts, caches, and sampling behavior
- `metrics`: quality, resource, profiling, and trace metrics
- `runtime`: environment setup, model preparation, device selection, and entry points
- `visual`: shared and model-specific visualization contracts

Run the suite from an activated development/root environment:

```bash
python -m pytest -q --import-mode=importlib
```

Useful focused checks:

```bash
python -m pytest tests/core/test_registry.py
python -m pytest tests/core/test_runner_end_to_end.py
python -m pytest tests/visual/public/test_dataset_trace_report.py
```

The repository includes a pure-Python mock adapter so generation, scoring, persistence, visualization, OOM handling, and resume behavior can be tested without a GPU.

## Reproducibility checklist

Before treating a row as reportable, confirm:

- prepared sample IDs match `_meta.json.selected_sample_ids` exactly
- model/config/checkpoint and code commit are recorded
- prompt, output ceiling, request config, and seed match the intended protocol
- hardware and measurement protocol are present
- timing, energy, VRAM, output length, step count, and required trace policy are satisfied
- no `oom_info.json` invalidates the row
- score fingerprints match current generation and scorer revisions
- cross-model resource charts use the same hardware and sample-set hash

Score and visualization perform the definitive artifact compatibility checks; this list is for human review.
