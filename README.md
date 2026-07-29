# dLLM Benchmark

Benchmark harness for diffusion LLMs (iLLaDA, DreamReasoner, W1, DiffusionGemma) vs
AR references (Qwen3-4B, Qwen3-8B, and the same-scale Gemma 4 26B-A4B), implementing the design in
`dLLM_benchmark_设计文档.md`: task quality, long-context robustness,
resource cost, and generation-process analysis (trace, parallelism,
commit-order, certainty).

## Status

The framework (unified model interface, all Part 3/4 metric math, all six formal
dataset scorers, resource measurement plumbing, the 3-stage pipeline, and
report/visualization generation) is fully implemented and unit-tested (see
[Testing](#testing)). A pure-Python mock model backend
(`models/mock.py`) exercises the entire pipeline end-to-end without a GPU.

On top of that, every **local** model's real sampling loop is implemented —
not just interface stubs:

| Model | Status | Verified against |
| --- | --- | --- |
| iLLaDA | **Real, ported sampler** (`models/illada.py`) | Official `ML-GSAI/LLaDA` `generate.py` plus the iLLaDA checkpoint card — checkpoint id, `mask_id=5` override, block-wise low-confidence unmasking, FP64 gumbel-noise formula. Algorithm-level tests in `tests/test_illada_sampling.py` (fake-logits, no GPU) check the actual selection order, not just wiring. |
| iLLaDA VarGen | **Real, official variable-canvas path** (`models/illada_vargen.py`) | Uses the same iLLaDA checkpoint and Best/Fast schedules, but follows official `var_generate`: append and denoise only the active block, then append the next. Future blocks are absent from both the model forward and that step's trace. Tests in `tests/test_illada_vargen_sampling.py` hard-check the growing forward lengths. |
| DiffusionGemma | **Real** (`models/diffusiongemma.py`) | Verified against the upstream `DiffusionGemmaForBlockDiffusion`/`EntropyBoundSampler` implementation. Trace capture wraps `accept_canvas` through `_prepare_sampler`. Tests live in `tests/test_diffusiongemma_sampling.py`. |
| Gemma 4 26B-A4B AR | **Real** (`models/gemma4_ar.py`) | Official `AutoProcessor` + `AutoModelForMultimodalLM` path in native BF16. It matches DiffusionGemma's 25.2B-total/3.8B-active MoE scale and reuses the common AR generation/trace protocol. |
| Gemma 4 + DFlash | **Real, deployment track** (`models/gemma_dflash.py`) | Official `google/gemma-4-26B-A4B-it` target plus `z-lab/gemma-4-26B-A4B-it-DFlash` draft through the official temporary vLLM Gemma4 build. It keeps the common result/score/resource interface and records vLLM acceptance counters, but is not a native AR baseline and exposes no per-token trace through the public serving API. |
| DreamReasoner | **Real, ported sampler** (`models/dreamreasoner.py`) | GitHub's `DreamLM/DreamReasoner` repo (the design doc's own link) ships only a README + assets, no python — so verified instead by fetching the real `generation_utils.py`/`modeling_dream.py`/`config.json` from the `Dream-org/DreamReasoner-8B` HF repo directly. Confirmed from that source (not assumed): `block_diffusion_generate` returns only `sequences`/`nfe`, no per-step history at all (unlike regular Dream-7B's `output_history`), so this adapter ports the real `_denoise_current_block`/`_select_transfer_index` loop itself (like iLLaDA) rather than calling the model's own convenience method; default `block_length`=`config.block_size`=32; default per-block `denoising_steps`=`block_length` unless `remasking_strategy='low_confidence_static'`; the library's own default remasking strategy (`low_confidence_dynamic`) is already confidence-based, so — unlike regular Dream, where the library default was overridden — no override is applied; `mask_token_id`=`config.mask_token_id`=151669, shipped directly; the real default path uses a prefix KV cache (ported faithfully, since it directly affects real Time/Energy/Compute cost, not just trace fidelity); never revises (same structural reason as iLLaDA). This is a genuinely different, independently trained model from regular Dream-7B (`Dream-org/Dream-v0-Instruct-7B`) — not a config variant of it — and the design doc's own model roster (section 5) no longer includes regular Dream at all, so that adapter/config/tests were removed rather than kept alongside this one. Tests in `tests/test_dreamreasoner_sampling.py`. |

The 2026-07-29 architecture audit also checked every formal checkpoint's
published config and generation path. Qwen3-4B/8B now load explicitly in
checkpoint-native BF16 and reconstruct their one-token AR trace after the
timed `generate()` call; they no longer retain full-vocabulary `output_scores`
inside the latency window. DreamReasoner explicitly uses the official
thinking template and restores the official `torch.no_grad()` inference
boundary. iLLaDA/DreamReasoner now honor exact tokenizer-level RULER input
targets and retain denoising-forward counts even when HelloBench trace capture
is disabled. DiffusionGemma no longer forces `disable_compile`; it records the
upstream `tokens_per_forward` result and removes post-EOS pad positions from
its final length and trace. Gemma 4 AR intentionally keeps its checkpoint's
shipped sampling config. W1 remains reference-only until the private API and
trace schema exist to validate.

What's still open, and why:

| Piece | Status | Why |
| --- | --- | --- |
| W1 | API adapter + configuration (`models/w1_api.py`, `configs/models/w1.yaml`) | The transport/timing path is implemented; the private endpoint and its trace payload still need to be validated against the real service before Part 4 is enabled. |
| HelloBench semantic judge | Deliberately excluded | Official HelloEval requires checklist-based LLM judging. This project reports a clearly named, judge-free `objective_quality_score` plus observable major-failure rates; it never labels that score HelloEval. |
| Real dataset loading | GSM8K + local files | `--no-demo` downloads pinned/checksummed official GSM8K; every dataset also accepts local JSON/JSONL through `--samples-file`. Remaining official task-bank downloaders are external preparation. |
| Batch experiment runner | Implemented with isolated environments | `run_bench.py` reads the matrix and delegates each model to its own script/venv. |
| Running the real models end-to-end | Diagnostic runs imported | Qwen3-4B, iLLaDA, and DreamReasoner outputs from RTX 4090 runs are available locally. Coverage, OOM rows, protocol failures, and missing model pairs are audited in `docs/CURRENT_RESULTS.md`; these results are not yet publication-ready. |

None of these block the framework from being extended — each is isolated
behind the same `ModelAdapter`/`Dataset` interface, with a TODO at the exact
point that needs a real checkpoint/API/judge/GPU to finish.

## Layout

```
run_bench.py                   # compatibility: same-machine all-in-one pipeline
run_model.py                   # server: generate model_output only
run_score.py                   # local: score transferred model_output
run_visualization.py           # local: visualize + build reports
setup_venv.py / run_tests.py   # venv dispatcher / test runner (see below)
prepare_model.py               # pre-warm a model's HF checkpoint cache (see below)
prepare_data.py                # prepare/cache every real dataset in the matrix
run_prepare.py                 # one-shot: all venvs + datasets + model snapshots
venv_scripts/                  # one Python venv/install/run script per model
docs/                          # benchmark audit, task taxonomy, and artifact layout
configs/
  models/       # one YAML per model, one or more named `configs:` variants (Appendix D)
  datasets/     # one YAML per dataset (section 1/6): dataset class + sample size + seed
  experiments/  # executable model x dataset matrices
src/dllm_bench/
  interfaces.py     # GenerationRequest/Result, TraceStep, ModelAdapter protocol
  registry.py       # YAML -> instantiated ModelAdapter/Dataset
  hf_cache.py       # project-relative HF cache directory (see below)
  models/           # base.py (resource-measurement wrapper), model_cache.py (shared
                     # loaded-weights cache), hf_ar.py (Qwen3-4B/Qwen3-8B),
                     # hf_diffusion.py (iLLaDA/DreamReasoner shared base + DiffusionStepConfig),
                     # illada.py, dreamreasoner.py (each model's real sampler — see Status),
                     # diffusiongemma.py, gemma4_ar.py, w1_api.py, mock.py
  datasets/         # base.py + gsm8k/mbpp/structeval_t/sudoku4/sudoku9/ruler/hellobench
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
output/             # ignored canonical generate/score/visualize artifacts
artifacts/          # ignored transfer archives and superseded local analyses
```

See `docs/PROJECT_LAYOUT.md` for the ownership and retention rules for every
top-level directory. The complete imported-run tables, task diagnostics,
matched comparisons, and figures live in `docs/TECHNICAL_DATA_REPORT.md`;
`docs/CURRENT_RESULTS.md` remains the compact result ledger.

Rebuild the tracked technical report from the current local result snapshot:

```bash
python scripts/build_technical_report.py
```

## Main entry points

Use the stage-specific entry points for the normal server/local workflow.
With no `-m`, each command covers every formal model and variant declared in
`configs/experiments/full_matrix.yaml`:

```bash
# GPU server: generation only
python run_model.py
python run_model.py -m illada
python run_model.py -m illada_vargen
python run_model.py -m illada dreamreasoner -d ruler hellobench
python run_model.py -m qwen3_8b -d sudoku  # all sudoku* matrix variants

# Local machine after copying output/model_output/
python run_score.py
python run_visualization.py
python run_score.py -m qwen3_8b -d sudoku
python run_visualization.py -m qwen3_8b -d sudoku
```

Useful controls:

```bash
python run_model.py --list-models
python run_model.py --dry-run -m illada
python run_model.py -m illada --n-samples 20
python run_model.py -m illada -v fast
python run_model.py -m illada -d ruler hellobench ruler_context_probe
python run_score.py --dry-run -m illada
python run_score.py -m dreamreasoner -d ruler hellobench --no-resume  # force re-score; never regenerates model output
```

`run_bench.py` remains available for backward-compatible same-machine runs.

Runs are resumable by default. The built-in demo dataset is the default. A
real-data run checks the normalized cache first and automatically invokes the
same preparation logic as `prepare_data.py` only when its artifact is absent.
Any sample OOM invalidates the complete `model × variant × dataset` output
directory: generation stops before later samples in that directory, writes
`oom_info.json`, and resume does not retry it. Other variants remain independent
dataset rows and may still run. Local scoring recognizes the marker and refuses
to create a score or aggregate summary for that invalid directory. OOM during
model loading or warmup follows the same rule and records `failure_stage` with
zero attempted formal samples. Matrix execution prints the invalid-test error
and continues later datasets. Scoring, visualization, and reporting exclude
only the marked row; even a stale score summary from an earlier run cannot enter
aggregate quality, latency, TPS/SPS/EPS, energy, or ranking statistics.

## Environment setup

```bash
python setup_venv.py                   # every model declared in the matrix
python setup_venv.py -m qwen3_8b       # only .venvs/qwen3_8b
python setup_venv.py -m illada         # only .venvs/illada
python setup_venv.py -m illada_vargen  # only .venvs/illada_vargen
python setup_venv.py -m dreamreasoner  # only .venvs/dreamreasoner
python setup_venv.py -m diffusiongemma # only .venvs/diffusiongemma
python setup_venv.py -m gemma          # only .venvs/gemma (reuses the legacy env if present)
```

`setup_venv.py` is only a dispatcher. It calls
`venv_scripts/<model>.py setup`. No model packages are installed
into the Python running `setup_venv.py`. Model environments are grouped under
the single `.venvs/` directory; the root `.venv/` remains the development and
test environment.

### Model scripts

Every model has one public script with the same four actions:

| Action | Purpose |
| --- | --- |
| `setup` | Create or update the model-specific virtualenv |
| `check` | Validate dependencies and construct every configured adapter |
| `prepare` | Download the checkpoint without loading it or generating samples |
| `run` | Internal compatibility action used by the top-level dispatcher |

Available entry points and environments:

| Model | Script | Environment |
| --- | --- | --- |
| Qwen3-4B AR | `venv_scripts/qwen3_4b.py` | `.venvs/qwen3_4b` |
| Qwen3-8B AR | `venv_scripts/qwen3_8b.py` | `.venvs/qwen3_8b` |
| iLLaDA | `venv_scripts/illada.py` | `.venvs/illada` |
| iLLaDA VarGen | `venv_scripts/illada_vargen.py` | `.venvs/illada_vargen` |
| DreamReasoner | `venv_scripts/dreamreasoner.py` | `.venvs/dreamreasoner` |
| DiffusionGemma | `venv_scripts/diffusiongemma.py` | `.venvs/diffusiongemma` |
| Gemma 4 26B-A4B AR | `venv_scripts/gemma.py` | `.venvs/gemma` |
| Gemma 4 + DFlash | `venv_scripts/gemma_dflash.py` | `.venvs/gemma_dflash` |
| W1 | `venv_scripts/w1.py` | `.venvs/w1` |
| Local non-model stages | `venv_scripts/root.py` | `.venvs/root` |

Example lifecycle:

```bash
python venv_scripts/qwen3_4b.py setup
python venv_scripts/qwen3_4b.py check
python venv_scripts/qwen3_4b.py prepare
python run_model.py -m qwen3_4b
```

Qwen3-8B is the dense same-scale AR reference for iLLaDA-8B and
DreamReasoner-8B. It uses the same greedy, non-thinking protocol as Qwen3-4B:

```bash
python venv_scripts/qwen3_8b.py setup
python venv_scripts/qwen3_8b.py check
python venv_scripts/qwen3_8b.py prepare
python run_model.py -m qwen3_8b -d gsm8k --n-samples 1
```

### A100 Gemma 4 comparison

Use the same A100 80GB in native BF16 for the Gemma 4 AR vs DiffusionGemma
comparison. A100 40GB cannot hold either roughly 50GB BF16 checkpoint entirely
on GPU. If Hugging Face requests model access, accept the repository terms and
export a read-capable `HF_TOKEN` first.

```bash
cd /workspace/dllm
export HF_TOKEN=hf_your_read_token

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python setup_venv.py -m gemma -m diffusiongemma --cuda-index cu124
python prepare_model.py -m gemma -m diffusiongemma

# One-sample generation checks, run sequentially on the same GPU.
.venvs/gemma/bin/python -m dllm_bench.cli generate \
  --model-config configs/models/gemma.yaml \
  --variant ar-baseline \
  --dataset-config configs/datasets/gsm8k.yaml \
  --demo --n-samples 1 --max-new-tokens 64 \
  --output-root output/a100_pair_check --no-resume

.venvs/diffusiongemma/bin/python -m dllm_bench.cli generate \
  --model-config configs/models/diffusiongemma.yaml \
  --variant official \
  --dataset-config configs/datasets/gsm8k.yaml \
  --demo --n-samples 1 --max-new-tokens 64 \
  --output-root output/a100_pair_check --no-resume
```

The deployment-optimized DFlash row is a normal selectable model in
`full_matrix.yaml`; `dg_comparison.yaml` is only a convenient three-way
subset. Its isolated setup installs the official Gemma4 DFlash vLLM PR build,
and preparation downloads both the Gemma target and 0.4B DFlash draft without
loading either checkpoint:

```bash
python run_prepare.py \
  -m gemma_dflash --skip-data

python run_model.py \
  -m gemma_dflash \
  --output-root output/formal --resume

# Local scoring uses the normal root environment; it never starts vLLM.
python run_score.py \
  -m gemma_dflash --output-root output/formal --resume
```

The setup scripts keep pip and uv caches, temporary build files, and Torch
extension builds under `DLLM_DATA_ROOT` (repository `data/` by default), so a
large vLLM wheel is not staged on RunPod's small `/tmp` filesystem. Override
the build location with `DLLM_BUILD_TMPDIR` when needed. DFlash setup uses the
compatible precompiled vLLM wheel and disables its persistent uv cache by
default, avoiding a second expanded copy of Torch on quota-limited volumes.
Set `VLLM_USE_PRECOMPILED=0` only when intentionally compiling vLLM from
source.

Run on the same exclusive A100 80GB used by the native pair. DFlash keeps the
normal measured timing, energy, peak-memory, TPS/SPS/EPS and dataset score
fields. Peak memory is sampled as total NVML device-used memory so it includes
both target and draft server processes; acceptance rate, mean acceptance
length, target verification passes, TTFT and TPOT are persisted in each
sample's `extra` object. Trace analysis is unavailable for this deployment row
because vLLM exposes aggregate acceptance counters, not its internal per-token
verification history. Set `DLLM_DFLASH_SERVER_URL` only when intentionally
using an already-running compatible server; otherwise the adapter manages the
local server itself.

The separate compact Sudoku trace case study keeps the same frozen 5 Easy +
5 Hard source rows and semantic scorer as `sudoku9`. Unlike the main reasoning
protocol, it forbids explanation and requests only the final 81 digits, writing
to its own `sudoku_trace` output/cache namespace. It is included in
`dg_comparison.yaml`, not the main matrix. After the pair smoke test,
validate three real puzzles before spending the full 10-sample budget:

```bash
python run_model.py \
  --matrix configs/experiments/dg_comparison.yaml \
  -m diffusiongemma gemma \
  -d sudoku_trace --real-data --n-samples 3 \
  --output-root output/a100_sudoku_trace_check --no-resume

python run_model.py \
  --matrix configs/experiments/dg_comparison.yaml \
  -m diffusiongemma gemma \
  -d sudoku_trace --real-data \
  --output-root output --resume
```

The run automatically materializes the new normalized JSONL from the already
cached Sudoku archive; no explicit `prepare_data.py` rerun or source download
is needed. Score and visualize the transferred outputs locally:

```bash
python run_score.py --matrix configs/experiments/dg_comparison.yaml \
  -m diffusiongemma gemma -d sudoku_trace \
  --output-root output --resume
python run_visualization.py --matrix configs/experiments/dg_comparison.yaml \
  -m diffusiongemma gemma -d sudoku_trace \
  --output-root output --n-representative 3
```

For DiffusionGemma, Sudoku revision analysis decodes complete 81-digit board
candidates from each denoising canvas rather than assuming one tokenizer token
equals one cell. Per-sample scores include blank-cell revision counts,
wrong-to-correct outcomes, and the fraction of trace steps with a parseable
board; visualization emits the generic token trace plus a decoded 9x9 GIF.

For formal comparisons, keep GPU type, precision, dataset sample set, output
caps, trace policy, and compute-profiling flag identical. Formal RULER uses a
shared 4096-token encoded input plus a 64-token answer allowance. A separate
one-sample diagnostic probes half of each model's declared context and is not
mixed into primary quality or resource aggregates. The complete execution
contract and acceptance checklist are in `docs/TEST_PROTOCOL.md`.
`run_model.py` creates the environment automatically when it is missing, then
launches the model's generation rows with that venv's Python executable. It does
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

Sudoku is split into two non-interchangeable tracks. `sudoku4` follows d1's
official zero-shot 4x4 prompt, `<reasoning>/<answer>` wrapper, test CSV, and
blank-cell accuracy. Preparation freezes 100 seeded rows from the official
500-row test file; every source puzzle has exactly eight blanks. The shared
main output cap is 256 tokens, the larger of d1's two reported settings, so
reasoning-oriented checkpoints have room to reach the answer tag. Whole-puzzle validity, reference exact match,
clue preservation, and answer-format compliance are reported alongside d1's
partial-credit primary score. There is no Easy/Hard split for 4x4.

`sudoku9` retains the Park split used by Ye et al. (ICLR 2025), the adapted
general-instruction prompt, and strict whole-puzzle constraint validity as its
primary score. Its output cap is 256 tokens. The prepared bank contains 50
Easy + 50 Hard rows; a 10-row override deterministically selects 5 + 5. The
Qwen3-4B, iLLaDA, iLLaDA VarGen, and DreamReasoner run one fixed row from each
direct Sudoku dataset because pilot runs show that they still expand into
reasoning instead of following the compact-answer request. Qwen3-8B and W1
run `sudoku4` on 100 rows and `sudoku9` on the 10-row probe.
DiffusionGemma and its Gemma-4 control reverse those budgets: `sudoku9` on
100 rows and `sudoku4` on 10 unstratified rows. Scores from 4x4 and 9x9 are
never merged or placed in the same Sudoku column. RULER selects 30
samples at a 4096 encoded-input target: 10 each for NIAH, multi-hop,
and aggregation, also balanced over front/middle/back answer positions. A
64-token answer allowance is added after the input target.

HelloBench is a focused long-output diagnostic rather than a full leaderboard
run. The default is one shared sample at each of 2K words
(`max_new_tokens=3072`) and 4K words (`max_new_tokens=6144`). In the formal
matrix, iLLaDA and DreamReasoner run only the 2K profile, one sample per
variant. These generation caps are attached
per sample by the runner, so the matrix-wide fallback cannot accidentally
reduce both groups to 256 tokens. Every model uses the same deterministic
configured shared subset. Per-sample wall-clock, output length, TPS, energy, peak VRAM,
objective quality, and major-failure flags describe long-output feasibility
and cost; the subset is not presented as a full HelloBench leaderboard score.
Its primary `objective_quality_score` is explicitly not official HelloEval:
it combines target-length fidelity, Seq-Rep-4, and repeated-segment quality,
then applies transparent penalties for empty/severely short or long output,
high repetition, exact segment loops, refusal, long prompt echo, and corrupt
control/replacement characters. The individual issue flags and issue-free
rate are preserved as auxiliary metrics. Semantic correctness, factuality,
coherence, style, and checklist satisfaction are not claimed without a judge.

Model weights, official/generated datasets, and package wheels are all kept
under the repository-root `data/` directory (`huggingface/`, `datasets/`,
and `pip-cache/`). This makes the cache follow the project onto a mounted
cloud volume. Set `DLLM_DATA_ROOT=/mounted/path/data` to override it. W1
additionally requires `W1_API_BASE_URL` and, when applicable, `W1_API_KEY` at
run time; data preparation does not need either. Reference-only W1 uses the
same formal 4096-token RULER input and its separate half-context probe also
targets 4096 input tokens because W1 declares an 8192-token maximum.

## Data preparation

Prepare every real dataset declared in the matrix before allocating a GPU:

```bash
python prepare_data.py
python prepare_data.py -d sudoku4
python prepare_data.py -d sudoku9 ruler
python prepare_data.py -d sudoku --force  # every sudoku* variant in the matrix
python prepare_data.py --force  # rebuild matching prepared artifacts
```

After a dataset is prepared successfully, only its active fingerprint directory
is retained under `data/datasets/prepared/<dataset>/`; older prepared versions
are removed automatically. Raw downloads and model caches are not removed.

The first command automatically creates the lightweight `.venvs/root`
environment when missing, installs only the base project dependencies there,
and restarts itself inside that environment. It does not modify the system
Python or install model/Torch dependencies. Model preparation similarly
dispatches through `.venvs/<model>` rather than the caller's Python.

Or prepare one dataset/source explicitly:

```bash
dllm-bench prepare-data \
  --dataset-config configs/datasets/gsm8k.yaml

dllm-bench prepare-data \
  --dataset-config configs/datasets/mbpp.yaml \
  --samples-file /mounted/raw/mbpp-sanitized.jsonl
```

Prepared samples land at
`data/datasets/prepared/<dataset>/<fingerprint>/samples.jsonl`, accompanied
by a manifest. The fingerprint covers the dataset YAML, loader implementation,
and raw source contents, so rerunning is idempotent while source/config changes
create a distinct cache artifact.

`python run_model.py` uses exactly the same function: cache hit
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

The atomic unit of testing is the **model**, not model+variant. iLLaDA and
DreamReasoner each have one official-algorithm traced implementation frozen to
commit `6dfd132`. Within one group, weights load once and Best/Fast only change the
generation-time sampling config (`models/model_cache.py`). So leaving out
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

Recommended cross-machine workflow:

```bash
# GPU server: system Python only dispatches into managed venvs
python prepare_data.py
python prepare_model.py
python run_model.py --output-root output

# Copy output/model_output/ to the local machine, then:
python prepare_data.py
python run_score.py --output-root output
python run_visualization.py --output-root output
```

`run_model.py` always runs generation only and uses `.venvs/<model>`.
`run_score.py` and `run_visualization.py` always use `.venvs/root`; they never
instantiate model adapters or load weights. `run_bench.py` remains as a
backward-compatible same-machine all-stage entry point, but is not the
recommended server/local workflow.

Before a model run, the dispatcher verifies the model venv's pinned Torch and
Transformers versions. If a pin changed, it updates only the stale runtime
package in the existing venv. In particular, DreamReasoner uses
`transformers==5.7.0`, matching the checkpoint metadata and remote model code;
the old 4.46.2 pin lacks `PretrainedConfig.validate_rope()` and cannot load the
checkpoint.

Formal `run_model.py` runs record timing, NVML energy, peak PyTorch VRAM, and
trace by default. FLOP/compute profiling is not part of the formal full-matrix
run because it replays generation and substantially increases GPU time; enable
it explicitly with `--measure-compute` only for a separate diagnostic run.
When enabled, all formal generations for a dataset finish first and compute
replays run afterward, outside the stored generation wall-clock/energy window.
Formal generation is persisted before replay, so a failed profiler can resume
and fill only missing compute without rerunning the model output. Formal runs
also enable `--require-all-metrics`; it requires compute only when
`--measure-compute` was explicitly selected. Use `--allow-missing-metrics`
only for an explicitly non-formal smoke run.

Compute can also be supplemented later in the same output tree. Run the same
model, matrix, sample selection, and `--output-root` with `--measure-compute`;
resume skips every persisted generation and profiles only successful sample
JSON files whose `compute_tflops` is still missing. Existing compute values are
left untouched.
Before the first measured sample
of each model/config/dataset, a short untimed warmup initializes kernels
(8 tokens normally; one valid 32-token block for `illada_vargen`);
tokenization, model loading, warmup, progress output, and persistence stay
outside the measured wall-clock window. Iterative adapters also pause the
aligned time/energy/VRAM measurement around trace-only entropy calculation,
tensor copies, token decoding, and trace construction, then accumulate the
remaining generation segments. Complete traces therefore still come from the
same generation, but their instrumentation cost only increases end-to-end job
duration, not the reported TPS/SPS/EPS windows. Energy defaults to the physical GPU
mapped to CUDA logical device 0; set `DLLM_NVML_GPU_INDICES=0,1` explicitly
for a future multi-GPU model.

iLLaDA and DreamReasoner are loaded in checkpoint-native BF16 precision. The
dtype is applied inside `from_pretrained` before the model moves to CUDA, so
the 8B-class checkpoints do not transiently become default-precision models
that exhaust a 24 GiB device. `inference_dtype` is persisted in each run's
`_meta.json` for reproducibility.

The shared runner does not wrap these adapters in a later-added inference or
OOM workaround. Each sampler therefore keeps the execution semantics of its
declared implementation. Enabled upstream execution features are persisted in
`_meta.json` and each sample JSON under `inference_optimizations`.

iLLaDA retains commit `6dfd132`'s fixed-canvas implementation of the official
sampler. Trace snapshots are copied after each forward and do not participate
in token selection.

`illada_vargen` is the parallel official `var_generate` execution-path
ablation. It uses the same BF16 checkpoint, prompt formatting, `mask_id=5`,
block length, Best/Fast transfer schedules, temperature and remasking rule as
`illada`; only canvas allocation changes. During a block's denoising forwards,
later blocks have not been appended, so they cannot contribute future mask
embeddings. Its trace grows one block at a time and records only positions that
exist at that forward. It has an independent `.venvs/illada_vargen` process
and can be prepared, generated, scored and visualized with `-m illada_vargen`.

DreamReasoner retains the checkpoint's prefix-KV-cache generation path from
[`generation_utils.py`](https://huggingface.co/Dream-org/DreamReasoner-8B/blob/main/generation_utils.py).
The default `dreamreasoner` adapter is frozen to commit `6dfd132`: it builds the complete block-triangular mask,
prefills all complete prompt blocks in one forward, passes the corresponding
mask slice during block denoising, and uses the official full-softmax
confidence calculation. The official loop is executed in the adapter solely
to expose trace: token/state snapshots are copied after each forward and
outside the measurement window, without changing logits, masks, cache state,
transfer decisions, or final output. The checkpoint convenience method exposes
sequences and NFE but no history; wrapping it separately would only remove
trace without introducing a distinct generation strategy, so it is not a
separate benchmark model.

Each model group shares one loaded checkpoint between its Best/Fast variants.

Optional compute profiling keeps the model's configured attention backend. For SDPA,
the profiler supplies a GQA-aware FLOP formula because PyTorch 2.6's built-in
counter assumes equal Q/K/V head counts and asserts on Qwen3's grouped-query
attention. This changes only FLOP accounting during the replay; it does not
replace SDPA with eager attention or alter formal generation.

Output run IDs append a variant only when it distinguishes configurations:
Qwen writes under `model_output/qwen3_4b/` and `model_output/qwen3_8b/`, while
multi-configuration models use names such as `illada_best` and `illada_fast`.
Local readers still accept the legacy `qwen3_4b_ar-baseline` directory.

Pass `--variant best` to a low-level model-config command to run one sampling
profile.
`score`/`visualize` deterministically reconstruct the same
sample list from `--demo`/`--no-demo`, `--n-samples`, and `--seed`; pass matching
values to every stage. The official GSM8K loader uses stable source indices as
sample IDs and a pinned source revision. It prepends the first four fixed
`gsm8k_cot` demonstrations from lm-evaluation-harness and applies the paper's
flexible final-number extraction.

Formal RULER records must provide `task_type`, `position`, `required_answers`,
and `context_length` in `reference`. The formal target is 4096 encoded prompt
tokens with a separate 64-token output allowance; `meta.context_window_tokens`
and `meta.input_tokens` state the total allowance and input target explicitly.
Formal HelloBench records provide `reference.target_length_words` as either
2000 or 4000. Dataset-aware sampling is deterministic under `--seed`.

The formal evaluation plan is diagnostic rather than a full-leaderboard run:

| Part | Samples per model configuration |
|---|---:|
| GSM8K | 100 |
| MBPP-Sanitized | 100 |
| StructEval-T | 100 |
| Sudoku4 direct | Qwen3-4B/iLLaDA/iLLaDA VarGen/DreamReasoner: 1 reference; Qwen3-8B/W1: 100; DG/Gemma-4: 10; no Easy/Hard split |
| Sudoku9 direct | Qwen3-4B/iLLaDA/iLLaDA VarGen/DreamReasoner: 1 reference; Qwen3-8B/W1: 10 (5 + 5); DG/Gemma-4: 100 (50 + 50) |
| Sudoku4/9 thinking | One fixed sample per model/variant and dataset, 2048-token cap; reference only |
| RULER | 30 at 4096 encoded input tokens (10 per task type) |
| HelloBench | iLLaDA/DreamReasoner: one 2K-word sample; others: one per configured profile |
| RULER half-context probe | one isolated capacity sample, excluded from formal aggregates |

Sudoku4's primary score is d1 blank-cell accuracy. Its strict
`puzzle_success_rate` requires a complete legal 4x4 board preserving every
clue. Sudoku9 uses the inverse emphasis: strict whole-puzzle constraint
validity is primary and blank-cell accuracy is diagnostic. Only Sudoku9 has
the Easy/Hard reporting split.

When a 256-token output ends in reasoning without a complete answer marker,
do not interpret that row alone as pure instruction-following failure. Every
model can run a temporary length probe through the normal entry point, for
example `python run_model.py -m qwen3_4b -d sudoku9 --real-data --n-samples 1
--max-new-tokens 512 --output-root output/sudoku_length_probe/qwen4b_512
--no-resume`. If the answer appears only there, report output-budget/verbosity
pressure; if it still does not appear, report persistent answer non-completion.
Temporary overrides are diagnostic and are never pooled with formal scores or
resource measurements. Use a fresh output root (recommended) or `--no-resume`;
resume rejects an existing sample generated with a different output cap.

For HelloBench, repeat `--hellobench-length` to select `2k`, `4k`, or both.
`--n-samples` is the total across the selected output profiles: selecting only
`4k --n-samples 3` runs three 4K samples, while selecting both with
`--n-samples 6` deterministically balances the run as three 2K plus three 4K.
With neither option, model-specific matrix defaults apply.

MBPP uses the official fixed 3-shot examples (task IDs 2, 3, and 4), includes
the public tests in the prompt, delimits code with `[BEGIN]` / `[DONE]`, and
reports pass@1: one candidate passes only when all official tests pass. Its
structure/content progress values are trace-only auxiliary diagnostics.
StructEval-T appends the upstream CLI's `<|BEGIN_CODE|>` / `<|END_CODE|>`
instruction and uses the official non-renderable formula as its primary metric,
`round(0.2 * strict_parse_success + 0.8 * required_path_coverage, 2)`; the
fault-tolerant formation score and strict all-fields-complete 0/1 result are
retained only as auxiliary diagnostics.

Formal RULER contains 30 samples at a shared 4096-token encoded-input target: 10 at
each of front/middle/back, balanced so that NIAH, multi-hop, and aggregation
also have 10 samples each. A separate `ruler_context_probe` sample uses half
of each model's declared maximum as input and runs last; it is a capacity/OOM
diagnostic, not an additional formal quality point. RULER allows 64 tokens for
the short answer. The
prompt ends with the official-style `Answer:` prefix. Its primary score follows
NVIDIA RULER's `string_match_all`: each required reference found in the output
receives equal fractional credit; `all_answers_match` is retained separately.
prepared filler is fitted again after the selected model's chat template and
tokenizer are applied, so the actual encoded input does not exceed that
target. Local HF model records include the observed count in
`extra.input_tokens`; W1 remains dependent on the external API's tokenizer.

HelloBench is the separate long-output axis. Its short-prompt 2K- and 4K-word
samples carry per-sample generation caps of 3072 and 6144 tokens respectively,
so they do not inherit the matrix-wide 256-token fallback.

Resource measurements reuse the formal task samples. Generation history is
captured and persisted for every sample whenever the model adapter exposes it,
except HelloBench: its 2K/4K long-output runs set `trace_scope: none` because
Task 2 uses final quality and resource metrics, while a full per-forward,
per-position trace would add very large observation and storage overhead.
`--n-representative` is applied only by the visualization stage to choose
which persisted traces receive per-sample plots; it never limits generation,
trace persistence, or dataset-level trace aggregation. Process/strategy
analysis can therefore use the complete run without scheduling a
separate 20–30 sample subset. W1 remains the exception until its API exposes a
validated per-step trace payload.

The aggregate speed summary records both TPS (tokens/s) and SPS (samples/s).
SPS is computed as `completed timed samples / total measured generation time`,
equivalently `1 / mean seconds per sample`; it is a ratio of totals, not the
mean of per-sample inverse latencies. This also exposes useful per-sample
totals without another run: `TPS / SPS` is tokens/sample, `EPS / SPS` is
joules/sample, and `1 / SPS` is seconds/sample. If optional compute profiling
was run, `CPS / SPS` additionally gives TFLOPs/sample.

Output lands under `output/` (override with `--output-root`), split by
stage, then by `<model>_<config>`, then by dataset — so `iLLaDA-best` and
`iLLaDA-fast` never collide, and you can `rsync`/copy just `model_output/`
off a GPU box:

```
output/
  model_output/<model>_<config>/<dataset>/
    _meta.json          # model/config/dataset name + run metadata (section 6)
    oom_info.json       # only when OOM marks the complete test invalid
    <sample_id>.json    # full GenerationResult, including trace
  score_output/<model>_<config>/<dataset>/
    <sample_id>.json    # ScoreResult for that sample
    summary.json        # RunSummary — section 3.4's raw-results-table row
  visualization_output/<model>_<config>/<dataset>/
    <sample_id>_*.png / .gif
```

To run either real Qwen AR baseline once `torch`/`transformers` are installed,
point at its config:

```bash
dllm-bench generate --model-config configs/models/qwen3_4b.yaml \
                     --dataset-config configs/datasets/gsm8k.yaml \
                     --no-demo --seed 42 --n-samples 5 --max-new-tokens 512

dllm-bench generate --model-config configs/models/qwen3_8b.yaml \
                     --dataset-config configs/datasets/gsm8k.yaml \
                     --demo --seed 42 --n-samples 1 --max-new-tokens 64
```

iLLaDA and DiffusionGemma work the same way (`configs/models/illada.yaml`,
`configs/models/diffusiongemma.yaml`). Their isolated scripts install the
appropriate transformer versions. Imported iLLaDA runs have been scored;
DiffusionGemma and its same-scale Gemma AR reference are still missing from the
current result snapshot — see `docs/CURRENT_RESULTS.md`.

## Model checkpoints and caching

Every model here loads via `from_pretrained(repo_id)` — `configs/models/
*.yaml`'s `model_name_or_path` is an HF Hub repo id (e.g. `Qwen/Qwen3-8B`),
never a local checkpoint path. `from_pretrained` downloads it on first use,
same as any other HF model.

By default, that download would land under `~/.cache/huggingface`. On a
cloud GPU box the large/persistent storage is usually a network volume
mounted at the project directory, while the home directory sits on small
local/ephemeral disk — so `hf_cache.py` points `HF_HOME` at the repository's
`data/huggingface` directory regardless of the launch working directory.
An explicit `HF_HOME`/`HF_HUB_CACHE`/`TRANSFORMERS_CACHE` still wins.
`cli.py` and `prepare_model.py` apply this before any model is touched.

To download a model snapshot ahead of time instead of lazily mid-benchmark:

```bash
# all models in the matrix, each through its own isolated environment
python prepare_model.py

# selected matrix models
python prepare_model.py -m illada -m qwen3_8b

# direct single-config mode inside the current compatible environment
python prepare_model.py --model-config configs/models/illada.yaml
# warms every variant declared in the file — for illada.yaml that's
# `best`+`fast`, but since they share one checkpoint (models/model_cache.py)
# this downloads the shared repository snapshot only once.
```

Environment mutation is intentionally outside this checkpoint-preparation
stage. `setup_venv.py` owns dependency/project installation;
`prepare_model.py` reads the checkpoint IDs from YAML and uses Hub snapshot
download only: it does not construct an adapter, import model code, load
weights into RAM, or touch the GPU. A compatibility repair
for a legacy incomplete editable install, if needed, is deferred until
`run_model.py` starts that model and remains outside the measured generation
window.

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

For StructEval-T and MBPP, framework features and substantive-content
features are classified separately at each checkpoint. `strategy_score.py`
uses their first-formation increments in a Kendall-like pairwise ordering:
structure earlier = 1, tie = 0.5, content earlier = 0. The resulting
`structure_first_score` is in `[0,1]`; 1 means a strong framework-first
generation preference, while 0.5 means synchronized or order-balanced
formation. It is a trace diagnostic only and never replaces official
StructEval `final_eval_score` or MBPP `pass_at_1`.

**Sudoku9** gets one more artifact on top: an animated 9x9 grid walking
through the solve (`report/sudoku_trace_viz.py`). Easy/Hard here means the
analysis-only naked-single-round stratum; it is not an official source label
and does not assert that a puzzle requires backtracking. Per-cell coloring:

- gray — not yet decided
- black text, white background — a given (prompt-supplied) cell, echoed
  correctly
- yellow — a given cell the model mis-transcribed
- green — a to-fill cell, filled correctly
- red — a to-fill cell, filled incorrectly (visible mid-solve for puzzles
  that need to revise a wrong trial before landing on the right digit)

This requires the trace's canvas to be exactly 81 positions, row-major
(`derive_sudoku_frames`). `simulate_sudoku_frames` remains a self-contained
demo/test fixture (the same role `models/mock.py` plays for the rest of the
framework); real Sudoku9 runs use decoded model trace canvases when available.

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
