# dLLM Arena: First-Draft Report Plan

Working title: **dLLM Arena: Compute-Normalized Evaluation of Diffusion
Language Models**

Target for the first draft: benchmark proposal plus an explicitly labeled
pilot study. Do not present the current imported runs as a final leaderboard.

## Draft Abstract

Diffusion language models expose inference controls that autoregressive models
do not, including denoising steps, block size, and parallel unmasking rate.
Published results are therefore difficult to compare: models are evaluated on
different tasks, with different output budgets, decoding schedules, hardware,
and cost accounting. We introduce dLLM Arena, a reproducible evaluation
protocol that treats a submission as a model-decoder configuration and reports
the Pareto frontier of task quality, wall-clock latency, energy, total compute,
and memory. The benchmark separates model-native, equal-compute, equal-latency,
and quality-matched tracks; uses strict task-specific output contracts; and
includes reasoning, code, structured generation, long context, long output,
and bidirectional editing. A pilot evaluation of Qwen3-4B, iLLaDA-8B, and
DreamReasoner-8B demonstrates why output protocol and inference budget must be
audited before architectural conclusions are drawn. A same-scale
DiffusionGemma/Gemma AR comparison is the next controlled experiment.

## 1. Research Question

The paper should ask:

> At a fixed end-to-end inference budget, where do diffusion language models
> lie on the quality-latency-compute frontier relative to autoregressive
> models, and which task properties make parallel refinement useful?

This is stronger and more defensible than asking whether dLLMs are simply
faster or better. The benchmark unit is `(checkpoint, sampler, inference
configuration)`, because steps, block size, and unmasking strategy can change
both quality and cost dramatically.

## 2. Why Another Benchmark

[Bertolani et al., *Diffusion Language Models: An Experimental Analysis*
(2026)](https://arxiv.org/abs/2606.19475) already provides an important unified
study of eight DLMs on eight benchmarks. It shows that:

- generation behavior depends strongly on steps, context length, block size,
  and unmasking ratio;
- aggressive global or intra-block parallel unmasking hurts reasoning and code
  quality;
- pure full-sequence diffusion can have very high cumulative generation FLOPs;
- block diffusion improves practical cost but retains a quality-compute tradeoff.

dLLM Arena must go beyond reproducing those tables. Its intended additions are:

1. an executable submission and artifact standard rather than a one-off study;
2. fixed-budget tracks and complete end-to-end resource accounting;
3. structured-output validity, long-output degradation, and long-context OOM;
4. editing and infilling tasks that directly test bidirectional conditioning;
5. hidden test sets and an ongoing public leaderboard;
6. optional anonymous pairwise evaluation for genuinely open-ended outputs.

The last item follows the useful institutional idea behind
[Chatbot Arena](https://arxiv.org/abs/2403.04132): continuously collect
standardized comparisons instead of treating one static table as permanent.
Objective tasks should still use deterministic graders; pairwise preference is
appropriate only where no reliable objective reference exists.

## 3. Evaluation Tracks

Do not collapse every submission into one unrestricted score. Publish these
tracks separately:

### Track 0: Canonical Reproduction

Before introducing a new Arena protocol, reproduce each checkpoint's published
or lm-evaluation-harness baseline with the same few-shot setting, prompt,
filtering, output allowance, and sampler. This validates the adapter and model
environment. It is a prerequisite, not the final fair-comparison track.

### Track A: Model-Native

Use each checkpoint's documented default sampler. This answers how a user sees
the released model, but it is not a controlled architectural comparison.

### Track B: Equal End-to-End Compute

Match total generation FLOPs per sample, including every denoising forward,
prefill, and replay-free decoding operation. Parameters are metadata, not the
budget. This is the primary inference-efficiency track.

### Track C: Equal Wall Time

Run on the same physical GPU with identical process isolation and compare
quality under fixed latency tiers, for example 1 s, 5 s, and 20 s per sample.

### Track D: Quality-Matched Cost

For a target quality threshold, report the minimum wall time, joules, and FLOPs
needed by each model configuration. This often communicates deployment value
more clearly than raw TPS.

### Track E: Controlled Training Compute

Use only models whose training tokens, data, precision, and total training
FLOPs are known or experimentally controlled. A training-efficiency ratio is
not valid for heterogeneous public checkpoints with undisclosed adaptation
costs.

## 4. Metrics

Report task quality and cost separately first. The primary result is a Pareto
frontier, not a single scalar.

### Required Per-Sample Measurements

- task score and strict format validity;
- answer-started and budget-exhausted flags;
- end-to-end wall time, excluding untimed warmup but including real decoding;
- total joules and peak VRAM;
- total forward passes and accepted tokens per forward;
- total generation FLOPs/TFLOPs when profiling is enabled;
- checkpoint, precision, GPU, software versions, prompt version, and sampler
  configuration.

### Derived Comparisons

For task `t`, compare each dLLM configuration against a designated AR control:

```text
quality_ratio_t = quality_dllm_t / quality_ar_t
compute_ratio_t = total_flops_dllm_t / total_flops_ar_t
energy_ratio_t  = joules_dllm_t / joules_ar_t
latency_ratio_t = wall_time_dllm_t / wall_time_ar_t

compute_efficiency_t = quality_ratio_t / compute_ratio_t
energy_efficiency_t  = quality_ratio_t / energy_ratio_t
```

Do not average raw GSM8K accuracy, MBPP pass@1, and HelloBench objective scores.
Normalize within each task first, report bootstrap confidence intervals, and
show task-level values beside any macro aggregate. Ratio metrics become unstable
when the AR denominator is near zero, so those rows should remain task-level or
use a predeclared baseline-adjusted normalization.

TPS is secondary: tokenizers differ, and diffusion canvases may contain fixed
or discarded positions. Wall time per successfully completed sample, total
compute, and energy are the cross-paradigm measurements that matter most.

## 5. Task Taxonomy

| Axis | Current task | Status | Needed addition |
|---|---|---|---|
| Arithmetic reasoning | GSM8K | Keep | Add harder reasoning if budget permits |
| Program synthesis | MBPP | Keep | Add HumanEval or equivalent hidden tests |
| Structured generation | StructEval-T | Keep after protocol repair | Add JSON-schema/CFG controlled tasks |
| Constraint reasoning | Sudoku | Reclassify; parser repair required | Do not call it structural generation |
| Long context | RULER | Keep | Standardize OOM/capacity reporting |
| Long output | HelloBench | Keep objective metrics | Add blinded preference/judge track |
| Bidirectional capability | None | Missing | Add infilling, editing, and constraint repair |

The structure-only controlled set should vary nesting depth, schema width,
cross-field constraints, and output length while holding semantic difficulty
constant. The editing set should provide both left and right context; this is a
more direct test of a claimed bidirectional advantage than Sudoku.

## 6. Current Pilot Findings

Use `CURRENT_RESULTS.md` for exact tables. The safe summary is:

1. Qwen3-4B follows the current StructEval output contract much more reliably
   than iLLaDA or DreamReasoner.
2. iLLaDA often spends its fixed 256-token canvas on `<think>` text and never
   reaches the requested structure. Its StructEval score is partly inflated by
   permissive CSV parsing.
3. iLLaDA Fast roughly doubles Best's accepted tokens per forward and reduces
   time/energy, but this does not recover MBPP or structured-output quality.
4. DreamReasoner is fast on short 256-token generations but has 45/100 OOMs on
   StructEval-T and weak strict structured-output validity.
5. Current Sudoku generations are protocol-invalid and cannot support a model
   or architecture claim.

These findings motivate the benchmark protocol; they are not yet a fair final
ranking of the model families.

## 7. Current Shortcomings

### Scientific Validity

1. The current matrix mixes reasoning, syntax, serialization, retrieval, and
   long generation without a predeclared taxonomy.
2. Qwen disables thinking while iLLaDA and DreamReasoner often expose reasoning
   inside the same output budget.
3. A single 256-token fallback is not enough for reasoning-first checkpoints
   and does not separate missing answers from wrong answers.
4. Best/Fast provides only two inference points. It is insufficient for a
   quality-compute Pareto curve.
5. Qwen3-4B versus 8B diffusion checkpoints confounds architecture, training
   data, model scale, and post-training.
6. Training-compute metadata is incomplete, so equal-pretraining-compute claims
   are not yet supportable.
7. There is no hidden test split or contamination audit.
8. Resource rows come from different code commits, with no repeated-run
   uncertainty or controlled idle-power record.
9. The current AR sanity baseline has not reproduced a canonical protocol.
   Bertolani et al. report Qwen3-4B at 81.19% on 4-shot GSM8K and 62.60%
   pass@1 on 3-shot MBPP, versus this pilot's 59% and 6% under different
   prompts/filtering. Those values cannot be compared until Track 0 is run.

### Scoring And Protocol

1. Sudoku can parse an echoed puzzle as an answer.
2. CSV parsing can award a render floor to prose with no required fields.
3. `valid`, `complete`, and semantic correctness are not consistently separated
   across every dataset.
4. HelloBench lacks a semantic judge or human preference track.
5. Raw score-per-joule is not comparable across tasks without normalization.

### Trace Interpretation

1. iLLaDA Best/Fast stabilizing exactly one/two tokens per forward is imposed by
   the sampler schedule, not evidence of an emergent parallel strategy.
2. Structure-first metrics are computed only for eligible traces and can look
   favorable when final output is invalid.
3. A token-position trace cannot be mapped directly to 81 Sudoku cells.
4. Trace collection and compute replay must be demonstrated not to alter timed
   generation or memory behavior.

### Coverage

- DiffusionGemma and same-scale Gemma AR are missing.
- W1 API and trace semantics are unvalidated.
- Qwen3-8B and optimized dLLM ablations are missing.
- iLLaDA RULER is all OOM; DreamReasoner RULER/HelloBench are unscored or
  incomplete; iLLaDA HelloBench has only one scored sample.

## 8. DiffusionGemma Experiment

The repository already contains a controlled matrix:
`configs/experiments/dg_comparison.yaml`. It pairs
`google/diffusiongemma-26B-A4B-it` with the same-scale
`google/gemma-4-26B-A4B-it` AR checkpoint. Run both sequentially on the same
physical A100 80 GB in BF16.

Do not launch the formal run until the answer protocol and scorer gate pass.

### Prepare Once On RunPod

```bash
cd /workspace/dllm
git pull

export HF_HOME=/workspace/dllm/.data/huggingface
export HF_TOKEN='YOUR_READ_TOKEN'

python prepare_data.py
python prepare_model.py \
  --matrix configs/experiments/dg_comparison.yaml \
  -m diffusiongemma \
  -m gemma4_26b
```

### One-Sample Protocol Gate

```bash
python run_model.py \
  --matrix configs/experiments/dg_comparison.yaml \
  -m diffusiongemma gemma4_26b \
  -d gsm8k structeval_t sudoku \
  --real-data \
  --n-samples 1 \
  --output-root output/dg_protocol_check \
  --no-measure-compute \
  --require-all-metrics \
  --no-resume
```

Inspect every `output_text` before proceeding. Required gate:

- the answer block starts;
- no prompt echo is parsed as an answer;
- strict parser behavior matches the task;
- output is not silently truncated before the answer;
- time, energy, and VRAM are non-null;
- DiffusionGemma trace capture does not change the output.

### Formal Quality And Direct Resource Run

```bash
python run_model.py \
  --matrix configs/experiments/dg_comparison.yaml \
  -m diffusiongemma gemma4_26b \
  --real-data \
  --output-root output/dg_a100 \
  --no-measure-compute \
  --require-all-metrics \
  --resume
```

### Deferred Compute Replay

Compute profiling requires generation replay and can roughly double GPU work.
Keep it separate in the analysis plan even though the command updates the same
generation records:

```bash
python run_model.py \
  --matrix configs/experiments/dg_comparison.yaml \
  -m diffusiongemma gemma4_26b \
  --real-data \
  --output-root output/dg_a100 \
  --measure-compute \
  --require-all-metrics \
  --resume
```

Transfer only `output/dg_a100/model_output` to the local machine, then run:

```bash
python run_score.py \
  --matrix configs/experiments/dg_comparison.yaml \
  -m diffusiongemma gemma4_26b \
  --real-data \
  --output-root output/dg_a100 \
  --no-resume

python run_visualization.py \
  --matrix configs/experiments/dg_comparison.yaml \
  -m diffusiongemma gemma4_26b \
  --real-data \
  --output-root output/dg_a100
```

## 9. Required Figures And Tables

The first draft should contain:

1. benchmark taxonomy and coverage table;
2. model/checkpoint/training-metadata table;
3. quality table with strict validity and completion beside the primary score;
4. quality versus wall-time Pareto plot;
5. quality versus joules Pareto plot;
6. quality versus total TFLOPs Pareto plot;
7. peak VRAM and OOM/capacity table;
8. steps-per-block and block-size ablation curves;
9. answer-started, truncation, and format-failure decomposition;
10. representative trace plots, clearly separated from final correctness;
11. same-scale DiffusionGemma versus Gemma AR case study;
12. editing/infilling results once that axis is implemented.

Every figure caption must state checkpoint, GPU, precision, sample count,
output cap, step count, block size, trace policy, and whether compute was
profiled or inferred.

## 10. First-Draft Work Plan

### Phase 1: Protocol Freeze

- classify every task;
- define canonical-reproduction configs separately from Arena configs;
- add answer boundaries and no-thinking normalization;
- repair Sudoku and CSV parsing;
- add answer-started and budget-exhausted diagnostics;
- assign a benchmark protocol version.

### Phase 2: Small Gate

- five samples per model/task configuration;
- include every StructEval output format;
- inspect raw outputs manually;
- do not proceed when a parser or prompt failure is systematic.

### Phase 3: Critical New Run

- DiffusionGemma/Gemma AR on one A100 80 GB;
- direct wall-time, energy, and VRAM pass;
- deferred total-compute replay;
- repeat a small subset three times for measurement variance.

### Phase 4: Draft

- freeze tables from versioned summaries;
- write methods before interpreting results;
- label existing Qwen/iLLaDA/Dream data as pilot;
- state unsupported claims explicitly;
- publish commands, configs, checksums, and raw summaries with the draft.

## 11. Claims To Avoid In The First Draft

Do not claim:

- bidirectional attention is intrinsically better or worse from the current
  three-model pilot;
- Sudoku demonstrates structure-generation ability;
- iLLaDA's exact one/two tokens per forward is learned parallelism;
- parameter matching is compute matching;
- current score-per-joule values form one cross-task leaderboard;
- one iLLaDA HelloBench sample is comparable to Qwen's 20-sample run;
- OOM rows are quality failures rather than capacity findings.

The strongest honest first-draft contribution is a benchmark protocol that
makes these distinctions unavoidable.
