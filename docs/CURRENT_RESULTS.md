# Current Result Audit

Snapshot date: 2026-07-28. These are diagnostic results imported from the
current local folders and rescored with the current scorer. They are not yet a
publication-ready comparison.

Machine-readable aggregate data is exported to
`output/report/current_summary.csv`. Exact per-sample generations and scores
remain under `output/model_output` and `output/score_output` respectively.

## What The Tasks Measure

| Dataset | Primary capability | Direct evidence for bidirectional structure formation? |
|---|---|---|
| GSM8K | Arithmetic reasoning | No |
| MBPP | Program synthesis: semantics plus executable syntax | Indirect only |
| StructEval-T | Constrained serialization and required-field coverage | Yes, the clearest current task |
| Sudoku | Global constraint reasoning | No; structure is fixed before generation |
| RULER | Long-context retrieval and aggregation | No |
| HelloBench | Long-form completion and repetition control | Weakly related |

Sudoku must not be presented as a structure-generation result. Every answer
already has the same 9 by 9 layout; success depends on solving globally coupled
constraints. Bidirectional attention may be useful to a solver, but this task
does not isolate or demonstrate that mechanism.

## Coverage

`success/selected`; `OOM` means every selected sample failed before a timed
result. A dash means no scored run is available.

| Run | GSM8K | MBPP | StructEval-T | Sudoku | RULER | HelloBench |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-4B AR | 100/100 | 100/100 | 100/100 | 100/100 | 60/60 | 20/20 |
| iLLaDA Best | 100/100 | 100/100 | 100/100 | 100/100 | 0/30 OOM | 1/1 |
| iLLaDA Fast | 100/100 | 100/100 | 100/100 | 100/100 | 0/30 OOM | - |
| DreamReasoner Best | 100/100 | 100/100 | 55/100 | 100/100 | - | - |
| DreamReasoner Fast | 100/100 | 100/100 | 55/100 | 100/100 | - | - |

The iLLaDA HelloBench value is a one-sample diagnostic and must not be compared
with Qwen's formal 20-sample result. DreamReasoner StructEval-T includes 45 OOM
records in the denominator. W1, DiffusionGemma, Gemma 4 AR, Qwen3-8B, and the
optimized ablations are not present in this snapshot.

## Quality Scores

Scores use each dataset's own `[0, 1]` primary metric and are not directly
interchangeable across columns.

| Run | GSM8K accuracy | MBPP pass@1 | StructEval final | Sudoku blank-cell* | RULER accuracy | Hello objective* |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-4B AR | 0.590 | 0.060 | 0.5803 | 0.0080 | 0.300 | 0.4921 |
| iLLaDA Best | 0.150 | 0.000 | 0.0600 | 0.0000 | 0.000 OOM | 0.2432 (n=1) |
| iLLaDA Fast | 0.210 | 0.000 | 0.0560 | 0.0000 | 0.000 OOM | - |
| DreamReasoner Best | 0.510 | 0.010 | 0.0193 | 0.0007 | - | - |
| DreamReasoner Fast | 0.570 | 0.020 | 0.0384 | 0.0007 | - | - |

`*` Sudoku is protocol-invalid in this snapshot: all outputs consumed the full
256-token allowance, and the old parser could mistake an echoed puzzle for an
answer. The nonzero values are retained for audit but must not be interpreted
as solving ability. HelloBench is the project's objective, judge-free score,
not official HelloEval.

## Throughput And Resource Data

All available rows were generated on an NVIDIA GeForce RTX 4090. TPS is not a
hardware-only number: tokenizer, output length, task, sampler, and successful
sample set all affect it.

### Tokens Per Second

| Run | GSM8K | MBPP | StructEval-T | Sudoku | RULER | HelloBench |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-4B AR | 23.62 | 23.48 | 23.55 | 23.40 | 8.96 | 23.30 |
| iLLaDA Best | 21.39 | 21.45 | 12.83 | 17.10 | - | 2.29 |
| iLLaDA Fast | 42.66 | 42.90 | 25.66 | 34.20 | - | - |
| DreamReasoner Best | 61.21 | 41.46 | 89.16 | 96.21 | - | - |
| DreamReasoner Fast | 90.95 | 67.61 | 127.16 | 125.75 | - | - |

### Mean Seconds Per Sample

| Run | GSM8K | MBPP | StructEval-T | Sudoku | RULER | HelloBench |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-4B AR | 8.891 | 9.709 | 9.125 | 10.940 | 5.033 | 125.341 |
| iLLaDA Best | 11.968 | 11.935 | 19.956 | 14.973 | - | 1340.493 |
| iLLaDA Fast | 6.001 | 5.968 | 9.975 | 7.486 | - | - |
| DreamReasoner Best | 4.183 | 6.174 | 2.871 | 2.661 | - | - |
| DreamReasoner Fast | 2.815 | 3.786 | 2.013 | 2.036 | - | - |

### Mean Joules Per Sample

| Run | GSM8K | MBPP | StructEval-T | Sudoku | RULER | HelloBench |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-4B AR | 995.8 | 1085.7 | 1033.4 | 1227.5 | 1809.9 | 14433.9 |
| iLLaDA Best | 3918.7 | 3792.9 | 7733.7 | 5358.2 | - | 601409.6 |
| iLLaDA Fast | 1939.8 | 1889.7 | 3933.4 | 2655.1 | - | - |
| DreamReasoner Best | 698.5 | 1113.4 | 503.5 | 449.7 | - | - |
| DreamReasoner Fast | 500.8 | 666.1 | 344.1 | 350.5 | - | - |

### Peak VRAM In GiB

| Run | GSM8K | MBPP | StructEval-T | Sudoku | RULER | HelloBench |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-4B AR | 7.76 | 7.74 | 7.84 | 7.76 | 16.22 | 11.23 |
| iLLaDA Best | 14.55 | 14.46 | 15.05 | 14.56 | OOM | 17.04 |
| iLLaDA Fast | 14.55 | 14.46 | 15.05 | 14.56 | OOM | - |
| DreamReasoner Best | 20.25 | 18.99 | 22.75 | 20.66 | - | - |
| DreamReasoner Fast | 20.24 | 18.97 | 22.76 | 20.66 | - | - |

## Structural Task Detail

### StructEval-T By Format

| Run | CSV | JSON | XML | YAML | TOML |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B AR | 0.9611 | 0.4273 | 0.1929 | 0.5763 | 0.0000 |
| iLLaDA Best | 0.2000 | 0.0000 | 0.0000 | 0.0125 | 0.0000 |
| iLLaDA Fast | 0.2000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| DreamReasoner Best | 0.0000 | 0.0000 | 0.0000 | 0.0603 | 0.0000 |
| DreamReasoner Fast | 0.0000 | 0.0455 | 0.0000 | 0.0887 | 0.0000 |

The iLLaDA CSV score is mostly an evaluator floor. Its natural-language
`<think>` output can be accepted by `csv.DictReader` as a header, earning the
0.2 parse component while matching none of the required fields. It is not
evidence of successful CSV generation.

### Final-Output Diagnostics

| Run | Struct format-valid | Struct complete-correct | Struct field coverage | MBPP executable | MBPP pass@1 |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B AR | 0.67 | 0.20 | 0.558 | 0.11 | 0.06 |
| iLLaDA Best | 0.30* | 0.00 | 0.000 | 0.01 | 0.00 |
| iLLaDA Fast | 0.28* | 0.00 | 0.000 | 0.00 | 0.00 |
| DreamReasoner Best | 0.05 | 0.00 | 0.000 | 0.01 | 0.01 |
| DreamReasoner Fast | 0.04 | 0.00 | 0.000 | 0.02 | 0.02 |

`*` Inflated by CSV's permissive parse floor.

### Trace Diagnostics

| Run | MBPP structure-first score / eligible | StructEval structure-first score / eligible | Mean tokens stabilized per forward |
|---|---:|---:|---:|
| Qwen3-4B AR | 0.058 / 99% | 0.402 / 61% | 1.0 |
| iLLaDA Best | 0.273 / 55% | 0.500 / 26% | 1.0 |
| iLLaDA Fast | 0.324 / 46% | 0.500 / 26% | 2.0 |
| DreamReasoner Best | 0.454 / 99% | 0.327 / 4% | task-dependent, 1.39-3.78 |
| DreamReasoner Fast | 0.437 / 100% | 0.415 / 5% | task-dependent, 2.27-5.40 |

The exact iLLaDA stabilization rates are imposed by the sampler schedule:
Best commits one token per forward and Fast commits two. They do not establish
that the model learned a globally parallel structural strategy. A high
structure-first score also does not imply final correctness; iLLaDA has
nonzero structure-first diagnostics while producing no passing MBPP program.

## Why The Diffusion Models Look Weak Here

1. Architecture is only an opportunity. Bidirectional attention exposes
   masked positions jointly, but the checkpoint still needs enough structured
   data and instruction tuning to use that information reliably.
2. iLLaDA is block-wise, not globally editable. It denoises 32-token blocks
   and permanently fixes earlier blocks. A closing brace, field, or code
   dependency hundreds of tokens later cannot repair an earlier decision.
3. Low-confidence unmasking selects confident tokens, not syntactic roles. It
   has no explicit grammar, schema, parser, or constraint solver.
4. The output protocol is currently unfair. Qwen explicitly disables thinking;
   iLLaDA and DreamReasoner spend much of the shared 256-token allowance on
   `<think>` or prose. Many records end before the requested structure starts.
5. The existing benchmark has only one direct structure task. MBPP mixes
   structure with algorithmic semantics, while Sudoku is constraint reasoning.
   The current matrix cannot support a broad causal claim about bidirectionality.

The defensible conclusion is therefore narrower: under the current checkpoints,
samplers, prompts, and fixed output budget, Qwen follows the structured-output
contract much more reliably than iLLaDA or DreamReasoner. The current data do
not prove that diffusion or bidirectional attention is intrinsically worse.

## Gate Before The Next Full Run

1. Add answer-first, no-explanation output contracts for StructEval-T, MBPP,
   and Sudoku; disable thinking through the checkpoint template where supported.
2. Record `answer_started`, `budget_exhausted`, and strict format validity so a
   missing answer is separated from an incorrect answer.
3. Validate five samples per format/model before scheduling 100 samples.
4. Keep Sudoku under reasoning, not structural evidence, and repair its parser
   before using its partial-credit score.
5. Add at least one controlled structure-only task and one bidirectional
   infilling/editing task before making an architectural claim.
6. Run the missing same-scale DiffusionGemma/Gemma AR pair and W1 only after
   the protocol gate passes.
