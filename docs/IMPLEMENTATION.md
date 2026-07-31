# Benchmark implementation guide

This file documents code behavior only. The experiment design, research
questions, metric interpretation, and reporting language live in the desktop
design document `dLLM_benchmark_设计文档.md`, which is the normative research
protocol. Generated results and figures belong under the ignored `output/`
tree, not under `docs/`.

## 1. Entry points and environments

- `run_prepare.py`: creates the requested virtual environments, prepares all
  selected datasets, and downloads model snapshots.
- `prepare_data.py`: downloads pinned upstream data and builds deterministic
  prepared sample banks under `data/datasets/prepared/`.
- `prepare_model.py`: downloads model/tokenizer files into
  `data/huggingface/`; it does not construct the adapter, move weights to a
  device, or execute a forward pass.
- `run_model.py`: server-side generation only. It dispatches each model to its
  own script in `venv_scripts/` and writes immutable generation JSON.
- `run_check.py`: read-only artifact audit under `.venvs/root`. It accepts the
  same `-m/-d/--matrix/--output-root` selection style and checks selected sample
  IDs/counts, per-sample generation ceilings and seeds, status, required
  timing/energy/VRAM/compute fields, trace policy, OOM invalidation, scoring
  completeness, and visualization presence against the current YAML matrix.
  Capacity diagnostics are informational by default and become required only
  when explicitly selected or when `--require-diagnostics` is passed.
- `run_score.py`: local CPU scoring from transferred generation JSON. It does
  not load model weights.
- `run_visualization.py`: local figures and reports derived from generation and
  score artifacts.
- `run_conversion.py`: optional, local pairwise quality/resource sensitivity
  analysis. It is not called by `run_visualization.py` and never changes the
  measured-only report.

The root Python only dispatches. Model dependencies are installed in
`.venvs/<model>/`; non-model local work uses `.venvs/root`. Missing model
environments are created by the preparation/run entry point before use.

## 2. Configuration ownership

- `configs/models/*.yaml`: checkpoint, adapter class, precision, sampler
  variants, and model-specific inference parameters.
- `configs/datasets/*.yaml`: dataset class, official primary metric, auxiliary
  fields, sample-bank size, and deterministic seed.
- `configs/experiments/full_matrix.yaml`: selected models/variants/datasets,
  per-model sample-count overrides, and generation ceilings.

iLLaDA, iLLaDA VarGen, and DreamReasoner use `p1`, `p2`, `p4`, and `p8`,
where the suffix is the planned token commits per denoising step for a
32-token block. The formal matrix defaults to `p1,p2`; explicit `-v` selection
may request future P4/P8 frontier runs without changing those defaults.
`migrate_parallelism_names.py` is the one-time, dry-run-by-default utility for
renaming historical `best`/`fast` aliases to explicit `p1`/`p2` in existing
outputs. It refuses existing targets and does not synthesize P4/P8 results.

Current formal ceilings are:

| Dataset | `max_new_tokens` |
| --- | ---: |
| GSM8K, MBPP, StructEval-T | 512 |
| Sudoku4 direct, Sudoku9 direct | 256 |
| Sudoku4 thinking, Sudoku9 thinking | 2048 |
| RULER and RULER context probe | 64 |
| HelloBench 2K / 4K | 3072 / 6144 |

Sudoku direct and thinking are distinct dataset classes, prepared banks,
output directories, and report rows. Direct requests marker-free copy-and-fill;
thinking uses the original marked reasoning contract. Both variants first
locate the final submitted fixed-size digit string/grid and then call the same
size-specific semantic scorer. A Sudoku9 81-cell submission may retain `0`
placeholders so blank-cell accuracy, clue preservation, and completion remain
observable; it still receives zero complete-sequence primary credit unless all
81 cells exactly equal the reference solution. A later explicit marker or
final-answer cue defines the submitted region even when truncated; in that case
older complete drafts are not scored. Without a located submission region,
copied or partial grids inside reasoning are not passed to the semantic scorer.
Extra thought in a direct response affects the
strict format/instruction-following auxiliary metric, not semantic correctness.
Their scores are never pooled.

## 3. Artifact contract

The canonical tree is:

```text
output/
  model_output/<model>_<variant>/<dataset>/
    _meta.json
    <sample_id>.json
    oom_info.json              # only when the dataset row is OOM-invalid
  score_output/<model>_<variant>/<dataset>/
    <sample_id>.json
    summary.json
  visualization_output/...
  report/
    raw_results.txt
    raw_results.csv
    raw_summary_details.json
    <dataset>/<comparison-scope>/...
  conversion_output/<A>__relative_to__<B>/<dataset>/beta-<b>_gamma-<g>/...
```

`_meta.json.selected_sample_ids` is ordered and defines the exact dataset row.
Generation is persisted after every sample. A row is reportable only when the
metadata marks it valid and complete and all selected sample files exist.

Each generation file contains the prompt/request, output text, status, timed
wall-clock interval, energy, peak VRAM, optional compute replay, forward-pass
count, final valid length, adapter extras, and trace. Trace collection occurs
after each timed forward boundary; serialization is outside the generation
timing denominator. HelloBench disables trace through its dataset policy.
Diffusion adapters define valid output as the token prefix before the first
checkpoint EOS token. Incremental iLLaDA VarGen finishes the active block and
then skips all later blocks, matching the upstream `var_generate` stop
boundary. Legacy traced artifacts whose `output_text` was decoded with special
tokens hidden are recovered from the final trace's token-level EOS marker at
load time; the recovered prefix and valid token length participate in score
fingerprints, summaries, and visualization.

## 4. Scoring contract

`run_score.py` reconstructs the deterministic sample bank and requires exact
ordered equality with `_meta.json.selected_sample_ids`.

- `SUCCESS`: run the dataset scorer normally.
- `TRUNCATED`: score the available text, force `complete=false`, and retain a
  truncation diagnostic. A correct answer already present is not erased.
- OOM, model-load failure, infrastructure failure, or a missing selected file:
  invalidate the entire model × variant × dataset row; never aggregate a
  partial row.

Per-sample score reuse requires an exact fingerprint match across dataset
revision, prompt/protocol revision, scorer revision, ordered sample-set hash,
generation-text hash, and primary metric. Legacy score JSON without this
metadata is rescored. `summary.json` records the combined fingerprint,
expected/actual sample counts, status counts, and micro-mean aggregation rule.
It also records `generation_protocol_revision`, which hashes each selected
sample's prompt, output budget, request config, and seed. Pairwise conversion
requires this value to match in addition to the ordered sample-set hash.

Primary and auxiliary fields have separate roles:

| Dataset | Primary scorer | Project auxiliary fields / boundary |
| --- | --- | --- |
| GSM8K | Four fixed `gsm8k_cot` examples; last flexible regex match; lm-eval exact match after ignoring comma, dollar sign, and trailing period | valid/complete/truncated rates and answer position; this is a fixed 4-shot subset, not the current default 8-shot full task |
| MBPP-Sanitized | One candidate, all official tests in a temporary Python subprocess, binary Pass@1 | executable/complete rates, answer position, and answer-local structure/content formation; score metadata records Python/OS, 10-second timeout, isolation, and extraction rule |
| StructEval-T | `round(0.2 * render + 0.8 * key_validation, 2)` using the official non-renderable parser and required-path validation | render, key validation, complete-correct, answer position, and answer-local structure/content formation |
| Sudoku4 direct/thinking | d1 answer extraction plus blank-cell accuracy, including short-answer padding and long-answer truncation | legal-puzzle success, strict 16-digit/direct-answer compliance, clue preservation, exact match, answer position; extra thought affects compliance but not an extracted correct primary answer |
| Sudoku9 direct/thinking | Ye et al. complete reference-sequence Accuracy after final-answer extraction | strict 81-digit/direct-answer compliance, constraint validity, blank-cell/cell accuracy, conflicts, and Easy/Hard audit strata; thought text is tolerated only when it ends in a non-rejected final submission |
| RULER-inspired | NVIDIA `string_match_all`: fraction of required references found case-insensitively | all-answers match, front/middle/back robustness, completion, truncation, and answer position; not the official 13-task leaderboard suite |
| HelloBench reference | transparent no-judge objective diagnostic | length, repetition, every major-issue flag, per-profile completion/time, and explicit `case_study_only=1`; never labelled HelloEval |

Every task except HelloBench persists a task-specific answer region. MBPP and
StructEval-T call the same final-answer locator before both task scoring and
trace diagnostics. It ignores drafts inside unfinished/closed thinking, then
selects the final task marker, code fence, or allowed fallback. GSM8K records
the exact numeric regex span; Sudoku prefers its formal marker, then the last
explicit final-answer cue, then a non-rejected complete digit answer; RULER begins at the first
non-whitespace output token. HelloBench hashes the whole output but intentionally
does not publish Answer Start.
The retired whole-trace `structure_first_score` is not published. Its
replacement is the conditional `answer_local_structure_first_score`, always
reported with answer-detection, token-span-mappable, and eligible ratios.

Auxiliary numeric fields are aggregated over the samples on which they exist;
missing fields produce an explicit eligible ratio instead of disappearing via
key intersection. Each score JSON stores the scored-payload SHA-256, answer
character span and detection method, marker completeness where applicable,
token span when trace mapping succeeds, primary score, validity, completion,
truncation, and the cache fingerprint. Non-numeric audit values remain in
per-sample score JSON.

## 5. Task 4 visualization implementation

`--n-representative` limits only the single-sample heatmap/GIF/result files.
The generic token-canvas GIF works for traced token canvases; the 9x9 board GIF
is Sudoku9-only and requires an 81-cell mapping. Redundant per-sample final
frame, first-commit scatter, and commit-speed files are not emitted.
`dataset_trace_summary.json` always consumes every generated trace available
for the selected model × config × dataset row. Normalized curves are first
binned within each sample and then aggregated with one value per sample per
bin, so a long trace cannot outweigh a short trace merely by contributing more
checkpoints. Dataset summaries persist Mean, Median, IQR and bootstrap 95% CI.

The headline dataset-level artifacts are TPF profiles, TPF-vs-Tps, a compact
parallelism signature (Peak/Mean TPF, busiest-10%-forward finalization share,
and P90 final-stable progress), a position × final-stable-forward density map,
commit-order tau by window, and Early/Middle/Late finalization shares.
Observed certainty/top-1, answer-local structure/content, and Sudoku revision
remain coverage-gated secondary analyses. Generic visible-token changes are
labelled Draft Volatility; they are not Sudoku correction. The redundant
final-stable CDF is not emitted. Cross-model curves retain bootstrap confidence
bands and remain partitioned by dataset, exact sample set and hardware.

Three block-diffusion-specific report families are derived from the same trace:

- `task4_update_geometry.png`: contiguous finalization-run length and enclosing
  span density;
- `task4_confidence_dynamics.png`: certainty-backslide step rate and mean
  backslide magnitude, labelled with entropy scope/coverage;
- `task4_visible_draft_correction.png`: observability gate, first-visible final
  match, wrong-draft exposure, helpful/lateral/harmful revision shares, and
  revision timing.

Visible-draft correction is N/A for commitment-only `MASKED -> ACCEPTED` traces;
the implementation never converts missing provisional token IDs into zero
revisions. DFlash has no public per-token verification trace, so its separate
`dflash_speculative_acceptance.png` uses aggregate acceptance counters. The
`task4_forward_yield.png` comparison prints its basis on every label: native
models use final-stable tokens/model-forward, while DFlash uses accepted
tokens/target-verification. Measured Tps still includes all execution overhead.

Certainty is never fabricated from a backend that did not record probability
data. AR rows therefore report entropy/top-1 scope as `unavailable` and do not
emit those curves. Entropy and top-1 each carry their own step rate, position
coverage and scope (`full_remaining`, `partial_or_active_subset`, or
`unavailable`), which is printed in cross-model labels.

MBPP and StructEval structure/content plots use only the mapped final-answer
token span. The report always emits answer-detection, trace-mapping and eligible
coverage; the model-level Structure-First figure is suppressed below 0.5
eligibility. Sudoku9 similarly requires at least 0.5 mappable trace-step
coverage per sample and at least 0.5 eligible samples in each Easy/Hard stratum.
Only then are Early/Middle/Late revision counts and wrong-visible correction
success plotted; otherwise the figure explicitly reports N/A instead of zero.
Native 81-position canvases remain mappable while cells are masked, whereas a
free-form/subword trace is mappable only at checkpoints containing one
unambiguous complete 81-cell grid.

## 6. Resource and failure boundaries

Model download, construction, device transfer, warmup, trace copying,
serialization, scoring, and visualization are outside the timed generation
window. Per-sample generation JSON is the raw source of truth: measured seconds,
joules, final output-token count, status, and peak VRAM. Summary JSON additionally
persists window-total seconds, joules, output tokens, timed-sample count, and
energy-sample count. Tps and Eps are ratios of complete window totals, while
Seconds/Sample and Energy/Sample divide their corresponding totals by completed
samples; none is a mean of per-sample ratios. Eps has units J/s and is displayed
as average power. Compute/Cps is absent from the formal first pass and can only
come from a separate optional replay.

The first OOM invalidates only the current dataset row. The runner writes
`oom_info.json`, stops attempting later samples in that dataset, then continues
with the next dataset. Invalid rows are excluded from all quality/resource
aggregates. The separate RULER context probe remains outside formal RULER.

## 7. Normal commands

```bash
python run_prepare.py
python run_model.py -m qwen3_4b
python run_score.py -m qwen3_4b --output-root output
python run_visualization.py -m qwen3_4b --output-root output
```

Selectors are parallel across stages:

```bash
python run_model.py -m illada dreamreasoner -d ruler hellobench
python run_score.py -m illada dreamreasoner -d ruler hellobench
```

`run_visualization.py -m/-d` forwards the same selection to the final raw
report, so unselected or unfinished models are not pulled in merely because an
older `summary.json` exists. The raw report writes only measured values:
Quality–Tps, Quality–Seconds/Sample, Quality–Energy/Sample, Score per Unit
Energy, P1-vs-P2, and answer-region diagnostics. Eps is labelled Average
Power. Chart directories separate different sample-set hashes and reported GPU
hardware, and every plotted label includes N.

Curated paper examples can be selected without reducing dataset-level Task 4
aggregation:

```bash
python run_visualization.py -m diffusiongemma \
  -d gsm8k mbpp structeval_t \
  --sample-ids gsm8k-test-0177,mbpp-sanitized-0131,structeval-t-180530

python run_visualization.py -m gemma dreamreasoner \
  -d mbpp structeval_t \
  --sample-ids mbpp-sanitized-0057,structeval-t-001841
```

`--sample-ids` controls only per-sample evidence. Every available trace still
contributes to `dataset_trace_summary.json` and cross-model Task 4 figures.

Optional conversion is a separate command:

```bash
python run_conversion.py \
  -m illada dreamreasoner \
  --base-model qwen3_8b --base-config ar-baseline \
  --beta 50 --gamma 30 --output-root output
```

Each selected model/config A produces an independent `A relative to B` output;
there is no cross-pair ranking. Speed uses `Seconds/Sample_B /
Seconds/Sample_A`, energy uses `Energy/Sample_B / Energy/Sample_A`, beta scales
the directional ideal-retry adjustment, and gamma is the energy-track weight
percentage (speed weight is `100-gamma`). The command rejects different sample
IDs, prompts/output budgets, dataset revisions, measurement boundaries, or
unmeasured timing. It also excludes HelloBench and the RULER context probe from
conversion.

Use `--no-resume` only when intentionally replacing a generation row. A
temporary `--max-new-tokens` diagnostic must use a separate `--output-root` so
it cannot be mixed with formal output.

## 8. Verification

Run the repository suite through the root environment:

```bash
python run_tests.py
```

Before reporting, verify exact selected IDs, row completion/validity, required
timing/energy/VRAM fields, dataset-specific ceilings, score fingerprints, and
absence of invalid probe rows from formal summaries.
