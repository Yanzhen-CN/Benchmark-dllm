# End-to-End Benchmark Test Protocol

This document is the authoritative execution protocol for the benchmark. It
separates comparable timed evaluation from deliberate capacity probes and
keeps every model on its own published inference path.

## 1. Non-negotiable rules

- Run one model family per OS process and one model-specific virtual
  environment. All declared variants run sequentially in that process and may
  share the already-loaded checkpoint weights.
- Do not add quantization, CPU offload, automatic batch/input reduction,
  speculative decoding, custom KV caches, `torch.cuda.empty_cache()`, or an
  OOM retry with changed settings.
- Keep model-native execution intact. iLLaDA uses its full-sequence official
  denoising loop without KV cache. DreamReasoner keeps its checkpoint's native
  prefix KV cache. AR models use their checkpoint `generate()` path.
- Dataset preparation, checkpoint download, model loading, and the short
  warmup are outside every sample timing window.
- Persist each sample immediately. `--resume` skips completed samples without
  regenerating or retiming them.
- A formal `model x variant x dataset` row is comparable only when every
  selected sample completes without OOM and all required resource fields are
  present. Never average a partial row.

## 2. Evaluation matrix

| Evaluation | Samples per model variant | Input/output rule | Reporting role |
|---|---:|---|---|
| GSM8K | 100 | Dataset protocol | Formal |
| MBPP-Sanitized | 100 | Dataset protocol | Formal |
| StructEval-T | 100 | Dataset protocol | Formal |
| Sudoku4 | 4B/8B/W1: 100; DG/Gemma-4: 10 | d1 zero-shot prompt, 128 output tokens | Formal compact track / large-model probe |
| Sudoku9 | DG/Gemma-4: 100 (50 Easy + 50 Hard); 4B/8B/W1: 10 (5 + 5) | General-instruction prompt, 256 output tokens | Formal 9x9 track / small-model feasibility probe |
| RULER | 30 | 4096 encoded-input target + at most 64 output tokens | Formal |
| HelloBench, iLLaDA | 1 | 2K-word profile, 3072-token generation cap | Reference diagnostic |
| HelloBench, DreamReasoner | 1 | 2K-word profile, 3072-token generation cap | Reference diagnostic |
| HelloBench, other models | 1 per configured 2K/4K profile | Per-profile cap | Reference diagnostic |
| RULER half-context probe | 1 | Input = floor(declared max context / 2), output cap = 64 | Capacity diagnostic only |

### Official-protocol boundary

| Dataset | Upstream/common protocol retained | Controlled deviation in this suite |
|---|---|---|
| GSM8K | lm-eval `gsm8k_cot`, first four demonstrations, flexible last-number extraction, and generate-until strings | Seeded 100/1319 test subset; fixed per-model decoding profiles |
| MBPP-Sanitized | Official tasks 2/3/4 three-shot prompt, `[BEGIN]`/`[DONE]`, execution of all tests, pass@1 | Sanitized source and seeded 100-row subset, so do not compare it directly with full-MBPP leaderboard numbers |
| StructEval-T | Official query, marker wrapper, strict parser, required-path coverage, and `0.2 render + 0.8 key validation` score | Seeded 100/950 subset and suite-controlled decoding profiles |
| Sudoku4 | d1's pinned 500-row test CSV, zero-shot prompt, `<reasoning>/<answer>` contract, and blank-cell accuracy | Seeded 100-row subset; strict complete-puzzle success is an auxiliary metric; one common 128-token setting rather than d1's 128/256 sweep |
| Sudoku9 | Ye et al. Park test split, zero-shot puzzle, clue preservation plus all Sudoku constraints as the primary 0/1 score | Easy/Hard is reporting-only; answer markers and tolerant extraction adapt the task to general instruction checkpoints |
| RULER | RULER-style task concepts, answer prefix, and fractional `string_match_all` | Reduced three-family, explicit-position, single-4096-input diagnostic; **not** the official 13-task RULER suite |
| HelloBench | Official length-constrained prompts and 2K/4K target profiles | 1 sample/profile and objective judge-free diagnostics; **not** official checklist-based HelloEval |

Consequently, only protocol-compatible metrics should use official metric
names. All reports must also show the subset size. RULER and HelloBench results
must be titled controlled/reference diagnostics and must never be presented as
official leaderboard scores.

Sudoku4 and Sudoku9 are separate tasks with separate sample IDs, output
directories, metrics, and report columns. Their scores must never be pooled.
Sudoku4 has no Easy/Hard label: every pinned d1 test puzzle has eight blanks,
and imposing a new difficulty classifier would create an unsupported split.

The formal RULER set contains 10 NIAH, 10 multi-hop, and 10 aggregation
questions. Front, middle, and back answer positions are also balanced across
the same 30 samples. For local HF models, filler is fitted after the model's
own chat template and tokenizer are applied, and the encoded prompt must equal
exactly 4096 tokens. Per-model input construction fails explicitly if an exact
fit cannot be constructed; it must not silently use a shorter input. The total request can be
up to 4160 tokens including the answer allowance. W1 remains a clearly labelled,
unverified 4096-word whitespace proxy because no local tokenizer is available.

The half-context probe is a separate dataset and output directory. It is not
included in formal RULER accuracy, TPS, SPS, energy, peak-VRAM, or ranking
aggregates. For the current declarations its target inputs are:

| Model | Declared maximum | Probe input |
|---|---:|---:|
| iLLaDA | 8192 | 4096 |
| DreamReasoner | 32768 | 16384 |
| Qwen3-4B / Qwen3-8B | 40960 | 20480 |
| W1 | 8192 | 4096 |
| DiffusionGemma / Gemma 4 | 262144 | 131072 |

## 3. One-process execution order

Each model subprocess executes jobs in this order:

1. Prepare/reuse normalized datasets and checkpoint files outside timing.
2. Load the checkpoint once and run an untimed eight-token warmup.
3. Run all regular formal datasets, persisting every sample immediately.
4. Run formal RULER at 4096 input tokens for all 30 samples.
5. Run the configured HelloBench reference profile(s). iLLaDA and
   DreamReasoner receive only the 2K-word, one-sample profile.
6. Run the isolated half-context RULER probe last.
7. Exit the model process, which releases all CPU/GPU state before the next
   model family starts.

This order means a capacity-probe OOM cannot prevent any formal timing row
from being collected. Best/Fast are still separate result rows even though
they reuse one loaded checkpoint.

## 4. OOM and failure policy

Formal-row OOM:

- Stop that `model x variant x dataset` row immediately.
- Persist the failed sample, `_meta.json`, and `oom_info.json` with stage,
  sample ordinal, requested lengths, GPU identity, and remaining count.
- Mark the whole `model x variant x dataset` row invalid and do not score it.
  Other variants remain independent rows and are still attempted.
- Continue with the next independent matrix job. There is no silent retry or
  changed inference setting.

Capacity-probe OOM:

- Record it as the expected diagnostic outcome for
  `ruler_context_probe`.
- Do not invalidate the separate 30-sample `ruler` row.
- Because the probe is last, no later formal timing can be lost.

Non-OOM sample failure is persisted with `failed` or `truncated` status and
handled by the dataset's completeness policy. Model-load OOM means the GPU
cannot host that checkpoint under the declared precision; no sample timing
from that model is valid on that machine.

## 5. Hardware gate

The harness cannot make an incompatible model fit without changing the model
being benchmarked. Before a full run, use the intended production GPU:

- 24 GiB-class GPU: Qwen3-4B, Qwen3-8B, iLLaDA-8B, and DreamReasoner-8B are
  the intended local group. The 4K formal RULER point is selected to avoid the
  observed iLLaDA 8K OOM. The half-context probe may still OOM, especially for
  DreamReasoner.
- 80 GiB-class GPU: DiffusionGemma-26B-A4B and Gemma-4-26B-A4B. Their 131072
  input-token probes are deliberately allowed to OOM.
- W1: API execution only. Provider latency/TPS is self-reported and must not
  be mixed with locally measured timing or energy. Without a provider
  tokenizer endpoint, RULER filler is fitted to a 4096-word whitespace proxy
  and persisted as `whitespace_proxy_unverified`; it is not evidence of an
  exact 4096-token input.

Available VRAM, driver/runtime versions, model precision, GPU identity, and
checkpoint revision must be recorded with the run. A successful short smoke
test does not prove that the capacity probe will fit.

## 6. Commands

Prepare environments, immutable data, and model snapshots before GPU timing:

```bash
python setup_venv.py
python prepare_data.py
python prepare_model.py
```

`prepare_model.py` only downloads the configured Hugging Face snapshots into
the repository-root `data/huggingface` cache. It does not construct the model
adapter, load weights into CPU/GPU memory, or execute a forward pass.

Run one complete model family at a time. `run_model.py` uses real data,
requires timing/energy/VRAM fields, disables optional FLOP replay, and resumes
completed samples by default:

```bash
python run_model.py -m qwen3_4b --output-root output/formal --resume
python run_model.py -m illada --output-root output/formal --resume
python run_model.py -m dreamreasoner --output-root output/formal --resume
```

Run the large matched pair on the 80 GiB machine and W1 only after its API
credentials and rate limits are confirmed:

```bash
python run_model.py -m diffusiongemma gemma4_26b_a4b --output-root output/formal --resume
python run_model.py -m w1 --output-root output/formal --resume
```

After copying `output/formal/model_output` to the scoring machine:

```bash
python run_score.py --output-root output/formal --resume
python run_visualization.py --output-root output/formal
```

For an isolated protocol check before the full schedule:

```bash
python run_model.py -m illada -d ruler hellobench ruler_context_probe \
  --output-root output/protocol-check --resume
```

Do not pass a global `--n-samples` during the formal run: it overrides the
dataset-specific matrix counts, including the 30-sample RULER selection.

## 7. Acceptance checklist

A model family is complete only when:

- Every expected formal run directory has `_meta.json` with
  `test_valid: true`, `test_complete: true`, matching selected/completed
  sample counts, and the current measurement protocol.
- Formal RULER has exactly 30 persisted samples per variant. Every local HF
  sample is exactly 4096 encoded input tokens; W1 is explicitly labelled as an
  unverified 4096-word whitespace proxy. All variants allow at most 64 output
  tokens.
- iLLaDA and DreamReasoner HelloBench each have exactly one 2K-profile sample
  per selected variant and no 4K-profile sample in that run.
- Every successful formal sample has positive wall time plus the required
  energy and peak-VRAM fields; output length and status are present.
- `ruler_context_probe` has exactly one persisted success or one explicit
  `oom_info.json` per attempted variant. Its outcome is not merged into the
  formal RULER summary.
- Scoring reconstructs the same deterministic sample IDs and creates no
  summary for an OOM-invalid formal row.
