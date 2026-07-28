"""DreamReasoner (Appendix D.2): block diffusion via the model's own
``block_diffusion_generate``/``_denoise_current_block``/``_select_transfer_index``
algorithm — ported directly from the real ``generation_utils.py`` shipped on
`Dream-org/DreamReasoner-8B`'s own HF repo (``trust_remote_code``).

This is a **different, independently trained model** from regular Dream-7B
(`Dream-org/Dream-v0-Instruct-7B`, formerly ``models/dream.py`` — removed,
since the design doc's model roster (section 5) no longer lists regular
Dream at all, only DreamReasoner-Best/Fast): DreamReasoner-8B adapts
Qwen3-8B-Base into a block diffusion model via block-size curriculum
learning, per its own paper/README, and exposes ``block_diffusion_generate``
(block-wise) rather than regular Dream's single-pass ``diffusion_generate``.

GitHub's `DreamLM/DreamReasoner` repo (the link the design doc points at)
ships only a README + assets — no python at all. The actual generation code
lives in the HF checkpoint's own `trust_remote_code` files, fetched and read
directly (not assumed):

- `Dream-org/DreamReasoner-8B/generation_utils.py` — the real
  ``block_diffusion_generate``/``_denoise_current_block``/
  ``_select_transfer_index``/``BlockDiffusionGenerationMixin``.
- `Dream-org/DreamReasoner-8B/modeling_dream.py` — confirms
  ``DreamForCausalLM(DreamPreTrainedModel, BlockDiffusionGenerationMixin,
  GenerationMixin)``, and that ``auto_map`` registers this same class for
  both ``AutoModel`` and ``AutoModelForCausalLM`` (this adapter still uses
  ``AutoModelForCausalLM`` explicitly, matching the design doc/README's own
  quick-start rather than relying on the ``AutoModel`` alias).
- `Dream-org/DreamReasoner-8B/config.json` — `block_size: 32`,
  `mask_token_id: 151669`, `model_type: "Dream"`.

Confirmed facts driving this port (not guessed):

- **No history/trace is exposed at all.** Unlike regular Dream's
  ``diffusion_generate`` (returns ``output.history``, a full snapshot per
  step), ``block_diffusion_generate`` returns only ``sequences``/``nfe`` —
  confirmed from source. So, like ``models/illada.py`` (and unlike
  ``models/dream.py``'s snapshot-diffing fallback), this adapter reimplements
  the real denoising loop itself rather than calling the model's own
  convenience method, to get a trace at all.
- Default `block_length` = `config.block_size` = 32 when unset (matches
  iLLaDA, and the GitHub README's own quick-start example).
- Default per-block `denoising_steps` = `block_length` whenever
  `remasking_strategy` isn't `'low_confidence_static'` (confirmed from
  source: only that one strategy defaults to a single step) — the library's
  own default `remasking_strategy` is `'low_confidence_dynamic'`, which is
  already confidence-based, so unlike regular Dream (whose library default,
  `alg="origin"`, was a random/time-scheduled commit order this benchmark
  deliberately overrode to `"entropy"`), **no override is needed or applied
  here** — the shipped default is kept as-is.
- The **last** step in every block force-commits every remaining masked
  position in that block (`force_accept = step == denoising_steps - 1`),
  ported as-is — the source's own guarantee that a block never moves on
  still partially masked.
- `mask_token_id` = `config.mask_token_id` = 151669, shipped directly in the
  checkpoint's own `config.json` (unlike regular Dream, where this was
  *never* auto-derivable and had to be resolved from a fallback chain).
- The real default path uses a prefix KV cache for already-committed blocks
  (`_default_use_kv_cache` -> `True` for `model_type in ('dream', 'dream1')`,
  which this checkpoint's config sets) — this directly affects real
  Time/Energy/Compute cost (design doc Part 3), so it's ported faithfully
  rather than skipped for implementation simplicity: each block does
  `denoising_steps` logit-producing forwards (`store_kv=False`, draft tokens
  not yet final) plus one final `store_kv=True` forward once the block is
  fully committed (to push its now-final tokens into the cache before the
  next block attends to them) — that finalize call changes no positions, so
  it produces no `TraceStep`, matching the source's own semantics exactly.
- Never revises: `_select_transfer_index` only ever selects from the
  *current* block's still-masked positions, computed fresh from `cur_x` each
  step, and a finished block's span in the running canvas is never written
  to again by any later block's loop — confirmed structurally from source,
  same as iLLaDA. This means design doc 4.2.6's `ErrorThenCorrect` is
  expected to come out ~0 for this model, same as iLLaDA — a real,
  ported-implementation-driven finding, not a modeling gap in this adapter.
- The real source's own `build_block_diffusion_attention_mask` materializes
  a dense `(total_length, total_length)` tensor up front — fine at the
  source's own short-context examples, but at RULER's long-context points
  (up to this model's own 32768-token window, see
  `configs/datasets/ruler.yaml`) that tensor alone is multiple GB, on top of
  the ~16 GiB bf16 model and the KV cache, and was observed to OOM a 24 GiB
  GPU. This adapter never materializes the full version: prefill runs in
  chunks of up to `_PREFILL_CHUNK_BLOCKS` blocks per call (not the whole
  prompt in one shot, and not one block per call either — see
  `_run_denoising`'s prefill comment for why chunking needs a real, but
  chunk-sized-not-sequence-sized, attention_mask, while a lone block per call
  needs none at all). Every call — whatever its chunk size — still sees
  exactly the same key/value set the source's explicit block-tril mask would
  give it (block i can attend to blocks 0..i and nothing after): already
  -cached earlier chunks are unconditionally visible (concatenated in by
  `past_key_values.update`), and only the chunk's own new tokens need the
  block-tril pattern applied explicitly, since those are the only positions
  where "block j come later, must stay invisible to block i" could otherwise
  be violated. The source's own full-sequence
  `build_block_diffusion_attention_mask` is therefore never ported at all —
  no call here ever needs a mask bigger than one chunk.

This benchmark's own addition (not from source): a full-vocab softmax over
each step's raw logits, purely to populate `entropy_by_position`/
`top1_confidence_by_position` for design doc 4.2.4's Certainty curve — the
real `_select_transfer_index` only ever sees the temperature/top-k/top-p
-adjusted sampling probability, not raw-logit entropy, so this is computed
separately and does not influence which positions get committed, mirroring
`models/illada.py`'s identical choice. Only the *currently active* block has
any logits at all at a given step (confirmed from the KV-cache path: future
blocks are never part of the forward call's input) — so, unlike iLLaDA
(which recomputes full-sequence logits every step and can report entropy for
every remaining position), this trace's `entropy_by_position`/
`top1_confidence_by_position` only ever cover the active block's still-masked
positions at each step. That's an accurate reflection of what the model
actually computes at that step, not a shortfall in this adapter.
"""

from __future__ import annotations

import math

from ..interfaces import PositionState, TraceStep
from .hf_diffusion import DiffusionStepConfig, HFDiffusionAdapter
from .model_cache import (
    cpu_offload_max_memory,
    get_or_load,
    offloaded_parameter_bytes,
    reload_with_offload,
)
from .prompting import tokenize_instruction_prompt

MASK_DISPLAY = "▢"

# How many blocks get prefilled per forward call. Bigger than 1 (unlike the
# denoising loop, which must do one real block at a time) trades a bounded,
# still-small attention_mask for far fewer forward calls at long context —
# see _run_denoising's prefill comment for the exact size/correctness
# argument. Keep the cap in the same 1024-token problem-size unit used by the
# benchmark's diffusion comparison: with the checkpoint's block_length=32,
# 32 blocks make one 1024-token prefill chunk. Shorter benchmark questions
# naturally use one smaller chunk containing only their actual full blocks;
# long RULER inputs are split into consecutive 1024-token chunks. At the
# 32768-token RULER point the largest bounded mask is therefore roughly
# (1024, 32768), about 128 MiB in float32, rather than the source's multi-GB
# full-sequence square mask.
_PREFILL_CHUNK_BLOCKS = 32


class DreamReasonerAdapter(HFDiffusionAdapter):
    """Appendix D.2. Best: block_length=32, steps_per_block=32 (1 token/step,
    the library's own default step count for one block). Fast: block_length=32,
    steps_per_block=16 (2 tokens/step) — mirrors iLLaDA's Best/Fast split
    exactly, since DreamReasoner's own README/model card gives no
    steps_per_block guidance beyond the library default this benchmark keeps
    for Best."""

    def __init__(
        self,
        model_name_or_path: str,
        step_config: DiffusionStepConfig,
        config_name: str,
        device: str | None = None,
    ) -> None:
        super().__init__(model_name_or_path, step_config, name="dreamreasoner", config_name=config_name, device=device)
        self._cpu_offloaded = False

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device

        def _load():
            return self._load_model_and_tokenizer(device, device_map_auto=False)

        # This can be a cache *hit* on a model a sibling variant (e.g. `best`,
        # if `fast` is what's loading here) already reloaded with offloading
        # after its own OOM (see `_reload_with_cpu_offload`) — check the
        # actual model, don't assume a fresh, 100%-GPU load.
        self._tokenizer, self._model = get_or_load(self._model_name, device, _load)
        self._cpu_offloaded_bytes = offloaded_parameter_bytes(self._model)
        self._cpu_offloaded = self._cpu_offloaded_bytes > 0

    def _load_model_and_tokenizer(self, device: str, *, device_map_auto: bool):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # The design doc/README's quick-start explicitly uses
        # AutoModelForCausalLM (config.json's auto_map also registers
        # AutoModel for the same class, but this adapter doesn't rely on
        # that alias).
        tokenizer = AutoTokenizer.from_pretrained(self._model_name, trust_remote_code=True)
        kwargs: dict = dict(trust_remote_code=True, torch_dtype=torch.bfloat16)
        if device_map_auto:
            kwargs["device_map"] = "auto"
            max_memory = cpu_offload_max_memory(device)
            if max_memory is not None:
                kwargs["max_memory"] = max_memory
        else:
            kwargs["low_cpu_mem_usage"] = True
        model = AutoModelForCausalLM.from_pretrained(self._model_name, **kwargs)
        if not device_map_auto:
            model.to(device)
        model.eval()
        return tokenizer, model

    def _reload_with_cpu_offload(self) -> bool:
        """GPU-only by default (matches every other model in this benchmark)
        — this only ever runs reactively, after `models/base.py`'s
        `generate()` has already tried a plain retry and a cache-cleared
        retry and *still* hit a real CUDA OOM. At RULER's own advertised max
        context (32768) this model's weights + KV cache alone leave almost
        no headroom on a 24 GiB GPU — confirmed from a real OOM ("23.00 GiB
        is allocated... 38.45 MiB reserved but unallocated": no fragmentation
        slack, a genuine capacity ceiling the mask/prefill-chunk fixes
        elsewhere in this file can't do anything further about). Falling
        back to `device_map="auto"` here lets `accelerate` offload part of
        the model to CPU so the sample can still *complete* with a real
        accuracy/quality score (task correctness is unaffected — same
        computation, just some of it on CPU) — see
        `models/base.py`'s `_reload_with_cpu_offload` docstring for how
        `self._cpu_offloaded`/`self._cpu_offloaded_bytes` get surfaced into
        every affected sample's own result, not just a single per-run flag.
        `model_cache.reload_with_offload` replaces the *shared* cache entry,
        so `fast` (if `best` is the one that actually hit the OOM) sees the
        already-offloaded model on its own next `_ensure_loaded()` call
        instead of separately hitting the same OOM itself. Run RULER as its
        own filtered invocation (`--dataset ruler`) to keep this from
        leaking into an otherwise-clean dataset that never needed it —
        once triggered, this stays offloaded for the rest of this process."""
        if self._cpu_offloaded:
            return True  # already offloaded from an earlier sample
        if self._model is None:
            return False  # never loaded at all yet — not this method's job

        def _load_with_offload():
            return self._load_model_and_tokenizer(self._device, device_map_auto=True)

        def _release_current() -> None:
            self._model = None
            self._tokenizer = None

        self._tokenizer, self._model = reload_with_offload(
            self._model_name,
            self._device,
            _load_with_offload,
            release_current=_release_current,
        )
        # Evicting the old (100%-GPU) copy before this reload frees its
        # memory first, so it's possible `device_map="auto"` finds enough
        # room to place everything back on the GPU after all — only flag
        # this run as offloaded if real parameter/buffer bytes actually
        # ended up off-GPU, not just because `device_map="auto"` was asked
        # for. Either way, the reload itself succeeded, so the caller should
        # still retry.
        self._cpu_offloaded_bytes = offloaded_parameter_bytes(self._model)
        self._cpu_offloaded = self._cpu_offloaded_bytes > 0
        return True

    def _resolve_mask_token_id(self, step_config: DiffusionStepConfig) -> int:
        candidates = (
            step_config.extra.get("mask_token_id"),
            # The checkpoint's own config.json ships this directly
            # (confirmed: mask_token_id=151669) — unlike regular Dream-7B,
            # where no single source reliably had it. Double-`getattr`
            # (not `self._model.config.mask_token_id`) so a model object
            # with no `.config` at all falls through instead of raising.
            getattr(getattr(self._model, "config", None), "mask_token_id", None),
            getattr(self._tokenizer, "mask_token_id", None),
        )
        for candidate in candidates:
            if candidate is not None:
                return candidate
        raise ValueError(
            "could not determine DreamReasoner's mask_token_id from the checkpoint's "
            "config.json or tokenizer; set step_config.extra['mask_token_id'] explicitly "
            "(see configs/models/dreamreasoner.yaml) — the real checkpoint ships "
            "mask_token_id=151669 directly, so this should only happen against a "
            "different/renamed checkpoint"
        )

    def _run_denoising(
        self, prompt: str, step_config: DiffusionStepConfig
    ) -> tuple[str, list[TraceStep], int]:
        import torch
        import torch.nn.functional as F
        from transformers.cache_utils import DynamicCache

        block_length = step_config.block_length or getattr(getattr(self._model, "config", None), "block_size", 32)
        gen_length = step_config.gen_length
        remasking_strategy = step_config.extra.get("remasking_strategy", "low_confidence_dynamic")
        # Real library default: 1 step for 'low_confidence_static', else
        # `block_length` — confirmed from source, ported exactly.
        denoising_steps = step_config.steps_per_block or (
            1 if remasking_strategy == "low_confidence_static" else block_length
        )
        temperature = float(step_config.extra.get("temperature", 0.0))
        top_k = int(step_config.extra.get("top_k", 0))
        top_p = float(step_config.extra.get("top_p", 1.0))
        confidence_threshold = float(step_config.extra.get("confidence_threshold", 0.9))
        eb_threshold = step_config.extra.get("eb_threshold", 0.35)
        mask_token_id = self._resolve_mask_token_id(step_config)

        device = self._device
        input_ids = tokenize_instruction_prompt(
            self._tokenizer, prompt, device=device
        )["input_ids"]
        prompt_len = input_ids.shape[1]
        self._start_measurement()

        num_blocks = max(1, math.ceil((prompt_len + gen_length) / block_length))
        total_length = num_blocks * block_length

        position_ids = torch.arange(total_length, device=device, dtype=torch.long).unsqueeze(0)

        x = torch.full((1, total_length), mask_token_id, dtype=torch.long, device=device)
        x[:, :prompt_len] = input_ids

        prefill_blocks = prompt_len // block_length
        past_key_values = DynamicCache()

        # Prefill in chunks of up to _PREFILL_CHUNK_BLOCKS blocks per call
        # (store_kv=True) instead of one block per call — fewer, bigger
        # forward calls means less per-call overhead (kernel launches, cache
        # bookkeeping) at long context, where the single-block version could
        # mean 1000+ tiny calls. A chunk spanning more than one block needs a
        # real (but bounded) attention_mask: within the chunk's own new
        # tokens, block i must still only see block j<=i (same block-tril
        # rule as the source's own mask, just scoped to one chunk instead of
        # the whole sequence) — already-cached earlier chunks are fully
        # visible to every row (concatenated in by `past_key_values.update`),
        # so the mask only needs an explicit 0 pattern over the chunk's own
        # tokens, sized (chunk_length, chunk_length), never
        # (total_length, total_length). A single-block chunk still needs no
        # mask at all (nothing "later" to wrongly become visible within it).
        chunk_index = 0
        while chunk_index < prefill_blocks:
            blocks_in_chunk = min(_PREFILL_CHUNK_BLOCKS, prefill_blocks - chunk_index)
            chunk_start = chunk_index * block_length
            chunk_length = blocks_in_chunk * block_length
            chunk_end = chunk_start + chunk_length

            attention_mask = None
            if blocks_in_chunk > 1:
                intra_chunk_tril = torch.tril(torch.ones(blocks_in_chunk, blocks_in_chunk, device=device))
                intra_chunk_mask = intra_chunk_tril.repeat_interleave(block_length, dim=0).repeat_interleave(
                    block_length, dim=1
                )
                if chunk_start > 0:
                    prefix_visible = torch.ones(chunk_length, chunk_start, device=device)
                    intra_chunk_mask = torch.cat([prefix_visible, intra_chunk_mask], dim=1)
                attention_mask = intra_chunk_mask.unsqueeze(0)

            self._model(
                x[:, chunk_start:chunk_end],
                attention_mask=attention_mask,
                position_ids=position_ids[:, chunk_start:chunk_end],
                past_key_values=past_key_values,
                use_cache=True,
                store_kv=True,
            )
            chunk_index += blocks_in_chunk

        num_transfer_tokens = _get_num_transfer_tokens(block_length, denoising_steps)
        trace: list[TraceStep] = []
        global_step = 0

        for num_block in range(prefill_blocks, num_blocks):
            block_start = num_block * block_length
            block_end = block_start + block_length
            cur_x = x[:, block_start:block_end].clone()
            cur_pos = position_ids[:, block_start:block_end]

            for step in range(denoising_steps + 1):
                mask_index = cur_x == mask_token_id
                if not mask_index.any():
                    # Block already fully committed (via the previous step's
                    # force_accept) — one clean forward to push its final
                    # tokens into the KV cache before the next block attends
                    # to them. No positions change, so no TraceStep.
                    self._model(
                        cur_x,
                        position_ids=cur_pos,
                        past_key_values=past_key_values,
                        use_cache=True,
                        store_kv=True,
                    )
                    break

                force_accept = step == denoising_steps - 1
                logits = self._model(
                    cur_x,
                    position_ids=cur_pos,
                    past_key_values=past_key_values,
                    use_cache=True,
                    store_kv=False,
                ).logits

                x0, x0_p = _sample_with_temperature_topk_topp(logits, temperature, top_k, top_p)
                x0 = torch.where(mask_index, x0, cur_x)
                transfer_index = _select_transfer_index(
                    remasking_strategy,
                    mask_index,
                    x0,
                    x0_p,
                    num_transfer_tokens,
                    step,
                    confidence_threshold,
                    eb_threshold,
                    force_accept=force_accept,
                )
                cur_x[transfer_index] = x0[transfer_index]
                x[:, block_start:block_end] = cur_x

                if self._trace_instrumentation_enabled():
                    with self._exclude_from_measurement():
                        transfer_index_full = torch.zeros_like(x, dtype=torch.bool)
                        transfer_index_full[:, block_start:block_end] = transfer_index
                        probs = F.softmax(logits.float(), dim=-1)
                        trace.append(
                            _build_trace_step(
                                forward_index=global_step,
                                x=x,
                                mask_token_id=mask_token_id,
                                transfer_index_full=transfer_index_full,
                                block_start=block_start,
                                probs=probs,
                                x0_p=x0_p,
                                prompt_len=prompt_len,
                                gen_length=gen_length,
                                tokenizer=self._tokenizer,
                            )
                        )
                global_step += 1

            x[:, block_start:block_end] = cur_x

        self._stop_measurement()
        self._last_num_forward_passes = global_step
        output_length = min(total_length, prompt_len + gen_length)
        final_ids = x[0, prompt_len:output_length].tolist()
        output_text = self._tokenizer.decode(final_ids, skip_special_tokens=True)
        return output_text, trace, len(final_ids)


def _top_k_logits(logits, k: int):
    if k <= 0:
        return logits
    import torch

    values, _ = torch.topk(logits, k)
    min_values = values[..., -1, None]
    return torch.where(logits < min_values, torch.full_like(logits, float("-inf")), logits)


def _top_p_logits(logits, p: float):
    import torch
    import torch.nn.functional as F

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_mask = cumulative_probs > p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False
    mask_indices = torch.scatter(
        torch.full_like(logits, False, dtype=torch.bool), -1, sorted_indices, sorted_mask
    )
    return logits.masked_fill(mask_indices, float("-inf"))


def _sample_with_temperature_topk_topp(logits, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0):
    """Ported exactly from the real source. This adjusted distribution (not
    raw-logit softmax) is what confidence-based remasking strategies select
    on — see this module's docstring for why the trace's own entropy/top1
    figures are computed separately, from raw logits."""
    import torch
    import torch.nn.functional as F

    orig_shape = logits.shape[:-1]
    vocab_size = logits.shape[-1]
    logits = logits.reshape(-1, vocab_size)

    if temperature > 0:
        logits = logits / temperature
    if top_k > 0:
        logits = _top_k_logits(logits, top_k)
    if top_p < 1.0:
        logits = _top_p_logits(logits, top_p)

    probs = F.softmax(logits, dim=-1)
    if temperature > 0:
        token = torch.multinomial(probs, num_samples=1)
    else:
        token = probs.argmax(dim=-1, keepdim=True)
    token_prob = torch.gather(probs, -1, token)
    return token.view(*orig_shape), token_prob.view(*orig_shape)


def _get_num_transfer_tokens(block_length: int, steps: int):
    """How many positions to commit at each of `steps` steps within one
    block, spreading `block_length` masked positions as evenly as possible —
    ported exactly (matches iLLaDA's `_transfer_schedule`, same underlying
    idea, kept as a tensor here to match the real source's own shape)."""
    import torch

    base = block_length // steps
    remainder = block_length % steps
    num_transfer_tokens = torch.zeros(steps, dtype=torch.int64) + base
    num_transfer_tokens[:remainder] += 1
    return num_transfer_tokens


def _select_transfer_index(
    remasking_strategy: str,
    mask_index,
    x0,
    x0_p,
    num_transfer_tokens,
    step: int,
    confidence_threshold: float,
    eb_threshold: float | None,
    *,
    force_accept: bool = False,
):
    """Ported exactly from the real `_select_transfer_index` — dispatches on
    `remasking_strategy` (this benchmark keeps the library's own default,
    `'low_confidence_dynamic'`, unless overridden via
    `step_config.extra['remasking_strategy']`)."""
    import torch

    if force_accept:
        return mask_index.clone()

    if remasking_strategy == "sequential":
        transfer_index = torch.zeros_like(x0, dtype=torch.bool)
        for j in range(x0.shape[0]):
            if not mask_index[j].any():
                continue
            first_mask_index = mask_index[j].nonzero(as_tuple=True)[0].min().item()
            end = first_mask_index + int(num_transfer_tokens[step].item())
            transfer_index[j, first_mask_index:end] = True
        return transfer_index

    if remasking_strategy == "low_confidence_static":
        confidence = torch.where(mask_index, x0_p, -torch.inf)
        transfer_index = torch.zeros_like(x0, dtype=torch.bool)
        k = max(1, int(num_transfer_tokens[step].item()))
        for j in range(confidence.shape[0]):
            _, idx = torch.topk(confidence[j], k)
            transfer_index[j, idx] = True
        return transfer_index

    if remasking_strategy == "low_confidence_dynamic":
        confidence = torch.where(mask_index, x0_p, -torch.inf)
        transfer_index = torch.zeros_like(x0, dtype=torch.bool)
        k = max(1, int(num_transfer_tokens[step].item()))
        for j in range(confidence.shape[0]):
            high_conf_mask = confidence[j] > confidence_threshold
            if int(high_conf_mask.sum().item()) >= k:
                transfer_index[j] = high_conf_mask
            else:
                _, idx = torch.topk(confidence[j], k)
                transfer_index[j, idx] = True
        return transfer_index

    if remasking_strategy == "entropy_bounded":
        if eb_threshold is None:
            raise ValueError("eb_threshold is required for entropy_bounded remasking.")
        eps = 1e-12
        entropies = -(x0_p.clamp_min(eps) * x0_p.clamp_min(eps).log())
        entropies = torch.where(mask_index, entropies, torch.inf)
        ent_sorted, order = torch.sort(entropies, dim=1, descending=False)
        cumsum = torch.cumsum(ent_sorted, dim=1)
        transfer_index = torch.zeros_like(x0, dtype=torch.bool)
        for j in range(x0_p.shape[0]):
            k = torch.searchsorted(cumsum[j], torch.tensor(eb_threshold, device=x0_p.device), right=False).item()
            k = max(1, min(k, int(mask_index[j].sum().item())))
            transfer_index[j, order[j, :k]] = True
        return transfer_index

    raise ValueError(f"Unknown remasking strategy: {remasking_strategy}")


def _build_trace_step(
    forward_index: int,
    x,
    mask_token_id: int,
    transfer_index_full,
    block_start: int,
    probs,
    x0_p,
    prompt_len: int,
    gen_length: int,
    tokenizer,
) -> TraceStep:
    mask_index_now = x == mask_token_id
    gen_slice = slice(prompt_len, prompt_len + gen_length)
    block_length = probs.shape[1]
    block_end = block_start + block_length

    gen_token_ids = x[0, gen_slice].tolist()
    gen_masked_now = mask_index_now[0, gen_slice].tolist()
    committed_local = sorted(
        p - prompt_len
        for p in transfer_index_full.nonzero(as_tuple=True)[1].tolist()
        if prompt_len <= p < prompt_len + gen_length
    )

    position_states = [
        PositionState.MASKED if gen_masked_now[i] else PositionState.ACCEPTED for i in range(gen_length)
    ]
    remaining_positions = [i for i, state in enumerate(position_states) if state == PositionState.MASKED]

    vocab_size = probs.shape[-1]
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)

    # Only the currently active block has any logits this step — future
    # blocks (still fully masked, not yet part of any forward call) get no
    # entry, matching what the model actually computed (see module
    # docstring).
    entropy_by_position: dict[int, float] = {}
    top1_confidence_by_position: dict[int, float] = {}
    for i in remaining_positions:
        global_pos = prompt_len + i
        if block_start <= global_pos < block_end:
            local_pos = global_pos - block_start
            entropy_by_position[i] = entropy[0, local_pos].item() / math.log(vocab_size)
            top1_confidence_by_position[i] = x0_p[0, local_pos].item()

    token_texts = [
        tokenizer.decode([gen_token_ids[i]]) if position_states[i] == PositionState.ACCEPTED else MASK_DISPLAY
        for i in range(gen_length)
    ]

    return TraceStep(
        forward_index=forward_index,
        token_ids=list(gen_token_ids),
        position_states=position_states,
        committed_positions=committed_local,
        decoded_text="".join(token_texts),
        entropy_by_position=entropy_by_position,
        top1_confidence_by_position=top1_confidence_by_position,
        token_texts=token_texts,
    )
