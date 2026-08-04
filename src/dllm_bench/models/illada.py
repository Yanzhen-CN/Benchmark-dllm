"""iLLaDA (Appendix D.1): block-wise, semi-autoregressive, confidence-based
unmasking.

Ported from `GSAI-ML/iLLaDA-8B-Instruct`'s own reference sampler (verified
against the official `ML-GSAI/LLaDA` implementation's `generate.py` — see
``models/hf_diffusion.py``'s module docstring for what's shared with
DreamReasoner vs. specific to this model). iLLaDA reuses LLaDA's inference code with
**`mask_id=5`** instead of LLaDA's own default `126336` — the reference
project's README states this explicitly ("By setting mask_id=5, iLLaDA can
directly reuse the existing LLaDA inference code"); getting this constant
wrong silently corrupts every mask/unmask decision, so it's hardcoded below
rather than left as a guessable config field.

Only the reference's *default* behavior is ported: `low_confidence`
remasking (top-k by predicted probability, restricted to the active block),
a uniform per-block step count, and gumbel-noise sampling. The reference
implementation also has research-ablation knobs this benchmark doesn't
expose (`token_selection_confidence_threshold`, custom per-block step
schedules, `decode_order='left_to_right'`) — those aren't part of the
P1/P2 configs this benchmark actually runs (Appendix D.1), so they're
deliberately not ported; add them to `step_config.extra` + this file's
`_run_denoising` if a future config needs them.
"""

from __future__ import annotations

import math

from ..interfaces import PositionState, TraceStep
from .hf_diffusion import (
    DiffusionStepConfig,
    HFDiffusionAdapter,
    decode_generated_ids_until_eos,
)
from .prompting import tokenize_instruction_prompt

MASK_ID = 5
MASK_DISPLAY = "▢"


class IlladaAdapter(HFDiffusionAdapter):
    """Appendix D.1. P1: 32 steps/block; P2: 16 steps/block."""

    def __init__(
        self,
        model_name_or_path: str,
        step_config: DiffusionStepConfig,
        config_name: str,
        device: str | None = None,
    ) -> None:
        super().__init__(model_name_or_path, step_config, name="illada", config_name=config_name, device=device)

    def _run_denoising(
        self,
        prompt: str,
        step_config: DiffusionStepConfig,
        target_input_tokens: int | None = None,
    ) -> tuple[str, list[TraceStep], int]:
        import torch

        block_length = step_config.block_length or step_config.gen_length
        steps_per_block = step_config.steps_per_block or 1
        temperature = float(step_config.extra.get("temperature", 0.0))
        remasking = step_config.extra.get("remasking", "low_confidence")

        gen_length = step_config.gen_length
        num_blocks = max(1, math.ceil(gen_length / block_length))
        padded_gen_length = num_blocks * block_length

        device = self._device
        input_ids = tokenize_instruction_prompt(
            self._tokenizer,
            prompt,
            device=device,
            target_input_tokens=target_input_tokens,
        )["input_ids"]
        prompt_len = input_ids.shape[1]
        self._last_input_tokens = int(prompt_len)
        self._start_measurement()

        x = torch.full((1, prompt_len + padded_gen_length), MASK_ID, dtype=torch.long, device=device)
        x[:, :prompt_len] = input_ids
        attention_mask = torch.ones_like(x)

        trace: list[TraceStep] = []
        global_step = 0

        for block_index in range(num_blocks):
            block_start = prompt_len + block_index * block_length
            block_end = block_start + block_length
            block_mask_count = int((x[:, block_start:block_end] == MASK_ID).sum().item())
            transfer_schedule = _transfer_schedule(block_mask_count, steps_per_block)

            for step_in_block in range(steps_per_block):
                mask_index = x == MASK_ID
                if not mask_index[:, block_start:block_end].any():
                    break  # block already fully committed (schedule exhausted early)

                with torch.no_grad():
                    logits = self._model(x, attention_mask=attention_mask).logits

                logits_for_pick = _add_gumbel_noise(logits, temperature)
                x0 = torch.argmax(logits_for_pick, dim=-1)
                probs, argmax_prob = _selected_token_probabilities(logits, x0)
                # Real predicted probability of the picked token — used for the
                # trace's certainty data regardless of remasking mode.

                if remasking == "random":
                    selection_score = torch.rand(x0.shape, device=device)
                else:
                    selection_score = argmax_prob

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, selection_score, torch.full_like(selection_score, -math.inf))
                confidence[:, block_end:] = -math.inf  # future blocks ineligible this step

                remaining_in_block = int(mask_index[:, block_start:block_end].sum().item())
                k = min(transfer_schedule[step_in_block], remaining_in_block)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                if k > 0:
                    _, select_index = torch.topk(confidence[0], k=k)
                    transfer_index[0, select_index] = True

                x[transfer_index] = x0[transfer_index]
                self._annotate_last_forward(
                    accepted_tokens=k,
                    active_tokens=block_end - block_start,
                    eligible_tokens=remaining_in_block,
                )

                if self._trace_instrumentation_enabled():
                    with self._exclude_from_measurement():
                        trace.append(
                            _build_trace_step(
                                forward_index=global_step,
                                x=x,
                                transfer_index=transfer_index,
                                probs=probs,
                                argmax_prob=argmax_prob,
                                prompt_len=prompt_len,
                                gen_length=gen_length,
                                tokenizer=self._tokenizer,
                            )
                        )
                global_step += 1

        self._stop_measurement()
        self._last_num_forward_passes = global_step
        final_ids = x[0, prompt_len : prompt_len + gen_length].tolist()
        output_text, final_valid_length, eos_token_id = (
            decode_generated_ids_until_eos(self._tokenizer, final_ids)
        )
        if eos_token_id is not None:
            self._last_stop_metadata = {
                "stop_reason": "eos",
                "stop_token_id": eos_token_id,
                "stop_position": final_valid_length,
            }
        return output_text, trace, final_valid_length


def _add_gumbel_noise(logits, temperature: float):
    """Exact formula from the reference implementation — not a standard
    Gumbel-softmax. At temperature=0 (this benchmark's P1/P2 default)
    it's a no-op: plain argmax over raw logits, fully deterministic."""
    if temperature == 0:
        return logits
    import torch

    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def _selected_token_probabilities(logits, token_ids):
    """Return full-softmax probabilities and the selected-token confidence.

    Keeping this as the sampler's single implementation point lets the
    algorithm-level test validate the exact confidence path used for token
    transfer, rather than a disconnected copy of the same formula.
    """
    import torch
    import torch.nn.functional as F

    probs = F.softmax(logits, dim=-1)
    selected = torch.gather(
        probs,
        dim=-1,
        index=token_ids.unsqueeze(-1),
    ).squeeze(-1)
    return probs, selected


def _transfer_schedule(mask_count: int, steps: int) -> list[int]:
    """How many positions to commit at each of `steps` steps within one
    block, spreading `mask_count` as evenly as possible (matches the
    reference `get_num_transfer_tokens`, specialized to batch size 1)."""
    if steps <= 0:
        return []
    base, remainder = divmod(mask_count, steps)
    return [base + 1 if i < remainder else base for i in range(steps)]


def _build_trace_step(
    forward_index: int,
    x,
    transfer_index,
    probs,
    argmax_prob,
    prompt_len: int,
    gen_length: int,
    tokenizer,
) -> TraceStep:
    import torch

    mask_index_now = x == MASK_ID
    gen_slice = slice(prompt_len, prompt_len + gen_length)

    gen_token_ids = x[0, gen_slice].tolist()
    gen_masked_now = mask_index_now[0, gen_slice].tolist()
    committed_local = sorted(
        p - prompt_len
        for p in transfer_index.nonzero(as_tuple=True)[1].tolist()
        if prompt_len <= p < prompt_len + gen_length
    )

    position_states = [
        PositionState.MASKED if gen_masked_now[i] else PositionState.ACCEPTED for i in range(gen_length)
    ]
    remaining_positions = [i for i, state in enumerate(position_states) if state == PositionState.MASKED]

    vocab_size = probs.shape[-1]
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
    gen_entropy = entropy[0, gen_slice].tolist()
    gen_top1 = argmax_prob[0, gen_slice].tolist()

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
        entropy_by_position={i: gen_entropy[i] / math.log(vocab_size) for i in remaining_positions},
        top1_confidence_by_position={i: gen_top1[i] for i in remaining_positions},
        token_texts=token_texts,
    )
