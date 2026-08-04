"""iLLaDA with DiffusionGemma's entropy-bound acceptance rule.

This is an intentionally narrow hybrid. It keeps iLLaDA's checkpoint,
tokenizer, absorbing-mask canvas, deterministic argmax proposal, block size,
and P2 forward budget. Only the fixed transfer schedule is replaced by the
cumulative entropy-bound rule used by DiffusionGemma's official sampler::

    cumsum(sorted_entropy) - sorted_entropy <= entropy_bound

DiffusionGemma-specific self-conditioning and random-token renoising are not
compatible with iLLaDA's model interface/noise process and are not used.
Unaccepted positions therefore remain ``MASK_ID``. On the final budgeted
forward, unresolved positions are finalized from the latest argmax canvas,
mirroring DiffusionGemma's use of its final argmax canvas as block output.
"""

from __future__ import annotations

import math

from ..interfaces import PositionState, TraceStep
from .hf_diffusion import (
    DiffusionStepConfig,
    decode_generated_ids_until_eos,
)
from .illada import IlladaAdapter, MASK_DISPLAY, MASK_ID
from .prompting import tokenize_instruction_prompt


class IlladaEntropyAdapter(IlladaAdapter):
    """iLLaDA P2 compute budget with adaptive entropy-bound acceptance."""

    def __init__(
        self,
        model_name_or_path: str,
        step_config: DiffusionStepConfig,
        config_name: str,
        device: str | None = None,
    ) -> None:
        super().__init__(model_name_or_path, step_config, config_name, device)
        self.name = "illada_entropy"
        self.trace_source = "illada_mask_with_dg_entropy_bound"

    def _run_denoising(
        self,
        prompt: str,
        step_config: DiffusionStepConfig,
        target_input_tokens: int | None = None,
    ) -> tuple[str, list[TraceStep], int]:
        import torch

        block_length = step_config.block_length or step_config.gen_length
        steps_per_block = step_config.steps_per_block or 1
        entropy_bound = float(step_config.extra.get("entropy_bound", 0.1))
        if entropy_bound <= 0:
            raise ValueError("entropy_bound must be positive")

        gen_length = step_config.gen_length
        num_blocks = max(1, math.ceil(gen_length / block_length))
        padded_gen_length = num_blocks * block_length

        input_ids = tokenize_instruction_prompt(
            self._tokenizer,
            prompt,
            device=self._device,
            target_input_tokens=target_input_tokens,
        )["input_ids"]
        prompt_len = input_ids.shape[1]
        self._last_input_tokens = int(prompt_len)
        self._last_stop_metadata.update(
            {
                "sampler_family": "dg_entropy_bound",
                "entropy_bound": entropy_bound,
                "proposal_strategy": "illada_argmax",
                "renoise_strategy": "illada_mask",
                "self_conditioning": "not_used_incompatible_interface",
                "finalization": "latest_argmax_on_last_budgeted_forward",
            }
        )
        self._start_measurement()

        x = torch.full(
            (1, prompt_len + padded_gen_length),
            MASK_ID,
            dtype=torch.long,
            device=self._device,
        )
        x[:, :prompt_len] = input_ids
        attention_mask = torch.ones_like(x)

        trace: list[TraceStep] = []
        global_step = 0

        for block_index in range(num_blocks):
            block_start = prompt_len + block_index * block_length
            block_end = block_start + block_length

            for step_in_block in range(steps_per_block):
                eligible = x[:, block_start:block_end] == MASK_ID
                if not eligible.any():
                    break

                with torch.no_grad():
                    logits = self._model(x, attention_mask=attention_mask).logits

                block_logits = logits[:, block_start:block_end]
                block_probs = torch.softmax(block_logits, dim=-1, dtype=torch.float32)
                argmax_block = torch.argmax(block_logits, dim=-1)
                argmax_probability = torch.gather(
                    block_probs, dim=-1, index=argmax_block.unsqueeze(-1)
                ).squeeze(-1)
                token_entropy = -(
                    block_probs * block_probs.clamp_min(1e-12).log()
                ).sum(dim=-1)

                accepted_local = entropy_bound_acceptance_mask(
                    token_entropy,
                    eligible,
                    entropy_bound,
                )
                final_budgeted_step = step_in_block == steps_per_block - 1
                if final_budgeted_step:
                    # DG finalizes each canvas from the latest argmax rather
                    # than requiring every position to pass its acceptance
                    # mask. Do the same while preserving iLLaDA commitment.
                    accepted_local = eligible

                block = x[:, block_start:block_end]
                block[accepted_local] = argmax_block[accepted_local]
                accepted_count = int(accepted_local.sum().item())
                remaining_before = int(eligible.sum().item())
                self._annotate_last_forward(
                    accepted_tokens=accepted_count,
                    active_tokens=block_length,
                    eligible_tokens=remaining_before,
                )

                if self._trace_instrumentation_enabled():
                    with self._exclude_from_measurement():
                        trace.append(
                            _build_entropy_trace_step(
                                forward_index=global_step,
                                x=x,
                                accepted_local=accepted_local,
                                token_entropy=token_entropy,
                                argmax_probability=argmax_probability,
                                vocab_size=block_probs.shape[-1],
                                block_start=block_start,
                                block_end=block_end,
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
            self._last_stop_metadata.update(
                {
                    "stop_reason": "eos",
                    "stop_token_id": eos_token_id,
                    "stop_position": final_valid_length,
                }
            )
        return output_text, trace, final_valid_length


def entropy_bound_acceptance_mask(token_entropy, eligible_mask, entropy_bound: float):
    """Apply DiffusionGemma's official cumulative entropy-bound formula.

    The calculation is restricted to iLLaDA positions that are still masked.
    It is batch-safe even though the benchmark currently generates one sample
    at a time.
    """
    import torch

    if token_entropy.shape != eligible_mask.shape:
        raise ValueError("token_entropy and eligible_mask must have the same shape")
    if entropy_bound <= 0:
        raise ValueError("entropy_bound must be positive")

    eligible_mask = eligible_mask.to(dtype=torch.bool)
    sortable_entropy = torch.where(
        eligible_mask,
        token_entropy,
        torch.full_like(token_entropy, torch.inf),
    )
    sorted_entropy, sorted_indices = torch.sort(
        sortable_entropy, dim=-1, descending=False
    )
    sorted_eligible = torch.gather(eligible_mask, dim=-1, index=sorted_indices)
    finite_entropy = torch.where(
        sorted_eligible, sorted_entropy, torch.zeros_like(sorted_entropy)
    )
    cumulative_entropy = torch.cumsum(finite_entropy, dim=-1)
    sorted_selection = sorted_eligible & (
        cumulative_entropy - finite_entropy <= float(entropy_bound)
    )
    return torch.scatter(
        input=torch.zeros_like(sorted_selection),
        dim=-1,
        index=sorted_indices,
        src=sorted_selection,
    )


def _build_entropy_trace_step(
    forward_index: int,
    x,
    accepted_local,
    token_entropy,
    argmax_probability,
    vocab_size: int,
    block_start: int,
    block_end: int,
    prompt_len: int,
    gen_length: int,
    tokenizer,
) -> TraceStep:
    gen_slice = slice(prompt_len, prompt_len + gen_length)
    gen_token_ids = x[0, gen_slice].tolist()
    gen_masked = (x[0, gen_slice] == MASK_ID).tolist()
    position_states = [
        PositionState.MASKED if masked else PositionState.ACCEPTED
        for masked in gen_masked
    ]
    committed_positions = [
        block_start - prompt_len + local_position
        for local_position in accepted_local[0].nonzero(as_tuple=True)[0].tolist()
        if block_start + local_position < prompt_len + gen_length
    ]

    normalization = math.log(max(2, int(vocab_size)))
    active_local_start = block_start - prompt_len
    entropy_by_position = {}
    confidence_by_position = {}
    for local_position in range(block_end - block_start):
        global_position = active_local_start + local_position
        if global_position >= gen_length:
            break
        if position_states[global_position] == PositionState.MASKED:
            entropy_by_position[global_position] = float(
                token_entropy[0, local_position].item() / normalization
            )
            confidence_by_position[global_position] = float(
                argmax_probability[0, local_position].item()
            )

    token_texts = [
        tokenizer.decode([gen_token_ids[index]])
        if position_states[index] == PositionState.ACCEPTED
        else MASK_DISPLAY
        for index in range(gen_length)
    ]
    return TraceStep(
        forward_index=forward_index,
        token_ids=list(gen_token_ids),
        position_states=position_states,
        committed_positions=committed_positions,
        decoded_text="".join(token_texts),
        entropy_by_position=entropy_by_position,
        top1_confidence_by_position=confidence_by_position,
        token_texts=token_texts,
    )
