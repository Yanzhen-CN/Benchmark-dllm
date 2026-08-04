"""iLLaDA using the official incremental ``var_generate`` canvas policy.

Unlike the fixed-canvas reference path, only the active block is appended to
the already committed prefix. Future blocks do not yet exist and therefore
cannot contribute mask embeddings to the current bidirectional forward pass.
The token-transfer rule, BF16 checkpoint, mask id and P1/P2 step schedules
remain identical to the existing iLLaDA adapter.
"""

from __future__ import annotations

import math

from ..interfaces import PositionState, TraceStep
from .hf_diffusion import (
    DiffusionStepConfig,
    HFDiffusionAdapter,
    decode_generated_ids_until_eos,
    first_eos_position,
)
from .illada import (
    MASK_DISPLAY,
    MASK_ID,
    _add_gumbel_noise,
    _selected_token_probabilities,
    _transfer_schedule,
)
from .prompting import tokenize_instruction_prompt


class IlladaVarGenAdapter(HFDiffusionAdapter):
    """Official per-block variable-canvas execution with benchmark tracing."""

    # Official var_generate only accepts complete blocks. The suite's generic
    # 8-token warmup is therefore invalid for this block_length=32 adapter.
    warmup_new_tokens = 32

    def __init__(
        self,
        model_name_or_path: str,
        step_config: DiffusionStepConfig,
        config_name: str,
        device: str | None = None,
    ) -> None:
        super().__init__(
            model_name_or_path,
            step_config,
            name="illada_vargen",
            config_name=config_name,
            device=device,
        )
        self.inference_optimizations = ["official-var-generate-incremental-canvas"]
        self.execution_path = "official-var-generate"
        self.trace_source = "instrumented-official-var-generate-loop"

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
        if gen_length % block_length != 0:
            raise ValueError(
                "official iLLaDA var_generate requires gen_length to be "
                "divisible by block_length"
            )

        with self._profile_stage("input_preparation"):
            input_ids = tokenize_instruction_prompt(
                self._tokenizer,
                prompt,
                device=self._device,
                target_input_tokens=target_input_tokens,
            )["input_ids"]
        initial_prompt_len = int(input_ids.shape[1])
        self._last_input_tokens = initial_prompt_len
        trace: list[TraceStep] = []
        global_step = 0
        generated_so_far = 0

        self._start_measurement()
        x = input_ids.clone()
        while generated_so_far < gen_length:
            active_length = min(block_length, gen_length - generated_so_far)
            active_masks = torch.full(
                (x.shape[0], active_length),
                MASK_ID,
                dtype=torch.long,
                device=self._device,
            )
            x = torch.cat([x, active_masks], dim=1)
            block_start = initial_prompt_len + generated_so_far
            block_end = block_start + active_length
            transfer_schedule = _transfer_schedule(active_length, steps_per_block)

            for step_in_block in range(steps_per_block):
                mask_index = x == MASK_ID
                if not mask_index[:, block_start:block_end].any():
                    break

                with self._profile_stage("denoise_step"):
                    with torch.no_grad():
                        # Official var_generate calls generate() without an
                        # attention mask for its single, unpadded prompt.
                        logits = self._model(x).logits

                with self._profile_stage("token_selection"):
                    logits_for_pick = _add_gumbel_noise(logits, temperature)
                    x0 = torch.argmax(logits_for_pick, dim=-1)
                    probs, argmax_prob = _selected_token_probabilities(logits, x0)
                if remasking == "low_confidence":
                    selection_score = argmax_prob
                elif remasking == "random":
                    selection_score = torch.rand(x0.shape, device=self._device)
                else:
                    raise NotImplementedError(remasking)
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(
                    mask_index,
                    selection_score,
                    torch.full_like(selection_score, -math.inf),
                )
                confidence[:, :block_start] = -math.inf

                remaining = int(mask_index[:, block_start:block_end].sum().item())
                k = min(transfer_schedule[step_in_block], remaining)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                if k > 0:
                    _, selected = torch.topk(confidence[0], k=k)
                    transfer_index[0, selected] = True
                with self._profile_stage("canvas_update"):
                    x[transfer_index] = x0[transfer_index]
                self._annotate_last_forward(
                    accepted_tokens=k,
                    active_tokens=active_length,
                    eligible_tokens=remaining,
                )

                if self._trace_instrumentation_enabled():
                    with self._exclude_from_measurement():
                        trace.append(
                            _build_variable_trace_step(
                                forward_index=global_step,
                                x=x,
                                transfer_index=transfer_index,
                                probs=probs,
                                argmax_prob=argmax_prob,
                                prompt_len=initial_prompt_len,
                                tokenizer=self._tokenizer,
                            )
                        )
                global_step += 1

            generated_so_far += active_length
            generated_ids = x[
                0, initial_prompt_len : initial_prompt_len + generated_so_far
            ].tolist()
            if first_eos_position(self._tokenizer, generated_ids) is not None:
                # Match official var_generate: finish the active block, then
                # skip all later blocks once EOS is present.
                break

        self._stop_measurement()
        self._last_num_forward_passes = global_step
        final_ids = x[0, initial_prompt_len:].tolist()
        with self._profile_stage("output_decode"):
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


def _build_variable_trace_step(
    *,
    forward_index: int,
    x,
    transfer_index,
    probs,
    argmax_prob,
    prompt_len: int,
    tokenizer,
) -> TraceStep:
    """Record only the canvas that actually exists at this forward pass."""
    current_ids = x[0, prompt_len:].tolist()
    current_masked = (x[0, prompt_len:] == MASK_ID).tolist()
    token_ids = current_ids
    position_states = [
        PositionState.MASKED if masked else PositionState.ACCEPTED
        for masked in current_masked
    ]
    committed_positions = sorted(
        position - prompt_len
        for position in transfer_index.nonzero(as_tuple=True)[1].tolist()
        if position >= prompt_len
    )

    vocab_size = probs.shape[-1]
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
    active_entropy = entropy[0, prompt_len:].tolist()
    active_top1 = argmax_prob[0, prompt_len:].tolist()
    remaining_allocated = [
        index for index, masked in enumerate(current_masked) if masked
    ]
    token_texts = [
        MASK_DISPLAY if state is PositionState.MASKED else tokenizer.decode([token_id])
        for token_id, state in zip(token_ids, position_states)
    ]

    return TraceStep(
        forward_index=forward_index,
        token_ids=token_ids,
        position_states=position_states,
        committed_positions=committed_positions,
        decoded_text="".join(token_texts),
        entropy_by_position={
            index: active_entropy[index] / math.log(vocab_size)
            for index in remaining_allocated
        },
        top1_confidence_by_position={
            index: active_top1[index] for index in remaining_allocated
        },
        token_texts=token_texts,
    )
