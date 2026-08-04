from __future__ import annotations

import math
from contextlib import contextmanager, nullcontext
from typing import Any

from .base import BaseModelAdapter
from ..interfaces import (
    EditingTraceStep,
    GenerationRequest,
    GenerationResult,
    PositionState,
    RunStatus,
    TraceStep,
)


class Llada21Adapter(BaseModelAdapter):
    """LLaDA2.1-mini adapter with an instrumented, intra-block T2T sampler."""

    name = "llada2_1"
    supports_trace = True
    deferred_measurement = True

    def __init__(self, model_name_or_path: str = "inclusionAI/LLaDA2.1-mini", revision: str | None = None,
                 config_name: str = "qmode",
                 device: str = "cuda", torch_dtype: str = "bfloat16", block_length: int = 32,
                 threshold: float = 0.7, editing_threshold: float = 0.5, max_post_steps: int = 16,
                 editing_enabled: bool = True, temperature: float = 0.0, mask_id: int = 156895,
                 eos_id: int = 156892, trust_remote_code: bool = True, **_: Any) -> None:
        super().__init__()
        self.config_name = config_name
        self.model_name_or_path = model_name_or_path
        self.revision = revision
        self.device = device
        self.torch_dtype = torch_dtype
        self.block_length = int(block_length)
        self.threshold = float(threshold)
        self.editing_threshold = float(editing_threshold)
        self.max_post_steps = int(max_post_steps)
        self.editing_enabled = bool(editing_enabled)
        self.temperature = float(temperature)
        self.mask_id = int(mask_id)
        self.eos_id = int(eos_id)
        self.trust_remote_code = bool(trust_remote_code)
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        common = {"revision": self.revision, "trust_remote_code": self.trust_remote_code}
        common = {key: value for key, value in common.items() if value is not None}
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, **common)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=getattr(torch, self.torch_dtype),
            low_cpu_mem_usage=True,
            **common,
        ).to(self.device)
        self._model.eval()

    @contextmanager
    def _capture_model_forwards(self, request, compute_handle=None):
        """Load before BaseModelAdapter installs optional per-forward hooks."""
        self._ensure_loaded()
        with super()._capture_model_forwards(request, compute_handle):
            yield

    def _token_text(self, token_id: int) -> str:
        return self._tokenizer.decode([int(token_id)], skip_special_tokens=False).strip()

    def _digit_token_ids(self, size: int) -> dict[str, int]:
        result: dict[str, int] = {}
        for digit in range(1, size + 1):
            text = str(digit)
            ids = self._tokenizer.encode(text, add_special_tokens=False)
            if len(ids) != 1 or self._token_text(ids[0]) != text:
                raise ValueError(f"Editable Sudoku requires one token per digit; {text!r} encoded as {ids}.")
            result[text] = int(ids[0])
        return result

    def _prompt_ids(self, prompt: str):
        import torch

        ids = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
        if not torch.is_tensor(ids):
            ids = torch.tensor([ids], dtype=torch.long)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        return ids.to(self.device)

    def _sample_predictions(self, logits):
        sampler = getattr(self._model, "_sample_with_temperature_topk_topp", None)
        if sampler is None:
            raise RuntimeError("The remote LLaDA2.1 model does not expose its official sampler helper.")
        return sampler(logits, self.temperature, 0, 1.0)

    def _generate_core(self, request: GenerationRequest) -> GenerationResult:
        import torch

        self._ensure_loaded()
        spec = dict(request.config.get("editable_sudoku") or {})
        answer_cells = int(spec.get("answer_cells", request.max_new_tokens))
        size = int(round(math.sqrt(answer_cells)))
        if size * size != answer_cells:
            raise ValueError(f"Editable Sudoku answer_cells must be square, got {answer_cells}.")
        digit_ids = self._digit_token_ids(size)
        capture_trace = bool(request.config.get("capture_trace", True))

        prompt_ids = self._prompt_ids(request.prompt)
        prompt_length = int(prompt_ids.shape[1])
        aligned_prompt_length = math.ceil(prompt_length / self.block_length) * self.block_length
        output_length = math.ceil(answer_cells / self.block_length) * self.block_length
        total_length = aligned_prompt_length + output_length
        x = torch.full((1, total_length), self.eos_id, dtype=torch.long, device=self.device)
        x[:, :prompt_length] = prompt_ids
        answer_start = aligned_prompt_length
        x[:, answer_start : answer_start + answer_cells] = self.mask_id

        immutable = torch.ones(total_length, dtype=torch.bool, device=self.device)
        immutable[answer_start : answer_start + answer_cells] = False
        for cell in {int(value) for value in spec.get("immutable_cells", [])}:
            immutable[answer_start + cell] = True
        seeded_grid = str(spec.get("seeded_grid", ""))
        if seeded_grid:
            if len(seeded_grid) != answer_cells:
                raise ValueError("editable_sudoku.seeded_grid length does not match answer_cells.")
            for cell, value in enumerate(seeded_grid):
                if value in digit_ids:
                    x[0, answer_start + cell] = digit_ids[value]
                elif value not in {"0", ".", "_"}:
                    raise ValueError(f"Unsupported seeded Sudoku value {value!r}.")

        block_count = total_length // self.block_length
        block_causal = torch.tril(torch.ones(block_count, block_count, dtype=torch.bool, device=self.device))
        attention_mask = block_causal.repeat_interleave(self.block_length, 0).repeat_interleave(
            self.block_length, 1
        ).unsqueeze(0)
        position_ids = torch.arange(total_length, device=self.device).unsqueeze(0)
        generic_trace: list[TraceStep] = []
        editing_trace: list[EditingTraceStep] = []
        forward_count = 0
        start_measurement = getattr(self, "_start_measurement", None)
        if callable(start_measurement):
            start_measurement()

        for output_block in range(output_length // self.block_length):
            block_start = answer_start + output_block * self.block_length
            block_end = block_start + self.block_length
            mapped = {local: output_block * self.block_length + local for local in range(self.block_length)
                      if output_block * self.block_length + local < answer_cells}
            post_step = 0
            block_steps: list[int] = []
            stop_reason = "block_complete"
            while True:
                old = x[0, block_start:block_end].clone()
                immutable_block = immutable[block_start:block_end]
                active_masks = old.eq(self.mask_id) & ~immutable_block
                if active_masks.any():
                    phase, current_post_step = "mask_filling", 0
                else:
                    if not self.editing_enabled or self.max_post_steps <= 0:
                        stop_reason = "editing_disabled" if not self.editing_enabled else "post_steps_disabled"
                        break
                    if post_step >= self.max_post_steps:
                        stop_reason = "max_post_steps"
                        break
                    post_step += 1
                    phase, current_post_step = "post_edit", post_step

                phase_context = getattr(self, "_forward_phase", None)
                context = phase_context(phase) if callable(phase_context) else nullcontext()
                with context, torch.no_grad():
                    model_output = self._model(
                        x[:, :block_end], attention_mask=attention_mask[:, :block_end, :block_end],
                        position_ids=position_ids[:, :block_end], output_attentions=False,
                    )
                    logits = model_output.logits[:, -self.block_length :, :]
                    predicted, confidence = self._sample_predictions(logits)
                predicted, confidence = predicted[0], confidence[0]
                mask_transfer = active_masks & confidence.gt(self.threshold)
                if active_masks.any() and not mask_transfer.any():
                    candidates = confidence.masked_fill(~active_masks, float("-inf"))
                    mask_transfer[candidates.argmax()] = True
                editable_candidates = old.ne(self.mask_id) & ~immutable_block
                editing_transfer = (editable_candidates & predicted.ne(old)
                                    & confidence.gt(self.editing_threshold) & self.editing_enabled)
                new = old.clone()
                transfer = mask_transfer | editing_transfer
                new[transfer] = predicted[transfer]
                x[0, block_start:block_end] = new
                forward_count += 1
                annotate = getattr(self, "_annotate_last_forward", None)
                profiles = getattr(self, "_forward_profiles", [])
                if profiles and profiles[-1].accepted_tokens is None:
                    profiles[-1].accepted_tokens = int(transfer.sum().item())
                    profiles[-1].active_tokens = int((~immutable_block).sum().item())
                    profiles[-1].eligible_tokens = int((active_masks | editable_candidates).sum().item())
                elif callable(annotate):
                    annotate(accepted_tokens=int(transfer.sum().item()),
                             active_tokens=int((~immutable_block).sum().item()),
                             eligible_tokens=int((active_masks | editable_candidates).sum().item()))

                if capture_trace:
                    old_ids = [int(value) for value in old.tolist()]
                    new_ids = [int(value) for value in new.tolist()]
                    predicted_ids = [int(value) for value in predicted.tolist()]
                    editing_trace.append(EditingTraceStep(
                        forward_index=forward_count, block_id=output_block, phase=phase,
                        old_block_tokens=old_ids, new_block_tokens=new_ids,
                        predicted_tokens=predicted_ids,
                        predicted_confidence=[float(value) for value in confidence.tolist()],
                        mask_positions=[int(value) for value in active_masks.nonzero().flatten().tolist()],
                        mask_transfer_positions=[int(value) for value in mask_transfer.nonzero().flatten().tolist()],
                        editable_positions=[int(value) for value in editable_candidates.nonzero().flatten().tolist()],
                        editing_transfer_positions=[int(value) for value in editing_transfer.nonzero().flatten().tolist()],
                        immutable_positions=[int(value) for value in immutable_block.nonzero().flatten().tolist()],
                        committed_positions=[], position_to_cell_map=mapped,
                        post_step_index=current_post_step, stop_reason=None,
                        old_block_token_texts=[self._token_text(value) for value in old_ids],
                        new_block_token_texts=[self._token_text(value) for value in new_ids],
                        predicted_token_texts=[self._token_text(value) for value in predicted_ids],
                    ))
                    block_steps.append(len(editing_trace) - 1)
                    visible_ids = [int(value) for value in x[0, answer_start:answer_start + answer_cells].tolist()]
                    accepted_cells = [mapped[pos] for pos in transfer.nonzero().flatten().tolist() if pos in mapped]
                    generic_trace.append(TraceStep(
                        forward_index=forward_count, token_ids=visible_ids,
                        token_texts=[self._token_text(value) for value in visible_ids],
                        position_states=[PositionState.MASKED if value == self.mask_id else PositionState.VISIBLE
                                         for value in visible_ids],
                        committed_positions=accepted_cells,
                        decoded_text="".join(self._token_text(value) for value in visible_ids
                                             if value != self.mask_id),
                        top1_confidence_by_position={mapped[pos]: float(confidence[pos].item())
                                                     for pos in mapped},
                    ))
                if phase == "post_edit" and not editing_transfer.any():
                    stop_reason = "no_edit_change"
                    break
            if block_steps:
                editing_trace[block_steps[-1]].committed_positions = list(mapped)
                editing_trace[block_steps[-1]].stop_reason = stop_reason

        output_ids = [int(value) for value in x[0, answer_start:answer_start + answer_cells].tolist()]
        output_text = "".join(self._token_text(value) for value in output_ids if value != self.mask_id)
        return GenerationResult(
            request=request, output_text=output_text, status=RunStatus.SUCCESS,
            num_forward_passes=forward_count,
            final_valid_length=answer_cells, trace=generic_trace, editing_trace=editing_trace,
            extra={"trace_capability": "editing_trace_v1", "editing_enabled": self.editing_enabled,
                   "threshold": self.threshold, "editing_threshold": self.editing_threshold,
                   "max_post_steps": self.max_post_steps,
                   "prompt_padding_tokens": aligned_prompt_length - prompt_length,
                   "output_token_ids": output_ids, "model_revision": self.revision},
        )
