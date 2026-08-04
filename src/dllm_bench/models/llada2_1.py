"""Official LLaDA2.1 Q/S generation with observation-only tracing.

The checkpoint's remote ``generate`` method remains the sole owner of mask
transfer and token editing. Temporary wrappers observe real canvases and
distributions, delegate unchanged, and are restored after every request.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any

from .base import BaseModelAdapter
from .device_transfer import move_model_to_device
from .model_cache import get_or_load
from .prompting import tokenize_instruction_prompt
from ..interfaces import GenerationRequest, GenerationResult, PositionState, RunStatus, TraceStep


class Llada21Adapter(BaseModelAdapter):
    name = "llada2_1"
    supports_trace = True
    deferred_measurement = True

    def __init__(
        self,
        model_name_or_path: str = "inclusionAI/LLaDA2.1-mini",
        revision: str | None = None,
        config_name: str = "qmode",
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        block_length: int = 32,
        steps: int = 32,
        threshold: float = 0.7,
        editing_threshold: float = 0.5,
        max_post_steps: int = 16,
        temperature: float = 0.0,
        top_p: float | None = None,
        top_k: int | None = None,
        eos_early_stop: bool = True,
        minimal_topk: int = 1,
        num_to_transfer: int = 1,
        mask_id: int = 156895,
        eos_id: int = 156892,
        trust_remote_code: bool = True,
        **_: Any,
    ) -> None:
        super().__init__()
        self.config_name = config_name
        self.model_name_or_path = model_name_or_path
        self.revision = revision
        self.device = device
        self.torch_dtype = torch_dtype
        self.block_length = int(block_length)
        self.steps = int(steps)
        self.threshold = float(threshold)
        self.editing_threshold = float(editing_threshold)
        self.max_post_steps = int(max_post_steps)
        self.temperature = float(temperature)
        self.top_p = top_p
        self.top_k = top_k
        self.eos_early_stop = bool(eos_early_stop)
        self.minimal_topk = int(minimal_topk)
        self.num_to_transfer = int(num_to_transfer)
        self.mask_id = int(mask_id)
        self.eos_id = int(eos_id)
        self.trust_remote_code = bool(trust_remote_code)
        self.execution_path = "official-transformers-remote-generate"
        self.trace_source = "observational-forward-and-sampler-wrapper"
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        common = {"revision": self.revision, "trust_remote_code": self.trust_remote_code}
        common = {key: value for key, value in common.items() if value is not None}

        def _load():
            tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, **common)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                torch_dtype=getattr(torch, self.torch_dtype),
                low_cpu_mem_usage=True,
                **common,
            )
            move_model_to_device(model, self.device, model_name=self.model_name_or_path)
            model.eval()
            return tokenizer, model

        key = f"{self.model_name_or_path}@{self.revision or 'main'}:{self.torch_dtype}"
        self._tokenizer, self._model = get_or_load(key, self.device, _load)

    @contextmanager
    def _capture_model_forwards(self, request, compute_handle=None):
        self._ensure_loaded()
        with super()._capture_model_forwards(request, compute_handle):
            yield

    def _prompt_ids(self, prompt: str):
        return tokenize_instruction_prompt(
            self._tokenizer,
            prompt,
            device=self.device,
        )["input_ids"]

    def _generate_core(self, request: GenerationRequest) -> GenerationResult:
        self._ensure_loaded()
        import torch

        prompt_ids = self._prompt_ids(request.prompt)
        prompt_length = int(prompt_ids.shape[1])
        observations: list[dict[str, Any]] = []
        pending_canvas = None
        forward_count = 0
        trace_enabled = self._trace_instrumentation_enabled()
        original_forward = self._model.forward
        original_sampler = self._model._sample_with_temperature_topk_topp

        def observed_forward(*args, **kwargs):
            nonlocal pending_canvas, forward_count
            input_ids = args[0] if args else kwargs.get("input_ids")
            forward_count += 1
            pending_canvas = input_ids.detach() if input_ids is not None else None
            with self._forward_phase("denoise_step"):
                return original_forward(*args, **kwargs)

        def observed_sampler(logits, temperature=0.0, top_k=None, top_p=None):
            sampled, confidence = original_sampler(
                logits, temperature=temperature, top_k=top_k, top_p=top_p
            )
            if trace_enabled and pending_canvas is not None:
                with self._exclude_from_measurement():
                    probabilities = torch.softmax(logits.float(), dim=-1)
                    entropy = -(
                        probabilities * probabilities.clamp_min(1e-12).log()
                    ).sum(dim=-1) / math.log(probabilities.shape[-1])
                    active_tokens = pending_canvas[:, -logits.shape[-2]:]
                    current_confidence = torch.gather(
                        probabilities,
                        dim=-1,
                        index=active_tokens.unsqueeze(-1),
                    ).squeeze(-1)
                    observations.append({
                        "canvas": pending_canvas[0].detach().cpu().tolist(),
                        "active_start": int(pending_canvas.shape[1] - logits.shape[-2]),
                        "entropy": entropy[0].detach().cpu().tolist(),
                        "confidence": confidence[0].detach().float().cpu().tolist(),
                        "current_confidence": current_confidence[0].detach().cpu().tolist(),
                        "proposed_tokens": sampled[0].detach().cpu().tolist(),
                    })
            return sampled, confidence

        self._model.forward = observed_forward
        self._model._sample_with_temperature_topk_topp = observed_sampler
        measurement_started = False
        try:
            self._start_measurement()
            measurement_started = True
            with torch.inference_mode():
                generated = self._model.generate(
                    inputs=prompt_ids,
                    temperature=self.temperature,
                    block_length=self.block_length,
                    steps=self.steps,
                    gen_length=int(request.max_new_tokens),
                    top_p=self.top_p,
                    top_k=self.top_k,
                    eos_early_stop=self.eos_early_stop,
                    minimal_topk=self.minimal_topk,
                    threshold=self.threshold,
                    editing_threshold=self.editing_threshold,
                    max_post_steps=self.max_post_steps,
                    eos_id=self.eos_id,
                    mask_id=self.mask_id,
                    num_to_transfer=self.num_to_transfer,
                )
        finally:
            if measurement_started:
                self._stop_measurement()
            self._model.forward = original_forward
            self._model._sample_with_temperature_topk_topp = original_sampler

        raw_ids = [int(value) for value in generated[0].tolist()]
        final_ids = raw_ids[:raw_ids.index(self.eos_id)] if self.eos_id in raw_ids else raw_ids
        output_text = self._tokenizer.decode(final_ids, skip_special_tokens=True)
        trace = _build_observational_trace(
            observations=observations,
            prompt_ids=[int(value) for value in prompt_ids[0].tolist()],
            final_ids=final_ids,
            prompt_length=prompt_length,
            mask_id=self.mask_id,
            tokenizer=self._tokenizer,
        )
        return GenerationResult(
            request=request,
            output_text=output_text,
            status=RunStatus.SUCCESS,
            trace=trace,
            num_forward_passes=forward_count,
            final_valid_length=len(final_ids),
            extra={
                "execution_path": self.execution_path,
                "trace_source": self.trace_source,
                "trace_capability": "official-token-editing-trace-v1",
                "model_revision": self.revision,
                "input_tokens": prompt_length,
                "block_length": self.block_length,
                "steps": self.steps,
                "threshold": self.threshold,
                "editing_threshold": self.editing_threshold,
                "max_post_steps": self.max_post_steps,
                "trace_revision_events": _count_revision_events(trace, self.mask_id),
            },
        )


def _canvas_tokens(canvas, prompt_length: int, output_length: int, mask_id: int):
    generated = list(canvas[prompt_length:prompt_length + output_length])
    return generated + [mask_id] * (output_length - len(generated))


def _build_observational_trace(
    *, observations, prompt_ids, final_ids, prompt_length, mask_id, tokenizer
) -> list[TraceStep]:
    """Pair every official decision distribution with its real post-canvas."""
    output_length = len(final_ids)
    if not observations or output_length <= 0:
        return []
    final_canvas = [*prompt_ids, *final_ids]
    trace = []
    for index, observation in enumerate(observations):
        before = _canvas_tokens(observation["canvas"], prompt_length, output_length, mask_id)
        after_source = observations[index + 1]["canvas"] if index + 1 < len(observations) else final_canvas
        after = _canvas_tokens(after_source, prompt_length, output_length, mask_id)
        changed = [i for i, (old, new) in enumerate(zip(before, after)) if old != new and new != mask_id]
        active_start = int(observation["active_start"]) - prompt_length
        entropy_by_position = {}
        confidence_by_position = {}
        current_confidence_by_position = {}
        proposed_token_ids_by_position = {}
        confidence_margin_by_position = {}
        editing_state_by_position = {}
        for local, value in enumerate(observation["entropy"]):
            position = active_start + local
            if 0 <= position < output_length:
                top1_confidence = float(observation["confidence"][local])
                current_confidence = float(observation["current_confidence"][local])
                proposed_token = int(observation["proposed_tokens"][local])
                entropy_by_position[position] = float(value)
                confidence_by_position[position] = top1_confidence
                current_confidence_by_position[position] = current_confidence
                proposed_token_ids_by_position[position] = proposed_token
                confidence_margin_by_position[position] = top1_confidence - current_confidence
                if before[position] == mask_id:
                    editing_state = "mask_fill" if after[position] != mask_id else "mask_wait"
                elif proposed_token == before[position]:
                    editing_state = "stable"
                elif after[position] != before[position]:
                    editing_state = "accepted_edit"
                else:
                    editing_state = "rejected_edit"
                editing_state_by_position[position] = editing_state
        states = [PositionState.MASKED if token == mask_id else PositionState.ACCEPTED for token in after]
        texts = ["" if token == mask_id else tokenizer.decode([token], skip_special_tokens=False) for token in after]
        trace.append(TraceStep(
            forward_index=index,
            token_ids=after,
            position_states=states,
            committed_positions=changed,
            decoded_text="".join(texts),
            entropy_by_position=entropy_by_position,
            top1_confidence_by_position=confidence_by_position,
            token_texts=texts,
            current_token_confidence_by_position=current_confidence_by_position,
            proposed_token_ids_by_position=proposed_token_ids_by_position,
            confidence_margin_by_position=confidence_margin_by_position,
            editing_state_by_position=editing_state_by_position,
        ))
    return trace


def _count_revision_events(trace: list[TraceStep], mask_id: int) -> int:
    previous = [mask_id] * len(trace[0].token_ids) if trace else []
    revisions = 0
    for step in trace:
        revisions += sum(old != mask_id and new != mask_id and old != new for old, new in zip(previous, step.token_ids))
        previous = step.token_ids
    return revisions
