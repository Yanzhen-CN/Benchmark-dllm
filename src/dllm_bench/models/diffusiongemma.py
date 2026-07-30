"""DiffusionGemma (Appendix D.4): official checkpoint, loaded straight from
``transformers`` — no vendoring, no ``trust_remote_code``.

Verified against the DiffusionGemma reference project's actual model-calling code
(``run.py``, and the vendored-but-unmodified ``generation_diffusion_gemma.py``
it imports from): ``DiffusionGemmaForBlockDiffusion`` and ``EntropyBoundSampler``
are real upstream classes, merged into ``transformers`` (confirmed present as
of ``transformers>=5.13.0`` — a much newer floor than this project's default
``hf`` extra, see ``pyproject.toml``). The reference project's ``vendor/transformers``
checkout is a genuine, unmodified clone of the public repo for everything
this adapter touches; the only project-specific patch is its
self-conditioning-alpha research sweep (reference-only instrumentation
layered on top, not part of DiffusionGemma's real interface) — deliberately
NOT ported here, since Appendix D.4 only calls for "shipped sampler and
recommended parameters".

Trace capture follows *How DiffusionGemma Actually Commits Tokens*, corrected
against the real method: ``accept_canvas`` is a **plain method** on a fresh
``EntropyBoundSampler`` instance built inside ``generate()`` by
``model._prepare_sampler(generation_config)`` — there is no
``model.sampler`` attribute to hook, and no ``nn.Module`` forward hook
applies (`accept_canvas` isn't a module's `forward`). The purely-observational
way in is to temporarily replace ``model._prepare_sampler`` with a wrapper
that, once the sampler is built, rebinds *that instance's* ``accept_canvas``
to record its inputs/outputs before delegating to the original — restored
in a ``finally`` block so this never leaks past one `generate()` call.
"""

from __future__ import annotations

import math

from ..interfaces import GenerationRequest, GenerationResult, PositionState, RunStatus, TraceStep
from .base import BaseModelAdapter
from .device_transfer import move_model_to_device
from .model_cache import get_or_load
from .prompting import tokenize_instruction_prompt

DEFAULT_DIFFUSIONGEMMA_CHECKPOINT = "google/diffusiongemma-26B-A4B-it"
DEFAULT_MAX_DENOISING_STEPS = 48


class DiffusionGemmaAdapter(BaseModelAdapter):
    deferred_measurement = True

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_DIFFUSIONGEMMA_CHECKPOINT,
        device: str | None = None,
        config_name: str = "official",
        steps: int | None = DEFAULT_MAX_DENOISING_STEPS,
    ) -> None:
        self.name = "diffusiongemma"
        self.config_name = config_name
        self.supports_trace = True
        self.natively_measures_resources = False
        self._model_name = model_name_or_path
        self._device = device
        self._default_steps = steps
        self._inference_dtype = "bfloat16"
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device

        def _load():
            # ``dtype="auto"`` reads the checkpoint's BF16 dtype instead of
            # materializing this 25.2B model as FP32. Placement stays fully
            # on the selected GPU so resource measurements remain comparable.
            return self._load_model_and_processor(device)

        self._processor, self._model = get_or_load(self._model_name, device, _load)

    def _load_model_and_processor(self, device: str):
        from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

        processor = AutoProcessor.from_pretrained(self._model_name)
        model = DiffusionGemmaForBlockDiffusion.from_pretrained(
            self._model_name, dtype="auto"
        )
        move_model_to_device(model, device, model_name=self._model_name)
        model.eval()
        return processor, model

    def _generate_core(self, request: GenerationRequest) -> GenerationResult:
        self._ensure_loaded()
        import torch

        gen_length = request.max_new_tokens
        steps = request.config.get("steps", self._default_steps) or DEFAULT_MAX_DENOISING_STEPS

        captured_steps: list[dict] = []
        forward_count = 0
        original_prepare_sampler = self._model._prepare_sampler

        def wrapped_prepare_sampler(generation_config):
            sampler = original_prepare_sampler(generation_config)
            original_accept_canvas = sampler.accept_canvas

            def wrapped_accept_canvas(current_canvas, denoiser_canvas, logits, cur_step):
                nonlocal forward_count
                forward_count += 1
                accepted_canvas = original_accept_canvas(current_canvas, denoiser_canvas, logits, cur_step)
                if self._trace_instrumentation_enabled():
                    with self._exclude_from_measurement():
                        entropy = torch.distributions.Categorical(logits=logits).entropy()
                        captured_steps.append(
                            {
                                "cur_step": cur_step,
                                "accepted_canvas": accepted_canvas.detach().to("cpu"),
                                "accepted_token_mask": sampler.accepted_token_mask.detach().to("cpu"),
                                "entropy": entropy.detach().to("cpu"),
                                "vocab_size": logits.shape[-1],
                            }
                        )
                return accepted_canvas

            sampler.accept_canvas = wrapped_accept_canvas
            return sampler

        # Instance-level override: a plain function stored directly in
        # `self._model.__dict__` is returned as-is on attribute access (no
        # auto-`self`-binding, since binding only happens for callables found
        # via the *class*), so `self._prepare_sampler(generation_config)`
        # inside `generate()` calls `wrapped_prepare_sampler` with exactly
        # the arguments below — no `self` parameter here on purpose.
        self._model._prepare_sampler = wrapped_prepare_sampler
        try:
            # The official checkpoint uses AutoProcessor.apply_chat_template;
            # the shared helper also applies RULER's post-template token cap.
            encoded = tokenize_instruction_prompt(
                self._processor,
                request.prompt,
                device=self._device,
                target_input_tokens=request.config.get("target_input_tokens"),
            )
            prompt_len = encoded["input_ids"].shape[1]
            self._start_measurement()
            with torch.inference_mode():
                output = self._model.generate(
                    **encoded,
                    max_new_tokens=gen_length,
                    max_denoising_steps=steps,
                    return_dict_in_generate=True,
                )
            self._stop_measurement()
        finally:
            self._model._prepare_sampler = original_prepare_sampler

        sequences = getattr(output, "sequences", output)
        generated_ids = sequences[0][prompt_len:].tolist()
        output_text = self._processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
        pad_token_id = getattr(
            getattr(self._model, "generation_config", None),
            "pad_token_id",
            None,
        )
        final_valid_length = (
            len(generated_ids)
            if pad_token_id is None
            else sum(token_id != pad_token_id for token_id in generated_ids)
        )
        official_tokens_per_forward = _scalar_tokens_per_forward(output)
        if official_tokens_per_forward is not None and official_tokens_per_forward > 0:
            official_forward_count = round(
                final_valid_length / official_tokens_per_forward
            )
            if official_forward_count > 0:
                forward_count = official_forward_count

        canvas_length = getattr(self._model.config, "canvas_length", gen_length)
        trace = _build_trace_from_captured_steps(captured_steps, canvas_length, self._processor.tokenizer)
        trace = _truncate_trace(trace, final_valid_length, self._processor.tokenizer)

        return GenerationResult(
            request=request,
            output_text=output_text,
            status=RunStatus.SUCCESS,
            trace=trace,
            num_forward_passes=forward_count,
            final_valid_length=final_valid_length,
            extra={
                "input_tokens": int(prompt_len),
                "official_tokens_per_forward": official_tokens_per_forward,
            },
        )


def _scalar_tokens_per_forward(output) -> float | None:
    """Read the official DiffusionGemma generation metric for batch size 1."""
    value = getattr(output, "tokens_per_forward", None)
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "reshape"):
        value = value.reshape(-1)[0]
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _truncate_trace(
    trace: list[TraceStep], final_valid_length: int, tokenizer
) -> list[TraceStep]:
    """Drop post-EOS padding positions from every persisted trace row."""
    if not trace or all(len(step.token_ids) <= final_valid_length for step in trace):
        return trace
    truncated: list[TraceStep] = []
    for step in trace:
        token_ids = step.token_ids[:final_valid_length]
        truncated.append(
            TraceStep(
                forward_index=step.forward_index,
                token_ids=token_ids,
                position_states=step.position_states[:final_valid_length],
                committed_positions=[
                    position
                    for position in step.committed_positions
                    if position < final_valid_length
                ],
                decoded_text=tokenizer.decode(token_ids, skip_special_tokens=True),
                entropy_by_position=(
                    {
                        position: value
                        for position, value in step.entropy_by_position.items()
                        if position < final_valid_length
                    }
                    if step.entropy_by_position
                    else None
                ),
                top1_confidence_by_position=(
                    {
                        position: value
                        for position, value in step.top1_confidence_by_position.items()
                        if position < final_valid_length
                    }
                    if step.top1_confidence_by_position
                    else None
                ),
                token_texts=(
                    step.token_texts[:final_valid_length]
                    if step.token_texts is not None
                    else None
                ),
            )
        )
    return truncated


def _assign_canvas_indices(captured_steps: list[dict]) -> list[int]:
    """`cur_step` counts *down* within one canvas/block; a step whose
    `cur_step` is higher than the previous row's signals a new canvas started
    (same heuristic the reference trace tooling uses)."""
    indices = []
    canvas_index = -1
    previous_step = None
    for step_data in captured_steps:
        cur_step = step_data["cur_step"]
        if previous_step is None or cur_step > previous_step:
            canvas_index += 1
        previous_step = cur_step
        indices.append(canvas_index)
    return indices


def _build_trace_from_captured_steps(
    captured_steps: list[dict], canvas_length: int, tokenizer
) -> list[TraceStep]:
    """Positions not accepted this step are rendered VISIBLE, not MASKED:
    DiffusionGemma always shows *some* token per position (accepted_canvas
    keeps the prior value where not accepted) rather than a blank mask
    placeholder — and, unlike iLLaDA, a position accepted in one step can be
    genuinely renoised and revised in a later step (verified: accepted_token_mask
    is fully recomputed every step, and renoise_canvas resamples every
    non-accepted position fresh each time), so VISIBLE-vs-ACCEPTED here can
    flip back and forth across the trace — that's real model behavior, not a
    bug in this conversion.
    """
    if not captured_steps:
        return []

    canvas_indices = _assign_canvas_indices(captured_steps)
    final_tokens_per_canvas: dict[int, list[int]] = {}
    for step_data, canvas_index in zip(captured_steps, canvas_indices):
        final_tokens_per_canvas[canvas_index] = step_data["accepted_canvas"][0].tolist()

    trace: list[TraceStep] = []
    previous_masks: dict[int, list[bool]] = {}
    previous_tokens: dict[int, list[int]] = {}
    for step_data, canvas_index in zip(captured_steps, canvas_indices):
        offset = canvas_index * canvas_length
        accepted_canvas = step_data["accepted_canvas"][0].tolist()
        accepted_mask = step_data["accepted_token_mask"][0].tolist()
        entropy = step_data["entropy"][0].tolist()
        vocab_size = step_data["vocab_size"]

        token_ids: list[int] = []
        for prior_index in range(canvas_index):
            token_ids.extend(final_tokens_per_canvas[prior_index])
        token_ids.extend(accepted_canvas)

        position_states = [PositionState.ACCEPTED] * offset + [
            PositionState.ACCEPTED if accepted else PositionState.VISIBLE for accepted in accepted_mask
        ]
        prior_mask = previous_masks.get(canvas_index, [False] * len(accepted_mask))
        prior_tokens = previous_tokens.get(canvas_index, accepted_canvas)
        committed_positions = [
            offset + index
            for index, accepted in enumerate(accepted_mask)
            if accepted
            and (
                not prior_mask[index]
                or accepted_canvas[index] != prior_tokens[index]
            )
        ]
        previous_masks[canvas_index] = list(accepted_mask)
        previous_tokens[canvas_index] = list(accepted_canvas)

        token_texts = [tokenizer.decode([t]) for t in token_ids]
        entropy_by_position = {
            offset + i: entropy[i] / math.log(vocab_size)
            for i, accepted in enumerate(accepted_mask)
            if not accepted
        }

        trace.append(
            TraceStep(
                forward_index=len(trace),
                token_ids=token_ids,
                position_states=position_states,
                committed_positions=committed_positions,
                decoded_text="".join(token_texts),
                entropy_by_position=entropy_by_position or None,
                token_texts=token_texts,
            )
        )
    return trace
