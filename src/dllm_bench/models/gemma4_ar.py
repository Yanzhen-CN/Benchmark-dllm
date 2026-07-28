"""Architecture-matched autoregressive control for DiffusionGemma.

``google/gemma-4-26B-A4B-it`` and DiffusionGemma share the same 25.2B-total /
3.8B-active MoE scale.  This adapter follows the official Transformers recipe:
``AutoProcessor``, ``AutoModelForMultimodalLM``, BF16 from the checkpoint, the
checkpoint's shipped sampling configuration, and thinking disabled in the chat
template.  The benchmark is text-only even though the checkpoint is multimodal.

AR trace rows are reconstructed from the completed token sequence after the
measured generation window.  This is exact for autoregressive decoding (one new
token commits per forward) and avoids retaining a full 262K-vocabulary logits
tensor for every generated token.
"""

from __future__ import annotations

from ..interfaces import GenerationRequest, GenerationResult, PositionState, RunStatus, TraceStep
from .base import BaseModelAdapter
from .model_cache import get_or_load

DEFAULT_GEMMA4_CHECKPOINT = "google/gemma-4-26B-A4B-it"


class Gemma4ARAdapter(BaseModelAdapter):
    deferred_measurement = True

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_GEMMA4_CHECKPOINT,
        device: str | None = None,
        config_name: str = "ar-baseline",
        capture_trace: bool = True,
        enable_thinking: bool = False,
    ) -> None:
        self.name = "gemma4_26b"
        self.config_name = config_name
        self.supports_trace = capture_trace
        self.natively_measures_resources = False
        self._model_name = model_name_or_path
        self._device = device
        self._capture_trace = capture_trace
        self._enable_thinking = enable_thinking
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device

        def _load():
            return self._load_model_and_processor(device)

        self._processor, self._model = get_or_load(self._model_name, device, _load)

    def _load_model_and_processor(self, device: str):
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        processor = AutoProcessor.from_pretrained(self._model_name)
        model = AutoModelForMultimodalLM.from_pretrained(
            self._model_name, dtype="auto"
        )
        model.to(device)
        model.eval()
        return processor, model

    def _generate_core(self, request: GenerationRequest) -> GenerationResult:
        self._ensure_loaded()
        import torch

        encoded = self._processor.apply_chat_template(
            [{"role": "user", "content": request.prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=self._enable_thinking,
            return_dict=True,
            return_tensors="pt",
        ).to(self._device)
        prompt_len = encoded["input_ids"].shape[1]

        self._start_measurement()
        with torch.inference_mode():
            output = self._model.generate(
                **encoded,
                max_new_tokens=request.max_new_tokens,
            )
        self._stop_measurement()

        generated_ids = output[0][prompt_len:].tolist()
        tokenizer = self._processor.tokenizer
        output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        capture_trace = self._capture_trace and self._trace_instrumentation_enabled()
        trace = _build_ar_trace(generated_ids, tokenizer) if capture_trace else []

        return GenerationResult(
            request=request,
            output_text=output_text,
            status=RunStatus.SUCCESS,
            trace=trace,
            num_forward_passes=len(generated_ids),
            final_valid_length=len(generated_ids),
        )


def _build_ar_trace(generated_ids: list[int], tokenizer) -> list[TraceStep]:
    """Build the exact one-token-per-forward AR commit history post-timing."""
    final_length = len(generated_ids)
    token_texts = [
        tokenizer.decode([token_id], skip_special_tokens=True)
        for token_id in generated_ids
    ]
    trace: list[TraceStep] = []
    for step_index in range(final_length):
        visible_count = step_index + 1
        trace.append(
            TraceStep(
                forward_index=step_index,
                token_ids=(
                    generated_ids[:visible_count]
                    + [-1] * (final_length - visible_count)
                ),
                position_states=(
                    [PositionState.ACCEPTED] * visible_count
                    + [PositionState.MASKED] * (final_length - visible_count)
                ),
                committed_positions=[step_index],
                decoded_text=tokenizer.decode(
                    generated_ids[:visible_count], skip_special_tokens=True
                ),
                token_texts=(
                    token_texts[:visible_count]
                    + [""] * (final_length - visible_count)
                ),
            )
        )
    return trace
