"""Qwen3 AR baselines: official HF checkpoints, standard
``generate()`` loop, greedy decoding. As a standard AR model it satisfies the
"same abstraction interface" requirement with no adapter shim needed —
``transformers`` already gives us the standard ``generate()`` call.

Trace capture treats each decoded token as accepted at its own forward step
(``ACCEPTED`` from the moment it is emitted, one commit per forward), matching
Appendix C's "AR decode as a ~1 token/forward parallelism reference" — this
is what Part 4 compares diffusion models' actual parallelism against, not a
full trace analysis subject in its own right (section 5 lists Qwen AR as
"并行度参考" only).
"""

from __future__ import annotations

from ..interfaces import GenerationRequest, GenerationResult, PositionState, RunStatus, TraceStep
from .base import BaseModelAdapter
from .model_cache import get_or_load
from .prompting import tokenize_instruction_prompt


class QwenARAdapter(BaseModelAdapter):
    deferred_measurement = True

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-4B",
        device: str | None = None,
        config_name: str = "ar-baseline",
        capture_trace: bool = True,
        enable_thinking: bool = False,
        adapter_name: str = "qwen3_4b",
    ) -> None:
        self.name = adapter_name
        self.config_name = config_name
        self.supports_trace = capture_trace
        self.natively_measures_resources = False
        self._model_name = model_name_or_path
        self._device = device
        self._capture_trace = capture_trace
        self._enable_thinking = enable_thinking
        self._inference_dtype = "bfloat16"
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device

        def _load():
            return self._load_model_and_tokenizer(device)

        # Shared across every config that points at this same checkpoint —
        # Each Qwen checkpoint has one config today, but this keeps the pattern
        # consistent with the diffusion adapters, which do have Best/Fast.
        self._tokenizer, self._model = get_or_load(self._model_name, device, _load)

    def _load_model_and_tokenizer(self, device: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        model.to(device)
        model.eval()
        return tokenizer, model

    def _generate_core(self, request: GenerationRequest) -> GenerationResult:
        self._ensure_loaded()
        import torch

        inputs = tokenize_instruction_prompt(
            self._tokenizer,
            request.prompt,
            device=self._device,
            chat_template_kwargs={"enable_thinking": self._enable_thinking},
            target_input_tokens=request.config.get("target_input_tokens"),
        )
        prompt_len = inputs["input_ids"].shape[1]
        self._start_measurement()
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                do_sample=False,
            )
        self._stop_measurement()

        sequence = output[0]
        generated_ids = sequence[prompt_len:]
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        generated_token_ids = generated_ids.tolist()
        capture_trace = self._capture_trace and self._trace_instrumentation_enabled()
        trace = _build_ar_trace(generated_token_ids, self._tokenizer) if capture_trace else []

        return GenerationResult(
            request=request,
            output_text=text,
            status=RunStatus.SUCCESS,
            trace=trace,
            num_forward_passes=len(generated_token_ids),
            final_valid_length=len(generated_token_ids),
            extra={"input_tokens": int(prompt_len)},
        )


def _build_ar_trace(generated_ids: list[int], tokenizer) -> list[TraceStep]:
    """Reconstruct the one-token-per-forward AR trace after timing.

    Retaining ``generate(..., output_scores=True)`` keeps one full-vocabulary
    logits tensor per token alive and changes the measured path. Token ids are
    sufficient for the benchmark's AR parallelism reference; entropy and
    confidence therefore remain absent instead of requiring a timed replay.
    """
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
