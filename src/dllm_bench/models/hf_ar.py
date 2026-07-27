"""Qwen3-4B AR baseline (Appendix D.5): official HF checkpoint, standard
``generate()`` loop, greedy decoding. As a standard AR model it satisfies the
"same abstraction interface" requirement with no adapter shim needed —
``transformers`` already gives us the standard ``generate()`` call.

Trace capture treats each decoded token as accepted at its own forward step
(``ACCEPTED`` from the moment it is emitted, one commit per forward), matching
Appendix C's "AR decode as a ~1 token/forward parallelism reference" — this
is what Part 4 compares diffusion models' actual parallelism against, not a
full trace analysis subject in its own right (section 5 lists Qwen3-4B as
"并行度参考" only).
"""

from __future__ import annotations

import math

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
    ) -> None:
        self.name = "qwen3_4b"
        self.config_name = config_name
        self.supports_trace = capture_trace
        self.natively_measures_resources = False
        self._model_name = model_name_or_path
        self._device = device
        self._capture_trace = capture_trace
        self._enable_thinking = enable_thinking
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device

        def _load():
            tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            model = AutoModelForCausalLM.from_pretrained(self._model_name)
            model.to(device)
            model.eval()
            return tokenizer, model

        # Shared across every config that points at this same checkpoint —
        # Qwen3-4B only has one config today, but this keeps the pattern
        # consistent with the diffusion adapters, which do have Best/Fast.
        self._tokenizer, self._model = get_or_load(self._model_name, device, _load)

    def _generate_core(self, request: GenerationRequest) -> GenerationResult:
        self._ensure_loaded()
        import torch

        inputs = tokenize_instruction_prompt(
            self._tokenizer,
            request.prompt,
            device=self._device,
            chat_template_kwargs={"enable_thinking": self._enable_thinking},
        )
        prompt_len = inputs["input_ids"].shape[1]

        self._start_measurement()
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                do_sample=False,
                output_scores=self._capture_trace,
                return_dict_in_generate=self._capture_trace,
            )
        self._stop_measurement()

        sequence = output.sequences[0] if self._capture_trace else output[0]
        generated_ids = sequence[prompt_len:]
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        trace: list[TraceStep] = []
        if self._capture_trace:
            generated_token_ids = generated_ids.tolist()
            generated_token_texts = [
                self._tokenizer.decode([token_id], skip_special_tokens=True)
                for token_id in generated_token_ids
            ]
            final_length = len(generated_token_ids)
            token_ids_accum: list[int] = []
            token_texts_accum: list[str] = []
            for step_index, (token_id, step_logits) in enumerate(
                zip(generated_token_ids, output.scores)
            ):
                token_ids_accum.append(token_id)
                token_texts_accum.append(generated_token_texts[step_index])
                position = len(token_ids_accum) - 1
                probs = torch.softmax(step_logits[0], dim=-1)
                top1_conf = probs.max().item()
                entropy = -(probs * probs.clamp_min(1e-12).log()).sum().item()
                normalized_entropy = entropy / math.log(probs.shape[-1])
                trace.append(
                    TraceStep(
                        forward_index=step_index,
                        token_ids=(
                            list(token_ids_accum)
                            + [-1] * (final_length - len(token_ids_accum))
                        ),
                        position_states=(
                            [PositionState.ACCEPTED] * len(token_ids_accum)
                            + [PositionState.MASKED] * (final_length - len(token_ids_accum))
                        ),
                        committed_positions=[position],
                        decoded_text=self._tokenizer.decode(
                            token_ids_accum, skip_special_tokens=True
                        ),
                        entropy_by_position={position: normalized_entropy},
                        top1_confidence_by_position={position: top1_conf},
                        token_texts=(
                            list(token_texts_accum)
                            + [""] * (final_length - len(token_texts_accum))
                        ),
                    )
                )

        return GenerationResult(
            request=request,
            output_text=text,
            status=RunStatus.SUCCESS,
            trace=trace,
            num_forward_passes=len(generated_ids),
            final_valid_length=len(generated_ids),
        )
