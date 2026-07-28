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
from .model_cache import get_or_load, offloaded_parameter_bytes, reload_with_offload
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

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device

        def _load():
            return self._load_model_and_tokenizer(device, device_map_auto=False)

        # Shared across every config that points at this same checkpoint —
        # Qwen3-4B only has one config today, but this keeps the pattern
        # consistent with the diffusion adapters, which do have Best/Fast.
        # This can be a cache *hit* on a model already reloaded with
        # offloading after an OOM elsewhere (see `_reload_with_cpu_offload`)
        # — check the actual model, don't assume a fresh, 100%-GPU load.
        self._tokenizer, self._model = get_or_load(self._model_name, device, _load)
        self._cpu_offloaded_bytes = offloaded_parameter_bytes(self._model)
        self._cpu_offloaded = self._cpu_offloaded_bytes > 0

    def _load_model_and_tokenizer(self, device: str, *, device_map_auto: bool):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        kwargs: dict = {}
        if device_map_auto:
            kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(self._model_name, **kwargs)
        if not device_map_auto:
            model.to(device)
        model.eval()
        return tokenizer, model

    def _reload_with_cpu_offload(self) -> bool:
        """See `models/base.py`'s `BaseModelAdapter._reload_with_cpu_offload`
        docstring — this only ever runs reactively, after a genuine capacity
        OOM neither a plain retry nor a cache-cleared one could recover
        from. Qwen3-4B is small enough that this is not expected to trigger
        in practice, but the same recovery path is available to every model
        in this benchmark rather than being special-cased to just the
        diffusion models that happened to need it first."""
        if getattr(self, "_cpu_offloaded", False):
            return True
        if self._model is None:
            return False

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
        self._cpu_offloaded_bytes = offloaded_parameter_bytes(self._model)
        self._cpu_offloaded = self._cpu_offloaded_bytes > 0
        return True

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
        capture_trace = self._capture_trace and self._trace_instrumentation_enabled()

        self._start_measurement()
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                do_sample=False,
                output_scores=capture_trace,
                return_dict_in_generate=capture_trace,
            )
        self._stop_measurement()

        sequence = output.sequences[0] if capture_trace else output[0]
        generated_ids = sequence[prompt_len:]
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        trace: list[TraceStep] = []
        if capture_trace:
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
