"""Shared HF-style diffusion adapter base for iLLaDA and DreamReasoner
(Appendix D.1/D.2).

Both models load via `trust_remote_code=True` (iLLaDA: `AutoModel`/
`AutoTokenizer`, the default `_ensure_loaded` here; DreamReasoner overrides
`_ensure_loaded` for `AutoModelForCausalLM` instead — see
`dreamreasoner.py`), and both are genuinely block-wise (unlike regular
Dream-7B's single-pass `diffusion_generate`, no longer part of this
benchmark's model roster — see design doc section 5). Their real per-block
denoising loops still differ enough in implementation detail (iLLaDA
recomputes full-sequence logits every step; DreamReasoner uses a real
per-block prefix KV cache with its own `_select_transfer_index` remasking
strategies) that each ports its own loop in its own module
(``illada.py``/``dreamreasoner.py``) subclassing :class:`HFDiffusionAdapter`
here and implementing :meth:`_run_denoising`. This base class only owns
what's genuinely shared: loading (via the process-wide weight cache, so
Best/Fast share one loaded copy), resource-measurement integration, and
merging per-request config overrides into a :class:`DiffusionStepConfig`.
"""

from __future__ import annotations

import math
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..interfaces import GenerationRequest, GenerationResult, RunStatus, TraceStep
from .base import BaseModelAdapter
from .model_cache import get_or_load


@dataclass
class DiffusionStepConfig:
    """Appendix D Best/Fast knobs. ``gen_length``/``steps``/``block_length``/
    ``steps_per_block`` are the fields shared conceptually across block/step
    diffusion models; ``extra`` carries whatever additional knobs one
    specific model's real sampler needs (e.g. DreamReasoner's
    `remasking_strategy`/`confidence_threshold`/`eb_threshold`/`top_k`/
    `top_p`/`temperature`) without forcing every model into one rigid shape.
    """

    gen_length: int
    steps: int | None = None
    block_length: int | None = None
    steps_per_block: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class HFDiffusionAdapter(BaseModelAdapter):
    deferred_measurement = True

    def __init__(
        self,
        model_name_or_path: str,
        step_config: DiffusionStepConfig,
        name: str,
        config_name: str,
        device: str | None = None,
    ) -> None:
        self.name = name
        self.config_name = config_name
        self.supports_trace = True
        self.natively_measures_resources = False
        self._model_name = model_name_or_path
        self._step_config = step_config
        self._device = device
        self._inference_dtype = "bfloat16"
        self._model = None
        self._tokenizer = None
        self.inference_optimizations: list[str] = []
        self.execution_path = str(
            step_config.extra.get("execution_path", "default")
        )
        self.sampling_profile = step_config.extra.get("sampling_profile")

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device

        def _load():
            return self._load_model_and_tokenizer(device)

        # Best/Fast point at the *same* checkpoint with a different
        # step_config — sharing this load is the whole point of nesting both
        # under one configs/models/*.yaml file (see README).
        self._tokenizer, self._model = get_or_load(self._model_name, device, _load)

    def _load_model_and_tokenizer(self, device: str):
        # This default (AutoModel/AutoTokenizer) is iLLaDA's own loading
        # path — a custom `trust_remote_code` model class, confirmed against
        # the reference project's own loading code. DreamReasoner overrides
        # both this and `_ensure_loaded` entirely (AutoModelForCausalLM
        # instead — see dreamreasoner.py), so this body never runs for it.
        import torch
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self._model_name, trust_remote_code=True)
        # iLLaDA is released as BF16 and its official model card requires
        # this dtype. Omitting it constructs an unnecessarily large
        # default-precision model that does not fit on a 24 GiB GPU.
        model = AutoModel.from_pretrained(
            self._model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        model.to(device)
        model.eval()
        return tokenizer, model

    @abstractmethod
    def _run_denoising(
        self,
        prompt: str,
        step_config: DiffusionStepConfig,
        target_input_tokens: int | None = None,
    ) -> tuple[str, list[TraceStep], int]:
        """Run the full denoising/unmasking schedule for one sample.

        Must return (output_text, trace, final_valid_length).
        """

    def _generate_core(self, request: GenerationRequest) -> GenerationResult:
        self._ensure_loaded()
        # max_new_tokens is always a hard cap. Dream's formal 1024-token
        # schedule is scaled down proportionally for shorter task/smoke runs;
        # iLLaDA leaves gen_length unset and follows the request directly.
        configured_gen_length = (
            request.config.get("gen_length")
            or self._step_config.gen_length
            or request.max_new_tokens
        )
        gen_length = min(int(configured_gen_length), request.max_new_tokens)

        steps = request.config.get("steps")
        if steps is None:
            steps = self._step_config.steps
            base_gen_length = self._step_config.gen_length
            if steps is not None and base_gen_length and gen_length < base_gen_length:
                steps = max(1, math.ceil(steps * gen_length / base_gen_length))

        extra = dict(self._step_config.extra)
        extra.update(request.config.get("extra", {}))
        step_config = DiffusionStepConfig(
            gen_length=gen_length,
            steps=steps,
            block_length=request.config.get("block_length", self._step_config.block_length),
            steps_per_block=request.config.get("steps_per_block", self._step_config.steps_per_block),
            extra=extra,
        )
        self._last_input_tokens = None
        output_text, trace, final_valid_length = self._run_denoising(
            request.prompt,
            step_config,
            target_input_tokens=request.config.get("target_input_tokens"),
        )
        result_extra = {}
        if self._last_input_tokens is not None:
            result_extra["input_tokens"] = self._last_input_tokens
        result_extra["inference_optimizations"] = list(
            self.inference_optimizations
        )
        result_extra["execution_path"] = self.execution_path
        trace_source = getattr(self, "trace_source", None)
        if trace_source is not None:
            result_extra["trace_source"] = trace_source
        if self.sampling_profile is not None:
            result_extra["sampling_profile"] = self.sampling_profile
        return GenerationResult(
            request=request,
            output_text=output_text,
            status=RunStatus.SUCCESS,
            trace=trace,
            num_forward_passes=int(
                getattr(self, "_last_num_forward_passes", len(trace))
            ),
            final_valid_length=final_valid_length,
            extra=result_extra,
        )
