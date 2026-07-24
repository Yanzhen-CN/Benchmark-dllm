"""Shared HF-style diffusion adapter base for iLLaDA and Dream (Appendix D.1/D.2).

Both models are expected to ship (or be wrapped into) standard HF
``PreTrainedModel`` checkpoints, so loading follows the same
``AutoTokenizer``/``AutoModel`` pattern as :mod:`hf_ar`. What differs — and
what the design doc explicitly flags as **unresolved until verified against
the real checkpoints** — is the actual denoising/unmasking loop: how many
positions get committed per step, what the sampler call signature looks like,
and whether it is already wired into a HF-compatible ``generate()``-like
entry point or only available as a standalone inference script that needs a
thin adapter shim (Appendix D.1/D.2's "如果官方发布的只是独立推理脚本...需要自己
包一层适配").

``HFDiffusionAdapter`` therefore implements everything that *is* fully
specified by the design doc (config wiring for Best/Fast block_length /
steps_per_block / steps / gen_length, resource-measurement integration via
:class:`~dllm_bench.models.base.BaseModelAdapter`) and leaves the actual
per-step sampling loop as :meth:`_run_denoising`, which subclasses must
implement once the checkpoint's real sampler API is confirmed. Calling
:meth:`generate` before that returns a ``RunStatus.FAILED`` result (via the
base class's exception handling) rather than crashing the whole benchmark
run or silently faking output.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass

from ..interfaces import GenerationRequest, GenerationResult, RunStatus, TraceStep
from .base import BaseModelAdapter
from .model_cache import get_or_load


@dataclass
class DiffusionStepConfig:
    """Appendix D Best/Fast knobs. Only the fields relevant to a given model
    need to be set (iLLaDA uses block_length/steps_per_block; Dream uses steps)."""

    gen_length: int
    steps: int | None = None
    block_length: int | None = None
    steps_per_block: int | None = None


class HFDiffusionAdapter(BaseModelAdapter):
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
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device

        def _load():
            # TODO(Appendix D.1/D.2): confirm the released checkpoint name and
            # whether it subclasses PreTrainedModel directly, or needs a custom
            # `trust_remote_code=True` class / a separate adapter shim.
            tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            model = AutoModel.from_pretrained(self._model_name)
            model.to(device)
            model.eval()
            return tokenizer, model

        # Best/Fast point at the *same* checkpoint with a different
        # step_config — sharing this load is the whole point of nesting both
        # under one configs/models/*.yaml file (see README).
        self._tokenizer, self._model = get_or_load(self._model_name, device, _load)

    @abstractmethod
    def _run_denoising(
        self, prompt: str, step_config: DiffusionStepConfig
    ) -> tuple[str, list[TraceStep], int]:
        """Run the full denoising/unmasking schedule for one sample.

        Must return (output_text, trace, final_valid_length). Left abstract:
        the exact sampler call signature is not yet confirmed against the
        official checkpoint (see module docstring / Appendix D.1-D.2).
        """

    def _generate_core(self, request: GenerationRequest) -> GenerationResult:
        self._ensure_loaded()
        # gen_length: request.config override > this adapter's configured
        # step_config.gen_length > request.max_new_tokens. The last fallback
        # matters for configs like illada.yaml's `best`/`fast` variants, which
        # intentionally leave gen_length unset ("follows each task's own
        # output length cap") — without it, `--max-new-tokens` on the CLI
        # would silently have no effect on how much this adapter generates.
        gen_length = (
            request.config.get("gen_length")
            or self._step_config.gen_length
            or request.max_new_tokens
        )
        step_config = DiffusionStepConfig(
            gen_length=gen_length,
            steps=request.config.get("steps", self._step_config.steps),
            block_length=request.config.get("block_length", self._step_config.block_length),
            steps_per_block=request.config.get("steps_per_block", self._step_config.steps_per_block),
        )
        output_text, trace, final_valid_length = self._run_denoising(request.prompt, step_config)
        return GenerationResult(
            request=request,
            output_text=output_text,
            status=RunStatus.SUCCESS,
            trace=trace,
            num_forward_passes=len(trace),
            final_valid_length=final_valid_length,
        )


class IlladaAdapter(HFDiffusionAdapter):
    """Appendix D.1. Best: block_length=32, steps_per_block=32 (1 token/step).
    Fast: block_length=32, steps_per_block=16 (2 tokens/step)."""

    def __init__(self, model_name_or_path: str, step_config: DiffusionStepConfig, config_name: str, device: str | None = None) -> None:
        super().__init__(model_name_or_path, step_config, name="illada", config_name=config_name, device=device)

    def _run_denoising(self, prompt, step_config):
        raise NotImplementedError(
            "iLLaDA sampler not wired in yet: confirm the official checkpoint name and "
            "whether it exposes a HF-compatible generate() or needs a custom sampling-loop "
            "adapter shim (Appendix D.1) before this can run for real."
        )


class DreamAdapter(HFDiffusionAdapter):
    """Appendix D.2. Best: steps=gen_length. Fast: steps=gen_length/2."""

    def __init__(self, model_name_or_path: str, step_config: DiffusionStepConfig, config_name: str, device: str | None = None) -> None:
        super().__init__(model_name_or_path, step_config, name="dream", config_name=config_name, device=device)

    def _run_denoising(self, prompt, step_config):
        raise NotImplementedError(
            "Dream sampler not wired in yet: confirm whether the shipped HF checkpoint's "
            "sampler is already integrated into generate() or is a standalone script needing "
            "an adapter shim (Appendix D.2) before this can run for real."
        )
