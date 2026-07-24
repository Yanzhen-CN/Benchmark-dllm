"""DiffusionGemma (Appendix D.4): official checkpoint, HF-loadable directly —
``google/diffusiongemma-26B-A4B-it`` / ``DiffusionGemmaForBlockDiffusion``,
``model_type diffusion_gemma``, no adapter shim needed for loading itself.

Trace capture follows *How DiffusionGemma Actually Commits Tokens*: a purely
observational forward hook on the sampler's ``accept_canvas`` method records
which positions were committed each call and their entropy, without touching
model weights or sampling decisions. The exact hook argument shape
(``accept_canvas``'s real signature) is not pinned down here — ``_on_accept``
extracts commonly-named fields defensively and is meant to be adjusted to
match the actual method signature once run against the real checkpoint,
per the same "verify before formal runs" caveat as the iLLaDA/Dream adapters.
"""

from __future__ import annotations

from ..interfaces import GenerationRequest, GenerationResult, PositionState, RunStatus, TraceStep
from .base import BaseModelAdapter
from .model_cache import get_or_load

DEFAULT_DG_CHECKPOINT = "google/diffusiongemma-26B-A4B-it"


class DGAdapter(BaseModelAdapter):
    def __init__(
        self,
        model_name_or_path: str = DEFAULT_DG_CHECKPOINT,
        device: str | None = None,
        config_name: str = "official",
    ) -> None:
        self.name = "dg"
        self.config_name = config_name
        self.supports_trace = True
        self.natively_measures_resources = False
        self._model_name = model_name_or_path
        self._device = device
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
            model = AutoModelForCausalLM.from_pretrained(self._model_name, trust_remote_code=True)
            model.to(device)
            model.eval()
            return tokenizer, model

        self._tokenizer, self._model = get_or_load(self._model_name, device, _load)

    def _generate_core(self, request: GenerationRequest) -> GenerationResult:
        self._ensure_loaded()
        import torch

        trace: list[TraceStep] = []

        def _on_accept(_module, _args, kwargs, _output) -> None:
            # TODO: confirm accept_canvas's real argument names against the
            # shipped sampler; this defensively looks for the fields the
            # reference hook technique describes (committed positions + their
            # entropy) so a mismatch degrades to an empty trace step instead
            # of crashing generation.
            try:
                positions = list(kwargs.get("accepted_positions", []))
                entropies = dict(kwargs.get("entropies", {}))
                canvas = kwargs.get("canvas")
                position_states = (
                    [PositionState.ACCEPTED if i in positions else PositionState.MASKED for i in range(len(canvas))]
                    if canvas is not None
                    else []
                )
                trace.append(
                    TraceStep(
                        forward_index=len(trace),
                        token_ids=list(canvas) if canvas is not None else [],
                        position_states=position_states,
                        committed_positions=positions,
                        decoded_text="",
                        entropy_by_position=entropies or None,
                    )
                )
            except Exception:  # noqa: BLE001 - observational hook must never break generation
                return

        handle = None
        sampler = getattr(self._model, "sampler", None)
        accept_canvas = getattr(sampler, "accept_canvas", None) if sampler is not None else None
        if accept_canvas is not None and hasattr(accept_canvas, "register_forward_hook"):
            handle = accept_canvas.register_forward_hook(_on_accept, with_kwargs=True)

        try:
            inputs = self._tokenizer(request.prompt, return_tensors="pt").to(self._device)
            prompt_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                output = self._model.generate(**inputs, max_new_tokens=request.max_new_tokens)
            generated_ids = output[0][prompt_len:]
            text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        finally:
            if handle is not None:
                handle.remove()

        return GenerationResult(
            request=request,
            output_text=text,
            status=RunStatus.SUCCESS,
            trace=trace,
            num_forward_passes=len(trace),
            final_valid_length=len(generated_ids),
        )
