"""Dream (Appendix D.2): confidence-based parallel unmasking via
``model.diffusion_generate(...)``.

**Unlike iLLaDA (models/illada.py) and DiffusionGemma (models/dg.py), no
local reference implementation was available for Dream in this repo** — per
project decision, this is implemented from Dream-7B's publicly documented HF
interface (``Dream-org/Dream-v0-Instruct-7B``'s model card:
``trust_remote_code=True``, a custom ``diffusion_generate`` method distinct
from standard ``generate()``, ``output_history`` for per-step snapshots, an
``alg`` parameter selecting the confidence/remasking strategy —
``"origin"``/``"maskgit_plus"``/``"topk_margin"``/``"entropy"``). Confirm
every parameter name/default here against the actual model card before
formal runs — this is a best-effort port, not a verified one, unlike its
two siblings.

Trace capture uses the shared snapshot-diffing fallback
(``models/trace_utils.py``) rather than a model-specific hook: Dream's public
interface exposes ``output_history`` (a full-sequence snapshot per step) but,
as far as this port can tell from public documentation, no direct per-step
per-position confidence/entropy the way iLLaDA/DG's own samplers do — so
``entropy_by_position``/``top1_confidence_by_position`` are left unset for
this model's trace. That's an honest reflection of what its native output
exposes, not a shortcut: don't fabricate confidence numbers Dream never gave us.
"""

from __future__ import annotations

from ..interfaces import TraceStep
from .hf_diffusion import DiffusionStepConfig, HFDiffusionAdapter
from .trace_utils import trace_steps_from_snapshots


class DreamAdapter(HFDiffusionAdapter):
    """Appendix D.2. Best: steps=gen_length. Fast: steps=gen_length/2."""

    def __init__(
        self,
        model_name_or_path: str,
        step_config: DiffusionStepConfig,
        config_name: str,
        device: str | None = None,
    ) -> None:
        super().__init__(model_name_or_path, step_config, name="dream", config_name=config_name, device=device)

    def _run_denoising(
        self, prompt: str, step_config: DiffusionStepConfig
    ) -> tuple[str, list[TraceStep], int]:
        import torch

        gen_length = step_config.gen_length
        steps = step_config.steps or gen_length
        temperature = float(step_config.extra.get("temperature", 0.2))
        top_p = step_config.extra.get("top_p", 0.95)
        alg = step_config.extra.get("alg", "entropy")
        alg_temp = float(step_config.extra.get("alg_temp", 0.0))

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        prompt_len = inputs["input_ids"].shape[1]

        mask_token_id = (
            step_config.extra.get("mask_token_id")
            or getattr(self._tokenizer, "mask_token_id", None)
            or getattr(self._model.config, "mask_token_id", None)
        )
        if mask_token_id is None:
            raise ValueError(
                "could not determine Dream's mask_token_id from the tokenizer/model config; "
                "set step_config.extra['mask_token_id'] explicitly (see configs/models/dream.yaml)"
            )

        with torch.no_grad():
            output = self._model.diffusion_generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=gen_length,
                output_history=True,
                return_dict_in_generate=True,
                steps=steps,
                temperature=temperature,
                top_p=top_p,
                alg=alg,
                alg_temp=alg_temp,
            )

        final_ids = output.sequences[0, prompt_len : prompt_len + gen_length].tolist()
        output_text = self._tokenizer.decode(final_ids, skip_special_tokens=True)

        gen_region_snapshots = [
            snapshot[0, prompt_len : prompt_len + gen_length].tolist() for snapshot in output.history
        ]
        trace = trace_steps_from_snapshots(gen_region_snapshots, mask_token_id, self._tokenizer)

        return output_text, trace, len(final_ids)
