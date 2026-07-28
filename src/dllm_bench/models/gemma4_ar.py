"""Gemma 4 26B-A4B autoregressive reference for DiffusionGemma.

The two checkpoints share the same 25.2B-total/3.8B-active Gemma 4 MoE
scale. Gemma 4 is loaded through its official multimodal auto classes, while
the text-only generation path reuses the benchmark's standard AR adapter.
"""

from __future__ import annotations

from .hf_ar import QwenARAdapter
from .model_cache import cpu_offload_max_memory


class Gemma4ARAdapter(QwenARAdapter):
    """Text-only Gemma 4 AR decoding with the common AR trace protocol."""

    def __init__(
        self,
        model_name_or_path: str = "google/gemma-4-26B-A4B-it",
        device: str | None = None,
        config_name: str = "ar-baseline",
        capture_trace: bool = True,
        enable_thinking: bool = False,
    ) -> None:
        super().__init__(
            model_name_or_path=model_name_or_path,
            device=device,
            config_name=config_name,
            capture_trace=capture_trace,
            enable_thinking=enable_thinking,
        )
        self.name = "gemma4_26b_a4b"
        self._inference_dtype = "bfloat16"

    def _load_model_and_tokenizer(self, device: str, *, device_map_auto: bool):
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        processor = AutoProcessor.from_pretrained(self._model_name)
        kwargs: dict = {"dtype": torch.bfloat16}
        if device_map_auto:
            kwargs["device_map"] = "auto"
            max_memory = cpu_offload_max_memory(device)
            if max_memory is not None:
                kwargs["max_memory"] = max_memory
        else:
            kwargs["low_cpu_mem_usage"] = True
        model = AutoModelForMultimodalLM.from_pretrained(
            self._model_name,
            **kwargs,
        )
        if not device_map_auto:
            model.to(device)
        model.eval()
        return processor, model
