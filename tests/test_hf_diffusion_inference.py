"""Regression tests for the shared HF diffusion inference boundary."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from types import ModuleType

from dllm_bench.interfaces import GenerationRequest, RunStatus
from dllm_bench.models.hf_diffusion import DiffusionStepConfig, HFDiffusionAdapter


class _InferenceProbeAdapter(HFDiffusionAdapter):
    def __init__(self, torch_probe: ModuleType) -> None:
        super().__init__(
            "unused-checkpoint",
            DiffusionStepConfig(
                gen_length=4,
                extra={
                    "execution_path": "optimized",
                    "sampling_profile": "best",
                },
            ),
            name="probe",
            config_name="test",
        )
        self._torch_probe = torch_probe
        self.grad_enabled: bool | None = None
        self.inference_enabled: bool | None = None

    def _ensure_loaded(self) -> None:
        pass

    def _run_denoising(self, prompt, step_config, target_input_tokens=None):
        self.grad_enabled = self._torch_probe.is_grad_enabled()
        self.inference_enabled = self._torch_probe.is_inference_mode_enabled()
        return "ok", [], 1


def _make_torch_probe() -> ModuleType:
    probe = ModuleType("torch")
    probe._grad_enabled = True
    probe._inference_enabled = False

    @contextmanager
    def inference_mode():
        previous_grad = probe._grad_enabled
        previous_inference = probe._inference_enabled
        probe._grad_enabled = False
        probe._inference_enabled = True
        try:
            yield
        finally:
            probe._grad_enabled = previous_grad
            probe._inference_enabled = previous_inference

    probe.inference_mode = inference_mode
    probe.is_grad_enabled = lambda: probe._grad_enabled
    probe.is_inference_mode_enabled = lambda: probe._inference_enabled
    return probe


def test_hf_diffusion_generate_core_enforces_inference_mode(monkeypatch):
    torch_probe = _make_torch_probe()
    monkeypatch.setitem(sys.modules, "torch", torch_probe)
    adapter = _InferenceProbeAdapter(torch_probe)

    result = adapter._generate_core(
        GenerationRequest(prompt="hello", max_new_tokens=4)
    )

    assert result.status is RunStatus.SUCCESS
    assert adapter.grad_enabled is False
    assert adapter.inference_enabled is True
    assert result.extra["execution_path"] == "optimized"
    assert result.extra["sampling_profile"] == "best"
