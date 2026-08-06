import pytest

from dllm_bench.resource.energy import EnergyUnavailableError, _energy_gpu_indices


def test_energy_defaults_to_first_benchmark_gpu(monkeypatch):
    monkeypatch.delenv("DLLM_NVML_GPU_INDICES", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert _energy_gpu_indices(4) == [0]


def test_energy_maps_numeric_cuda_visible_device(monkeypatch):
    monkeypatch.delenv("DLLM_NVML_GPU_INDICES", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,1")
    assert _energy_gpu_indices(4) == [3]


def test_energy_supports_explicit_multi_gpu_selection(monkeypatch):
    monkeypatch.setenv("DLLM_NVML_GPU_INDICES", "1,3")
    assert _energy_gpu_indices(4) == [1, 3]


def test_energy_rejects_invalid_explicit_device(monkeypatch):
    monkeypatch.setenv("DLLM_NVML_GPU_INDICES", "7")
    with pytest.raises(EnergyUnavailableError):
        _energy_gpu_indices(2)
