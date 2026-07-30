from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from dllm_bench.interfaces import GenerationRequest, RunStatus
from dllm_bench.models import gemma_dflash
from dllm_bench.registry import build_model_adapter


def test_config_builds_parallel_adapter_without_starting_server():
    adapter = build_model_adapter("configs/models/gemma_dflash.yaml", variant="dflash")

    assert adapter.name == "gemma"
    assert adapter.config_name == "dflash"
    assert adapter.supports_trace is False
    assert adapter._model_name == "google/gemma-4-26B-A4B-it"
    assert adapter._draft_model_name == "z-lab/gemma-4-26B-A4B-it-DFlash"
    assert adapter._startup_timeout_seconds == 3600


def test_prometheus_parser_accepts_total_suffix_and_labels():
    parsed = gemma_dflash._parse_prometheus_metrics(
        """
        # HELP ignored ignored
        vllm:spec_decode_num_drafts_total{engine=\"0\"} 2
        vllm:spec_decode_num_draft_tokens_total{engine=\"0\"} 30
        vllm:spec_decode_num_accepted_tokens_total{engine=\"0\"} 12
        """
    )

    assert parsed["vllm:spec_decode_num_drafts"] == 2
    assert parsed["vllm:spec_decode_num_draft_tokens"] == 30
    assert parsed["vllm:spec_decode_num_accepted_tokens"] == 12


def test_managed_vllm_server_forces_offline_model_resolution(
    monkeypatch, tmp_path
):
    class FakeProcess:
        def poll(self):
            return None

    captured = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["environment"] = kwargs["env"]
        return FakeProcess()

    class HealthyResponse:
        ok = True

    monkeypatch.setattr(gemma_dflash.subprocess, "Popen", fake_popen)
    monkeypatch.setitem(
        gemma_dflash.sys.modules,
        "requests",
        SimpleNamespace(
            get=lambda *args, **kwargs: HealthyResponse(),
            RequestException=Exception,
        ),
    )
    monkeypatch.setattr(gemma_dflash, "data_root", lambda: tmp_path)
    monkeypatch.setattr(gemma_dflash.sys, "executable", str(tmp_path / "python"))
    executable = tmp_path / (
        "vllm.exe" if gemma_dflash.os.name == "nt" else "vllm"
    )
    executable.touch()

    server = gemma_dflash._ManagedVLLMServer()
    try:
        server.ensure(
            target="target",
            draft="draft",
            host="127.0.0.1",
            port=8000,
            startup_timeout_seconds=1,
            num_speculative_tokens=15,
            max_model_len=16384,
            max_num_batched_tokens=32768,
            gpu_memory_utilization=0.9,
        )
    finally:
        server._process = None
        server.close()

    assert captured["environment"]["HF_HUB_OFFLINE"] == "1"
    assert captured["environment"]["TRANSFORMERS_OFFLINE"] == "1"
    assert captured["environment"]["PATH"].split(gemma_dflash.os.pathsep)[0] == str(
        executable.parent
    )
    assert "--language-model-only" in captured["command"]


def test_generate_keeps_common_metrics_interface_and_adds_dflash_stats(monkeypatch):
    adapter = gemma_dflash.GemmaDFlashAdapter()
    adapter._base_url = "http://127.0.0.1:8000"
    adapter._tokenizer = SimpleNamespace()
    monkeypatch.setattr(adapter, "_ensure_loaded", lambda: None)
    monkeypatch.setattr(adapter, "_encode_prompt", lambda request: [1, 2, 3])
    snapshots = iter(
        [
            {
                "vllm:spec_decode_num_drafts": 10,
                "vllm:spec_decode_num_draft_tokens": 100,
                "vllm:spec_decode_num_accepted_tokens": 60,
            },
            {
                "vllm:spec_decode_num_drafts": 12,
                "vllm:spec_decode_num_draft_tokens": 130,
                "vllm:spec_decode_num_accepted_tokens": 72,
            },
        ]
    )
    monkeypatch.setattr(adapter, "_read_spec_metrics", lambda: next(snapshots))
    monkeypatch.setattr(
        adapter,
        "_stream_completion",
        lambda input_ids, request: ("answer", 13, 0.01),
    )

    @contextmanager
    def fake_vram():
        yield SimpleNamespace(available=True, peak_gb=55.0)

    @contextmanager
    def fake_energy(*, synchronize=True):
        assert synchronize is False
        yield SimpleNamespace(available=True, joules=42.0)

    monkeypatch.setattr(gemma_dflash, "measure_peak_device_memory_gb", fake_vram)
    monkeypatch.setattr(gemma_dflash, "measure_energy_joules", fake_energy)

    result = adapter.generate(
        GenerationRequest(prompt="question", max_new_tokens=32, sample_id="one")
    )

    assert result.status is RunStatus.SUCCESS
    assert result.timing is not None and result.timing.wall_clock_seconds > 0
    assert result.energy_joules == 42.0
    assert result.peak_vram_gb == 55.0
    assert result.final_valid_length == 13
    assert result.trace == []
    assert result.extra["target_verification_passes"] == 2
    assert result.extra["drafted_tokens"] == 30
    assert result.extra["accepted_draft_tokens"] == 12
    assert result.extra["draft_acceptance_rate"] == pytest.approx(0.4)
    assert result.extra["mean_acceptance_length"] == pytest.approx(7.0)
