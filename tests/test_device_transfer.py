from __future__ import annotations

from threading import Event

import pytest

from dllm_bench.models import device_transfer


class _Model:
    def __init__(self) -> None:
        self.devices = []

    def to(self, device):
        self.devices.append(device)
        return self


class _Device:
    def __init__(self, device_type: str) -> None:
        self.type = device_type


class _VramMonitor:
    def __init__(self) -> None:
        self.closed = False

    def snapshot(self) -> str:
        return "GPU 0 VRAM 20.0/80.0 GiB (25.0%)"

    def close(self) -> None:
        self.closed = True


def test_cuda_transfer_reports_phase_and_real_elapsed_time(monkeypatch, capsys):
    times = iter((10.0, 12.25))
    monkeypatch.setattr(device_transfer, "perf_counter", lambda: next(times))
    monitor = _VramMonitor()
    monkeypatch.setattr(device_transfer, "_GpuVramMonitor", lambda: monitor)
    model = _Model()
    device = _Device("cuda")

    assert device_transfer.move_model_to_device(
        model, device, model_name="org/checkpoint"
    ) is model

    assert model.devices == [device]
    assert monitor.closed
    assert capsys.readouterr().out.splitlines() == [
        "Moving checkpoint to GPU ... 0.0s elapsed | GPU 0 VRAM 20.0/80.0 GiB (25.0%)",
        "Moved checkpoint to GPU in 2.2s | GPU 0 VRAM 20.0/80.0 GiB (25.0%)",
    ]


def test_cpu_transfer_keeps_plain_model_to_without_progress_output(capsys):
    model = _Model()
    device = _Device("cpu")

    assert device_transfer.move_model_to_device(model, device) is model

    assert model.devices == [device]
    assert capsys.readouterr().out == ""


def test_slow_cuda_transfer_reports_elapsed_heartbeat(monkeypatch, capsys):
    heartbeat_reported = Event()
    release_transfer = Event()

    class _SlowModel(_Model):
        def to(self, device):
            self.devices.append(device)
            assert heartbeat_reported.wait(timeout=1.0)
            release_transfer.set()
            return self

    def report_once(stop, *, label, started, vram):
        del started
        print(
            f"Moving {label} to GPU ... 5.0s elapsed | {vram.snapshot()}",
            flush=True,
        )
        heartbeat_reported.set()
        stop.wait(timeout=1.0)

    monkeypatch.setattr(device_transfer, "_report_transfer_heartbeat", report_once)
    times = iter((10.0, 15.25))
    monkeypatch.setattr(device_transfer, "perf_counter", lambda: next(times))
    monitor = _VramMonitor()
    monkeypatch.setattr(device_transfer, "_GpuVramMonitor", lambda: monitor)

    model = _SlowModel()
    device = _Device("cuda")
    assert device_transfer.move_model_to_device(
        model, device, model_name="Qwen/Qwen3-8B"
    ) is model

    assert release_transfer.is_set()
    assert monitor.closed
    assert capsys.readouterr().out.splitlines() == [
        "Moving Qwen3-8B to GPU ... 0.0s elapsed | GPU 0 VRAM 20.0/80.0 GiB (25.0%)",
        "Moving Qwen3-8B to GPU ... 5.0s elapsed | GPU 0 VRAM 20.0/80.0 GiB (25.0%)",
        "Moved Qwen3-8B to GPU in 5.2s | GPU 0 VRAM 20.0/80.0 GiB (25.0%)",
    ]


def test_failed_cuda_transfer_still_closes_vram_monitor(monkeypatch):
    class _FailingModel(_Model):
        def to(self, device):
            del device
            raise RuntimeError("transfer failed")

    monitor = _VramMonitor()
    monkeypatch.setattr(device_transfer, "_GpuVramMonitor", lambda: monitor)

    with pytest.raises(RuntimeError, match="transfer failed"):
        device_transfer.move_model_to_device(_FailingModel(), _Device("cuda"))

    assert monitor.closed
