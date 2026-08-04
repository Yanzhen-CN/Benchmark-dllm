from __future__ import annotations

from io import StringIO
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
        return "GPU 0 VRAM [#####---------------]  25.0% (20.0/80.0 GiB)"

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
    output = capsys.readouterr().out
    assert output.count("\r") == 0
    assert output.count("\n") == 1
    assert "GPU 0 VRAM [#####---------------]  25.0% (20.0/80.0 GiB)" in output
    assert "Moved checkpoint to GPU in 2.2s" in output


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

    def report_once(stop, *, label, started, vram, display):
        del started
        display.update(
            f"Moving {label} to GPU ... 5.0s elapsed | {vram.snapshot()}"
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
    output = capsys.readouterr().out
    assert output.count("\r") == 0
    assert output.count("\n") == 1
    assert "Moved Qwen3-8B to GPU in 5.2s" in output


def test_interactive_display_clears_and_reuses_one_bounded_terminal_line():
    class _InteractiveStream(StringIO):
        def isatty(self):
            return True

    stream = _InteractiveStream()
    display = device_transfer._TransferDisplay(stream, terminal_columns=40)
    display.update("Moving a-very-long-checkpoint-name to GPU with VRAM details")
    display.update("Moving checkpoint to GPU ... 5.0s")
    display.finish("Moved checkpoint to GPU in 6.0s")

    output = stream.getvalue()
    assert output.count("\r\x1b[2K") == 3
    assert output.count("\n") == 1
    rendered = [part.split("\n", 1)[0] for part in output.split("\r\x1b[2K") if part]
    assert all(len(line) <= 39 for line in rendered)
    assert "..." in rendered[0]


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


def test_vram_bar_is_bounded_and_has_fixed_width():
    assert device_transfer._vram_bar(25.0) == "[#####---------------]"
    assert device_transfer._vram_bar(-10.0) == "[--------------------]"
    assert device_transfer._vram_bar(120.0) == "[####################]"
