from __future__ import annotations

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


def test_cuda_transfer_reports_phase_and_real_elapsed_time(monkeypatch, capsys):
    times = iter((10.0, 12.25))
    monkeypatch.setattr(device_transfer, "perf_counter", lambda: next(times))
    model = _Model()
    device = _Device("cuda")

    assert device_transfer.move_model_to_device(
        model, device, model_name="org/checkpoint"
    ) is model

    assert model.devices == [device]
    assert capsys.readouterr().out.splitlines() == [
        "Moving checkpoint to GPU ...",
        "Moved checkpoint to GPU in 2.2s",
    ]


def test_cpu_transfer_keeps_plain_model_to_without_progress_output(capsys):
    model = _Model()
    device = _Device("cpu")

    assert device_transfer.move_model_to_device(model, device) is model

    assert model.devices == [device]
    assert capsys.readouterr().out == ""
