"""Behavior-preserving model device transfer with honest phase logging."""

from __future__ import annotations

from threading import Event, Thread
from time import perf_counter


_TRANSFER_HEARTBEAT_SECONDS = 1.0
_GIB = 1024**3
_VRAM_BAR_WIDTH = 20


def _vram_bar(percent: float) -> str:
    bounded = min(100.0, max(0.0, percent))
    filled = round(_VRAM_BAR_WIDTH * bounded / 100.0)
    return f"[{'#' * filled}{'-' * (_VRAM_BAR_WIDTH - filled)}]"


class _TransferDisplay:
    """Render transfer status by replacing one terminal line in place."""

    def __init__(self) -> None:
        self._width = 0

    def update(self, message: str) -> None:
        self._width = max(self._width, len(message))
        print(f"\r{message.ljust(self._width)}", end="", flush=True)

    def finish(self, message: str) -> None:
        self._width = max(self._width, len(message))
        print(f"\r{message.ljust(self._width)}", flush=True)


class _GpuVramMonitor:
    """Best-effort NVML reader for live whole-device memory usage."""

    def __init__(self) -> None:
        self._pynvml = None
        self._devices: list[tuple[int, object]] = []
        try:
            import pynvml

            from ..resource.energy import _energy_gpu_indices

            pynvml.nvmlInit()
            self._pynvml = pynvml
            indices = _energy_gpu_indices(pynvml.nvmlDeviceGetCount())
            self._devices = [
                (index, pynvml.nvmlDeviceGetHandleByIndex(index))
                for index in indices
            ]
            if not self._devices:
                self.close()
        except Exception:
            self.close()

    def snapshot(self) -> str | None:
        if self._pynvml is None:
            return None
        try:
            parts = []
            for index, device in self._devices:
                memory = self._pynvml.nvmlDeviceGetMemoryInfo(device)
                used = int(memory.used)
                total = int(memory.total)
                percent = (100.0 * used / total) if total else 0.0
                parts.append(
                    f"GPU {index} VRAM {_vram_bar(percent)} {percent:5.1f}% "
                    f"({used / _GIB:.1f}/{total / _GIB:.1f} GiB)"
                )
            return ", ".join(parts) or None
        except Exception:
            return None

    def close(self) -> None:
        pynvml = self._pynvml
        self._pynvml = None
        self._devices = []
        if pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


def _transfer_status(label: str, elapsed: float, vram: _GpuVramMonitor) -> str:
    status = f"Moving {label} to GPU ... {elapsed:.1f}s elapsed"
    snapshot = vram.snapshot()
    return f"{status} | {snapshot}" if snapshot else status


def _report_transfer_heartbeat(
    stop: Event,
    *,
    label: str,
    started: float,
    vram: _GpuVramMonitor,
    display: _TransferDisplay,
) -> None:
    while not stop.wait(_TRANSFER_HEARTBEAT_SECONDS):
        elapsed = perf_counter() - started
        display.update(_transfer_status(label, elapsed, vram))


def run_gpu_loading_operation(operation, *, label: str):
    """Run a non-Transformers GPU load through the shared loading display.

    Backends such as vLLM own model construction and device placement inside
    their startup operation rather than exposing a ``model.to(device)`` call.
    This adapter preserves that backend path while reusing the same elapsed
    time and NVML VRAM heartbeat shown for in-process model transfers.
    """

    class _OperationAdapter:
        result = None

        def to(self, _device) -> None:
            self.result = operation()

    adapter = _OperationAdapter()
    move_model_to_device(adapter, "cuda", model_name=label)
    return adapter.result


def move_model_to_device(model, device, *, model_name: str | None = None):
    """Run ``model.to(device)`` and report CUDA transfer wall time.

    PyTorch exposes no reliable fractional progress callback for a whole-model
    ``to`` operation. Report an elapsed-time heartbeat instead of displaying a
    percentage bar that can only remain at zero until the transfer finishes.
    """
    device_type = getattr(device, "type", str(device).split(":", 1)[0])
    if str(device_type).lower() != "cuda":
        model.to(device)
        return model

    label = (model_name or model.__class__.__name__).rsplit("/", 1)[-1]
    started = perf_counter()
    vram = _GpuVramMonitor()
    display = _TransferDisplay()
    display.update(_transfer_status(label, 0.0, vram))
    stop_heartbeat = Event()
    heartbeat = Thread(
        target=_report_transfer_heartbeat,
        kwargs={
            "stop": stop_heartbeat,
            "label": label,
            "started": started,
            "vram": vram,
            "display": display,
        },
        daemon=True,
    )
    heartbeat.start()
    final_vram = None
    succeeded = False
    try:
        model.to(device)
        succeeded = True
    finally:
        stop_heartbeat.set()
        heartbeat.join()
        final_vram = vram.snapshot()
        vram.close()
        elapsed = perf_counter() - started
        suffix = f" | {final_vram}" if final_vram else ""
        if succeeded:
            display.finish(f"Moved {label} to GPU in {elapsed:.1f}s{suffix}")
        else:
            display.finish(
                f"Failed moving {label} to GPU after {elapsed:.1f}s{suffix}"
            )
    return model
