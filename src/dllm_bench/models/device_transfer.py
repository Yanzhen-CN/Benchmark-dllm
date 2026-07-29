"""Behavior-preserving model device transfer with honest phase logging."""

from __future__ import annotations

from time import perf_counter


def move_model_to_device(model, device, *, model_name: str | None = None):
    """Run ``model.to(device)`` and report CUDA transfer wall time.

    PyTorch exposes no reliable fractional progress callback for a whole-model
    ``to`` operation. Do not display a percentage bar that can only remain at
    zero until the entire transfer finishes.
    """
    device_type = getattr(device, "type", str(device).split(":", 1)[0])
    if str(device_type).lower() != "cuda":
        model.to(device)
        return model

    label = (model_name or model.__class__.__name__).rsplit("/", 1)[-1]
    print(f"Moving {label} to GPU ...", flush=True)
    started = perf_counter()
    model.to(device)
    elapsed = perf_counter() - started
    print(f"Moved {label} to GPU in {elapsed:.1f}s", flush=True)
    return model
