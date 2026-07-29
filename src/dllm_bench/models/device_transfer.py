"""Visible, behavior-preserving model device transfer."""

from __future__ import annotations

from threading import Event, Thread


def move_model_to_device(model, device, *, model_name: str | None = None):
    """Run ``model.to(device)`` with a visible CUDA transfer phase.

    PyTorch exposes no reliable fractional progress callback for a whole-model
    ``to`` operation. The bar therefore represents the transfer as one phase
    and refreshes its elapsed time while the original operation is running.
    """
    device_type = getattr(device, "type", str(device).split(":", 1)[0])
    if str(device_type).lower() != "cuda":
        model.to(device)
        return model

    from tqdm.auto import tqdm

    label = (model_name or model.__class__.__name__).rsplit("/", 1)[-1]
    with tqdm(
        total=1,
        desc=f"Move {label} to GPU",
        unit="model",
        dynamic_ncols=True,
    ) as progress:
        finished = Event()

        def refresh_elapsed() -> None:
            while not finished.wait(0.5):
                progress.refresh()

        refresher = Thread(target=refresh_elapsed, daemon=True)
        refresher.start()
        try:
            model.to(device)
            progress.update(1)
        finally:
            finished.set()
            refresher.join(timeout=1.0)
    return model
