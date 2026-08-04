"""Discovery for model-owned visualization declarations."""

from __future__ import annotations

import importlib

from ..base import ModelVisual, public_model_visual


def load_model_visual(model_name: str) -> ModelVisual:
    """Load one model declaration without maintaining a central model list."""
    module_key = model_name.replace("-", "_")
    module_name = f"{__name__}.{module_key}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            return public_model_visual(model_name)
        raise

    visual = getattr(module, "MODEL_VISUAL", None)
    if not isinstance(visual, ModelVisual):
        raise RuntimeError(f"{module_name} must define MODEL_VISUAL")
    if visual.model_name not in {model_name, module_key}:
        raise RuntimeError(
            f"{module_name} declares model {visual.model_name!r}, "
            f"expected {model_name!r}"
        )
    return visual
