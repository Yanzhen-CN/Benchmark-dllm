"""Points the HuggingFace cache at a project-relative directory by default.

``transformers``/``huggingface_hub`` normally cache downloaded checkpoints
under the user's home directory (``~/.cache/huggingface``). On a cloud GPU
box the persistent/large storage is often a network volume mounted at the
project directory, while the home directory sits on small local/ephemeral
disk — so the default silently fills up the wrong disk. Every model here is
loaded via ``from_pretrained(repo_id)`` (no local-path checkpoints needed;
see ``configs/models/*.yaml``'s comments), so where that download lands is
the only thing to control.

:func:`configure_default_cache_dir` binds every Hugging Face cache variable
to ``<DLLM_DATA_ROOT>/huggingface``. Generic variables inherited from a cloud
image are deliberately replaced so that no model venv can silently redirect
downloads to an ephemeral home directory. Must run
before ``transformers``/``huggingface_hub`` are imported anywhere — both
``cli.py`` and ``prepare_model.py`` call this first, before any adapter's
lazy HF import happens.
"""

from __future__ import annotations

import os
from pathlib import Path

def configure_default_cache_dir(base_dir: str | Path | None = None) -> Path:
    """Return and export the single HF cache shared by every model venv."""
    if base_dir is None:
        from .data_paths import ensure_data_layout

        cache_dir = ensure_data_layout()["huggingface"]
    else:
        cache_dir = (Path(base_dir) / "data" / "huggingface").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    hub_dir = cache_dir / "hub"
    hub_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(hub_dir)
    os.environ["HF_XET_CACHE"] = str(cache_dir / "xet")
    os.environ["HF_ASSETS_CACHE"] = str(cache_dir / "assets")
    os.environ.pop("TRANSFORMERS_CACHE", None)
    os.environ.pop("HUGGINGFACE_HUB_CACHE", None)
    return cache_dir
