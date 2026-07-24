"""Points the HuggingFace cache at a project-relative directory by default.

``transformers``/``huggingface_hub`` normally cache downloaded checkpoints
under the user's home directory (``~/.cache/huggingface``). On a cloud GPU
box the persistent/large storage is often a network volume mounted at the
project directory, while the home directory sits on small local/ephemeral
disk — so the default silently fills up the wrong disk. Every model here is
loaded via ``from_pretrained(repo_id)`` (no local-path checkpoints needed;
see ``configs/models/*.yaml``'s comments), so where that download lands is
the only thing to control.

:func:`configure_default_cache_dir` sets ``HF_HOME`` to ``<cwd>/.hf_cache``
unless the caller has already set ``HF_HOME``/``HF_HUB_CACHE``/
``TRANSFORMERS_CACHE`` themselves (any of those always wins). Must run
before ``transformers``/``huggingface_hub`` are imported anywhere — both
``cli.py`` and ``prepare_model.py`` call this first, before any adapter's
lazy HF import happens.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_CACHE_DIRNAME = ".hf_cache"
_OVERRIDE_ENV_VARS = ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE")


def configure_default_cache_dir(base_dir: str | Path | None = None) -> Path:
    """Returns the effective HF cache directory, setting `HF_HOME` to a
    project-relative default only if the user hasn't already pinned one."""
    for var in _OVERRIDE_ENV_VARS:
        existing = os.environ.get(var)
        if existing:
            return Path(existing)

    cache_dir = Path(base_dir or Path.cwd()) / DEFAULT_CACHE_DIRNAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    return cache_dir
