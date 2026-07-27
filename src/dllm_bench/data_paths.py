"""Persistent data directories shared by every model environment."""

from __future__ import annotations

import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def data_root() -> Path:
    """Return the configured data root without creating it."""
    configured = os.environ.get("DLLM_DATA_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return REPOSITORY_ROOT / ".data"


def ensure_data_layout(root: str | Path | None = None) -> dict[str, Path]:
    """Create and return the persistent cache/data directory layout."""
    resolved_root = (
        Path(root).expanduser().resolve() if root is not None else data_root()
    )
    paths = {
        "root": resolved_root,
        "huggingface": resolved_root / "huggingface",
        "datasets": resolved_root / "datasets",
        "pip_cache": resolved_root / "pip-cache",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths
