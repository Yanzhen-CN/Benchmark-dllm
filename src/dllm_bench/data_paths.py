"""Persistent data directories shared by every model environment."""

from __future__ import annotations

import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT_ENV = "DLLM_DATA_ROOT"
DATA_CACHE_ENV = "DLLM_DATA_CACHE"


def project_root() -> Path:
    """Backward-compatible accessor for the repository root."""
    return REPOSITORY_ROOT


def data_root() -> Path:
    """Return the configured data root without creating it."""
    configured = os.environ.get(DATA_ROOT_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    return REPOSITORY_ROOT / "data"


def ensure_data_layout(root: str | Path | None = None) -> dict[str, Path]:
    """Create and return the persistent cache/data directory layout."""
    resolved_root = (
        Path(root).expanduser().resolve() if root is not None else data_root()
    )
    paths = {
        "root": resolved_root,
        "huggingface": resolved_root / "huggingface",
        "datasets": resolved_root / "datasets",
        "tmp": resolved_root / "tmp",
        "torch_extensions": resolved_root / "torch-extensions",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    # Keep the legacy dataset-cache variable synchronized. DLLM_DATA_ROOT (or
    # the explicit ``root`` argument) is authoritative for the unified layout.
    os.environ[DATA_CACHE_ENV] = str(paths["datasets"])
    return paths
