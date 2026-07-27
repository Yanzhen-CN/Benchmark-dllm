"""Pinned, checksummed downloads for official benchmark source files."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from ..data_paths import ensure_data_layout


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_download(
    dataset_name: str,
    filename: str,
    *,
    url: str,
    sha256: str,
) -> Path:
    target = ensure_data_layout()["datasets"] / "sources" / dataset_name / filename
    if target.is_file() and sha256_file(target).lower() == sha256.lower():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "dllm-bench/0.1"})
    try:
        with urlopen(request, timeout=120) as response:  # noqa: S310 - pinned HTTPS URL
            payload = response.read()
    except (OSError, URLError) as exc:
        raise RuntimeError(f"failed to download {dataset_name} source: {exc}") from exc

    digest = hashlib.sha256(payload).hexdigest()
    if digest.lower() != sha256.lower():
        raise RuntimeError(
            f"{dataset_name} checksum mismatch: expected {sha256}, got {digest}"
        )
    partial = target.with_suffix(target.suffix + ".part")
    try:
        partial.write_bytes(payload)
        os.replace(partial, target)
    finally:
        if partial.exists():
            partial.unlink()
    return target
