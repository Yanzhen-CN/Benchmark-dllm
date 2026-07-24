"""Process-wide cache of loaded ``(tokenizer, model)`` pairs, keyed by
``(model_name_or_path, device)``.

Best/Fast (or standard/jump/gidd) are different *generation-time* configs
of the *same* checkpoint — the whole reason `configs/models/illada.yaml`
nests them under one file (see README) is to test them together without
paying the model-load cost twice. The CLI achieves the "once per process"
part by sweeping every variant in one invocation
(``cli.py``'s `generate`/`score`/`visualize` default to all variants); this
module achieves the "actually shared" part: building a second adapter
instance for the same checkpoint reuses the first one's in-memory weights
instead of calling `from_pretrained` and `.to(device)` again.

Not meaningful across separate process invocations — it's an in-memory
cache, not a disk one (that's `hf_cache.py`'s job).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_CACHE: dict[tuple[str, str], Any] = {}


def get_or_load(model_name_or_path: str, device: str, loader: Callable[[], Any]) -> Any:
    key = (model_name_or_path, device)
    if key not in _CACHE:
        _CACHE[key] = loader()
    return _CACHE[key]


def clear() -> None:
    """Mainly for tests — drops every cached (tokenizer, model) pair."""
    _CACHE.clear()
