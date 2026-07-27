"""Compute-per-sample profiling (Appendix B).

    ComputePerSample = TotalFLOPsOfFullGeneration / 1e12   [TFLOP/sample]

The design doc allows this to be measured via a separate profiling replay on
the same input/config/execution path rather than inline with every timed
run (FLOP counting has overhead and would contaminate the timing window).
``measure_compute_tflops`` follows that: call it as its own pass, not nested
inside :func:`resource.timing.measure_wall_clock`.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class ComputeHandle:
    tflops: float | None = None
    available: bool = True


def gqa_sdpa_flop_count(
    query_shape,
    key_shape,
    value_shape,
    *args,
    out_shape=None,
    **kwargs,
) -> int:
    """Count SDPA FLOPs for MHA, MQA and grouped-query attention.

    PyTorch 2.6's built-in formula asserts that Q/K/V have identical head
    counts. Qwen3 uses GQA, where K/V heads are shared across groups of query
    heads. The actual score and value products still execute once per query
    head, so both batched matmuls use ``query_heads`` in the FLOP formula.
    """
    batch, query_heads, query_tokens, query_dim = query_shape
    key_batch, key_heads, key_tokens, key_dim = key_shape
    value_batch, value_heads, value_tokens, value_dim = value_shape
    if batch != key_batch or batch != value_batch:
        raise ValueError("SDPA query/key/value batch sizes must match")
    if key_heads != value_heads or query_heads % key_heads != 0:
        raise ValueError("SDPA GQA requires matching KV heads that divide query heads")
    if query_dim != key_dim or key_tokens != value_tokens:
        raise ValueError("SDPA query/key dimensions and key/value lengths must match")

    score_flops = 2 * batch * query_heads * query_tokens * key_tokens * query_dim
    value_flops = 2 * batch * query_heads * query_tokens * key_tokens * value_dim
    return score_flops + value_flops


def _sdpa_custom_mapping():
    import torch

    mapping = {}
    for name in (
        "_scaled_dot_product_efficient_attention",
        "_scaled_dot_product_flash_attention",
        "_scaled_dot_product_cudnn_attention",
    ):
        packet = getattr(torch.ops.aten, name, None)
        if packet is not None:
            mapping[packet] = gqa_sdpa_flop_count
    return mapping


@contextmanager
def measure_compute_tflops() -> Iterator[ComputeHandle]:
    handle = ComputeHandle()
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        handle.available = False
        yield handle
        return

    counter = FlopCounterMode(
        display=False,
        custom_mapping=_sdpa_custom_mapping(),
    )
    try:
        with counter:
            yield handle
    finally:
        total_flops = counter.get_total_flops()
        handle.tflops = total_flops / 1e12
