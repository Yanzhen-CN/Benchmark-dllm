"""Unified Torch operator/module profiling for one deterministic replay."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import json
from pathlib import Path
from typing import Callable


def _category(name: str) -> str:
    lowered = name.lower()
    if any(part in lowered for part in ("attention", "attn", "scaled_dot_product", "flash")):
        return "attention"
    if any(part in lowered for part in ("moe", "expert", "router", "grouped_gemm")):
        return "moe"
    if any(part in lowered for part in ("mlp", "feed_forward", "feedforward")):
        return "mlp"
    if any(part in lowered for part in ("linear", "addmm", "matmul", "aten::mm")):
        return "linear"
    if any(part in lowered for part in ("norm", "rms")):
        return "normalization"
    if any(part in lowered for part in ("embed", "embedding")):
        return "embedding"
    if any(part in lowered for part in ("topk", "argmax", "softmax", "multinomial", "sort")):
        return "sampling"
    if "cache" in lowered:
        return "kv_cache"
    return "other"


def _module_category(name: str, module) -> str | None:
    category = _category(f"{name} {module.__class__.__name__}")
    return category if category != "other" else None


@contextmanager
def _module_ranges(model):
    """Add stable per-layer/module labels to eager-mode Torch profiles."""
    import torch

    handles = []
    active = defaultdict(list)
    for name, module in model.named_modules():
        if not name:
            continue
        category = _module_category(name, module)
        if category is None:
            continue

        def before(current, _inputs, *, module_name=name, module_category=category):
            marker = torch.autograd.profiler.record_function(
                f"dllm::module::{module_category}::{module_name}"
            )
            marker.__enter__()
            active[id(current)].append(marker)

        def after(current, _inputs, _output):
            stack = active.get(id(current))
            if stack:
                stack.pop().__exit__(None, None, None)

        handles.append(module.register_forward_pre_hook(before))
        handles.append(module.register_forward_hook(after))
    try:
        yield
    finally:
        for stack in active.values():
            while stack:
                stack.pop().__exit__(None, None, None)
        for handle in handles:
            handle.remove()


def _device_time_us(event) -> float:
    return float(
        getattr(event, "device_time_total", None)
        or getattr(event, "cuda_time_total", None)
        or 0.0
    )


def run_torch_profile(model, execute: Callable[[], None], artifact_dir: str | Path | None) -> dict:
    """Profile one replay and persist comparable operator/module summaries."""
    if artifact_dir is None:
        execute()
        return {"status": "disabled"}

    import torch

    out = Path(artifact_dir)
    out.mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    ) as profiler:
        with _module_ranges(model):
            execute()

    trace_path = out / "torch_trace.json"
    profiler.export_chrome_trace(str(trace_path))
    operator_rows = []
    module_rows = []
    categories = defaultdict(
        lambda: {
            "calls": 0,
            "cpu_time_us": 0.0,
            "device_time_us": 0.0,
            "flops": 0,
        }
    )
    for event in profiler.key_averages():
        key = str(event.key)
        row = {
            "name": key,
            "calls": int(event.count),
            "cpu_time_us": float(event.cpu_time_total),
            "device_time_us": _device_time_us(event),
            "flops": int(event.flops or 0),
        }
        if key.startswith("dllm::module::"):
            module_rows.append(row)
            continue
        operator_rows.append(row)
        bucket = categories[_category(key)]
        for metric in ("calls", "cpu_time_us", "device_time_us", "flops"):
            bucket[metric] += row[metric]

    sort_key = lambda row: (row["device_time_us"], row["cpu_time_us"])
    operator_rows.sort(key=sort_key, reverse=True)
    module_rows.sort(key=sort_key, reverse=True)
    summary = {
        "status": "complete",
        "trace_path": str(trace_path),
        "category_summary": dict(categories),
        "operators": operator_rows,
        "modules": module_rows,
    }
    summary_path = out / "torch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "status": "complete",
        "trace_path": str(trace_path),
        "summary_path": str(summary_path),
        "category_summary": dict(categories),
    }
