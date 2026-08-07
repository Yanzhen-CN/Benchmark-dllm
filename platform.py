"""Root launcher for the local Benchmark-dllm platform.

Run ``python platform.py`` from the repository root.  When imported by code
that expects Python's standard-library ``platform`` module, this file proxies
that module instead of exposing the launcher.
"""

from __future__ import annotations


if __name__ == "__main__":
    import runpy
    from pathlib import Path

    runpy.run_path(
        str(Path(__file__).resolve().parent / "platform" / "start.py"),
        run_name="__main__",
    )
else:
    import importlib.util
    import sysconfig
    from pathlib import Path

    _stdlib_path = Path(sysconfig.get_path("stdlib")) / "platform.py"
    _stdlib_spec = importlib.util.spec_from_file_location(
        "_benchmark_stdlib_platform", _stdlib_path
    )
    if _stdlib_spec is None or _stdlib_spec.loader is None:
        raise ImportError(f"Cannot load the standard-library platform module: {_stdlib_path}")
    _stdlib_module = importlib.util.module_from_spec(_stdlib_spec)
    _stdlib_spec.loader.exec_module(_stdlib_module)
    for _name in dir(_stdlib_module):
        if _name not in {"__name__", "__package__", "__loader__", "__spec__"}:
            globals()[_name] = getattr(_stdlib_module, _name)

