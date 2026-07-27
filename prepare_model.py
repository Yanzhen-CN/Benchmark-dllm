#!/usr/bin/env python3
"""Pre-downloads/loads a model's HF checkpoint without running any
generation — useful for warming the cache before a benchmark run, especially
on a server where the default HF cache location (``~/.cache/huggingface``)
would land on local/ephemeral disk instead of the mounted network volume
(see ``dllm_bench.hf_cache``, which this script points at the repository's
``.data/huggingface`` directory by default).

Every model here is loaded via ``from_pretrained(repo_id)`` — `--model-config`
just needs to point `init_kwargs.model_name_or_path` at the real HF repo ID
(see each `configs/models/*.yaml`'s comments); nothing needs a local
checkpoint path.

With no arguments, this dispatches every model in the full matrix through its
own isolated environment. By default direct ``--model-config`` mode warms
*every* variant declared in that model config (Best and Fast share one
checkpoint, so this only downloads/loads it once — see ``models/model_cache.py``).
Pass ``-m`` to select matrix models or ``--variant``/``--variants`` in direct
mode to narrow variants.

    python prepare_model.py
    python prepare_model.py -m illada -m qwen3_4b
    python prepare_model.py --model-config configs/models/illada.yaml
    python prepare_model.py --model-config configs/models/illada.yaml --variant best
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _running_in_venv() -> bool:
    return bool(os.environ.get("DLLM_VENV")) or sys.prefix != sys.base_prefix


def _prepare_one(model_config: str, variant: str | None, variants_arg: str | None) -> None:
    from dllm_bench.hf_cache import configure_default_cache_dir
    from dllm_bench.registry import build_model_adapter, list_model_variants

    cache_dir = configure_default_cache_dir()

    if variant and variants_arg:
        raise SystemExit("pass either --variant or --variants, not both")
    if variant:
        variants = [variant]
    elif variants_arg:
        variants = [value.strip() for value in variants_arg.split(",") if value.strip()]
    else:
        variants = list_model_variants(model_config)

    print(f"HF cache directory: {cache_dir}")
    print(f"Warming variants {variants} of {model_config} ...")
    for resolved_variant in variants:
        adapter = build_model_adapter(model_config, variant=resolved_variant)
        if not hasattr(adapter, "warm"):
            print(
                f"[{resolved_variant}] {adapter.name}: "
                "no local weights to warm (API-backed) — skipping"
            )
            continue
        print(f"[{resolved_variant}] {adapter.name}: loading ...")
        adapter.warm()
        print(f"[{resolved_variant}] {adapter.name}: ready")
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-config", help="Direct mode: path to one configs/models/*.yaml")
    parser.add_argument("--variant", default=None, help="Warm just this one named config")
    parser.add_argument("--variants", default=None, help="Comma-separated named configs to warm (default: every variant in the file)")
    parser.add_argument("-m", "--model", action="append", default=[], help="Matrix mode: model name; repeat or comma-separate (default: all)")
    parser.add_argument("--matrix", default=str(PROJECT_ROOT / "configs" / "experiments" / "full_matrix.yaml"))
    parser.add_argument("--venv-scripts-dir", dest="scripts_dir", default=str(PROJECT_ROOT / "venv_scripts"))
    parser.add_argument("--dry-run", action="store_true", help="Print per-model prepare commands without running them")
    args = parser.parse_args()

    if args.variant and args.variants:
        raise SystemExit("pass either --variant or --variants, not both")

    if args.model_config and _running_in_venv() and not args.dry_run:
        if args.model:
            raise SystemExit("pass either --model-config or -m/--model, not both")
        _prepare_one(args.model_config, args.variant, args.variants)
        return

    if not args.model_config and (args.variant or args.variants):
        raise SystemExit("--variant/--variants require --model-config")

    from run_bench import (
        dispatch_model_scripts,
        matrix_model_names,
        normalize_model_names,
    )

    if args.model_config:
        if args.model:
            raise SystemExit("pass either --model-config or -m/--model, not both")
        model_name = Path(args.model_config).stem
        environment: dict[str, str] = {}
        if args.variant:
            environment["PREPARE_MODEL_VARIANT"] = args.variant
        if args.variants:
            environment["PREPARE_MODEL_VARIANTS"] = args.variants
        dispatch_model_scripts(
            [model_name],
            action="prepare",
            scripts_dir=args.scripts_dir,
            env_updates=environment,
            dry_run=args.dry_run,
        )
        return

    matrix_path = Path(args.matrix).resolve()
    available = matrix_model_names(matrix_path)
    try:
        selected = normalize_model_names(args.model, available)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Matrix: {matrix_path}")
    print(f"Models: {', '.join(selected)}")
    dispatch_model_scripts(
        selected,
        action="prepare",
        scripts_dir=args.scripts_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
