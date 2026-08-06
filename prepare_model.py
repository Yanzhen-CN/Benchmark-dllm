#!/usr/bin/env python3
"""Downloads a model's HF checkpoint without loading it or running any
generation — useful for filling the cache before a benchmark run, especially
on a server where the default HF cache location (``~/.cache/huggingface``)
would land on local/ephemeral disk instead of the mounted network volume
(see ``dllm_bench.hf_cache``, which this script points at the repository's
``data/huggingface`` directory by default).

Every model here is loaded via ``from_pretrained(repo_id)`` — `--model-config`
just needs to point `init_kwargs.model_name_or_path` at the real HF repo ID
(see each `configs/models/*.yaml`'s comments); nothing needs a local
checkpoint path.

With no arguments, this dispatches every model in the full matrix through its
own isolated environment. By default direct ``--model-config`` mode prepares
*every* variant declared in that model config (P1 and P2 share one
checkpoint, so its repository snapshot is downloaded only once).
Pass ``-m`` to select matrix models or ``--variant``/``--variants`` in direct
mode to narrow variants.

    python prepare_model.py
    python prepare_model.py -m illada -m qwen3_4b
    python prepare_model.py -m illada_vargen
    python prepare_model.py --model-config configs/models/illada.yaml
    python prepare_model.py --model-config configs/models/illada.yaml --variant p1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "src"
for source_path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))


def _running_in_model_venv() -> bool:
    """True only for a per-model wrapper, never for an arbitrary active venv."""
    return bool(os.environ.get("DLLM_MODEL") and os.environ.get("DLLM_VENV"))


def _download_snapshot(repo_id: str, revision: str | None, cache_dir: Path) -> str:
    """Download one complete model repository without importing model code."""
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=repo_id,
        revision=revision,
        cache_dir=cache_dir / "hub",
    )


def _prepare_one(model_config: str, variant: str | None, variants_arg: str | None) -> None:
    from dllm_bench.hf_cache import configure_default_cache_dir
    from dllm_bench.registry import list_model_variants, load_yaml

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
    config = load_yaml(model_config)
    print(f"Preparing variants {variants} of {model_config} ...")
    snapshots: dict[tuple[str, str | None], list[str]] = {}
    for resolved_variant in variants:
        if resolved_variant not in config["configs"]:
            available = list(config["configs"])
            raise SystemExit(
                f"unknown variant {resolved_variant!r}; available: {available}"
            )
        init_kwargs = config["configs"][resolved_variant].get("init_kwargs", {})
        repositories: list[tuple[str, str | None]] = []
        for key, value in init_kwargs.items():
            if not key.endswith("model_name_or_path") or not value:
                continue
            revision_key = (
                "revision"
                if key == "model_name_or_path"
                else f"{key.removesuffix('_model_name_or_path')}_revision"
            )
            repositories.append((str(value), init_kwargs.get(revision_key)))
        if not repositories:
            print(
                f"[{resolved_variant}] {config['model']}: "
                "no Hugging Face checkpoint to download — skipping"
            )
            continue
        for repo_id, revision in repositories:
            snapshots.setdefault((repo_id, revision), []).append(resolved_variant)

    for (repo_id, revision), shared_variants in snapshots.items():
        revision_label = revision or "default revision"
        print(
            f"[{','.join(shared_variants)}] downloading {repo_id} "
            f"({revision_label}) ..."
        )
        snapshot_path = _download_snapshot(repo_id, revision, cache_dir)
        print(f"[{','.join(shared_variants)}] cached: {snapshot_path}")
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-config", help="Direct mode: path to one configs/models/*.yaml")
    parser.add_argument("--variant", default=None, help="Warm just this one named config")
    parser.add_argument("--variants", default=None, help="Comma-separated named configs to warm (default: every variant in the file)")
    parser.add_argument(
        "-m", "--model", action="extend", nargs="+", default=[],
        help="Matrix mode: model names; space-separate, repeat, or comma-separate (default: all)",
    )
    parser.add_argument("--matrix", default=str(PROJECT_ROOT / "configs" / "experiments" / "full_matrix.yaml"))
    parser.add_argument("--venv-scripts-dir", dest="scripts_dir", default=str(PROJECT_ROOT / "venv_scripts"))
    parser.add_argument("--dry-run", action="store_true", help="Print per-model prepare commands without running them")
    args = parser.parse_args()

    if args.variant and args.variants:
        raise SystemExit("pass either --variant or --variants, not both")

    if args.model_config and _running_in_model_venv() and not args.dry_run:
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
