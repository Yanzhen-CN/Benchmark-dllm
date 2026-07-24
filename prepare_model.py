#!/usr/bin/env python3
"""Pre-downloads/loads a model's HF checkpoint without running any
generation — useful for warming the cache before a benchmark run, especially
on a server where the default HF cache location (``~/.cache/huggingface``)
would land on local/ephemeral disk instead of the mounted network volume
(see ``dllm_bench.hf_cache``, which this script points at
``<cwd>/.hf_cache`` by default).

Every model here is loaded via ``from_pretrained(repo_id)`` — `--model-config`
just needs to point `init_kwargs.model_name_or_path` at the real HF repo ID
(see each `configs/models/*.yaml`'s comments); nothing needs a local
checkpoint path.

By default this warms *every* variant declared in the model config (Best and
Fast share one checkpoint, so this only downloads/loads it once — see
``models/model_cache.py``); pass ``--variant``/``--variants`` to narrow it.

    python prepare_model.py --model-config configs/models/illada.yaml
    python prepare_model.py --model-config configs/models/illada.yaml --variant best
"""

from __future__ import annotations

import argparse

from dllm_bench.hf_cache import configure_default_cache_dir

cache_dir = configure_default_cache_dir()

from dllm_bench.registry import build_model_adapter, list_model_variants


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-config", required=True, help="Path to configs/models/*.yaml")
    parser.add_argument("--variant", default=None, help="Warm just this one named config")
    parser.add_argument("--variants", default=None, help="Comma-separated named configs to warm (default: every variant in the file)")
    args = parser.parse_args()

    if args.variant and args.variants:
        raise SystemExit("pass either --variant or --variants, not both")
    if args.variant:
        variants = [args.variant]
    elif args.variants:
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    else:
        variants = list_model_variants(args.model_config)

    print(f"HF cache directory: {cache_dir}")
    print(f"Warming variants {variants} of {args.model_config} ...")

    for variant in variants:
        adapter = build_model_adapter(args.model_config, variant=variant)
        if not hasattr(adapter, "warm"):
            print(f"[{variant}] {adapter.name}: no local weights to warm (API-backed) — skipping")
            continue
        print(f"[{variant}] {adapter.name}: loading ...")
        adapter.warm()
        print(f"[{variant}] {adapter.name}: ready")

    print("Done.")


if __name__ == "__main__":
    main()
