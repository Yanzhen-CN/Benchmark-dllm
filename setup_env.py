#!/usr/bin/env python3
"""Installs dllm-bench (editable) into whatever Python is currently active.

    python setup_env.py                 # core + dev extra (pytest)
    python setup_env.py --extras hf,gpu # + torch/transformers + pynvml

No virtualenv is created — this just runs `pip install -e .[...]` against
the interpreter you invoke it with. Activate/create a venv yourself first if
you want isolation.
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extras",
        default="",
        help="comma-separated optional-dependency groups to install in addition to 'dev' (e.g. hf,gpu)",
    )
    args = parser.parse_args()

    groups = ["dev"] + [g.strip() for g in args.extras.split(",") if g.strip()]
    spec = ",".join(dict.fromkeys(groups))  # de-dupe, keep order

    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", f".[{spec}]"], check=True)

    print(f"\nDone. Installed extras: {spec}")
    print("Run tests with: python run_tests.py")


if __name__ == "__main__":
    main()
