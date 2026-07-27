#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible alias for the Qwen3-4B AR environment.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/runpod_model_env.sh" qwen3_4b
