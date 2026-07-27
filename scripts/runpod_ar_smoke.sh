#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible alias. Passing another model name is still supported.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/runpod_model_pipeline.sh" "${1:-qwen3_4b}"
