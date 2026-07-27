#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?Usage: model_command.sh <model> [setup|check|prepare|run]}"
ACTION="${2:-run}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/model_profile.sh"
dllm_load_model_profile "${MODEL}"

cd "${REPO_ROOT}"
VENV_DIR="${DLLM_VENV_DIR:-${REPO_ROOT}/${DLLM_VENV_NAME}}"

print_usage() {
  echo "Usage: bash scripts/${DLLM_MODEL_ID}.sh [setup|check|prepare|run]"
  echo "  setup    Create or update the model environment"
  echo "  check    Validate packages and adapter construction"
  echo "  prepare  Download and load the model checkpoint"
  echo "  run      Generate, score, and report (default)"
}

ensure_environment() {
  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    DLLM_VENV_DIR="${VENV_DIR}" bash "${SCRIPT_DIR}/model_env.sh" "${DLLM_MODEL_ID}"
  fi
}

activate_environment() {
  set +u
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
  set -u
  export DLLM_MODEL="${DLLM_MODEL_ID}"
  export DLLM_MODEL_CONFIG
  export DLLM_VENV="${VENV_DIR}"
}

case "${ACTION}" in
  help|-h|--help)
    print_usage
    ;;
  setup)
    DLLM_VENV_DIR="${VENV_DIR}" bash "${SCRIPT_DIR}/model_env.sh" "${DLLM_MODEL_ID}"
    ;;
  check)
    ensure_environment
    activate_environment
    python -m pip check
    python - <<PY
from dllm_bench.registry import build_model_adapter, list_model_variants

config = "${DLLM_MODEL_CONFIG}"
variants = list_model_variants(config)
for variant in variants:
    adapter = build_model_adapter(config, variant=variant)
    print(f"{adapter.name}:{adapter.config_name} adapter OK")
PY
    if [[ "${DLLM_MODEL_ID}" == "diffusiongemma" ]]; then
      python -c "from transformers import DiffusionGemmaForBlockDiffusion; print('DiffusionGemma class OK')"
    fi
    ;;
  prepare)
    ensure_environment
    activate_environment
    python prepare_model.py --model-config "${DLLM_MODEL_CONFIG}"
    ;;
  run)
    ensure_environment
    activate_environment
    exec bash "${SCRIPT_DIR}/model_pipeline.sh" "${DLLM_MODEL_ID}"
    ;;
  *)
    echo "Unknown action '${ACTION}'. Use setup, check, prepare, or run." >&2
    exit 2
    ;;
esac
