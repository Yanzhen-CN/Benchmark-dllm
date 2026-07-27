#!/usr/bin/env bash

# Source this file to create (on first use) and activate one model's env:
#   source scripts/use_model_env.sh qwen3_4b

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "This script must be sourced so activation persists in your shell." >&2
  echo "Run: source scripts/use_model_env.sh <model>" >&2
  exit 2
fi

_dllm_use_model_env() {
  local model="${1:-qwen3_4b}"
  local script_dir repo_root default_venv model_config venv_dir venv_path
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  repo_root="$(cd "${script_dir}/.." && pwd)"

  case "${model}" in
    qwen3_4b|qwen|ar)
      model="qwen3_4b"
      default_venv=".venv-qwen3-ar"
      model_config="configs/models/qwen3_4b.yaml"
      ;;
    illada)
      default_venv=".venv-illada"
      model_config="configs/models/illada.yaml"
      ;;
    dream)
      default_venv=".venv-dream"
      model_config="configs/models/dream.yaml"
      ;;
    dg|diffusiongemma)
      model="dg"
      default_venv=".venv-dg"
      model_config="configs/models/dg.yaml"
      ;;
    w1)
      default_venv=".venv-w1"
      model_config="configs/models/w1.yaml"
      ;;
    mock)
      default_venv=".venv-mock"
      model_config="configs/models/mock.yaml"
      ;;
    *)
      echo "Unknown model '${model}'." >&2
      echo "Use one of: qwen3_4b, illada, dream, dg, w1, mock." >&2
      return 2
      ;;
  esac

  venv_dir="${DLLM_VENV_DIR:-${default_venv}}"
  case "${venv_dir}" in
    /*) venv_path="${venv_dir}" ;;
    *) venv_path="${repo_root}/${venv_dir}" ;;
  esac

  if [[ ! -x "${venv_path}/bin/python" ]]; then
    VENV_DIR="${venv_dir}" bash "${script_dir}/runpod_model_env.sh" "${model}" || return $?
  fi

  # shellcheck disable=SC1091
  source "${venv_path}/bin/activate"
  cd "${repo_root}"

  export DLLM_MODEL="${model}"
  export DLLM_MODEL_CONFIG="${model_config}"
  export DLLM_VENV="${venv_path}"

  echo "Active model: ${DLLM_MODEL}"
  echo "Environment: ${DLLM_VENV}"
  echo "Model config: ${DLLM_MODEL_CONFIG}"
}

_dllm_use_model_env "$@"
