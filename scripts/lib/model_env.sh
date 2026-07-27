#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?Usage: model_env.sh <model>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/model_profile.sh"
dllm_load_model_profile "${MODEL}"

CUDA_INDEX="${CUDA_INDEX:-cu124}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${DLLM_VENV_DIR:-${REPO_ROOT}/${DLLM_VENV_NAME}}"
export PIP_CACHE_DIR="${DLLM_PIP_CACHE_DIR:-${REPO_ROOT}/.pip_cache}"

case "${CUDA_INDEX}" in
  cu118|cu121|cu124|cu126) ;;
  *)
    echo "Unsupported CUDA_INDEX=${CUDA_INDEX}. Use cu118, cu121, cu124, or cu126." >&2
    exit 2
    ;;
esac

if [[ -n "${DLLM_TORCH_VERSION}" && " ${DLLM_TORCH_CUDA_INDEXES} " != *" ${CUDA_INDEX} "* ]]; then
  echo "torch ${DLLM_TORCH_VERSION} for ${DLLM_MODEL_ID} has no ${CUDA_INDEX} wheel." >&2
  echo "Supported CUDA indexes: ${DLLM_TORCH_CUDA_INDEXES}." >&2
  exit 2
fi

cd "${REPO_ROOT}"
mkdir -p "${PIP_CACHE_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"

set +u
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
set -u

python -m pip install --upgrade pip setuptools wheel

if [[ -n "${DLLM_TORCH_VERSION}" ]]; then
  python -m pip install --upgrade \
    "torch==${DLLM_TORCH_VERSION}" \
    --index-url "https://download.pytorch.org/whl/${CUDA_INDEX}"
  python -m pip uninstall -y torchvision torchaudio || true
fi

if [[ -n "${DLLM_TRANSFORMERS_VERSION}" ]]; then
  python -m pip install --upgrade \
    "transformers==${DLLM_TRANSFORMERS_VERSION}" \
    "accelerate==1.14.0" \
    "safetensors>=0.8.0" \
    sentencepiece
fi

python -m pip install -e ".[${DLLM_EXTRAS}]"
python -m pip check

python - <<'PY'
import importlib.util

print("Environment import check")
if importlib.util.find_spec("torch"):
    import torch
    print("torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA runtime:", torch.version.cuda)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

if importlib.util.find_spec("transformers"):
    import transformers
    print("transformers:", transformers.__version__)

if importlib.util.find_spec("requests"):
    import requests
    print("requests:", requests.__version__)
PY

echo
echo "Environment ready: ${DLLM_MODEL_ID}"
echo "Path: ${VENV_DIR}"
