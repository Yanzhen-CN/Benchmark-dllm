#!/usr/bin/env bash
set -euo pipefail

# Create a model-specific RunPod environment.
#
# Usage:
#   bash scripts/runpod_model_env.sh qwen3_4b
#   bash scripts/runpod_model_env.sh illada
#   bash scripts/runpod_model_env.sh dream
#   bash scripts/runpod_model_env.sh w1
#   bash scripts/runpod_model_env.sh dg
#   bash scripts/runpod_model_env.sh mock
#   bash scripts/runpod_model_env.sh all
#
# Optional:
#   CUDA_INDEX=cu126 bash scripts/runpod_model_env.sh qwen3_4b
#   CUDA_INDEX=cu121 bash scripts/runpod_model_env.sh dream
#   VENV_DIR=.venv-custom bash scripts/runpod_model_env.sh qwen3_4b

MODEL="${1:-qwen3_4b}"
CUDA_INDEX="${CUDA_INDEX:-cu124}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

if [[ "${MODEL}" == "all" ]]; then
  for model_name in qwen3_4b illada dream dg w1 mock; do
    CUDA_INDEX="${CUDA_INDEX}" PYTHON_BIN="${PYTHON_BIN}" \
      bash "${BASH_SOURCE[0]}" "${model_name}"
  done
  exit 0
fi

case "${CUDA_INDEX}" in
  cu118|cu121|cu124|cu126) ;;
  *)
    echo "Unsupported CUDA_INDEX=${CUDA_INDEX}. Use cu118, cu121, cu124, or cu126." >&2
    exit 2
    ;;
esac

case "${MODEL}" in
  qwen3_4b|qwen|ar)
    MODEL="qwen3_4b"
    DEFAULT_VENV=".venv-qwen3-ar"
    EXTRAS="dev,hf,gpu"
    TORCH_VERSION="2.6.0"
    TRANSFORMERS_VERSION="5.14.1"
    TORCH_CUDA_INDEXES="cu118 cu124 cu126"
    MODEL_CONFIG="configs/models/qwen3_4b.yaml"
    ;;
  illada)
    DEFAULT_VENV=".venv-illada"
    EXTRAS="dev,hf,gpu"
    TORCH_VERSION="2.6.0"
    TRANSFORMERS_VERSION="4.57.1"
    TORCH_CUDA_INDEXES="cu118 cu124 cu126"
    MODEL_CONFIG="configs/models/illada.yaml"
    ;;
  dream)
    DEFAULT_VENV=".venv-dream"
    EXTRAS="dev,hf,gpu"
    TORCH_VERSION="2.5.1"
    TRANSFORMERS_VERSION="4.46.2"
    TORCH_CUDA_INDEXES="cu118 cu121 cu124"
    MODEL_CONFIG="configs/models/dream.yaml"
    ;;
  dg|diffusiongemma)
    MODEL="dg"
    DEFAULT_VENV=".venv-dg"
    EXTRAS="dev,dg,gpu"
    TORCH_VERSION="2.6.0"
    TRANSFORMERS_VERSION="5.14.1"
    TORCH_CUDA_INDEXES="cu118 cu124 cu126"
    MODEL_CONFIG="configs/models/dg.yaml"
    ;;
  w1)
    DEFAULT_VENV=".venv-w1"
    EXTRAS="dev,api"
    TORCH_VERSION=""
    TRANSFORMERS_VERSION=""
    TORCH_CUDA_INDEXES=""
    MODEL_CONFIG="configs/models/w1.yaml"
    ;;
  mock)
    DEFAULT_VENV=".venv-mock"
    EXTRAS="dev"
    TORCH_VERSION=""
    TRANSFORMERS_VERSION=""
    TORCH_CUDA_INDEXES=""
    MODEL_CONFIG="configs/models/mock.yaml"
    ;;
  *)
    echo "Unknown model '${MODEL}'." >&2
    echo "Use one of: qwen3_4b, illada, dream, dg, w1, mock, all." >&2
    exit 2
    ;;
esac

VENV_DIR="${VENV_DIR:-${DEFAULT_VENV}}"

if [[ -n "${TORCH_VERSION}" && " ${TORCH_CUDA_INDEXES} " != *" ${CUDA_INDEX} "* ]]; then
  echo "torch ${TORCH_VERSION} for ${MODEL} has no ${CUDA_INDEX} wheel." >&2
  echo "Supported CUDA indexes for this model: ${TORCH_CUDA_INDEXES}." >&2
  exit 2
fi

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

if [[ -n "${TORCH_VERSION}" ]]; then
  python -m pip install --upgrade "torch==${TORCH_VERSION}" --index-url "https://download.pytorch.org/whl/${CUDA_INDEX}"
  python -m pip uninstall -y torchvision torchaudio || true
fi

if [[ -n "${TRANSFORMERS_VERSION}" ]]; then
  python -m pip install --upgrade \
    "transformers==${TRANSFORMERS_VERSION}" \
    "accelerate==1.14.0" \
    "safetensors>=0.8.0" \
    "sentencepiece"
fi

python -m pip install -e ".[${EXTRAS}]"
python -m pip check

python - <<'PY'
import importlib.util

print("environment import check")
if importlib.util.find_spec("torch"):
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda:", torch.version.cuda)
    print("device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
else:
    print("torch: not installed")

if importlib.util.find_spec("transformers"):
    import transformers
    print("transformers:", transformers.__version__)
else:
    print("transformers: not installed")

if importlib.util.find_spec("requests"):
    import requests
    print("requests:", requests.__version__)
PY

echo
echo "RunPod env ready for ${MODEL}"
echo "Activate with: source ${VENV_DIR}/bin/activate"
echo "Warm with: python prepare_model.py --model-config ${MODEL_CONFIG}"
