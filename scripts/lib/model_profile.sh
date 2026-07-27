#!/usr/bin/env bash

# Load the dependency and path profile for one benchmark model.
dllm_load_model_profile() {
  local requested_model="${1:-}"

  case "${requested_model}" in
    qwen3_4b|qwen|ar)
      DLLM_MODEL_ID="qwen3_4b"
      DLLM_VENV_NAME=".venv-qwen3-ar"
      DLLM_MODEL_CONFIG="configs/models/qwen3_4b.yaml"
      DLLM_EXTRAS="dev,hf,gpu"
      DLLM_TORCH_VERSION="2.6.0"
      DLLM_TRANSFORMERS_VERSION="5.14.1"
      DLLM_TORCH_CUDA_INDEXES="cu118 cu124 cu126"
      ;;
    illada)
      DLLM_MODEL_ID="illada"
      DLLM_VENV_NAME=".venv-illada"
      DLLM_MODEL_CONFIG="configs/models/illada.yaml"
      DLLM_EXTRAS="dev,hf,gpu"
      DLLM_TORCH_VERSION="2.6.0"
      DLLM_TRANSFORMERS_VERSION="4.57.1"
      DLLM_TORCH_CUDA_INDEXES="cu118 cu124 cu126"
      ;;
    dream)
      DLLM_MODEL_ID="dream"
      DLLM_VENV_NAME=".venv-dream"
      DLLM_MODEL_CONFIG="configs/models/dream.yaml"
      DLLM_EXTRAS="dev,hf,gpu"
      DLLM_TORCH_VERSION="2.5.1"
      DLLM_TRANSFORMERS_VERSION="4.46.2"
      DLLM_TORCH_CUDA_INDEXES="cu118 cu121 cu124"
      ;;
    dg|diffusiongemma)
      DLLM_MODEL_ID="diffusiongemma"
      DLLM_VENV_NAME=".venv-diffusiongemma"
      DLLM_MODEL_CONFIG="configs/models/dg.yaml"
      DLLM_EXTRAS="dev,dg,gpu"
      DLLM_TORCH_VERSION="2.6.0"
      DLLM_TRANSFORMERS_VERSION="5.14.1"
      DLLM_TORCH_CUDA_INDEXES="cu118 cu124 cu126"
      ;;
    w1)
      DLLM_MODEL_ID="w1"
      DLLM_VENV_NAME=".venv-w1"
      DLLM_MODEL_CONFIG="configs/models/w1.yaml"
      DLLM_EXTRAS="dev,api"
      DLLM_TORCH_VERSION=""
      DLLM_TRANSFORMERS_VERSION=""
      DLLM_TORCH_CUDA_INDEXES=""
      ;;
    mock)
      DLLM_MODEL_ID="mock"
      DLLM_VENV_NAME=".venv-mock"
      DLLM_MODEL_CONFIG="configs/models/mock.yaml"
      DLLM_EXTRAS="dev"
      DLLM_TORCH_VERSION=""
      DLLM_TRANSFORMERS_VERSION=""
      DLLM_TORCH_CUDA_INDEXES=""
      ;;
    *)
      echo "Unknown model '${requested_model}'." >&2
      return 2
      ;;
  esac
}
