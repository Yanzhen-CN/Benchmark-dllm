#!/usr/bin/env bash
set -euo pipefail

# Generate, score, and report for any configured model. Defaults make this a
# one-sample smoke run; override N_SAMPLES/MAX_NEW_TOKENS/OUTPUT_ROOT for a
# formal run.

MODEL="${1:-${DLLM_MODEL:-qwen3_4b}}"
DATASET_CONFIG="${DATASET_CONFIG:-configs/datasets/gsm8k.yaml}"
DATASET_NAME="${DATASET_NAME:-gsm8k}"
N_SAMPLES="${N_SAMPLES:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"

case "${MODEL}" in
  qwen3_4b|qwen|ar)
    MODEL_CONFIG="configs/models/qwen3_4b.yaml"
    OUTPUT_ROOT="${OUTPUT_ROOT:-.tmp_qwen_smoke}"
    ;;
  illada)
    MODEL_CONFIG="configs/models/illada.yaml"
    OUTPUT_ROOT="${OUTPUT_ROOT:-.tmp_illada_smoke}"
    ;;
  dream)
    MODEL_CONFIG="configs/models/dream.yaml"
    OUTPUT_ROOT="${OUTPUT_ROOT:-.tmp_dream_smoke}"
    ;;
  dg|diffusiongemma)
    MODEL_CONFIG="configs/models/dg.yaml"
    OUTPUT_ROOT="${OUTPUT_ROOT:-.tmp_dg_smoke}"
    ;;
  w1)
    MODEL_CONFIG="configs/models/w1.yaml"
    OUTPUT_ROOT="${OUTPUT_ROOT:-.tmp_w1_smoke}"
    if [[ -z "${W1_API_BASE_URL:-}" ]]; then
      echo "W1_API_BASE_URL must be set before running w1." >&2
      exit 2
    fi
    ;;
  mock)
    MODEL_CONFIG="configs/models/mock.yaml"
    OUTPUT_ROOT="${OUTPUT_ROOT:-.tmp_mock_smoke}"
    ;;
  *)
    echo "Unknown model '${MODEL}'." >&2
    echo "Use one of: qwen3_4b, illada, dream, dg, w1, mock." >&2
    exit 2
    ;;
esac

python -m dllm_bench.cli generate \
  --model-config "${MODEL_CONFIG}" \
  --dataset-config "${DATASET_CONFIG}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --n-samples "${N_SAMPLES}" \
  --output-root "${OUTPUT_ROOT}"

python -m dllm_bench.cli score \
  --model-config "${MODEL_CONFIG}" \
  --dataset-config "${DATASET_CONFIG}" \
  --n-samples "${N_SAMPLES}" \
  --output-root "${OUTPUT_ROOT}"

python -m dllm_bench.cli report \
  --output-root "${OUTPUT_ROOT}" \
  --dataset "${DATASET_NAME}"
