#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?Usage: model_pipeline.sh <model>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/model_profile.sh"
dllm_load_model_profile "${MODEL}"

DATASET_CONFIG="${DATASET_CONFIG:-configs/datasets/gsm8k.yaml}"
DATASET_NAME="${DATASET_NAME:-gsm8k}"
DATA_SOURCE="${DATA_SOURCE:-demo}"
N_SAMPLES="${N_SAMPLES:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
SEED="${SEED:-42}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/checks/${DLLM_MODEL_ID}}"

case "${DATA_SOURCE}" in
  demo) DATA_FLAG="--demo" ;;
  real) DATA_FLAG="--no-demo" ;;
  *)
    echo "Unknown DATA_SOURCE='${DATA_SOURCE}'. Use demo or real." >&2
    exit 2
    ;;
esac

if [[ "${DLLM_MODEL_ID}" == "w1" && -z "${W1_API_BASE_URL:-}" ]]; then
  echo "W1_API_BASE_URL must be set before running W1." >&2
  exit 2
fi

cd "${REPO_ROOT}"

python -m dllm_bench.cli generate \
  --model-config "${DLLM_MODEL_CONFIG}" \
  --dataset-config "${DATASET_CONFIG}" \
  "${DATA_FLAG}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --n-samples "${N_SAMPLES}" \
  --seed "${SEED}" \
  --output-root "${OUTPUT_ROOT}"

python -m dllm_bench.cli score \
  --model-config "${DLLM_MODEL_CONFIG}" \
  --dataset-config "${DATASET_CONFIG}" \
  "${DATA_FLAG}" \
  --n-samples "${N_SAMPLES}" \
  --seed "${SEED}" \
  --output-root "${OUTPUT_ROOT}"

python -m dllm_bench.cli report \
  --output-root "${OUTPUT_ROOT}" \
  --dataset "${DATASET_NAME}"
