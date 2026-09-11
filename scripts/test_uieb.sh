#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GPU_ID="${GPU_ID:-0}"
SPLIT_FILE="${SPLIT_FILE:-splits/UIEB}"
CHECKPOINT="${CHECKPOINT:-pretrained/ocfr_uie_uieb_best.pth}"
DATA_ROOT="${DATA_ROOT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/uieb_test}"
METRICS="${METRICS:-all}"

cmd=(
  python test.py
  --split-file "${SPLIT_FILE}"
  --split test
  --checkpoint "${CHECKPOINT}"
  --output-dir "${OUTPUT_DIR}"
  --metrics "${METRICS}"
)

if [[ -n "${DATA_ROOT}" ]]; then
  cmd+=(--data-root "${DATA_ROOT}")
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${cmd[@]}"
