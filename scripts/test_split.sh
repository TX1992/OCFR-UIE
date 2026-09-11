#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GPU_ID="${1:?Usage: bash scripts/test_split.sh GPU_ID SPLIT_FILE DATA_ROOT CHECKPOINT [SPLIT] [OUT_DIR] [METRICS]}"
SPLIT_FILE="${2:?Missing split manifest}"
DATA_ROOT="${3:?Missing dataset root}"
CHECKPOINT="${4:?Missing checkpoint}"
SPLIT="${5:-test}"
OUT_DIR="${6:-outputs/test}"
METRICS="${7:-all}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python test.py \
  --split-file "${SPLIT_FILE}" \
  --data-root "${DATA_ROOT}" \
  --split "${SPLIT}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUT_DIR}" \
  --metrics "${METRICS}" \
  --save-images
