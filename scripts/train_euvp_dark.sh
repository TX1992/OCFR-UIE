#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GPU_IDS="${GPU_IDS:-0}"
SPLIT_FILE="${SPLIT_FILE:-splits/EUVP-Dark}"
DATA_ROOT="${DATA_ROOT:-}"
DINO_REPO="${DINO_REPO:-third_party/DINOv2}"
DINO_CHECKPOINT="${DINO_CHECKPOINT:-pretrained/dinov2_vitb14_pretrain.pth}"

cmd=(
  python train.py
  --config configs/euvp_dark_ocfr.json
  --split-file "${SPLIT_FILE}"
  --dinov2-repo "${DINO_REPO}"
  --dinov2-checkpoint "${DINO_CHECKPOINT}"
)

if [[ -n "${DATA_ROOT}" ]]; then
  cmd+=(--data-root "${DATA_ROOT}")
fi

CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${cmd[@]}"
