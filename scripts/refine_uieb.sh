#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GPU_ID="${GPU_ID:-0}"
: "${DATA_ROOT:?Set DATA_ROOT to the UIEB dataset directory}"
: "${DINO_REPO:?Set DINO_REPO to the official DINOv2 repository}"
: "${DINO_CHECKPOINT:?Set DINO_CHECKPOINT to the ViT-B/14 checkpoint}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python experiments/quality_calibration/finetune.py \
  --config configs/quality_calibration/uieb.json
