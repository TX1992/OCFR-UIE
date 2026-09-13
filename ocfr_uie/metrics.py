"""Full-reference metrics used for the reported paired results."""

from __future__ import annotations

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def quantized_numpy_batch(images: torch.Tensor) -> np.ndarray:
    arrays = images.detach().float().clamp(0.0, 1.0).cpu().numpy()
    arrays = np.transpose(arrays, (0, 2, 3, 1))
    return np.rint(arrays * 255.0).astype(np.uint8).astype(np.float32) / 255.0


def float_numpy_batch(images: torch.Tensor) -> np.ndarray:
    arrays = images.detach().float().clamp(0.0, 1.0).cpu().numpy()
    return np.transpose(arrays, (0, 2, 3, 1))


def psnr(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(target, prediction, data_range=1.0))


def ssim(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(
        structural_similarity(
            target,
            prediction,
            data_range=1.0,
            channel_axis=-1,
            win_size=5,
        )
    )
