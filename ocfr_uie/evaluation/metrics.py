import math
from typing import List, Mapping, Optional, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


IMAGE_EPS = 1e-12
PYIQA_METRICS = ("uranker", "musiq", "paq2piq", "niqe")
NUMPY_NO_REFERENCE_METRICS = ("uiqm", "uciqe")
FULL_REFERENCE_METRICS = ("psnr", "ssim")
NO_REFERENCE_METRICS = PYIQA_METRICS + NUMPY_NO_REFERENCE_METRICS
DEFAULT_METRICS = PYIQA_METRICS + NUMPY_NO_REFERENCE_METRICS + FULL_REFERENCE_METRICS


def normalize_metric_names(metrics: Optional[Sequence[str] | str]) -> List[str]:
    if metrics is None or metrics == "all":
        return list(DEFAULT_METRICS)
    if isinstance(metrics, str):
        metrics = [item.strip() for item in metrics.split(",") if item.strip()]
    normalized = []
    valid = set(DEFAULT_METRICS)
    aliases = {
        "niqm": "niqe",
        "uciqe_lab": "uciqe",
        "uciqe_cielab": "uciqe",
    }
    for name in metrics:
        key = str(name).strip().lower()
        key = aliases.get(key, key)
        if key not in valid:
            raise ValueError(
                f"Unknown metric '{name}'. Valid metrics: {', '.join(DEFAULT_METRICS)}"
            )
        if key not in normalized:
            normalized.append(key)
    return normalized


def to_numpy_batch(image) -> np.ndarray:
    if torch.is_tensor(image):
        arr = image.detach().float().cpu().numpy()
    else:
        arr = np.asarray(image)

    if arr.ndim == 2:
        arr = arr[None, :, :, None]
    elif arr.ndim == 3:
        if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        arr = arr[None]
    elif arr.ndim == 4:
        if arr.shape[1] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (0, 2, 3, 1))
    else:
        raise ValueError(f"Expected image with 2/3/4 dims, got shape {arr.shape}")

    arr = arr.astype(np.float32, copy=False)
    if arr.size and arr.max() > 1.5:
        arr = arr / 255.0
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)


def _float01_to_uint8(rgb_float: np.ndarray) -> np.ndarray:
    return (np.clip(rgb_float, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def get_uciqe(rgb_float: np.ndarray) -> float:
    """Original CIELab UCIQE definition released by the metric authors.

    The Lab channels and 65,536-bin luminance CDF follow the authors' released
    OpenCV implementation. The 1%/99% CDF interval is the Python counterpart of
    the default ``stretchlim`` interval in their original MATLAB implementation.
    """

    rgb_uint8 = _float01_to_uint8(rgb_float)
    lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB).astype(np.float64)
    luminance = lab[:, :, 0] / 255.0 + IMAGE_EPS
    channel_a = lab[:, :, 1] / 255.0
    channel_b = lab[:, :, 2] / 255.0

    chroma = np.sqrt(channel_a * channel_a + channel_b * channel_b)
    saturation = chroma / np.sqrt(chroma * chroma + luminance * luminance + IMAGE_EPS)
    mean_chroma = float(np.mean(chroma))
    chroma_dispersion = math.sqrt(
        float(np.mean(np.abs(1.0 - np.square(mean_chroma / (chroma + IMAGE_EPS)))))
    )
    histogram, _ = np.histogram(luminance, bins=65_536)
    cumulative = np.cumsum(histogram, dtype=np.float64) / max(1, luminance.size)
    low_index = int(np.flatnonzero(cumulative > 0.01)[0])
    high_index = int(np.flatnonzero(cumulative >= 0.99)[0])
    luminance_contrast = float((high_index - low_index) / 65_535.0)
    mean_saturation = float(np.mean(saturation))
    return float(
        0.4680 * chroma_dispersion + 0.2745 * luminance_contrast + 0.2576 * mean_saturation
    )


def _trimmed_stats(values: np.ndarray, trim_ratio: float = 0.1) -> tuple[float, float]:
    values = np.sort(values.reshape(-1))
    lo = int(math.floor(trim_ratio * values.size))
    hi = int(math.ceil((1.0 - trim_ratio) * values.size))
    values = values[lo:hi]
    return float(np.mean(values)), float(np.var(values))


def _iter_blocks(arr: np.ndarray, block_size: int):
    h, w = arr.shape[:2]
    for y in range(max(1, math.ceil(h / block_size))):
        for x in range(max(1, math.ceil(w / block_size))):
            yield arr[
                y * block_size : min((y + 1) * block_size, h),
                x * block_size : min((x + 1) * block_size, w),
            ]


def _eme(arr: np.ndarray, block_size: int = 4) -> float:
    blocks = list(_iter_blocks(arr.astype(np.float64), block_size))
    if not blocks:
        return 0.0
    score = 0.0
    for block in blocks:
        bmin = max(float(np.min(block)), 1.0)
        bmax = max(float(np.max(block)), 1.0)
        if bmax >= bmin:
            score += (2.0 / len(blocks)) * math.log(bmax / bmin)
    return float(score)


def get_uiqm(rgb_float: np.ndarray) -> float:
    rgb = _float01_to_uint8(rgb_float)
    rgb64 = rgb.astype(np.float64)
    r, g, b = rgb64[:, :, 0], rgb64[:, :, 1], rgb64[:, :, 2]
    rg = r - g
    yb = (r + g) / 2.0 - b
    mean_rg, var_rg = _trimmed_stats(rg)
    mean_yb, var_yb = _trimmed_stats(yb)
    uicm = -0.0268 * math.sqrt(mean_rg * mean_rg + mean_yb * mean_yb) + 0.1586 * math.sqrt(
        var_rg + var_yb
    )

    uism = 0.0
    for channel, weight in enumerate((0.299, 0.587, 0.114)):
        img = rgb64[:, :, channel]
        sx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        edge = np.clip(np.round(np.sqrt(sx * sx + sy * sy)), 0.0, 255.0) * img
        uism += weight * _eme(edge)

    intensity = np.mean(rgb64, axis=2)
    blocks = list(_iter_blocks(intensity, 4))
    uiconm = 0.0
    for block in blocks:
        bmin = float(np.min(block))
        bmax = float(np.max(block))
        top = bmax - bmin
        bottom = bmax + bmin
        if top > IMAGE_EPS and bottom > IMAGE_EPS:
            ratio = min(max(top / bottom, IMAGE_EPS), 1.0)
            uiconm += ratio * math.log(ratio)
    uiconm = -uiconm / max(1, len(blocks))
    return float(0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm)


def get_psnr(pred_rgb: np.ndarray, gt_rgb: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(gt_rgb, pred_rgb, data_range=1.0))


def get_ssim(pred_rgb: np.ndarray, gt_rgb: np.ndarray) -> float:
    h, w = pred_rgb.shape[:2]
    win_size = 5 if min(h, w) >= 5 else max(3, min(h, w) // 2 * 2 - 1)
    return float(
        structural_similarity(gt_rgb, pred_rgb, data_range=1.0, channel_axis=-1, win_size=win_size)
    )


def resize_for_iqa(x: torch.Tensor, size: Optional[int]) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    if size and x.shape[-2:] != (int(size), int(size)):
        x = F.interpolate(x, size=(int(size), int(size)), mode="bilinear", align_corners=False)
    return x


def build_pyiqa_metrics(
    names: Sequence[str] = PYIQA_METRICS,
    device: str | torch.device = "cuda",
) -> Mapping[str, object]:
    import pyiqa

    device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
    metrics = {}
    for name in names:
        key = str(name).lower()
        if key in PYIQA_METRICS:
            metrics[key] = pyiqa.create_metric(key, device=device)
    return metrics
