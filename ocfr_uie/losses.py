"""Exact training objective used by the validated OCFR-UIE checkpoint."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG16_Weights, vgg16

from .dinov2_loss import DINOContrastiveRestorationLoss


class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.epsilon = float(epsilon)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        difference = torch.add(prediction, -target)
        error = torch.sqrt(difference * difference + self.epsilon)
        return torch.mean(error)


class HVITransform(nn.Module):
    """Frozen RGB-to-HVI transform used by the validated OCFR-UIE run.

    The density value is the exact `trans.density_k` value from the pretrained
    CIDNet checkpoint. The full CIDNet is unnecessary because training calls
    only this deterministic transform.
    """

    def __init__(self, density: float = 0.37945818901062012) -> None:
        super().__init__()
        self.register_buffer("density", torch.tensor([density], dtype=torch.float32))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        epsilon = 1e-8
        value = image.max(1)[0].to(image.dtype)
        minimum = image.min(1)[0].to(image.dtype)
        denominator = value - minimum + epsilon
        hue = torch.zeros_like(value)
        blue_mask = image[:, 2] == value
        hue = torch.where(
            blue_mask,
            4.0 + (image[:, 0] - image[:, 1]) / denominator,
            hue,
        )
        green_mask = image[:, 1] == value
        hue = torch.where(
            green_mask,
            2.0 + (image[:, 2] - image[:, 0]) / denominator,
            hue,
        )
        red_mask = image[:, 0] == value
        hue = torch.where(
            red_mask,
            ((image[:, 1] - image[:, 2]) / denominator) % 6.0,
            hue,
        )
        hue = torch.where(minimum == value, torch.zeros_like(hue), hue)
        hue = hue / 6.0

        saturation = (value - minimum) / (value + epsilon)
        saturation = torch.where(value == 0, torch.zeros_like(saturation), saturation)

        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)
        density = self.density.to(dtype=image.dtype)
        color_sensitive = ((value * 0.5 * math.pi).sin() + epsilon).pow(density)
        horizontal = color_sensitive * saturation * (2.0 * math.pi * hue).cos()
        vertical = color_sensitive * saturation * (2.0 * math.pi * hue).sin()
        return torch.cat([horizontal, vertical, value], dim=1)


def _gaussian_window(window_size: int, sigma: float = 1.5) -> torch.Tensor:
    kernel = torch.tensor(
        [
            math.exp(-((index - window_size // 2) ** 2) / float(2.0 * sigma**2))
            for index in range(window_size)
        ],
        dtype=torch.float32,
    )
    kernel = kernel / kernel.sum()
    kernel = kernel.unsqueeze(1)
    return kernel.mm(kernel.t()).unsqueeze(0).unsqueeze(0)


class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 5, channels: int = 3) -> None:
        super().__init__()
        self.window_size = int(window_size)
        self.channels = int(channels)
        self.register_buffer(
            "window", _gaussian_window(window_size).expand(channels, 1, -1, -1).contiguous()
        )

    def forward(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        window = self.window.to(device=first.device, dtype=first.dtype)
        mean_first = F.conv2d(first, window, groups=self.channels)
        mean_second = F.conv2d(second, window, groups=self.channels)
        mean_first_sq = mean_first.pow(2)
        mean_second_sq = mean_second.pow(2)
        mean_product = mean_first * mean_second
        variance_first = F.conv2d(first * first, window, groups=self.channels) - mean_first_sq
        variance_second = F.conv2d(second * second, window, groups=self.channels) - mean_second_sq
        covariance = F.conv2d(first * second, window, groups=self.channels) - mean_product
        c1 = 0.01**2
        c2 = 0.03**2
        contrast_numerator = 2.0 * covariance + c2 + 1e-8
        contrast_denominator = variance_first + variance_second + c2 + 1e-8
        ssim_map = ((2.0 * mean_product + c1) * contrast_numerator) / (
            (mean_first_sq + mean_second_sq + c1) * contrast_denominator
        )
        loss_map = 1.0 - ssim_map
        if weight is not None:
            weight = F.interpolate(weight, size=loss_map.shape[-2:], mode="area")
            loss_map = loss_map * weight
        return loss_map.mean()


class VGGPerceptualLoss(nn.Module):
    """Unnormalized VGG16 feature MSE used by the validated training run."""

    feature_indices = {"3", "8", "15"}

    def __init__(self) -> None:
        super().__init__()
        self.features = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].eval()
        for parameter in self.features.parameters():
            parameter.requires_grad = False

    def _extract(self, image: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        for name, module in self.features._modules.items():
            image = module(image)
            if name in self.feature_indices:
                outputs.append(image)
        return outputs

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prediction_features = self._extract(prediction)
        with torch.no_grad():
            target_features = self._extract(target)
        losses = []
        for prediction_value, target_value in zip(prediction_features, target_features):
            difference = (prediction_value - target_value).square()
            if weight is not None:
                feature_weight = F.interpolate(weight, size=difference.shape[-2:], mode="area")
                difference = difference * feature_weight
            losses.append(difference.mean())
        return sum(losses) / len(losses)


class SobelEdgeLoss(nn.Module):
    def __init__(self, channels: int = 3) -> None:
        super().__init__()
        kernel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        kernel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
        self.channels = channels
        self.register_buffer("kernel_x", kernel_x[None, None].repeat(channels, 1, 1, 1))
        self.register_buffer("kernel_y", kernel_y[None, None].repeat(channels, 1, 1, 1))

    def _magnitude(self, image: torch.Tensor) -> torch.Tensor:
        # Preserve the validated objective's repeated grouped-Sobel reduction.
        # The repeated responses are mathematically redundant, but changing the
        # reduction graph changes floating-point gradients and the trajectory.
        kernel_x = self.kernel_x.repeat(self.channels, 1, 1, 1).to(
            device=image.device, dtype=image.dtype
        )
        kernel_y = self.kernel_y.repeat(self.channels, 1, 1, 1).to(
            device=image.device, dtype=image.dtype
        )
        gradient_x = F.conv2d(image, kernel_x, padding=1, groups=self.channels)
        gradient_y = F.conv2d(image, kernel_y, padding=1, groups=self.channels)
        return torch.sqrt(gradient_x.square() + gradient_y.square() + 1e-6)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        difference = (self._magnitude(prediction) - self._magnitude(target)).square()
        if weight is not None:
            difference = difference * weight
        return difference.mean()


class OCFRTrainingLoss(nn.Module):
    """Validated OCFR-UIE objective with frozen perceptual extractors."""

    def __init__(
        self,
        *,
        charbonnier_weight: float = 1.0,
        hvi_weight: float = 0.5,
        ssim_weight: float = 0.1,
        perceptual_weight: float = 0.1,
        dinov2_dpc_weight: float = 0.00065,
        edge_weight: float = 0.1,
        dinov2_repo: str = "",
        dinov2_checkpoint: str = "",
        dinov2_model: str = "dinov2_vitb14",
        dinov2_layers: tuple[int, ...] = (3, 7),
        dinov2_feature_size: int = 224,
        dinov2_chunk_size: int = 4,
    ) -> None:
        super().__init__()
        self.charbonnier_weight = float(charbonnier_weight)
        self.hvi_weight = float(hvi_weight)
        self.ssim_weight = float(ssim_weight)
        self.perceptual_weight = float(perceptual_weight)
        self.dinov2_dpc_weight = float(dinov2_dpc_weight)
        self.edge_weight = float(edge_weight)
        weights = (
            self.charbonnier_weight,
            self.hvi_weight,
            self.ssim_weight,
            self.perceptual_weight,
            self.dinov2_dpc_weight,
            self.edge_weight,
        )
        if any(weight < 0 for weight in weights):
            raise ValueError("training loss weights must be non-negative")
        if not any(weight > 0 for weight in weights):
            raise ValueError("at least one training loss weight must be positive")

        self.charbonnier = CharbonnierLoss()
        self.hvi_transform = HVITransform() if self.hvi_weight else None
        self.ssim = SSIMLoss(window_size=5) if self.ssim_weight else None
        self.perceptual = VGGPerceptualLoss() if self.perceptual_weight else None
        self.dinov2_dpc = (
            DINOContrastiveRestorationLoss(
                repo_path=dinov2_repo,
                checkpoint=dinov2_checkpoint,
                model_name=dinov2_model,
                layers=dinov2_layers,
                feature_size=dinov2_feature_size,
                chunk_size=dinov2_chunk_size,
            )
            if self.dinov2_dpc_weight
            else None
        )
        self.edge = SobelEdgeLoss() if self.edge_weight else None

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        degraded: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        zero = prediction.new_zeros(())
        char = self.charbonnier(prediction, target) if self.charbonnier_weight else zero
        if self.hvi_transform is not None:
            prediction_hvi = self.hvi_transform(prediction.clamp(0.0, 1.0))
            with torch.no_grad():
                target_hvi = self.hvi_transform(target)
            hvi = self.charbonnier(prediction_hvi, target_hvi)
        else:
            hvi = zero
        ssim = self.ssim(prediction, target) if self.ssim is not None else zero
        perceptual = self.perceptual(prediction, target) if self.perceptual is not None else zero
        dpc = self.dinov2_dpc(prediction, target, degraded) if self.dinov2_dpc is not None else zero
        edge = self.edge(prediction, target) if self.edge is not None else zero

        total = (
            self.charbonnier_weight * char
            + self.hvi_weight * hvi
            + self.ssim_weight * ssim
            + self.perceptual_weight * perceptual
            + self.edge_weight * edge
            + self.dinov2_dpc_weight * dpc
        )
        values = {
            "charbonnier": (self.charbonnier_weight, char),
            "hvi": (self.hvi_weight, hvi),
            "ssim": (self.ssim_weight, ssim),
            "vgg": (self.perceptual_weight, perceptual),
            "edge": (self.edge_weight, edge),
            "dinov2_dpc": (self.dinov2_dpc_weight, dpc),
        }
        return total, {name: value.detach() for name, (weight, value) in values.items() if weight}
