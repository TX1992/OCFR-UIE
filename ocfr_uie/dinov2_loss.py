"""Frozen DINOv2 dense-token loss used only during training."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint


SUPPORTED_DINOV2_BACKBONES = {
    "dinov2_vits14",
    "dinov2_vitb14",
    "dinov2_vitb14_reg",
}


def load_official_dinov2(
    repo_path: str,
    checkpoint: str,
    model_name: str,
) -> nn.Module:
    """Load an official DINOv2 backbone from local source and weights."""
    repo = Path(repo_path).expanduser().resolve()
    weights = Path(checkpoint).expanduser().resolve()
    if not (repo / "dinov2" / "hub" / "backbones.py").is_file():
        raise RuntimeError(f"invalid DINOv2 repository: {repo}")
    if not weights.is_file():
        raise RuntimeError(f"missing DINOv2 checkpoint: {weights}")
    if model_name not in SUPPORTED_DINOV2_BACKBONES:
        raise ValueError(f"unsupported DINOv2 backbone: {model_name}")

    repo_string = str(repo)
    if repo_string not in sys.path:
        sys.path.insert(0, repo_string)
    backbones = importlib.import_module("dinov2.hub.backbones")
    imported_path = Path(backbones.__file__).resolve()
    if repo not in imported_path.parents:
        raise RuntimeError(f"imported DINOv2 from {imported_path}, expected {repo}")

    backbone = getattr(backbones, model_name)(pretrained=False)
    try:
        state_dict = torch.load(weights, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(weights, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"invalid DINOv2 checkpoint object: {type(state_dict)}")
    backbone.load_state_dict(state_dict, strict=True)
    return backbone


class DINOv2DenseStructureLoss(nn.Module):
    """Cosine distance between frozen shallow/middle DINOv2 patch tokens."""

    def __init__(
        self,
        repo_path: str = "",
        checkpoint: str = "",
        model_name: str = "dinov2_vitb14",
        layers: Sequence[int] = (3, 7),
        feature_size: int = 224,
        max_samples: int = 4,
        backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.layers = tuple(int(layer) for layer in layers)
        if not self.layers or min(self.layers) < 0:
            raise ValueError(f"DINOv2 layers must be non-empty and non-negative: {self.layers}")
        self.feature_size = int(feature_size)
        if self.feature_size <= 0 or self.feature_size % 14:
            raise ValueError("DINOv2 feature_size must be positive and divisible by 14")
        self.max_samples = int(max_samples)
        if self.max_samples < 0:
            raise ValueError("DINOv2 max_samples must be non-negative")

        if backbone is None:
            backbone = load_official_dinov2(repo_path, checkpoint, model_name)
        if not hasattr(backbone, "get_intermediate_layers"):
            raise TypeError("DINOv2 backbone must expose get_intermediate_layers")
        self.backbone = backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))

    def train(self, mode: bool = True) -> "DINOv2DenseStructureLoss":
        super().train(mode)
        self.backbone.eval()
        return self

    def _prepare(self, image: torch.Tensor) -> torch.Tensor:
        image = F.interpolate(
            image.clamp(0.0, 1.0),
            size=(self.feature_size, self.feature_size),
            mode="bilinear",
            align_corners=False,
            # CUDA antialiased bilinear backward is nondeterministic in PyTorch 2.1.
            antialias=False,
        )
        return (image - self.mean.to(image)) / self.std.to(image)

    def _features(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs = self.backbone.get_intermediate_layers(
            self._prepare(image),
            n=self.layers,
            reshape=False,
            return_class_token=False,
            norm=True,
        )
        if len(outputs) != len(self.layers):
            raise RuntimeError(f"DINOv2 returned {len(outputs)} features for layers {self.layers}")
        return tuple(outputs)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction/target shape mismatch: {prediction.shape} != {target.shape}"
            )
        if self.max_samples and prediction.shape[0] > self.max_samples:
            prediction = prediction[: self.max_samples]
            target = target[: self.max_samples]

        prediction_features = self._features(prediction)
        with torch.no_grad():
            target_features = self._features(target)

        losses = []
        for prediction_feature, target_feature in zip(
            prediction_features, target_features, strict=True
        ):
            if prediction_feature.shape != target_feature.shape:
                raise RuntimeError(
                    "DINOv2 prediction/target feature mismatch: "
                    f"{prediction_feature.shape} != {target_feature.shape}"
                )
            prediction_feature = F.normalize(prediction_feature.float(), dim=-1, eps=1e-6)
            target_feature = F.normalize(target_feature.float(), dim=-1, eps=1e-6)
            losses.append(1.0 - (prediction_feature * target_feature).sum(dim=-1).mean())
        return torch.stack(losses).mean()


class DINOContrastiveRestorationLoss(DINOv2DenseStructureLoss):
    """Full-batch DINO perception contrastive loss for paired restoration."""

    def __init__(
        self,
        repo_path: str = "",
        checkpoint: str = "",
        model_name: str = "dinov2_vitb14",
        layers: Sequence[int] = (3, 7),
        feature_size: int = 224,
        chunk_size: int = 4,
        eps: float = 1e-6,
        backbone: nn.Module | None = None,
    ) -> None:
        super().__init__(
            repo_path=repo_path,
            checkpoint=checkpoint,
            model_name=model_name,
            layers=layers,
            feature_size=feature_size,
            max_samples=0,
            backbone=backbone,
        )
        self.chunk_size = int(chunk_size)
        self.eps = float(eps)
        if self.chunk_size < 0:
            raise ValueError("DINOv2 chunk_size must be non-negative")
        if self.eps <= 0:
            raise ValueError("DINOv2 eps must be positive")

    @staticmethod
    def _feature_distance(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        if first.shape != second.shape:
            raise RuntimeError(f"DINOv2 feature mismatch: {first.shape} != {second.shape}")
        return (first.float() - second.float()).abs().flatten(1).mean(dim=1)

    def _prediction_features(self, prediction: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if prediction.requires_grad:
            return activation_checkpoint(self._features, prediction, use_reentrant=False)
        return self._features(prediction)

    def _forward_components(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        degraded: torch.Tensor,
    ) -> torch.Tensor:
        if prediction.shape != target.shape or prediction.shape != degraded.shape:
            raise ValueError(
                "prediction/target/degraded shape mismatch: "
                f"{prediction.shape}, {target.shape}, {degraded.shape}"
            )

        batch_size = prediction.shape[0]
        if batch_size == 0:
            raise ValueError("DINO contrastive loss requires a non-empty batch")
        chunk_size = self.chunk_size or batch_size
        sample_losses: list[torch.Tensor] = []
        samples_seen = 0

        for start in range(0, batch_size, chunk_size):
            end = min(start + chunk_size, batch_size)
            prediction_features = self._prediction_features(prediction[start:end])
            with torch.no_grad():
                target_features = self._features(target[start:end])
                degraded_features = self._features(degraded[start:end])

            layer_losses = []
            for prediction_feature, target_feature, degraded_feature in zip(
                prediction_features,
                target_features,
                degraded_features,
                strict=True,
            ):
                positive_distance = self._feature_distance(prediction_feature, target_feature)
                negative_distance = self._feature_distance(prediction_feature, degraded_feature)
                layer_losses.append(positive_distance / (negative_distance + self.eps))

            per_sample_loss = torch.stack(layer_losses, dim=0).mean(dim=0)
            sample_losses.append(per_sample_loss)
            samples_seen += end - start

        if samples_seen != batch_size:
            raise RuntimeError(f"DINO contrastive coverage failure: {samples_seen} != {batch_size}")
        return torch.cat(sample_losses, dim=0)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        degraded: torch.Tensor,
    ) -> torch.Tensor:
        return self._forward_components(prediction, target, degraded).mean()
