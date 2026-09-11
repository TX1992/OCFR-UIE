"""Typed configuration for reproducible OCFR-UIE training."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class TrainConfig:
    split_file: str = "splits/UIEB"
    data_root: str = ""
    output_root: str = "outputs"
    run_name: str = "ocfr_uie_uieb"
    seed: int = 7
    feature_channels: int = 28
    image_size: int = 256
    resize: bool = True
    epochs: int = 100
    batch_size: int = 24
    eval_batch_size: int = 24
    num_workers: int = 4
    learning_rate: float = 2e-3
    weight_decay: float = 1e-5
    scheduler: str = "cosine"
    charbonnier_weight: float = 1.0
    hvi_weight: float = 0.5
    ssim_weight: float = 0.1
    perceptual_weight: float = 0.1
    dinov2_dpc_weight: float = 0.00065
    edge_weight: float = 0.1
    dinov2_repo: str = "third_party/DINOv2"
    dinov2_checkpoint: str = "pretrained/dinov2_vitb14_pretrain.pth"
    dinov2_model: str = "dinov2_vitb14"
    dinov2_layers: str = "3,7"
    dinov2_feature_size: int = 224
    dinov2_chunk_size: int = 4
    checkpoint_metric: str = "ssim_tie_psnr"
    ssim_tie_epsilon: float = 0.002
    data_parallel: bool = True
    activation_checkpointing: bool = False
    deterministic: bool = True
    initialization_checkpoint: str = ""

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = sorted(set(payload) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown configuration keys: {unknown}")
        config = cls(**payload)
        config.validate()
        return config

    def validate(self) -> None:
        if not self.split_file:
            raise ValueError("split_file must identify a fixed train/val/test manifest")
        if self.feature_channels % 4:
            raise ValueError("feature_channels must be divisible by four")
        if self.image_size % 4:
            raise ValueError("image_size must be divisible by four")
        if self.epochs <= 0 or self.batch_size <= 0 or self.eval_batch_size <= 0:
            raise ValueError("epochs and batch sizes must be positive")
        if self.scheduler != "cosine":
            raise ValueError("OCFR-UIE supports cosine scheduling only")
        weights = (
            self.charbonnier_weight,
            self.hvi_weight,
            self.ssim_weight,
            self.perceptual_weight,
            self.dinov2_dpc_weight,
            self.edge_weight,
        )
        if any(weight < 0 for weight in weights):
            raise ValueError("loss weights must be non-negative")
        if not any(weight > 0 for weight in weights):
            raise ValueError("at least one loss weight must be positive")
        layers = [part.strip() for part in self.dinov2_layers.split(",") if part.strip()]
        if not layers or any(int(layer) < 0 for layer in layers):
            raise ValueError("dinov2_layers must contain non-negative layer indices")
        if self.dinov2_model not in {
            "dinov2_vits14",
            "dinov2_vitb14",
            "dinov2_vitb14_reg",
        }:
            raise ValueError("unsupported dinov2_model")
        if self.dinov2_feature_size <= 0 or self.dinov2_feature_size % 14:
            raise ValueError("dinov2_feature_size must be positive and divisible by 14")
        if self.dinov2_chunk_size < 0:
            raise ValueError("dinov2_chunk_size must be non-negative")
        if self.checkpoint_metric != "ssim_tie_psnr":
            raise ValueError("checkpoint_metric must be ssim_tie_psnr")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
